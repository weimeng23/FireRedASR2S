"""Teacher-forced AED scoring with independent audio and candidate batches."""

import math
import threading
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .config import Settings


class InputError(ValueError):
    pass


class InputTooLarge(InputError):
    pass


@dataclass(frozen=True)
class PreparedInput:
    uid: str
    feature: torch.Tensor
    duration_s: float
    texts: tuple[str, ...]
    targets: tuple[tuple[int, ...], ...]

    @property
    def token_count(self):
        return sum(map(len, self.targets))


class PPLScorer:
    def __init__(self, settings: Settings, *, wrapper=None):
        self.settings = settings
        if wrapper is None:
            from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config

            if not settings.model.model_dir:
                raise ValueError("model_dir is required")
            # Set encoder/decoder precision separately, without the wrapper's auto dtype.
            wrapper = FireRedAsr2.from_pretrained(
                "aed", settings.model.model_dir, FireRedAsr2Config(use_gpu=False)
            )
        self.wrapper = wrapper
        self.device = torch.device("cuda" if settings.runtime.use_gpu else "cpu")
        dtypes = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        self.encoder_dtype = dtypes[settings.runtime.encoder_precision]
        self.decoder_dtype = dtypes[settings.runtime.decoder_precision]
        self.encoder = wrapper.model.encoder.eval().to(self.device, self.encoder_dtype)
        self.decoder = wrapper.model.decoder.eval().to(self.device, self.decoder_dtype)
        self.tokenizer = wrapper.tokenizer
        self._local = threading.local()
        self.stats = {"encoder_batches": 0, "decoder_batches": 0, "audios": 0,
                      "candidates": 0, "oom_splits": 0}

    def prepare(self, uid, waveform, sample_rate, texts):
        """CPU worker entry point; each worker owns the mutable FBank extractor."""
        if not hasattr(self._local, "extractor"):
            from pathlib import Path
            from fireredasr2s.fireredasr2.data.asr_feat import ASRFeatExtractor

            self._local.extractor = ASRFeatExtractor(
                str(Path(self.settings.model.model_dir) / "cmvn.ark")
            )
        feats, lengths, durations, _, _ = self._local.extractor(
            [(sample_rate, waveform)], [uid]
        )
        if feats is None:
            raise InputError("audio is too short to extract features")
        targets = []
        for text in texts:
            _, ids = self.tokenizer.tokenize(text)
            ids = list(ids)
            if self.settings.model.include_eos:
                ids.append(self.decoder.eos_id)
            if not ids or all(token == self.decoder.pad_id for token in ids):
                raise InputError("candidate has no scored tokens; enable include_eos or supply text")
            if len(ids) > self.settings.model.max_text_tokens:
                raise InputTooLarge("candidate exceeds max_text_tokens; text is not truncated")
            if len(ids) > self.decoder.positional_encoding.pe.size(1):
                raise InputTooLarge("candidate exceeds decoder position limit")
            targets.append(tuple(ids))
        return PreparedInput(uid, feats[0, :int(lengths[0])], durations[0],
                             tuple(texts), tuple(targets))

    def process_features(self, items):
        if not items:
            return []
        # Leave failed tensor frames before attempting recovery, so their memory is released.
        try:
            return self._score_batch(items)
        except torch.cuda.OutOfMemoryError:
            if len(items) == 1:
                raise
        self.stats["oom_splits"] += 1
        torch.cuda.empty_cache()
        middle = len(items) // 2
        return self.process_features(items[:middle]) + self.process_features(items[middle:])

    @torch.inference_mode()
    def _score_batch(self, items):
        feats = pad_sequence([item.feature for item in items], batch_first=True)
        lengths = torch.tensor([len(item.feature) for item in items], device=self.device)
        encoder_out, encoder_lengths, encoder_mask = self.encoder(
            feats.to(self.device, self.encoder_dtype), lengths
        )
        self.stats["encoder_batches"] += 1
        self.stats["audios"] += len(items)
        # Position in this batch, never uid, identifies an audio: uids may repeat across callers.
        rows = [(audio_index, candidate_index, target)
                for audio_index, item in enumerate(items)
                for candidate_index, target in enumerate(item.targets)]
        enc_lengths = encoder_lengths.cpu().tolist()
        rows.sort(key=lambda row: (len(row[2]), enc_lengths[row[0]]))
        results = [[None] * len(item.texts) for item in items]
        for batch in self._candidate_batches(rows, enc_lengths):
            self._decode_with_recovery(batch, encoder_out, encoder_mask, enc_lengths, items, results)
        return results

    def _candidate_batches(self, rows, enc_lengths):
        config = self.settings.scheduler
        batch, max_text, max_audio = [], 0, 0
        for row in rows:
            text_len, audio_len = len(row[2]), enc_lengths[row[0]]
            next_text, next_audio = max(max_text, text_len), max(max_audio, audio_len)
            count = len(batch) + 1
            fits = (count <= config.decoder_max_batch_size
                    and count * next_text <= config.decoder_max_padded_tokens
                    and count * (next_text ** 2 + next_text * next_audio)
                    <= config.decoder_max_attention_elements)
            if batch and not fits:
                yield batch
                batch, max_text, max_audio = [], 0, 0
            if (text_len > config.decoder_max_padded_tokens
                    or text_len ** 2 + text_len * audio_len > config.decoder_max_attention_elements):
                raise InputTooLarge("one candidate exceeds decoder batch budget")
            batch.append(row)
            max_text, max_audio = max(max_text, text_len), max(max_audio, audio_len)
        if batch:
            yield batch

    def _decode_with_recovery(self, batch, encoder_out, encoder_mask, enc_lengths, items, results):
        try:
            self._decode(batch, encoder_out, encoder_mask, enc_lengths, items, results)
            return
        except torch.cuda.OutOfMemoryError:
            if len(batch) == 1:
                raise
        self.stats["oom_splits"] += 1
        torch.cuda.empty_cache()
        middle = len(batch) // 2
        self._decode_with_recovery(batch[:middle], encoder_out, encoder_mask, enc_lengths, items, results)
        self._decode_with_recovery(batch[middle:], encoder_out, encoder_mask, enc_lengths, items, results)

    def _decode(self, batch, encoder_out, encoder_mask, enc_lengths, items, results):
        decoder = self.decoder
        ys_in = pad_sequence([
            torch.tensor((decoder.sos_id,) + row[2][:-1], device=self.device) for row in batch
        ], batch_first=True, padding_value=decoder.pad_id)
        ys_out = pad_sequence([
            torch.tensor(row[2], device=self.device) for row in batch
        ], batch_first=True, padding_value=decoder.pad_id)
        indices = torch.tensor([row[0] for row in batch], device=self.device)
        max_audio = max(enc_lengths[row[0]] for row in batch)
        enc = encoder_out[:, :max_audio].index_select(0, indices).to(self.decoder_dtype)
        mask = encoder_mask[:, :, :max_audio].index_select(0, indices)
        target_mask = decoder.ignored_target_position_is_0(ys_in, decoder.pad_id)
        output = decoder.dropout(decoder.tgt_word_emb(ys_in) * decoder.scale
                                 + decoder.positional_encoding(ys_in))
        for layer in decoder.layer_stack:
            output = layer(output, enc, target_mask, mask)
        output = decoder.layer_norm_out(output)
        # Keep the legacy temperature application and full vocabulary normalization.
        logits = (decoder.tgt_word_prj(output) / self.settings.model.softmax_smoothing).float()
        losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), ys_out.reshape(-1),
                                 ignore_index=decoder.pad_id, reduction="none").view(len(batch), -1)
        valid = ys_out.ne(decoder.pad_id)
        total = losses.masked_fill(~valid, 0.0).sum(dim=1)
        counts = valid.sum(dim=1)
        averages = total / counts.clamp_min(1)
        # One host transfer per batch instead of per-candidate .item() synchronizations.
        values = torch.stack((averages, total, counts), dim=1).cpu().tolist()
        self.stats["decoder_batches"] += 1
        self.stats["candidates"] += len(batch)
        for (audio_index, candidate_index, _), (avg, nll, count) in zip(batch, values, strict=True):
            finite = math.isfinite(avg) and math.isfinite(nll) and count > 0
            results[audio_index][candidate_index] = {
                "uid": items[audio_index].uid,
                "text": items[audio_index].texts[candidate_index],
                "ppl": round(math.exp(min(avg, 80.0)), 6) if finite else None,
                "avg_nll": round(avg, 6) if finite else None,
                "total_nll": round(nll, 6) if finite else None,
                "token_count": int(count),
            }
