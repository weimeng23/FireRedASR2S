"""One GPU worker, bounded admission, and cross-request audio micro-batches."""

import asyncio
import logging
from dataclasses import dataclass
from time import monotonic

import torch

from .config import Settings
from .scorer import InputError, InputTooLarge, PreparedInput

logger = logging.getLogger(__name__)


class QueueFullError(RuntimeError):
    pass


class SchedulerClosedError(RuntimeError):
    pass


@dataclass(eq=False)
class Reservation:
    owner: object
    candidates: int = 0
    tokens: int = 0
    active: bool = True


@dataclass(eq=False)
class Pending:
    item: PreparedInput
    future: asyncio.Future
    reservation: Reservation
    enqueued_at: float


class PPLBatchScheduler:
    def __init__(self, *, engine, executor, settings: Settings):
        self.engine, self.executor, self.settings = engine, executor, settings
        self._pending = []
        self._reservations = set()
        self._candidates = self._tokens = self._in_flight = 0
        self._wake = asyncio.Event()
        self._worker = None
        self._stopping = False
        self._failed = False
        self.stats = {"batches": 0, "requests": 0, "max_batch_size": 0,
                      "queue_wait_ms_total": 0.0, "inference_ms_total": 0.0}

    @property
    def is_healthy(self):
        return (not self._stopping and not self._failed and self._worker is not None
                and not self._worker.done())

    def snapshot(self):
        return {**self.stats, "admitted_requests": len(self._reservations),
                "pending_requests": len(self._pending), "in_flight_requests": self._in_flight,
                "admitted_candidates": self._candidates, "admitted_tokens": self._tokens}

    async def start(self):
        if self._worker is not None:
            raise RuntimeError("scheduler already started")
        self._worker = asyncio.create_task(self._run())

    def reserve(self):
        if not self.is_healthy:
            raise SchedulerClosedError("inference scheduler is not ready")
        if len(self._reservations) >= self.settings.server.queue_capacity:
            raise QueueFullError("PPL request queue is full")
        reservation = Reservation(self)
        self._reservations.add(reservation)
        return reservation

    def set_workload(self, reservation, candidates, tokens):
        if reservation.owner is not self or not reservation.active:
            raise ValueError("invalid reservation")
        if candidates < 0 or tokens < 0:
            raise ValueError("workload must be nonnegative")
        next_candidates = self._candidates - reservation.candidates + candidates
        next_tokens = self._tokens - reservation.tokens + tokens
        limits = self.settings.server
        if next_candidates > limits.queue_max_candidates or next_tokens > limits.queue_max_tokens:
            raise QueueFullError("PPL candidate/token queue budget is full")
        self._candidates, self._tokens = next_candidates, next_tokens
        reservation.candidates, reservation.tokens = candidates, tokens

    def release(self, reservation):
        if reservation.owner is not self:
            raise ValueError("reservation belongs to another scheduler")
        if reservation.active:
            self._reservations.remove(reservation)
            self._candidates -= reservation.candidates
            self._tokens -= reservation.tokens
            reservation.active = False

    def _bucket(self, item):
        for index, bucket in enumerate(self.settings.scheduler.buckets):
            if item.duration_s <= bucket.max_seconds:
                return index
        raise InputTooLarge("audio exceeds scheduler duration limits")

    def _fits_encoder(self, tasks):
        config = self.settings.scheduler
        frames = max(len(task.item.feature) for task in tasks)
        count = len(tasks)
        return (count * frames <= config.encoder_max_padded_frames
                and count * ((frames + 3) // 4) ** 2 <= config.encoder_max_attention_elements)

    def submit(self, reservation, item):
        if not self.is_healthy:
            raise SchedulerClosedError("inference scheduler is not ready")
        self._bucket(item)
        future = asyncio.get_running_loop().create_future()
        task = Pending(item, future, reservation, monotonic())
        if not self._fits_encoder([task]):
            raise InputTooLarge("audio exceeds encoder batch budget")
        encoded_length = (len(item.feature) + 3) // 4
        config = self.settings.scheduler
        if any(len(target) > config.decoder_max_padded_tokens
               or len(target) ** 2 + len(target) * encoded_length > config.decoder_max_attention_elements
               for target in item.targets):
            raise InputTooLarge("candidate exceeds decoder batch budget")
        self.set_workload(reservation, len(item.targets), item.token_count)
        self._pending.append(task)
        future.add_done_callback(lambda _: self._wake.set())
        self._wake.set()
        return future

    def _discard_cancelled(self):
        active = []
        for task in self._pending:
            if task.future.done():
                self.release(task.reservation)
            else:
                active.append(task)
        self._pending = active

    async def _next_batch(self):
        while True:
            self._discard_cancelled()
            if self._stopping:
                return None
            if not self._pending:
                self._wake.clear()
                await self._wake.wait()
                continue
            anchor = self._pending[0]  # Oldest request anchors selection: no long-audio starvation.
            bucket_index = self._bucket(anchor.item)
            maximum = self.settings.scheduler.buckets[bucket_index].max_batch_size
            batch = []
            budget_full = False
            for task in self._pending:
                if self._bucket(task.item) != bucket_index:
                    continue
                if len(batch) == maximum:
                    break
                if not self._fits_encoder(batch + [task]):
                    budget_full = True
                    continue
                batch.append(task)
            delay = self.settings.scheduler.max_batch_delay_ms / 1000
            remaining = anchor.enqueued_at + delay - monotonic()
            if len(batch) == maximum or budget_full or remaining <= 0:
                selected = set(batch)
                self._pending = [task for task in self._pending if task not in selected]
                return batch
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=remaining)
            except TimeoutError:
                pass

    def _fail_pending(self, error):
        pending, self._pending = self._pending, []
        for task in pending:
            if not task.future.done():
                task.future.set_exception(error)
            self.release(task.reservation)

    async def _run(self):
        loop = asyncio.get_running_loop()
        try:
            while (batch := await self._next_batch()) is not None:
                self._in_flight = len(batch)
                started = monotonic()
                self.stats["queue_wait_ms_total"] += sum(
                    (started - task.enqueued_at) * 1000 for task in batch
                )
                try:
                    results = await loop.run_in_executor(
                        self.executor, self.engine.process_features, [task.item for task in batch]
                    )
                    if len(results) != len(batch) or any(
                        len(result) != len(task.item.texts)
                        for task, result in zip(batch, results, strict=True)
                    ):
                        raise RuntimeError("PPL result count mismatch")
                    for task, result in zip(batch, results, strict=True):
                        if not task.future.done():
                            task.future.set_result(result)
                except Exception as error:
                    logger.exception("PPL inference batch failed")
                    for task in batch:
                        if not task.future.done():
                            task.future.set_exception(error)
                    # A single oversized batch can fail while the worker remains usable.
                    if not isinstance(error, (InputError, torch.cuda.OutOfMemoryError)):
                        raise
                finally:
                    self.stats["batches"] += 1
                    self.stats["requests"] += len(batch)
                    self.stats["max_batch_size"] = max(self.stats["max_batch_size"], len(batch))
                    self.stats["inference_ms_total"] += (monotonic() - started) * 1000
                    self._in_flight = 0
                    for task in batch:
                        self.release(task.reservation)
        except Exception:
            self._failed = True
            self._fail_pending(SchedulerClosedError("PPL worker failed; restart the service"))

    async def stop(self):
        self._stopping = True
        self._fail_pending(SchedulerClosedError("PPL service is stopping"))
        self._wake.set()
        if self._worker is not None:
            await self._worker
