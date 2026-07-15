#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import onnx
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fireredasr2s.fireredlid.lid import load_fireredlid_model


class EncoderExportWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, features, feature_lengths):
        return self.encoder(features, feature_lengths)


def finalize_external_data_layout(staged_path, output_path):
    staged_path = Path(staged_path)
    output_path = Path(output_path)
    try:
        data_directory = staged_path.parent.relative_to(output_path.parent)
    except ValueError as exc:
        raise ValueError(
            "staged ONNX must be inside the output directory"
        ) from exc
    if data_directory == Path(".") or ".." in data_directory.parts:
        raise ValueError("staged ONNX must be inside a data subdirectory")

    model = onnx.load(staged_path, load_external_data=False)
    external_locations = set()
    for tensor in model.graph.initializer:
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        for item in tensor.external_data:
            if item.key != "location":
                continue
            source_location = Path(item.value)
            if source_location.is_absolute() or ".." in source_location.parts:
                raise ValueError(
                    f"external data location must be relative: {item.value}"
                )
            external_path = staged_path.parent / source_location
            if not external_path.is_file():
                raise FileNotFoundError(
                    f"external data file does not exist: {external_path}"
                )
            external_locations.add(source_location)
            item.value = (data_directory / source_location).as_posix()

    onnx.save_model(model, staged_path)
    staged_path.replace(output_path)
    return len(external_locations)


def export_encoder(
    encoder,
    output_path,
    sample_features,
    sample_lengths,
):
    wrapper = EncoderExportWrapper(encoder.float().cpu().eval())
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_directory = output_path.parent / "data"
    data_directory.mkdir(parents=True, exist_ok=True)
    staged_path = data_directory / output_path.name
    torch.onnx.export(
        wrapper,
        (
            sample_features.float().cpu(),
            sample_lengths.long().cpu(),
        ),
        str(staged_path),
        input_names=["features", "feature_lengths"],
        output_names=[
            "encoder_outputs",
            "encoder_lengths",
            "encoder_mask",
        ],
        dynamic_axes={
            "features": {0: "batch", 1: "frames"},
            "feature_lengths": {0: "batch"},
            "encoder_outputs": {0: "batch", 1: "encoder_frames"},
            "encoder_lengths": {0: "batch"},
            "encoder_mask": {0: "batch", 2: "encoder_frames"},
        },
        opset_version=17,
        do_constant_folding=True,
        external_data=True,
        dynamo=False,
    )
    finalize_external_data_layout(staged_path, output_path)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export the original FireRedLID Encoder to dynamic FP32 ONNX."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-frames", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model_dir) / "model.pth.tar"
    model = load_fireredlid_model(model_path)
    output_path = Path(args.output_dir) / "encoder.fp32.onnx"
    sample_features = torch.zeros(1, args.sample_frames, 80)
    sample_lengths = torch.tensor([args.sample_frames], dtype=torch.long)
    export_encoder(
        model.encoder,
        output_path,
        sample_features,
        sample_lengths,
    )
    print(output_path)


if __name__ == "__main__":
    main()
