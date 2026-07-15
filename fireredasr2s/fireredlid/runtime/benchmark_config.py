def validate_backend_device_precision(backends, device, precision):
    if "tensorrt" in backends and (
        device != "cuda" or precision != "fp16"
    ):
        raise ValueError(
            "TensorRT benchmarks require device=cuda and precision=fp16"
        )


def resolve_device(device, cuda_available):
    if device == "auto":
        return "cuda" if cuda_available else "cpu"
    return device
