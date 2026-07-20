import argparse
import base64
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen


def build_payload(audio_path, uttid):
    audio_path = Path(audio_path)
    audio_base64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    return json.dumps(
        {
            "inputs": [
                {
                    "uttid": uttid or audio_path.stem,
                    "audio_base64": audio_base64,
                }
            ]
        }
    ).encode("utf-8")


def send_request(
    url,
    payload,
    timeout_s,
    *,
    open_url=urlopen,
    clock=time.perf_counter,
):
    request = Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = clock()
    with open_url(request, timeout=timeout_s) as response:
        result = json.loads(response.read())
    return result, clock() - started


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Send a WAV file to the FireRedLID FastAPI server",
    )
    parser.add_argument("audio", help="path to a 16 kHz mono WAV file")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/v1/lid",
        help="FireRedLID server endpoint",
    )
    parser.add_argument("--uttid", help="utterance id (defaults to file stem)")
    parser.add_argument(
        "--repeat", type=int, default=1, help="number of requests to send"
    )
    parser.add_argument(
        "--timeout", type=float, default=300.0, help="per-request timeout in seconds"
    )
    args = parser.parse_args(argv)
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def main(argv=None, request_sender=send_request):
    args = parse_args(argv)
    payload = build_payload(args.audio, args.uttid)
    for index in range(1, args.repeat + 1):
        result, latency_s = request_sender(
            args.url,
            payload,
            args.timeout,
        )
        print(f"request {index}: client_latency={latency_s:.3f}s")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
