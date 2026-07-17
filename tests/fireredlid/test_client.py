import base64
import importlib.util
import json
from pathlib import Path


SCRIPT = Path("runtime/fireredlid/client.py")


def load_client_module():
    spec = importlib.util.spec_from_file_location(
        "fireredlid_client",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *unused_args):
        return False

    def read(self):
        return self._payload


def test_build_payload_encodes_audio_and_defaults_uttid_to_filename(tmp_path):
    module = load_client_module()
    audio_path = tmp_path / "hello.wav"
    audio_path.write_bytes(b"wav-bytes")

    payload = json.loads(module.build_payload(audio_path, uttid=None))

    assert payload == {
        "inputs": [
            {
                "uttid": "hello",
                "audio_base64": base64.b64encode(b"wav-bytes").decode(
                    "ascii"
                ),
            }
        ]
    }


def test_send_request_posts_json_and_measures_client_latency():
    module = load_client_module()
    captured = {}
    times = iter((10.0, 10.25))

    def open_url(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse({"backend": "eager", "results": []})

    result, latency_s = module.send_request(
        "http://127.0.0.1:12345/v1/lid",
        b'{"inputs":[]}',
        timeout_s=30.0,
        open_url=open_url,
        clock=lambda: next(times),
    )

    request = captured["request"]
    assert request.full_url == "http://127.0.0.1:12345/v1/lid"
    assert request.get_method() == "POST"
    assert request.get_header("Content-type") == "application/json"
    assert request.data == b'{"inputs":[]}'
    assert captured["timeout"] == 30.0
    assert result == {"backend": "eager", "results": []}
    assert latency_s == 0.25


def test_main_repeats_requests_and_prints_each_response(tmp_path, capsys):
    module = load_client_module()
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"audio")
    calls = []

    def send(url, payload, timeout_s):
        calls.append((url, json.loads(payload), timeout_s))
        return {"backend": "eager", "results": []}, 0.125

    module.main(
        [
            str(audio_path),
            "--url",
            "http://server:8000/v1/lid",
            "--uttid",
            "custom-id",
            "--repeat",
            "2",
            "--timeout",
            "60",
        ],
        request_sender=send,
    )

    assert len(calls) == 2
    assert calls[0] == (
        "http://server:8000/v1/lid",
        {
            "inputs": [
                {
                    "uttid": "custom-id",
                    "audio_base64": base64.b64encode(b"audio").decode(
                        "ascii"
                    ),
                }
            ]
        },
        60.0,
    )
    output = capsys.readouterr().out
    assert output.count("client_latency=0.125s") == 2
    assert output.count('"backend": "eager"') == 2
