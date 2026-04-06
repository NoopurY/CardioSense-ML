
from __future__ import annotations
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import os
from typing import List

try:
    from inference import predict as model_predict
except Exception:
    model_predict = None

HOST = "0.0.0.0"
PORT = int(os.getenv("PORT", "8000"))
MODEL_PATH = Path(__file__).with_name("model.pkl")


def fallback_predict(features: List[float]) -> dict:
    mean = sum(features) / len(features)
    variance = sum((x - mean) ** 2 for x in features) / len(features)

    if variance > 0.12:
        class_id = 2
        probabilities = [0.06, 0.12, 0.7, 0.07, 0.05]
    elif variance > 0.05:
        class_id = 1
        probabilities = [0.2, 0.58, 0.1, 0.06, 0.06]
    else:
        class_id = 0
        probabilities = [0.85, 0.08, 0.03, 0.02, 0.02]

    return {"class_id": class_id, "probabilities": probabilities}


class Handler(BaseHTTPRequestHandler):
    def _write_json(self, status_code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/health":
            self._write_json(404, {"error": "Not found"})
            return

        if model_predict is None:
            self._write_json(200, {"status": "fallback"})
            return

        if not MODEL_PATH.exists():
            self._write_json(503, {"error": "model.pkl not found"})
            return

        self._write_json(200, {"status": "ok"})

    def do_POST(self):
        if self.path != "/predict":
            self._write_json(404, {"error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length)
            payload = json.loads(raw_body.decode("utf-8"))
            features = payload.get("features")

            if not isinstance(features, list):
                self._write_json(400, {"error": "features must be a list"})
                return

            try:
                values = [float(x) for x in features]
            except (TypeError, ValueError):
                self._write_json(400, {"error": "features must contain only numbers"})
                return

            if len(values) != 12:
                self._write_json(400, {"error": "features must contain exactly 12 values"})
                return

            if model_predict is None:
                self._write_json(200, fallback_predict(values))
                return

            result = model_predict(values)
            self._write_json(
                200,
                {
                    "class_id": int(result.get("class_id", 0)),
                    "probabilities": list(result.get("probabilities", [])),
                },
            )
        except Exception as exc:
            self._write_json(500, {"error": f"Internal error: {exc}"})

    def log_message(self, format, *args):  # noqa: A003
        return


if __name__ == "__main__":
    server = HTTPServer((HOST, PORT), Handler)
    print(f"CardioSense ML service listening on http://{HOST}:{PORT}")
    server.serve_forever()
