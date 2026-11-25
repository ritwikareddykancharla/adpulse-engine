"""AWS Lambda Function URL & Local HTTP Handler (`adpulse/aws_lambda_handler.py`).

Pure Python HTTP handler compatible with AWS Lambda Function URLs (`us-west-2`,
Payload Format v2.0 and v1.0):
  - `GET  /`            -> Interactive HTML overview or JSON index
  - `GET  /api/health`  -> Runtime health & subsystem inventory
  - `POST /api/run`     -> Live execution for `demo-all | ranker | pacing | ope | diagnostics`
                           plus ad-hoc query ranking (`mode="rank-query"`).
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import platform
import time
from typing import Any, Dict
import urllib.parse

from . import __version__
from .catalog_ranker import ShopperQuery, TwoStageAdRanker, generate_synthetic_catalog_and_queries
from .cli import SUPPORTED_MODES, execute_cli_mode


DOCS_INDEX_CANDIDATES = (
    Path(__file__).resolve().parent.parent / "web" / "index.html",
    Path(__file__).resolve().parent.parent / "docs" / "index.html",
    Path(__file__).resolve().parent / "docs" / "index.html",
)

CORS_HEADERS: Dict[str, str] = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
}


def _json_response(payload: Dict[str, Any], status_code: int = 200) -> Dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {
            **CORS_HEADERS,
            "Content-Type": "application/json; charset=utf-8",
        },
        "body": json.dumps(payload, indent=2),
    }


def _html_response(html_body: str, status_code: int = 200) -> Dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {
            **CORS_HEADERS,
            "Content-Type": "text/html; charset=utf-8",
            "Cache-Control": "no-cache",
        },
        "body": html_body,
    }


def _load_index_html() -> str | None:
    for candidate in DOCS_INDEX_CANDIDATES:
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    return None


def _execute_payload(body: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(body.get("mode", "demo-all")).strip()
    seed = int(body.get("seed", 42))

    if mode in SUPPORTED_MODES:
        return execute_cli_mode(mode=mode, seed=seed)

    if mode == "rank-query":
        t0 = time.perf_counter()
        catalog, _ = generate_synthetic_catalog_and_queries(seed=seed)
        ranker = TwoStageAdRanker(catalog)
        raw_q = str(body.get("query", "wireless over ear headphones anc under 180"))
        max_price = float(body["max_price"]) if body.get("max_price") is not None else None
        q = ShopperQuery(
            query_id="LIVE-Q",
            raw_query=raw_q,
            target_category=body.get("category"),
            product_type_intent=str(body.get("product_type", "")),
            max_price=max_price,
            brand_intent=body.get("brand"),
            required_attributes=dict(body.get("required_attributes") or {}),
            relevant_sku_ids=(),
        )
        top_k = max(1, min(10, int(body.get("top_k", 3))))
        candidates = [asdict(c) for c in ranker.rank(q, top_k=top_k)]
        return {
            "engine": "AdPulse",
            "version": __version__,
            "mode": mode,
            "wall_time_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            "candidates": candidates,
        }

    raise ValueError(
        f"Unsupported mode '{mode}'. Valid modes: {', '.join(SUPPORTED_MODES)}, rank-query"
    )


def handler(event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """AWS Lambda Function URL entrypoint."""
    method = (
        event.get("requestContext", {}).get("http", {}).get("method")
        or event.get("httpMethod")
        or "GET"
    ).upper()
    raw_path = event.get("rawPath") or event.get("path") or "/"
    path = urllib.parse.urlparse(raw_path).path.rstrip("/") or "/"

    if method == "OPTIONS":
        return {"statusCode": 204, "headers": CORS_HEADERS, "body": ""}

    if method == "GET" and path == "/api/health":
        return _json_response(
            {
                "status": "ok",
                "engine": "AdPulse",
                "version": __version__,
                "python_version": platform.python_version(),
                "supported_modes": list(SUPPORTED_MODES) + ["rank-query"],
            }
        )

    if method == "GET" and path == "/":
        html = _load_index_html()
        if html is not None:
            return _html_response(html)
        return _json_response(
            {
                "engine": "AdPulse",
                "version": __version__,
                "endpoints": {
                    "health": "GET /api/health",
                    "run": "POST /api/run",
                },
                "supported_modes": list(SUPPORTED_MODES) + ["rank-query"],
            }
        )

    if method == "POST" and path in ("/", "/api/run"):
        raw_body = event.get("body") or "{}"
        if event.get("isBase64Encoded"):
            raw_body = base64.b64decode(raw_body).decode("utf-8")
        try:
            body = json.loads(raw_body) if isinstance(raw_body, str) and raw_body.strip() else {}
            result = _execute_payload(body)
            return _json_response(result, status_code=200)
        except (ValueError, json.JSONDecodeError) as exc:
            return _json_response({"error": str(exc)}, status_code=400)

    if path in (
        "/api/cax/recommend",
        "/api/offsearch/pdp",
        "/api/pal/calibrate",
        "/api/storefront/search",
        "/api/pacing/simulate",
        "/api/copilot/recommend",
        "/api/bootstrap",
    ):
        from .web_api import lambda_handler as web_lambda_handler

        return web_lambda_handler(event, context)

    return _json_response({"error": f"Route not found: {method} {path}"}, status_code=404)


def run_local_server(port: int = 8086) -> None:
    class _Handler(BaseHTTPRequestHandler):
        def do_OPTIONS(self) -> None:  # noqa: N802
            resp = handler({"httpMethod": "OPTIONS", "path": self.path})
            self._write(resp)

        def do_GET(self) -> None:  # noqa: N802
            resp = handler({"httpMethod": "GET", "path": self.path})
            self._write(resp)

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            resp = handler({"httpMethod": "POST", "path": self.path, "body": raw})
            self._write(resp)

        def _write(self, resp: Dict[str, Any]) -> None:
            self.send_response(int(resp["statusCode"]))
            for k, v in resp.get("headers", {}).items():
                self.send_header(k, v)
            self.end_headers()
            body = resp.get("body", "")
            if body:
                self.wfile.write(body.encode("utf-8"))

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    print(f"AdPulse local HTTP server listening on http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run AdPulse local HTTP server")
    ap.add_argument("--port", type=int, default=8086)
    parsed = ap.parse_args()
    run_local_server(port=parsed.port)
