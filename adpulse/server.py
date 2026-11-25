"""HTTP & AWS Lambda Server Router for AdPulse v2 (`adpulse/server.py`).

Exposes all AdPulse v1 and v2 API routes:
  - `GET  /api/health`
  - `GET  /api/bootstrap`
  - `POST /api/storefront/search`
  - `POST /api/pacing/simulate`
  - `POST /api/copilot/recommend`
  - `GET|POST /api/cax/recommend`
  - `GET|POST /api/offsearch/pdp`
  - `GET|POST /api/pal/calibrate`
  - `POST /api/run`
"""

from __future__ import annotations

import os

from .web_api import (
    AdPulseStudioRuntime,
    CORS_HEADERS,
    LocalAdPulseServerHandler,
    STUDIO,
    lambda_handler,
    route_api_request,
    run_local_server,
)

__all__ = [
    "AdPulseStudioRuntime",
    "CORS_HEADERS",
    "LocalAdPulseServerHandler",
    "STUDIO",
    "lambda_handler",
    "route_api_request",
    "run_local_server",
]


if __name__ == "__main__":
    run_local_server(port=int(os.environ.get("PORT", "8086")))
