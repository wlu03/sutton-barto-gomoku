"""Run with: python -m gomoku.web"""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(prog="gomoku.web")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true", help="auto-reload on file changes")
    args = p.parse_args()

    import uvicorn

    uvicorn.run(
        "gomoku.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
