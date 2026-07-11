from __future__ import annotations

import uvicorn

from .config import get_settings


def main() -> None:
    s = get_settings()
    uvicorn.run(
        "app.main:app",
        host=s.host,
        port=s.port,
        log_level=s.log_level,
        reload=False,
    )


if __name__ == "__main__":
    main()
