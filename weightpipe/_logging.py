"""Package logging: quiet by default, opt-in readable output."""

import logging
import sys
from typing import TextIO

ROOT_LOGGER_NAME = "weightpipe"
LOG_FORMAT = "%(asctime)s  %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_HANDLER_FLAG = "_weightpipe_handler"

_root = logging.getLogger(ROOT_LOGGER_NAME)
# Libraries should not emit anything unless the application asks for it.
_root.addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """Logger for a module inside the package (``weightpipe.<module>``)."""
    return logging.getLogger(name)


def setup_logging(
    level: int | str = "INFO",
    *,
    stream: TextIO | None = None,
    fmt: str = LOG_FORMAT,
    datefmt: str = DATE_FORMAT,
) -> logging.Logger:
    """Send weightpipe messages to ``stream`` with a compact format.

    Calling this repeatedly replaces the handler instead of stacking new ones,
    and it leaves the application's root logger configuration untouched.
    """
    for handler in [h for h in _root.handlers if getattr(h, _HANDLER_FLAG, False)]:
        _root.removeHandler(handler)

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    setattr(handler, _HANDLER_FLAG, True)

    _root.addHandler(handler)
    _root.setLevel(level)
    _root.propagate = False
    return _root


def set_log_level(level: int | str) -> None:
    """Change the verbosity of weightpipe messages."""
    _root.setLevel(level)
