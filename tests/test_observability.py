# Tests for centralized structured logging.

import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jyagent.observability import log_event, setup_logging


def test_setup_logging_creates_jsonl_file_and_is_idempotent(tmp_path):
    logger = logging.getLogger("jyagent")
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate

    log_file = tmp_path / "logs" / "jyagent.jsonl"

    try:
        setup_logging(level="INFO", log_file=log_file)
        first_managed = [handler for handler in logger.handlers if getattr(handler, "_jyagent_managed", False)]
        setup_logging(level="INFO", log_file=log_file)
        second_managed = [handler for handler in logger.handlers if getattr(handler, "_jyagent_managed", False)]

        assert len(first_managed) == 2
        assert len(second_managed) == 2

        log_event(logger, logging.INFO, "test.event", value="ok")
        for handler in logger.handlers:
            handler.flush()

        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        payload = json.loads(lines[0])
        assert payload["event"] == "test.event"
        assert payload["payload"]["value"] == "ok"
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
        for handler in original_handlers:
            logger.addHandler(handler)
        logger.setLevel(original_level)
        logger.propagate = original_propagate
