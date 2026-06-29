import io
import json
import logging

import pytest

from config import config

def test_json_logging_format():
    """Verify that score events are logged as valid JSON with required fields."""
    # We create a custom handler to capture the exact formatted string
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    
    from pythonjsonlogger import jsonlogger
    formatter = jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    
    logger = logging.getLogger("test_json_logger")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    # Simulate a score event
    logger.info("Wallet scored", extra={
        "wallet": "GABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890ABCDEFGHIJKLMNOP",
        "score": 85.5,
        "latency_ms": 12.34,
        "model_version": "v1.2.3",
        "asset_pair": "USDC:GA5Z.../XLM:native"
    })
    
    log_output = stream.getvalue().strip()
    assert log_output, "Log output should not be empty"
    
    try:
        log_data = json.loads(log_output)
    except json.JSONDecodeError:
        pytest.fail(f"Log output is not valid JSON: {log_output}")
        
    assert log_data["message"] == "Wallet scored"
    assert log_data["wallet"] == "GABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890ABCDEFGHIJKLMNOP"
    assert log_data["score"] == 85.5
    assert log_data["latency_ms"] == 12.34
    assert log_data["model_version"] == "v1.2.3"
    assert log_data["asset_pair"] == "USDC:GA5Z.../XLM:native"
    assert "asctime" in log_data
    assert "levelname" in log_data

def test_json_logging_error_format():
    """Verify that errors are logged with tracebacks and required fields."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    
    from pythonjsonlogger import jsonlogger
    formatter = jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    
    logger = logging.getLogger("test_json_logger_error")
    logger.setLevel(logging.ERROR)
    logger.addHandler(handler)
    
    try:
        raise ValueError("Something went terribly wrong")
    except Exception as e:
        logger.error("Scoring error", exc_info=True, extra={
            "wallet": "G123",
            "error_type": type(e).__name__,
            "error_message": str(e)
        })
        
    log_output = stream.getvalue().strip()
    log_data = json.loads(log_output)
    
    assert log_data["message"] == "Scoring error"
    assert log_data["wallet"] == "G123"
    assert log_data["error_type"] == "ValueError"
    assert log_data["error_message"] == "Something went terribly wrong"
    # python-json-logger puts traceback in exc_info
    assert "exc_info" in log_data
    assert "ValueError: Something went terribly wrong" in log_data["exc_info"]
