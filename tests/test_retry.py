"""Tests for utils/retry.py retry_with_backoff decorator."""

from unittest.mock import MagicMock, call, patch

import pytest

from utils.retry import retry_with_backoff


def _mock_func(side_effect=None, return_value=None):
    """Return a MagicMock with __name__ set (required by the logger inside retry)."""
    m = MagicMock(side_effect=side_effect, return_value=return_value)
    m.__name__ = "mock_func"
    return m


def test_succeeds_on_first_attempt():
    func = _mock_func(return_value=42)
    decorated = retry_with_backoff()(func)
    with patch("time.sleep") as mock_sleep:
        result = decorated()
    assert result == 42
    func.assert_called_once()
    mock_sleep.assert_not_called()


def test_retries_on_specified_exception():
    func = _mock_func(side_effect=[ValueError("x"), ValueError("x"), "ok"])
    decorated = retry_with_backoff(exceptions=(ValueError,))(func)
    with patch("time.sleep") as mock_sleep:
        result = decorated()
    assert result == "ok"
    assert func.call_count == 3
    assert mock_sleep.call_count == 2


def test_raises_after_max_attempts():
    func = _mock_func(side_effect=RuntimeError("boom"))
    decorated = retry_with_backoff(max_attempts=3)(func)
    with patch("time.sleep") as mock_sleep:
        with pytest.raises(RuntimeError, match="boom"):
            decorated()
    assert func.call_count == 3
    assert mock_sleep.call_count == 2


def test_does_not_retry_unspecified_exception():
    func = _mock_func(side_effect=ValueError("nope"))
    decorated = retry_with_backoff(exceptions=(ConnectionError,))(func)
    with patch("time.sleep") as mock_sleep:
        with pytest.raises(ValueError, match="nope"):
            decorated()
    func.assert_called_once()
    mock_sleep.assert_not_called()


def test_exponential_backoff_delay_sequence():
    func = _mock_func(side_effect=[Exception("e"), Exception("e"), "done"])
    decorated = retry_with_backoff(base_delay_seconds=1.0, backoff_factor=3.0)(func)
    with patch("time.sleep") as mock_sleep:
        result = decorated()
    assert result == "done"
    assert mock_sleep.call_args_list == [call(1.0), call(3.0)]


def test_default_max_attempts_is_three():
    func = _mock_func(side_effect=OSError("fail"))
    with patch("time.sleep"):
        with pytest.raises(OSError):
            retry_with_backoff()(func)()
    assert func.call_count == 3


def test_wraps_preserves_function_metadata():
    def my_func():
        """My docstring."""

    decorated = retry_with_backoff()(my_func)
    assert decorated.__name__ == "my_func"
    assert decorated.__doc__ == "My docstring."
