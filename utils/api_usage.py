import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_FILENAME = "api_usage.json"


def _usage_path() -> Path:
    configured = os.environ.get("API_USAGE_PATH")
    if configured:
        return Path(configured)

    output_path = os.environ.get("OUTPUT_PATH")
    if output_path:
        return Path(output_path) / DEFAULT_FILENAME

    return Path.cwd() / DEFAULT_FILENAME


def reset_usage(path: Optional[Path] = None) -> Path:
    usage_path = path or _usage_path()
    usage_path.parent.mkdir(parents=True, exist_ok=True)
    usage_path.write_text(
        json.dumps(
            {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "by_provider": {},
                "by_endpoint": {},
                "by_model": {},
                "events": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return usage_path


def load_usage(path: Optional[Path] = None) -> Dict[str, Any]:
    usage_path = path or _usage_path()
    if not usage_path.exists():
        reset_usage(usage_path)

    try:
        return json.loads(usage_path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "by_provider": {},
            "by_endpoint": {},
            "by_model": {},
            "events": [],
        }


def record_api_call(
    provider: str,
    endpoint: str,
    model: Optional[str] = None,
    success: bool = True,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    usage_path = _usage_path()
    usage_path.parent.mkdir(parents=True, exist_ok=True)
    data = load_usage(usage_path)

    data["total_calls"] = int(data.get("total_calls", 0)) + 1
    status_key = "successful_calls" if success else "failed_calls"
    data[status_key] = int(data.get(status_key, 0)) + 1

    for key, value in (
        ("by_provider", provider),
        ("by_endpoint", endpoint),
        ("by_model", model or "unknown"),
    ):
        bucket = data.setdefault(key, {})
        bucket[value] = int(bucket.get(value, 0)) + 1

    event = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "provider": provider,
        "endpoint": endpoint,
        "model": model or "unknown",
        "success": success,
    }
    if error:
        event["error"] = error[:500]
    data.setdefault("events", []).append(event)

    usage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def usage_summary(path: Optional[Path] = None) -> str:
    data = load_usage(path)
    return (
        f"API calls total={data.get('total_calls', 0)}, "
        f"success={data.get('successful_calls', 0)}, "
        f"failed={data.get('failed_calls', 0)}"
    )
