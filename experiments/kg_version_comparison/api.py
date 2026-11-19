"""Recorded API calls and model capability probes for the paired KG study."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
import urllib.request

from common.models import credentials, QWENCLOUD_URL


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


class ProviderFailure(RuntimeError):
    pass


def request(config, messages, limit=None):
    value = {"model": config["model"], "messages": messages}
    if config.get("temperature") is not None:
        value["temperature"] = config["temperature"]
    cap = limit or config["max_output_tokens"]
    if config["transport"] == "openai":
        value.update(max_completion_tokens=cap, reasoning_effort="low", store=False,
                     service_tier="default")
    elif config["transport"] == "qwencloud":
        value.update(max_tokens=cap, reasoning_effort=config["reasoning"]["effort"],
                     extra_body={"enable_thinking": True})
    else:
        value.update(max_tokens=cap, extra_body={"provider": config["provider"]})
        if config.get("reasoning") is not None:
            value["extra_body"]["reasoning"] = config["reasoning"]
    return value


def reasoning_text(response):
    """Only expose reasoning actually returned by the API; never reconstruct it."""
    message = ((response.get("choices") or [{}])[0].get("message") or {})
    parts = []
    for key in ("reasoning", "reasoning_content"):
        if isinstance(message.get(key), str) and message[key] not in parts:
            parts.append(message[key])
    for detail in message.get("reasoning_details") or []:
        if isinstance(detail, dict):
            for key in ("text", "summary"):
                if isinstance(detail.get(key), str) and detail[key] not in parts:
                    parts.append(detail[key])
    return "\n\n".join(parts)


def has_completion(body):
    if not isinstance(body, dict):
        return False
    return bool(body.get("choices") or body.get("usage") or body.get("id"))


def call(config, messages, directory, *, stopped=lambda: False, limit=None):
    """Resume receipts; retry only definite HTTP rejections without a completion."""
    from openai import OpenAI

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = request(config, messages, limit)
    request_file, receipt_file = directory / "request.json", directory / "api_call.json"
    if request_file.exists() and read(request_file) != payload:
        raise RuntimeError("A saved request differs from the frozen inputs.")
    if receipt_file.exists():
        receipt = read(receipt_file)
        if receipt.get("response"):
            return receipt["response"]
        raise ProviderFailure(receipt.get("error", "Recorded API failure."))
    write(request_file, payload)
    key = credentials(config["transport"])
    if not key:
        raise ProviderFailure("API credential is unavailable.")
    url = {"openai": "https://api.openai.com/v1", "openrouter": "https://openrouter.ai/api/v1",
           "qwencloud": QWENCLOUD_URL}[config["transport"]]
    for index in range(3):
        folder = directory / "submissions" / f"submission_{index + 1:02d}"
        receipt_path = folder / "receipt.json"
        if receipt_path.exists():
            previous = read(receipt_path)
            if previous.get("response"):
                write(receipt_file, previous)
                return previous["response"]
            if not previous.get("retryable_http_rejection"):
                raise ProviderFailure(previous.get("error", "Recorded API failure."))
            continue
        if (folder / "submitted.json").exists():
            raise ProviderFailure("A submitted request has no receipt; reconcile before resubmission.")
        if stopped():
            raise InterruptedError("Study stopped before API submission.")
        if index:
            previous = read(directory / "submissions" / f"submission_{index:02d}" / "receipt.json")
            delay = max(0, 60 - (datetime.now(timezone.utc) -
                                datetime.fromisoformat(previous["received_at"])).total_seconds())
            deadline = time.monotonic() + delay
            while time.monotonic() < deadline:
                if stopped():
                    raise InterruptedError("Study stopped during HTTP retry backoff.")
                time.sleep(min(0.5, max(0, deadline - time.monotonic())))
        submitted = now()
        write(folder / "submitted.json", {"submitted_at": submitted, "base_url": url})
        started = time.monotonic()
        try:
            with OpenAI(api_key=key, base_url=url, timeout=900, max_retries=0) as client:
                raw = client.chat.completions.with_raw_response.create(**payload)
                response = raw.parse().model_dump(mode="json")
            if response.get("error"):
                raise ProviderFailure(str(response["error"]))
            receipt = {"response": response, "status_code": raw.status_code,
                       "request_id": raw.headers.get("x-request-id"),
                       "seconds": time.monotonic() - started,
                       "submitted_at": submitted, "received_at": now()}
            write(receipt_path, receipt)
            write(receipt_file, receipt)
            write(directory / "response.json", response)
            message = ((response.get("choices") or [{}])[0].get("message") or {})
            write(directory / "reasoning.json", {
                "available": bool(reasoning_text(response) or message.get("reasoning_details")),
                **{k: message[k] for k in ("reasoning", "reasoning_content", "reasoning_details") if k in message},
                "note": "API-returned content only; hidden reasoning is not available."})
            (directory / "reasoning.md").write_text(reasoning_text(response))
            content = message.get("content") or ""
            (directory / "answer.md").write_text(content if isinstance(content, str) else json.dumps(content))
            return response
        except Exception as error:
            status = getattr(error, "status_code", None)
            body = getattr(error, "body", None)
            safe_error = str(error).replace(key, "[REDACTED]")[:8000]
            safe_body = json.loads(json.dumps(body, default=str).replace(key, "[REDACTED]"))
            retryable = status == 429 and not has_completion(body)
            receipt = {"error": safe_error, "body": safe_body, "status_code": status,
                       "retryable_http_rejection": retryable,
                       "seconds": time.monotonic() - started,
                       "submitted_at": submitted, "received_at": now()}
            write(receipt_path, receipt)
            if not retryable or index == 2:
                write(receipt_file, receipt)
                raise ProviderFailure(safe_error) from None
    raise ProviderFailure("HTTP retry limit exhausted.")


def preflight(model, directory, *, gpt_temperature_exception=False, strict_low=False, omit_temperature=False):
    directory = Path(directory)
    destination = directory / "preflight.json"
    if destination.exists():
        previous = read(destination)
        if previous.get("status") == "ready":
            return previous
        if previous.get("status") != "awaiting_temperature_decision" or not gpt_temperature_exception:
            return previous
    config = {"model": model["id"], "transport": model["transport"], "temperature": None if omit_temperature else 0,
              "reasoning": {"effort": "low"}, "max_output_tokens": 65536}
    if config["transport"] == "qwencloud":
        config["base_url"] = QWENCLOUD_URL
    try:
        if config["transport"] == "openrouter":
            with urllib.request.urlopen("https://openrouter.ai/api/v1/models/" + model["id"] + "/endpoints", timeout=45) as response:
                endpoints = json.load(response)
            write(directory / "endpoints.json", endpoints)
            active = [x for x in endpoints["data"]["endpoints"]
                      if x.get("status") == 0 and "temperature" in x.get("supported_parameters", [])]
            if not active:
                raise ProviderFailure("No active temperature-capable endpoint.")
            caps = [x["max_completion_tokens"] for x in active if x.get("max_completion_tokens")]
            config["max_output_tokens"] = min(65536, max(caps)) if caps else 8192
            eligible = [x for x in active if not x.get("max_completion_tokens")
                        or x["max_completion_tokens"] >= config["max_output_tokens"]]
            config["provider"] = {"require_parameters": True, "allow_fallbacks": True}
            if any(x.get("quantization") in ("fp8", "bf16", "fp16", "fp32") for x in eligible):
                config["provider"]["quantizations"] = ["fp8", "bf16", "fp16", "fp32"]
            levels = [("low", {"effort": "low"}), ("minimal", {"effort": "minimal"}),
                      ("minimum_budget_1024", {"max_tokens": 1024}),
                      ("disabled", {"enabled": False})]
            if strict_low:
                levels = [("low", {"effort": "low"})]
        else:
            if gpt_temperature_exception:
                config["temperature"] = None
            levels = [("low", {"effort": "low"})]
        errors = []
        for label, reasoning in levels:
            config["reasoning"] = reasoning
            probe_name = label + ("_temperature_default" if config["temperature"] is None else "_temperature_0")
            try:
                response = call(config, [{"role": "user", "content": "Reply OK."}],
                                directory / "probes" / probe_name, limit=4096)
                if not response.get("choices"):
                    raise ProviderFailure("Probe returned no choices.")
                value = {"status": "ready", "created_at": now(), "config": config,
                         "reasoning_label": label, "excluded_from_coding_metrics": True,
                         "temperature_exception": config["temperature"] is None,
                         "effective_temperature": 1 if config["temperature"] is None else 0,
                         "fallback_errors": errors}
                write(destination, value)
                return value
            except ProviderFailure as error:
                errors.append(str(error))
                if config["transport"] == "openai" and config["temperature"] == 0 and "temperature" in str(error):
                    value = {"status": "awaiting_temperature_decision", "errors": errors, "created_at": now()}
                    write(destination, value)
                    return value
                receipt = read(directory / "probes" / probe_name / "api_call.json")
                if receipt.get("status_code") not in (400, 404, 422):
                    raise
        raise ProviderFailure("No accepted reasoning/sampling configuration: " + str(errors))
    except Exception as error:
        value = {"status": "preflight_error", "error": str(error)[:8000], "created_at": now()}
        write(destination, value)
        return value
