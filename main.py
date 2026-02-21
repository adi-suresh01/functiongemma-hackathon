import sys
from pathlib import Path
import json, os, re, time
from difflib import SequenceMatcher


def _configure_cactus_import_path():
    root = Path(__file__).resolve().parent
    candidates = []

    env_path = os.environ.get("CACTUS_PYTHON_SRC")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.extend([
        root / "cactus" / "python" / "src",
        root.parent / "cactus" / "python" / "src",
    ])

    for path in candidates:
        if (path / "cactus.py").exists() or (path / "cactus").is_dir():
            sys.path.insert(0, str(path))
            return


def _resolve_functiongemma_path():
    root = Path(__file__).resolve().parent
    candidates = []

    env_path = os.environ.get("CACTUS_FUNCTIONGEMMA_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.extend([
        root / "cactus" / "weights" / "functiongemma-270m-it",
        root.parent / "cactus" / "weights" / "functiongemma-270m-it",
    ])

    for path in candidates:
        if path.is_dir():
            return str(path)

    # Keep backward compatibility with original relative path.
    return "cactus/weights/functiongemma-270m-it"


_configure_cactus_import_path()
functiongemma_path = _resolve_functiongemma_path()

try:
    from cactus import cactus_init, cactus_complete, cactus_destroy
except ModuleNotFoundError as exc:
    if exc.name != "cactus":
        raise
    raise ModuleNotFoundError(
        "Could not import 'cactus'. Set CACTUS_PYTHON_SRC to your cactus/python/src path "
        "(for example: /Users/adi/Desktop/cactus/python/src), or clone cactus into "
        "./cactus inside this repo."
    ) from exc


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = cactus_init(functiongemma_path)

    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": "You are a helpful assistant that can use tools."}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    cactus_destroy(model)

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
        }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    try:
        from google import genai
        from google.genai import types
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Cloud fallback requires the `google-genai` package. "
            "Install it with: pip install google-genai"
        ) from exc

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Cloud fallback requires GEMINI_API_KEY to be set.")

    client = genai.Client(api_key=api_key)

    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            )
            for t in tools
        ])
    ]

    contents = [m["content"] for m in messages if m["role"] == "user"]

    start_time = time.time()

    gemini_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(tools=gemini_tools),
    )

    total_time_ms = (time.time() - start_time) * 1000

    function_calls = []
    for candidate in gemini_response.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append({
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args),
                })

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


_STOPWORDS = {
    "a", "an", "and", "at", "be", "by", "for", "from", "get", "in", "is", "it",
    "me", "my", "of", "on", "or", "the", "to", "up", "with", "what", "whats",
    "what's", "please", "current", "like", "can", "you",
}

_ACTION_HINTS = {
    "weather": ["weather", "forecast", "temperature", "rain", "sunny", "city", "location"],
    "alarm": ["alarm", "wake", "wake up", "morning", "hour", "minute", "clock"],
    "timer": ["timer", "countdown", "minutes", "seconds", "duration"],
    "message": ["message", "text", "sms", "send", "recipient", "contact"],
    "reminder": ["remind", "reminder", "remember", "title", "time"],
    "search": ["search", "find", "lookup", "look up", "contact", "query"],
    "contact": ["contact", "contacts", "person", "name", "find", "lookup"],
    "play": ["play", "music", "song", "playlist", "listen", "track"],
}


def _extract_user_text(messages):
    parts = []
    for m in messages:
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, str):
                parts.append(content.strip())
    return " ".join(p for p in parts if p).strip()


def _norm_key(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _split_words(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _clean_span(text):
    return str(text).strip().strip("\"'").strip(".,!?;:").strip()


def _tool_schema(tool):
    params = tool.get("parameters", {})
    if not isinstance(params, dict):
        return {}, []
    properties = params.get("properties", {})
    required = params.get("required", [])
    if not isinstance(properties, dict):
        properties = {}
    if not isinstance(required, list):
        required = []
    return properties, required


def _match_tool_name(name, tools_by_name):
    if name in tools_by_name:
        return name
    if not name:
        return None
    target = _norm_key(name)
    for candidate in tools_by_name:
        if _norm_key(candidate) == target:
            return candidate
    best_name, best_ratio = None, 0.0
    for candidate in tools_by_name:
        ratio = SequenceMatcher(None, target, _norm_key(candidate)).ratio()
        if ratio > best_ratio:
            best_name, best_ratio = candidate, ratio
    if best_ratio >= 0.78:
        return best_name
    return None


def _match_arg_key(key, properties):
    if key in properties:
        return key
    target = _norm_key(key)
    if not target:
        return None

    for candidate in properties:
        if _norm_key(candidate) == target:
            return candidate

    best_key, best_ratio = None, 0.0
    for candidate in properties:
        ratio = SequenceMatcher(None, target, _norm_key(candidate)).ratio()
        if ratio > best_ratio:
            best_key, best_ratio = candidate, ratio
    if best_ratio >= 0.75:
        return best_key
    return None


def _extract_time_tuple(text):
    match = re.search(r"\b(\d{1,2})(?::([0-5]\d))?\s*(am|pm)\b", text, flags=re.IGNORECASE)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        am_pm = match.group(3).lower()
        if am_pm == "am" and hour == 12:
            hour = 0
        elif am_pm == "pm" and hour != 12:
            # Keep 24h integer representation for numeric fields.
            hour += 12
        return hour, minute

    match = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text)
    if match:
        return int(match.group(1)), int(match.group(2))

    return None


def _extract_time_text(text):
    match = re.search(r"\b(\d{1,2})(?::([0-5]\d))?\s*(AM|PM)\b", text, flags=re.IGNORECASE)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        am_pm = match.group(3).upper()
        return f"{hour}:{minute:02d} {am_pm}"

    match = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text)
    if match:
        return f"{int(match.group(1))}:{int(match.group(2)):02d}"

    return None


def _extract_int_for_key(key, text):
    key_l = key.lower()
    time_tuple = _extract_time_tuple(text)

    if key_l == "hour" and time_tuple:
        return time_tuple[0]
    if key_l == "minute" and time_tuple:
        return time_tuple[1]

    if "minute" in key_l:
        match = re.search(r"(-?\d+)\s*(?:minutes?|mins?)\b", text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

    if "second" in key_l:
        match = re.search(r"(-?\d+)\s*(?:seconds?|secs?)\b", text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

    match = re.search(r"-?\d+", text)
    if match:
        return int(match.group(0))
    return None


def _extract_string_for_key(key, call_name, text):
    key_l = key.lower()
    call_l = str(call_name).lower()

    if key_l in {"location", "city", "place"}:
        patterns = [
            r"(?:weather|forecast|temperature)\s+(?:in|at|for)\s+([a-z0-9' \-]+?)(?:[,.!?]|$|\band\b)",
            r"(?:in|at|for)\s+([a-z0-9' \-]+?)(?:[,.!?]|$|\band\b)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return _clean_span(match.group(1)).title()

    if key_l in {"recipient", "contact", "query", "name", "person"}:
        patterns = [
            r"(?:message|text|send(?:\s+a)?\s+message)\s+to\s+([a-z][a-z' \-]+?)(?:\s+saying\b|[,.!?]|$|\band\b)",
            r"(?:find|look up|search(?: for)?)\s+([a-z][a-z' \-]+?)(?:\s+in\b|[,.!?]|$|\band\b)",
            r"\bto\s+([a-z][a-z' \-]+?)(?:\s+saying\b|[,.!?]|$|\band\b)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                candidate = _clean_span(match.group(1))
                if candidate and candidate.lower() not in {"him", "her", "them", "me", "you"}:
                    return candidate.title()

    if key_l in {"message", "text", "body", "content"}:
        patterns = [
            r"(?:saying|that says|saying that)\s+(.+?)(?:[,.!?]|$|\band\b)",
            r"(?:message|text)\s+(.+?)(?:[,.!?]|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return _clean_span(match.group(1))

    if key_l in {"song", "music", "track", "playlist"}:
        patterns = [
            r"\bplay\s+(?:some\s+)?(.+?)(?:\s+music\b|[,.!?]|$|\band\b)",
            r"\blisten to\s+(.+?)(?:[,.!?]|$|\band\b)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return _clean_span(match.group(1))

    if key_l == "title" or ("reminder" in call_l and key_l in {"task", "subject"}):
        pattern = r"\bremind me(?:\s+to|\s+about)?\s+(.+?)(?:\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)\b|[,.!?]|$|\band\b)"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _clean_span(match.group(1))

    if key_l in {"time", "when", "datetime"}:
        return _extract_time_text(text)

    quote_match = re.search(r"\"([^\"]+)\"|'([^']+)'", text)
    if quote_match:
        return _clean_span(quote_match.group(1) or quote_match.group(2))

    return None


def _infer_argument(key, arg_type, text, call_name):
    arg_type_l = str(arg_type).lower()
    if arg_type_l in {"integer", "number"}:
        return _extract_int_for_key(key, text)
    if arg_type_l == "boolean":
        lower = text.lower()
        if re.search(r"\b(true|yes|on|enable)\b", lower):
            return True
        if re.search(r"\b(false|no|off|disable)\b", lower):
            return False
        return None
    return _extract_string_for_key(key, call_name, text)


def _coerce_argument(value, arg_type, key, user_text):
    arg_type_l = str(arg_type).lower()
    key_l = key.lower()

    if arg_type_l == "integer":
        if isinstance(value, bool):
            parsed = int(value)
        elif isinstance(value, int):
            parsed = value
        elif isinstance(value, float):
            parsed = int(round(value))
        elif isinstance(value, str):
            match = re.search(r"-?\d+", value)
            parsed = int(match.group(0)) if match else None
        else:
            parsed = None

        if parsed is None:
            parsed = _extract_int_for_key(key_l, user_text)
        if parsed is None:
            return None

        if key_l in {"minutes", "minute", "hour"} and parsed < 0:
            parsed = abs(parsed)

        if key_l == "minutes" and parsed >= 300 and re.search(r"\bminute", user_text, flags=re.IGNORECASE):
            if parsed % 60 == 0:
                parsed = parsed // 60

        if key_l == "minute" and parsed > 59:
            parsed = parsed % 60

        return parsed

    if arg_type_l == "number":
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            match = re.search(r"-?\d+(?:\.\d+)?", value)
            if match:
                return float(match.group(0))
        inferred = _extract_int_for_key(key_l, user_text)
        if inferred is not None:
            return float(inferred)
        return None

    if arg_type_l == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lower = value.strip().lower()
            if lower in {"true", "yes", "1", "on"}:
                return True
            if lower in {"false", "no", "0", "off"}:
                return False
        return None

    if value is None:
        inferred = _extract_string_for_key(key_l, "", user_text)
        return inferred if inferred is not None else ""

    return str(value).strip()


def _sanitize_calls(calls, tools, user_text):
    tools_by_name = {t["name"]: t for t in tools}
    repaired = []

    for call in calls or []:
        if not isinstance(call, dict):
            continue

        name = _match_tool_name(call.get("name"), tools_by_name)
        if not name:
            continue

        tool = tools_by_name[name]
        properties, required = _tool_schema(tool)
        raw_args = call.get("arguments", {})
        if not isinstance(raw_args, dict):
            raw_args = {}

        args = {}
        for raw_key, raw_value in raw_args.items():
            key = _match_arg_key(raw_key, properties)
            if not key:
                continue
            expected_type = properties.get(key, {}).get("type", "string")
            coerced = _coerce_argument(raw_value, expected_type, key, user_text)
            if coerced is not None and coerced != "":
                args[key] = coerced

        for req_key in required:
            if req_key in args and args[req_key] not in ("", None):
                continue
            req_type = properties.get(req_key, {}).get("type", "string")
            inferred = _infer_argument(req_key, req_type, user_text, name)
            if inferred is None:
                continue
            coerced = _coerce_argument(inferred, req_type, req_key, user_text)
            if coerced is not None and coerced != "":
                args[req_key] = coerced

        if "hour" in args and "minute" in args:
            parsed_time = _extract_time_tuple(user_text)
            if parsed_time:
                args["hour"], args["minute"] = parsed_time

        missing = [k for k in required if k not in args or args[k] in ("", None)]
        if missing:
            continue

        repaired.append({
            "name": name,
            "arguments": args,
        })

    unique = []
    seen = set()
    for call in repaired:
        key = (call["name"], json.dumps(call["arguments"], sort_keys=True, ensure_ascii=False))
        if key in seen:
            continue
        seen.add(key)
        unique.append(call)
    return unique


def _contains_refusal(text):
    lower = str(text).lower()
    markers = [
        "i apologize",
        "i cannot",
        "i can't",
        "unable to",
        "not able to",
        "my capabilities are limited",
    ]
    return any(marker in lower for marker in markers)


def _split_clauses(user_text):
    if not user_text:
        return []
    clauses = re.split(r"\s*(?:,|;|\band then\b|\bthen\b|\band\b)\s*", user_text, flags=re.IGNORECASE)
    cleaned = [_clean_span(c) for c in clauses if _clean_span(c)]
    return cleaned or [_clean_span(user_text)]


def _tool_keywords(tool):
    words = set()
    words.update(_split_words(tool.get("name", "")))
    words.update(_split_words(tool.get("description", "")))

    properties, _ = _tool_schema(tool)
    for key, schema in properties.items():
        words.update(_split_words(key))
        if isinstance(schema, dict):
            words.update(_split_words(schema.get("description", "")))

    name_l = str(tool.get("name", "")).lower()
    for action, hints in _ACTION_HINTS.items():
        if action in name_l:
            for hint in hints:
                words.update(_split_words(hint))

    return {w for w in words if w and w not in _STOPWORDS}


def _tool_relevance(tool, text):
    text_l = str(text).lower()
    if not text_l.strip():
        return 0
    score = 0
    for kw in _tool_keywords(tool):
        if re.search(rf"\b{re.escape(kw)}\b", text_l):
            score += 2
        elif kw in text_l:
            score += 1
    return score


def _choose_clause_tools(clause, tools, max_tools=6):
    ranked = sorted(((tool, _tool_relevance(tool, clause)) for tool in tools), key=lambda x: x[1], reverse=True)
    if not ranked:
        return tools
    positive = [tool for tool, score in ranked if score > 0]
    if not positive:
        return tools
    return positive[:max_tools]


def _run_local_candidate(messages, tools, system_prompt=None, tool_rag_top_k=0):
    model = cactus_init(functiongemma_path)
    cactus_tools = [{"type": "function", "function": t} for t in tools]
    if not system_prompt:
        system_prompt = "You are a helpful assistant that can use tools."

    kwargs = {
        "tools": cactus_tools,
        "force_tools": True,
        "max_tokens": 256,
        "stop_sequences": ["<|im_end|>", "<end_of_turn>"],
        "tool_rag_top_k": tool_rag_top_k,
    }
    try:
        raw_str = cactus_complete(
            model,
            [{"role": "system", "content": system_prompt}] + messages,
            **kwargs,
        )
    except TypeError:
        kwargs.pop("tool_rag_top_k", None)
        raw_str = cactus_complete(
            model,
            [{"role": "system", "content": system_prompt}] + messages,
            **kwargs,
        )
    finally:
        cactus_destroy(model)

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
            "response": raw_str,
        }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
        "response": raw.get("response", ""),
    }


def _merge_calls(primary_calls, extra_calls):
    merged = []
    seen = set()
    for call in (primary_calls or []) + (extra_calls or []):
        key = (call.get("name"), json.dumps(call.get("arguments", {}), sort_keys=True, ensure_ascii=False))
        if key in seen:
            continue
        seen.add(key)
        merged.append(call)
    return merged


def _estimate_intent_count(user_text, tools):
    if not user_text:
        return 1
    clauses = [c for c in _split_clauses(user_text) if len(c.split()) >= 2]
    count = len(clauses) if clauses else 1
    if tools:
        count = min(count, len(tools))
    return max(1, count)


def _score_local_candidate(calls, tools, user_text, confidence=0.0, response=""):
    if not tools:
        return 0.0

    tools_by_name = {t["name"]: t for t in tools}
    valid_calls = 0
    required_total = 0
    required_hit = 0

    for call in calls:
        name = call.get("name")
        if name not in tools_by_name:
            continue
        valid_calls += 1
        properties, required = _tool_schema(tools_by_name[name])
        required_total += len(required)
        args = call.get("arguments", {})
        if not isinstance(args, dict):
            args = {}
        for req in required:
            if req in args and args[req] not in ("", None):
                required_hit += 1

    valid_ratio = (valid_calls / len(calls)) if calls else 0.0
    required_ratio = (required_hit / required_total) if required_total else (1.0 if calls else 0.0)
    expected = _estimate_intent_count(user_text, tools)
    coverage = min(len(calls), expected) / max(1, expected)
    overflow_penalty = max(0, len(calls) - expected) * 0.15
    refusal_penalty = 0.20 if _contains_refusal(response) and not calls else 0.0
    conf = max(0.0, min(1.0, float(confidence or 0.0)))

    score = (
        0.35 * valid_ratio +
        0.25 * required_ratio +
        0.25 * coverage +
        0.15 * conf -
        overflow_penalty -
        refusal_penalty
    )
    return max(0.0, min(1.0, score))


def _synthesize_calls_from_text(user_text, tools):
    if not user_text or not tools:
        return []

    synthesized = []
    clauses = _split_clauses(user_text)
    for clause in clauses:
        ranked = sorted(((tool, _tool_relevance(tool, clause)) for tool in tools), key=lambda x: x[1], reverse=True)
        if not ranked or ranked[0][1] <= 0:
            continue
        tool = ranked[0][0]
        properties, required = _tool_schema(tool)
        args = {}
        for req in required:
            arg_type = properties.get(req, {}).get("type", "string")
            value = _infer_argument(req, arg_type, clause, tool.get("name", ""))
            if value is None:
                value = _infer_argument(req, arg_type, user_text, tool.get("name", ""))
            if value is None:
                args = None
                break
            coerced = _coerce_argument(value, arg_type, req, user_text)
            if coerced is None or coerced == "":
                args = None
                break
            args[req] = coerced
        if args:
            synthesized.append({"name": tool["name"], "arguments": args})

    if not synthesized:
        ranked = sorted(((tool, _tool_relevance(tool, user_text)) for tool in tools), key=lambda x: x[1], reverse=True)
        if ranked and ranked[0][1] > 0:
            tool = ranked[0][0]
            properties, required = _tool_schema(tool)
            args = {}
            for req in required:
                arg_type = properties.get(req, {}).get("type", "string")
                value = _infer_argument(req, arg_type, user_text, tool.get("name", ""))
                if value is None:
                    args = None
                    break
                coerced = _coerce_argument(value, arg_type, req, user_text)
                if coerced is None or coerced == "":
                    args = None
                    break
                args[req] = coerced
            if args:
                synthesized.append({"name": tool["name"], "arguments": args})

    return synthesized


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """
    Hybrid routing with local-first repair/recovery, then selective cloud fallback.
    Keeps interface compatible with benchmark.py.
    """
    user_text = _extract_user_text(messages)
    expected_calls = _estimate_intent_count(user_text, tools)

    primary = _run_local_candidate(messages, tools, tool_rag_top_k=0)
    total_local_time = primary.get("total_time_ms", 0) or 0
    local_confidence = float(primary.get("confidence", 0) or 0)
    primary_response = primary.get("response", "")

    repaired_primary_calls = _sanitize_calls(primary.get("function_calls", []), tools, user_text)
    best_calls = repaired_primary_calls
    best_score = _score_local_candidate(best_calls, tools, user_text, local_confidence, primary_response)

    need_clause_recovery = (
        len(best_calls) == 0 or
        len(best_calls) < expected_calls or
        best_score < 0.72
    )
    if need_clause_recovery and tools:
        clause_calls = []
        for clause in _split_clauses(user_text)[:4]:
            clause_tools = _choose_clause_tools(clause, tools)
            clause_messages = [{"role": "user", "content": clause}]
            clause_result = _run_local_candidate(clause_messages, clause_tools, tool_rag_top_k=0)
            total_local_time += clause_result.get("total_time_ms", 0) or 0
            clause_fixed = _sanitize_calls(clause_result.get("function_calls", []), clause_tools, clause)
            if clause_fixed:
                # Each clause should usually map to one primary tool call.
                clause_calls.append(clause_fixed[0])

        merged_calls = _sanitize_calls(_merge_calls(best_calls, clause_calls), tools, user_text)
        merged_score = _score_local_candidate(
            merged_calls,
            tools,
            user_text,
            local_confidence,
            primary_response,
        )
        if merged_score > best_score:
            best_calls = merged_calls
            best_score = merged_score

    if (not best_calls or best_score < 0.55) and tools:
        synthesized = _sanitize_calls(_synthesize_calls_from_text(user_text, tools), tools, user_text)
        synth_score = _score_local_candidate(synthesized, tools, user_text, local_confidence, "")
        if synth_score >= best_score:
            best_calls = synthesized
            best_score = synth_score

    local = {
        "function_calls": best_calls,
        "total_time_ms": total_local_time,
        "confidence": local_confidence,
    }

    risk = 0
    if len(best_calls) == 0:
        risk += 2
    if len(best_calls) < expected_calls:
        risk += 1
    if best_score < 0.45:
        risk += 2
    if local_confidence < min(0.85, confidence_threshold):
        risk += 1
    if _contains_refusal(primary_response) and not best_calls:
        risk += 1

    should_try_cloud = bool(os.environ.get("GEMINI_API_KEY")) and risk >= 3
    if should_try_cloud:
        try:
            cloud = generate_cloud(messages, tools)
        except Exception as exc:
            # Keep benchmark execution alive in offline/local-only setups.
            local["source"] = "on-device"
            local["cloud_error"] = str(exc)
            return local

        cloud_calls = _sanitize_calls(cloud.get("function_calls", []), tools, user_text)
        cloud_score = _score_local_candidate(cloud_calls, tools, user_text, 1.0, "")
        if cloud_score >= best_score + 0.02:
            cloud["function_calls"] = cloud_calls
            cloud["source"] = "cloud (fallback)"
            cloud["local_confidence"] = local_confidence
            cloud["total_time_ms"] += total_local_time
            return cloud

    local["source"] = "on-device"
    return local


def print_result(label, result):
    """Pretty-print a generation result."""
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    if "local_confidence" in result:
        print(f"Local confidence (below threshold): {result['local_confidence']:.4f}")
    print(f"Total time: {result['total_time_ms']:.2f}ms")
    for call in result["function_calls"]:
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")


############## Example usage ##############

if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["location"],
        },
    }]

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)
