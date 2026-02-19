import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import httpx
import markdown

import re

BASE_DIR = Path(__file__).parent


def load_config() -> dict:
    return json.loads((BASE_DIR / "config.json").read_text())


def save_config(config: dict):
    (BASE_DIR / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def get_output_dir(config: dict) -> Path:
    d = BASE_DIR / config["output_dir"]
    d.mkdir(exist_ok=True)
    return d


MODEL_ORDER = ["opus", "sonnet", "haiku"]


FLAG_ORDER = ["opus", "sonnet", "haiku", "ollama"]


def flags_from_str(flags_str: str) -> dict[str, bool]:
    flags = {name: (ch == "+") for name, ch in zip(FLAG_ORDER, flags_str)}
    if len(flags_str) == 3:
        flags["ollama"] = True
    return flags


def flags_to_str(flags: dict[str, bool]) -> str:
    return "".join("+" if flags.get(n, True) else "-" for n in FLAG_ORDER)


def parse_model_flags(raw: str, default_flags: str = "++++") -> tuple[str, dict[str, bool]]:
    match = re.search(r'(?:^|\s)([+\-]{3,4})(?:\s|$)', raw)
    if match:
        flags_str = match.group(1)
        prompt = raw[:match.start()] + raw[match.end():]
        prompt = prompt.strip()
        flags = flags_from_str(flags_str)
        enabled = [n for n, v in flags.items() if v]
        status(f"model flags: {flags_str} → {', '.join(enabled) or 'none'}")
        return prompt, flags
    flags = flags_from_str(default_flags)
    enabled = [n for n, v in flags.items() if v]
    status(f"model flags: {default_flags} (default) → {', '.join(enabled)}")
    return raw, flags


def resolve_prompt(raw: str) -> str:
    def replace_file_ref(match):
        filepath = Path(match.group(1)).expanduser()
        if filepath.exists():
            content = filepath.read_text()
            status(f"attached {filepath} ({len(content)} chars)")
            return f"\n--- FILE: {filepath.name} ---\n{content}\n--- END FILE ---\n"
        status(f"file not found: {filepath}")
        return match.group(0)

    return re.sub(r"@(\S+)", replace_file_ref, raw)


def status(msg: str):
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def extract_search_results(content_blocks) -> list[dict]:
    results = []
    for block in content_blocks:
        if getattr(block, "type", None) == "server_tool_use" and block.name == "web_search":
            results.append({"query": block.input.get("query", "")})
        elif getattr(block, "type", None) == "web_search_tool_result":
            for sr in getattr(block, "content", []):
                if getattr(sr, "type", None) == "web_search_result":
                    results.append({
                        "title": sr.title,
                        "url": sr.url,
                        "snippet": getattr(sr, "page_snippet", ""),
                    })
    return results


async def call_model(client: anthropic.AsyncAnthropic, config: dict, model_id: str, name: str, prompt: str) -> dict:
    status(f"{name} — sending request...")
    kwargs = dict(
        model=model_id,
        max_tokens=config["max_tokens"],
        messages=[{"role": "user", "content": prompt}],
    )
    if config.get("temperature") is not None:
        kwargs["temperature"] = config["temperature"]
    if config.get("top_p") is not None:
        kwargs["top_p"] = config["top_p"]
    if config.get("top_k") is not None:
        kwargs["top_k"] = config["top_k"]
    if config.get("system"):
        kwargs["system"] = config["system"]
    if config.get("stop_sequences"):
        kwargs["stop_sequences"] = config["stop_sequences"]
    if config.get("web_search"):
        kwargs["tools"] = [{
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": config.get("web_search_max_uses", 3),
        }]

    start = time.perf_counter()
    response = await client.messages.create(**kwargs)
    elapsed = time.perf_counter() - start

    text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
    search_results = extract_search_results(response.content) if config.get("web_search") else []

    status(f"{name} — done ({response.usage.output_tokens} tok, {elapsed:.1f}s)")
    return {
        "model": response.model,
        "model_id_requested": model_id,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "stop_reason": response.stop_reason,
        "latency_seconds": round(elapsed, 3),
        "response_text": text,
        "search_results": search_results,
    }


async def call_ollama(config: dict, prompt: str, responses: dict) -> dict:
    ollama_cfg = config["ollama"]
    base_url = ollama_cfg["base_url"]
    model = ollama_cfg["model"]

    active = [n for n in MODEL_ORDER if n in responses]
    names_str = ", ".join(n.capitalize() for n in active)

    comparison_prompt = (
        f"The following prompt was sent to {len(active)} AI model(s) ({names_str}):\n\n"
        f"PROMPT: {prompt}\n\n"
    )
    for name in active:
        r = responses[name]
        if "error" in r:
            comparison_prompt += f"--- {name.upper()} ---\n[ERROR: {r['error']}]\n\n"
        else:
            comparison_prompt += f"--- {name.upper()} ---\n{r['response_text']}\n\n"

    if len(active) == 1:
        comparison_prompt += (
            "Summarize and critique this response. Evaluate its depth, accuracy, "
            "completeness, and note any errors or omissions. Be concise but thorough."
        )
    else:
        comparison_prompt += (
            "Compare and contrast these responses. Identify key differences in "
            "depth, accuracy, style, and completeness. Note any unique insights each model provided "
            "and any errors or omissions. Be concise but thorough."
        )

    status(f"ollama ({model}) — sending comparison request...")
    start = time.perf_counter()

    options = {}
    for key in ["temperature", "top_p", "top_k", "num_predict", "repeat_penalty"]:
        if ollama_cfg.get(key) is not None:
            options[key] = ollama_cfg[key]

    payload = {"model": model, "prompt": comparison_prompt, "stream": False}
    if options:
        payload["options"] = options

    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as http:
        resp = await http.post(f"{base_url}/api/generate", json=payload)
        resp.raise_for_status()
        body = resp.json()

    elapsed = time.perf_counter() - start
    status(f"ollama ({model}) — done ({elapsed:.1f}s)")

    return {
        "model": model,
        "response_text": body.get("response", ""),
        "total_duration_ns": body.get("total_duration"),
        "eval_count": body.get("eval_count"),
        "latency_seconds": round(elapsed, 3),
    }


import asyncio

async def run_all(config: dict, prompt: str, model_flags: dict[str, bool] | None = None) -> dict:
    client = anthropic.AsyncAnthropic(api_key=os.environ.get(config["api_key_env"]))
    models = config["models"]
    if model_flags is None:
        model_flags = {name: True for name in models}

    tasks = {
        name: call_model(client, config, model_id, name, prompt)
        for name, model_id in models.items()
        if model_flags.get(name, True)
    }

    results = {}
    gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
    for name, result in zip(tasks.keys(), gathered):
        if isinstance(result, Exception):
            results[name] = {"error": str(result), "model_id_requested": models[name]}
            status(f"{name} — ERROR: {result}")
        else:
            results[name] = result

    comparison = None
    if config.get("ollama") and model_flags.get("ollama", True):
        try:
            comparison = await call_ollama(config, prompt, results)
        except Exception as e:
            comparison = {"error": str(e)}
            status(f"ollama — ERROR: {e}")

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt,
        "model_flags": model_flags,
        "config": {
            k: config.get(k) for k in
            ["max_tokens", "temperature", "top_p", "top_k", "system", "stop_sequences", "web_search"]
        },
        "results": results,
        "comparison": comparison,
    }


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def md(text: str) -> str:
    return markdown.markdown(text, extensions=["tables", "fenced_code", "nl2br"])


def build_html(data: dict) -> str:
    prompt_escaped = esc(data["prompt"])
    flags = data.get("model_flags", {name: True for name in MODEL_ORDER})
    active_models = [n for n in MODEL_ORDER if flags.get(n, False)]
    ncols = len(active_models) or 1

    cards = []
    for name in active_models:
        r = data["results"].get(name, {})
        if "error" in r:
            body = f'<p class="error">{esc(r["error"])}</p>'
            meta = ""
        else:
            meta = (
                f'<div class="meta">'
                f'Model: {r["model"]} | '
                f'In: {r["input_tokens"]} tok | '
                f'Out: {r["output_tokens"]} tok | '
                f'Stop: {r["stop_reason"]} | '
                f'Latency: {r["latency_seconds"]}s'
                f'</div>'
            )
            body = f'<div class="md-body">{md(r["response_text"])}</div>'

            sources = [s for s in r.get("search_results", []) if "url" in s]
            if sources:
                links = "".join(
                    f'<li><a href="{s["url"]}">{esc(s["title"])}</a></li>' for s in sources
                )
                body += f'<div class="sources"><strong>Sources:</strong><ul>{links}</ul></div>'

        cards.append(
            f'<div class="col">'
            f'<h2>{name.upper()}</h2>'
            f'{meta}'
            f'{body}'
            f'</div>'
        )

    comparison_html = ""
    comp = data.get("comparison")
    if comp:
        if "error" in comp:
            comparison_html = f'<div class="comparison"><h2>Comparison (Ollama)</h2><p class="error">{esc(comp["error"])}</p></div>'
        else:
            comp_meta = (
                f'<div class="meta">'
                f'Model: {comp["model"]} | '
                f'Eval tokens: {comp.get("eval_count", "n/a")} | '
                f'Latency: {comp["latency_seconds"]}s'
                f'</div>'
            )
            comparison_html = (
                f'<div class="comparison">'
                f'<h2>Comparison — {esc(comp["model"])}</h2>'
                f'{comp_meta}'
                f'<div class="md-body">{md(comp["response_text"])}</div>'
                f'</div>'
            )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Anthropic Suite — {data["timestamp"]}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: system-ui, -apple-system, sans-serif; background: #0f0f0f; color: #e0e0e0; padding: 24px; }}
.prompt {{ background: #1a1a2e; border: 1px solid #333; border-radius: 8px; padding: 16px; margin-bottom: 24px; white-space: pre-wrap; font-size: 14px; }}
.prompt-label {{ color: #888; font-size: 12px; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 1px; }}
.ts {{ color: #666; font-size: 12px; margin-bottom: 16px; }}
.grid {{ display: grid; grid-template-columns: repeat({ncols}, 1fr); gap: 16px; margin-bottom: 24px; }}
.col {{ background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 8px; padding: 16px; overflow: auto; }}
.col h2 {{ font-size: 16px; margin-bottom: 8px; color: #c9a0ff; }}
.meta {{ font-size: 11px; color: #777; margin-bottom: 12px; line-height: 1.6; }}
.md-body {{ font-size: 13px; line-height: 1.6; }}
.md-body h1, .md-body h2, .md-body h3, .md-body h4 {{ color: #c9a0ff; margin: 12px 0 6px 0; }}
.md-body h1 {{ font-size: 18px; }} .md-body h2 {{ font-size: 16px; }} .md-body h3 {{ font-size: 14px; }}
.md-body p {{ margin: 8px 0; }}
.md-body ul, .md-body ol {{ margin: 8px 0 8px 20px; }}
.md-body li {{ margin: 4px 0; }}
.md-body strong {{ color: #f0f0f0; }}
.md-body em {{ color: #ccc; }}
.md-body a {{ color: #7ab8ff; text-decoration: none; }}
.md-body a:hover {{ text-decoration: underline; }}
.md-body table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 12px; }}
.md-body th {{ background: #2a2a3e; color: #c9a0ff; padding: 8px; text-align: left; border: 1px solid #444; }}
.md-body td {{ padding: 6px 8px; border: 1px solid #333; vertical-align: top; }}
.md-body tr:nth-child(even) {{ background: #1e1e2e; }}
.md-body pre {{ background: #161622; border: 1px solid #333; border-radius: 4px; padding: 10px; overflow-x: auto; margin: 8px 0; }}
.md-body code {{ background: #1e1e2e; padding: 2px 5px; border-radius: 3px; font-size: 12px; }}
.md-body pre code {{ background: none; padding: 0; }}
.md-body blockquote {{ border-left: 3px solid #555; padding-left: 12px; color: #999; margin: 8px 0; }}
.md-body hr {{ border: none; border-top: 1px solid #333; margin: 16px 0; }}
.error {{ color: #ff6b6b; }}
.sources {{ margin-top: 12px; font-size: 12px; border-top: 1px solid #333; padding-top: 8px; }}
.sources ul {{ margin: 4px 0 0 16px; }}
.sources li {{ margin-bottom: 4px; }}
.sources a {{ color: #7ab8ff; text-decoration: none; }}
.sources a:hover {{ text-decoration: underline; }}
.comparison {{ background: #1a1a2e; border: 1px solid #3a3a5e; border-radius: 8px; padding: 20px; }}
.comparison h2 {{ font-size: 16px; margin-bottom: 8px; color: #ffa657; }}
</style>
</head>
<body>
<div class="ts">{data["timestamp"]}</div>
<div class="prompt-label">Prompt</div>
<div class="prompt">{prompt_escaped}</div>
<div class="grid">
{"".join(cards)}
</div>
{comparison_html}
</body>
</html>"""


def save_and_open(config: dict, data: dict) -> tuple[Path, Path]:
    output_dir = get_output_dir(config)
    slug = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"{slug}.json"
    html_path = output_dir / f"{slug}.html"

    json_path.write_text(json.dumps(data, indent=2))
    html_path.write_text(build_html(data))

    status(f"saved {json_path}")
    status(f"saved {html_path}")

    if config.get("open_html", True):
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(html_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform == "linux":
            subprocess.Popen(["xdg-open", str(html_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return json_path, html_path
