#!/usr/bin/env python3
import asyncio
import glob
import os
import re
import readline
import sys

from core import (
    FLAG_ORDER, flags_from_str, flags_to_str, load_config, parse_model_flags,
    resolve_prompt, run_all, save_and_open, save_config, status,
)


def setup_completer():
    commands = ["/quit", "/q", "/exit", "/config", "/reload", "/set"]
    _cache = {"matches": [], "prefix": None}

    def completer(text, state):
        if state == 0:
            line = readline.get_line_buffer()
            cursor = readline.get_endidx()

            at_pos = line.rfind("@", 0, cursor)
            if at_pos >= 0 and (at_pos == 0 or line[at_pos - 1] == " "):
                partial = line[at_pos + 1:cursor]
                expanded = os.path.expanduser(partial)
                raw_matches = glob.glob(expanded + "*")
                results = []
                for m in sorted(raw_matches):
                    suffix = "/" if os.path.isdir(m) else ""
                    if partial.startswith("~"):
                        home = os.path.expanduser("~")
                        display = "~" + m[len(home):]
                    else:
                        display = m
                    results.append(display + suffix)
                _cache["matches"] = results
            elif text.startswith("/"):
                _cache["matches"] = [c for c in commands if c.startswith(text)]
            else:
                _cache["matches"] = []

        if state < len(_cache["matches"]):
            return _cache["matches"][state]
        return None

    readline.set_completer(completer)
    readline.set_completer_delims(" \t\n@")
    if "libedit" in (readline.__doc__ or ""):
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")

BANNER = """
╔══════════════════════════════════════════╗
║         Anthropic Suite — Chat           ║
║                                          ║
║  Type a prompt and press Enter.          ║
║  Use @path/to/file.txt to attach files.  ║
║  Commands:                               ║
║    /quit or /q   — exit                  ║
║    /set ++-+     — set default flags      ║
║    /config       — show current config   ║
║    /reload       — reload config.json    ║
╚══════════════════════════════════════════╝
"""


def show_config(config: dict):
    print("\n  Current config:")
    print(f"    max_tokens:    {config['max_tokens']}")
    print(f"    temperature:   {config.get('temperature')}")
    print(f"    top_p:         {config.get('top_p')}")
    print(f"    top_k:         {config.get('top_k')}")
    print(f"    system:        {config.get('system') or '(none)'}")
    print(f"    web_search:    {config.get('web_search')}")
    print(f"    models:        {', '.join(config['models'].keys())}")
    ollama = config.get("ollama", {})
    print(f"    ollama:        {ollama.get('model', 'disabled')} @ {ollama.get('base_url', 'n/a')}")
    df = config.get("default_flags", "++++")
    labels = [f"{n}={'on' if ch == '+' else 'off'}" for n, ch in zip(FLAG_ORDER, df)]
    print(f"    default_flags: {df} ({', '.join(labels)})")
    print(f"    open_html:     {config.get('open_html', True)}")
    print()


async def main():
    config = load_config()
    round_num = 0

    setup_completer()
    print(BANNER)

    while True:
        try:
            prompt = input("\n\033[1;36mprompt>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not prompt:
            continue

        if prompt.lower() in ("/quit", "/q", "/exit"):
            print("Bye.")
            break

        if prompt.lower() == "/config":
            show_config(config)
            continue

        if prompt.lower() == "/reload":
            config = load_config()
            print("  Config reloaded.")
            continue

        if prompt.lower().startswith("/set"):
            parts = prompt.split()
            if len(parts) == 2 and re.match(r'^[+\-]{3,4}$', parts[1]):
                flags_str = parts[1] if len(parts[1]) == 4 else parts[1] + "+"
                config["default_flags"] = flags_str
                save_config(config)
                flags = flags_from_str(flags_str)
                enabled = [n for n, v in flags.items() if v]
                print(f"  Default flags saved: {flags_str} → {', '.join(enabled)}")
            else:
                print("  Usage: /set ++-+  (opus, sonnet, haiku, ollama)")
            continue

        prompt, model_flags = parse_model_flags(prompt, config.get("default_flags", "++++"))
        prompt = resolve_prompt(prompt)
        active = [n for n, v in model_flags.items() if v]

        round_num += 1
        print(f"\n--- Round {round_num} ---")
        print(f"  Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print(f"  Sending to {', '.join(active)}...\n")

        data = await run_all(config, prompt, model_flags)
        save_and_open(config, data)

        print(f"\n--- Round {round_num} complete ---")


if __name__ == "__main__":
    asyncio.run(main())
