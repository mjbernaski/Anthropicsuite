#!/usr/bin/env python3
import asyncio
import glob
import os
import readline
import sys

from core import load_config, parse_model_flags, resolve_prompt, run_all, save_and_open, status


def setup_completer():
    commands = ["/quit", "/q", "/exit", "/config", "/reload"]
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

        prompt, model_flags = parse_model_flags(prompt)
        prompt = resolve_prompt(prompt)
        active = [n for n, v in model_flags.items() if v]

        round_num += 1
        print(f"\n--- Round {round_num} ---")
        print(f"  Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print(f"  Sending to {', '.join(active)} + Ollama comparison...\n")

        data = await run_all(config, prompt, model_flags)
        save_and_open(config, data)

        print(f"\n--- Round {round_num} complete ---")


if __name__ == "__main__":
    asyncio.run(main())
