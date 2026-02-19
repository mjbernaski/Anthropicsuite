#!/usr/bin/env python3
import asyncio
import sys

from core import load_config, run_all, save_and_open, status

BANNER = """
╔══════════════════════════════════════════╗
║         Anthropic Suite — Chat           ║
║                                          ║
║  Type a prompt and press Enter.          ║
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

        round_num += 1
        print(f"\n--- Round {round_num} ---")
        print(f"  Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print(f"  Sending to {len(config['models'])} models + Ollama comparison...\n")

        data = await run_all(config, prompt)
        save_and_open(config, data)

        print(f"\n--- Round {round_num} complete ---")


if __name__ == "__main__":
    asyncio.run(main())
