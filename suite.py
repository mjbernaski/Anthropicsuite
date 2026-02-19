#!/usr/bin/env python3
import asyncio
import sys

from core import load_config, parse_model_flags, resolve_prompt, run_all, save_and_open


async def main():
    if len(sys.argv) < 2:
        print("Usage: python suite.py \"Your prompt here\"")
        sys.exit(1)

    config = load_config()
    raw = " ".join(sys.argv[1:])
    prompt, model_flags = parse_model_flags(raw)
    prompt = resolve_prompt(prompt)
    active = [n for n, v in model_flags.items() if v]
    print(f"Running Anthropic Suite ({', '.join(active)} + Ollama comparison):")

    data = await run_all(config, prompt, model_flags)
    save_and_open(config, data)
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
