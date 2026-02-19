#!/usr/bin/env python3
import asyncio
import sys

from core import load_config, run_all, save_and_open


async def main():
    if len(sys.argv) < 2:
        print("Usage: python suite.py \"Your prompt here\"")
        sys.exit(1)

    config = load_config()
    prompt = " ".join(sys.argv[1:])
    print(f"Running Anthropic Suite ({len(config['models'])} models + Ollama comparison):")

    data = await run_all(config, prompt)
    save_and_open(config, data)
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
