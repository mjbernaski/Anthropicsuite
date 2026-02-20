# Anthropic Suite

Compare responses from Anthropic's Claude models (Opus, Sonnet, Haiku) side-by-side with optional web search and an Ollama-powered critique/comparison.

## Setup

```bash
python3 -m venv venv
venv/bin/pip install anthropic markdown
export ANTHROPIC_API_KEY="your-key-here"
```

## Usage

### One-shot mode

```bash
venv/bin/python suite.py "your prompt here"
```

### Interactive chat

```bash
venv/bin/python chat.py
```

Chat commands:
- `/set ++-+` — set default model flags (persists to config.json)
- `/config` — show current configuration
- `/reload` — hot-reload config.json
- `/quit` — exit

## Model flags

Control which models run using a `+`/`-` string. Position order: **Opus, Sonnet, Haiku, Ollama**.

| Flag | Models |
|------|--------|
| `++++` | All three + Ollama critique |
| `+++-` | All three, no critique |
| `-+-+` | Sonnet + Ollama critique |
| `--+-` | Haiku only, no critique |
| `+++` | 3-char shorthand (Ollama defaults to on) |

Place flags anywhere in the prompt — they're stripped before sending:

```
prompt> ++-+ why is the sky blue
prompt> --+- what is karst
```

Or set a default with `/set`:

```
prompt> /set -+--
```

## File attachments

Use `@` to inline file contents into the prompt:

```
prompt> summarize @~/notes.txt
prompt> review @src/app.py and @src/utils.py
```

Tab completion works for file paths in chat mode.

## Configuration

All settings live in `config.json`:

| Key | Description |
|-----|-------------|
| `max_tokens` | Max output tokens per model |
| `temperature` | Sampling temperature (0.0–1.0) |
| `top_p` | Nucleus sampling |
| `top_k` | Top-k sampling |
| `system` | System prompt sent to all models |
| `stop_sequences` | Strings that stop generation |
| `web_search` | Enable web search tool |
| `web_search_max_uses` | Max searches per model call |
| `default_flags` | Default model flags (e.g. `++++`) |
| `open_html` | Auto-open HTML output in browser |
| `ollama.base_url` | Ollama server address |
| `ollama.model` | Ollama model for critique |
| `ollama.temperature` | Ollama sampling temperature |

## Output

Each run saves two files to `output/`:
- `YYYYMMDD_HHMMSS.json` — full prompt, config, per-model metadata, responses, search results, and comparison
- `YYYYMMDD_HHMMSS.html` — dark-themed page with markdown-rendered responses in a responsive grid, source citations, and the Ollama comparison
