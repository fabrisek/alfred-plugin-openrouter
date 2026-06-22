# @alfred-plugin/openrouter

[Alfred](https://github.com/) plugin that exposes [OpenRouter](https://openrouter.ai)
as an LLM provider. Single API key, ~400 models (Anthropic Claude, OpenAI
GPT/o-series, Google Gemini, Meta Llama, Mistral, DeepSeek, Qwen, ...), pay
per token, automatic provider routing.

## Install

Drop this folder under your Alfred installation's `data/plugins/`:

```
data/plugins/openrouter/
├── package.json
├── tsconfig.json
└── src/...
```

On next Alfred start (or via `POST /api/plugins/openrouter/reload`) the loader
runs `npm install`, bundles the source, and spawns the plugin worker.

## Configure

Open **Settings → Plugins → OpenRouter** and set:

| Field | Required | Notes |
| --- | --- | --- |
| `apiKey` | yes | Get one at https://openrouter.ai/keys |
| `baseUrl` | no | Defaults to `https://openrouter.ai/api/v1` |
| `appName` | no | Sent as `X-Title` (default `Alfred`) |
| `siteUrl` | no | Sent as `HTTP-Referer` |
| `modelFilter` | no | Substring filter on model ids (e.g. `anthropic/`) |
| `freeOnly` | no | Only expose `:free` models — useful while exploring |
| `modelsCacheMinutes` | no | Default 60min — the catalog rarely changes |

A `.env` next to `package.json` works too:

```
apiKey=sk-or-v1-...
modelFilter=anthropic/
```

## Per-model parameters

Configurable per model in **Agent → Model**:

- **Sampling**: `temperature`, `top_p`, `top_k`, `frequency_penalty`,
  `presence_penalty`, `seed`
- **Generation**: `max_tokens`
- **Reasoning**: `reasoning_effort` (`low`/`medium`/`high`) for o3, o4, Claude
  thinking, ...
- **Routing**: `provider_order` (pin upstream providers), `allow_fallbacks`,
  `transforms_middle_out` (compress over-context prompts)

## Admin actions

Two actions are exposed for the Settings UI:

- `testConnection` — verifies the key with `GET /auth/key` and returns the
  current usage/limit.
- `listModelsRaw` — returns the resolved model list (after filter/freeOnly)
  for debugging.

## License

MIT.
