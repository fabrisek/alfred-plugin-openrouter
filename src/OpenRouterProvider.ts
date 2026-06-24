import type {
  ChatChunk,
  ChatRequest,
  ChatResponse,
  LLMProvider,
  Logger,
  Message,
  MessageAttachment,
  ModelCapabilities,
  ModelDetails,
  ModelInfo,
  ModelPricing,
  ToolCall,
  ToolSchema,
} from '@alfred/sdk';

const noopLogger: Logger = {
  debug: () => {},
  info: () => {},
  warn: () => {},
  error: () => {},
};

export type OpenRouterConfig = {
  apiKey: string;
  baseUrl: string;
  appName: string;
  siteUrl: string;
  modelFilter: string;
  freeOnly: boolean;
  modelsCacheMinutes: number;
};

export type OpenRouterProviderOptions = {
  config: OpenRouterConfig;
  logger?: Logger;
};

type OpenRouterModelEntry = {
  id: string;
  name?: string;
  description?: string;
  context_length?: number;
  architecture?: {
    modality?: string;
    input_modalities?: string[];
    output_modalities?: string[];
    tokenizer?: string;
  };
  pricing?: {
    prompt?: string;
    completion?: string;
  };
  supported_parameters?: string[];
  top_provider?: {
    context_length?: number;
    max_completion_tokens?: number;
  };
};

type OpenRouterModelsResponse = {
  data?: OpenRouterModelEntry[];
};

type OpenAIToolCall = {
  id?: string;
  type?: string;
  function?: { name?: string; arguments?: string };
};

type OpenAIMessage = {
  role?: string;
  content?: string | null;
  reasoning?: string | null;
  reasoning_content?: string | null;
  tool_calls?: OpenAIToolCall[];
};

type OpenAIChoice = {
  index?: number;
  message?: OpenAIMessage;
  delta?: OpenAIMessage;
  finish_reason?: string | null;
};

type OpenAIUsage = {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
};

type OpenAIChatResponse = {
  id?: string;
  choices?: OpenAIChoice[];
  usage?: OpenAIUsage;
  error?: { message?: string; code?: string | number };
};

type CachedModels = {
  fetchedAt: number;
  models: ModelInfo[];
};

export class OpenRouterProvider implements LLMProvider {
  readonly id = 'openrouter';
  readonly displayName = 'OpenRouter';
  readonly kind = 'remote' as const;
  readonly capabilities = {
    tools: true,
    thinking: true,
    vision: true,
    streaming: true,
    modelManagement: false,
  };

  private readonly log: Logger;
  private cache: CachedModels | null = null;

  constructor(private readonly opts: OpenRouterProviderOptions) {
    this.log = opts.logger ?? noopLogger;
  }

  getRuntimeConfig(): Record<string, unknown> {
    const { baseUrl, appName, siteUrl, modelFilter, freeOnly, modelsCacheMinutes } =
      this.opts.config;
    return { baseUrl, appName, siteUrl, modelFilter, freeOnly, modelsCacheMinutes };
  }

  async healthCheck(): Promise<boolean> {
    try {
      const res = await fetch(`${this.opts.config.baseUrl}/auth/key`, {
        headers: this.authHeaders(),
      });
      return res.ok;
    } catch {
      return false;
    }
  }

  async listModels(): Promise<ModelInfo[]> {
    const ttlMs = Math.max(0, this.opts.config.modelsCacheMinutes) * 60_000;
    if (this.cache && ttlMs > 0 && Date.now() - this.cache.fetchedAt < ttlMs) {
      return this.cache.models;
    }
    const url = `${this.opts.config.baseUrl}/models`;
    const res = await fetch(url, { headers: this.authHeaders() }).catch((err: unknown) => {
      throw new Error(`OpenRouter listModels fetch failed: ${(err as Error).message}`);
    });
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      throw new Error(`OpenRouter listModels failed: ${res.status} ${text.slice(0, 300)}`);
    }
    const data = (await res.json()) as OpenRouterModelsResponse;
    const filter = this.opts.config.modelFilter.trim().toLowerCase();
    const freeOnly = this.opts.config.freeOnly;
    const entries = (data.data ?? []).filter((e) => {
      if (!e.id) return false;
      if (freeOnly && !e.id.endsWith(':free')) return false;
      if (filter && !e.id.toLowerCase().includes(filter)) return false;
      return true;
    });
    const models: ModelInfo[] = entries.map((e) => ({
      id: e.id,
      providerId: this.id,
      displayName: e.name ?? e.id,
      contextWindow: e.top_provider?.context_length ?? e.context_length,
      capabilities: deriveCapabilities(e),
      description: e.description?.trim() || undefined,
      pricing: derivePricing(e.pricing),
    }));
    models.sort((a, b) => a.id.localeCompare(b.id));
    this.cache = { fetchedAt: Date.now(), models };
    this.log.info('listed models', { count: models.length, filter, freeOnly });
    return models;
  }

  async getModelInfo(name: string): Promise<ModelDetails> {
    const models = await this.listModels().catch(() => [] as ModelInfo[]);
    const m = models.find((x) => x.id === name);
    return { contextLength: m?.contextWindow };
  }

  async chat(req: ChatRequest): Promise<ChatResponse> {
    const body = this.buildBody(req, false);
    const url = `${this.opts.config.baseUrl}/chat/completions`;
    this.log.info('chat request', {
      model: req.model,
      messages: req.messages.length,
      tools: req.tools?.length ?? 0,
    });
    const res = await fetch(url, {
      method: 'POST',
      headers: {
        ...this.authHeaders(),
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
      signal: req.signal,
    }).catch((err: unknown) => {
      if ((err as Error).name === 'AbortError') throw err;
      throw new Error(`OpenRouter chat fetch failed: ${(err as Error).message}`);
    });
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      throw new Error(`OpenRouter chat failed: ${res.status} ${text.slice(0, 500)}`);
    }
    const data = (await res.json()) as OpenAIChatResponse;
    if (data.error) {
      throw new Error(`OpenRouter error: ${data.error.message ?? 'unknown'}`);
    }
    const choice = data.choices?.[0];
    const msg = choice?.message ?? {};
    const content = typeof msg.content === 'string' ? msg.content : '';
    const thinking =
      (msg.reasoning ?? msg.reasoning_content ?? '').toString().trim() || undefined;
    const toolCalls = parseToolCalls(msg.tool_calls);
    return {
      content: content.trim(),
      thinking,
      toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
      finishReason: mapFinishReason(choice?.finish_reason),
      usage: computeUsage(data.usage),
    };
  }

  async *stream(req: ChatRequest): AsyncIterable<ChatChunk> {
    const body = this.buildBody(req, true);
    const url = `${this.opts.config.baseUrl}/chat/completions`;
    this.log.info('stream request', {
      model: req.model,
      messages: req.messages.length,
      tools: req.tools?.length ?? 0,
    });

    let res: Response;
    try {
      res = await fetch(url, {
        method: 'POST',
        headers: {
          ...this.authHeaders(),
          'Content-Type': 'application/json',
          Accept: 'text/event-stream',
        },
        body: JSON.stringify(body),
        signal: req.signal,
      });
    } catch (err) {
      if ((err as Error).name === 'AbortError') return;
      throw new Error(`OpenRouter stream fetch failed: ${(err as Error).message}`);
    }
    if (!res.ok || !res.body) {
      const text = await res.text().catch(() => '');
      throw new Error(`OpenRouter stream failed: ${res.status} ${text.slice(0, 500)}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let assembledContent = '';
    let assembledThinking = '';
    const toolBuffers = new Map<number, { id: string; name: string; argsRaw: string }>();
    const emittedTools = new Map<number, ToolCall>();
    let finishReason: ChatResponse['finishReason'] = 'stop';
    let lastUsage: OpenAIUsage | undefined;

    try {
      while (true) {
        let chunk: ReadableStreamReadResult<Uint8Array>;
        try {
          chunk = await reader.read();
        } catch (err) {
          if ((err as Error).name === 'AbortError') return;
          throw err;
        }
        if (chunk.done) break;
        buffer += decoder.decode(chunk.value, { stream: true });

        // OpenRouter (and many proxies) keepalive with ": OPENROUTER PROCESSING"
        // SSE comments while the upstream warms up — strip those silently.
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';
        for (const rawLine of lines) {
          const line = rawLine.trim();
          if (!line) continue;
          if (line.startsWith(':')) continue;
          if (!line.startsWith('data:')) continue;
          const payload = line.slice(5).trim();
          if (payload === '[DONE]') continue;
          let parsed: OpenAIChatResponse;
          try {
            parsed = JSON.parse(payload) as OpenAIChatResponse;
          } catch {
            continue;
          }
          if (parsed.error) {
            throw new Error(
              `OpenRouter stream error: ${parsed.error.message ?? 'unknown'}`,
            );
          }
          const choice = parsed.choices?.[0];
          const delta = choice?.delta ?? {};

          const reasoningDelta = (delta.reasoning ?? delta.reasoning_content ?? '') as string;
          if (typeof reasoningDelta === 'string' && reasoningDelta.length > 0) {
            assembledThinking += reasoningDelta;
            yield { type: 'thinking', delta: reasoningDelta };
          }
          if (typeof delta.content === 'string' && delta.content.length > 0) {
            assembledContent += delta.content;
            yield { type: 'text', delta: delta.content };
          }
          if (delta.tool_calls && delta.tool_calls.length > 0) {
            for (const incoming of delta.tool_calls) {
              const idx = (incoming as { index?: number }).index ?? 0;
              const existing = toolBuffers.get(idx) ?? { id: '', name: '', argsRaw: '' };
              const idDelta =
                incoming.id && incoming.id !== existing.id ? incoming.id : undefined;
              const nameDelta = incoming.function?.name || undefined;
              const argsDelta =
                typeof incoming.function?.arguments === 'string' &&
                incoming.function.arguments.length > 0
                  ? incoming.function.arguments
                  : undefined;
              if (incoming.id) existing.id = incoming.id;
              if (incoming.function?.name) existing.name = incoming.function.name;
              if (argsDelta) existing.argsRaw += argsDelta;
              toolBuffers.set(idx, existing);
              if (idDelta || nameDelta || argsDelta) {
                yield {
                  type: 'tool_call_delta',
                  index: idx,
                  id: idDelta,
                  nameDelta,
                  argsDelta,
                };
              }
              if (existing.name && !emittedTools.has(idx)) {
                const args = tryParseJson(existing.argsRaw);
                if (args && existing.argsRaw.trim().endsWith('}')) {
                  const call: ToolCall = {
                    id: existing.id || `openrouter-${idx}`,
                    name: existing.name,
                    arguments: args,
                  };
                  emittedTools.set(idx, call);
                  yield { type: 'tool_call', call };
                }
              }
            }
          }
          if (choice?.finish_reason) finishReason = mapFinishReason(choice.finish_reason);
          if (parsed.usage) lastUsage = parsed.usage;
        }
      }

      for (const [idx, buf] of toolBuffers) {
        if (emittedTools.has(idx)) continue;
        if (!buf.name) continue;
        const args = tryParseJson(buf.argsRaw) ?? {};
        const call: ToolCall = {
          id: buf.id || `openrouter-${idx}`,
          name: buf.name,
          arguments: args,
        };
        emittedTools.set(idx, call);
        yield { type: 'tool_call', call };
      }

      const toolCalls = Array.from(emittedTools.values());
      const response: ChatResponse = {
        content: assembledContent.trim(),
        thinking: assembledThinking.trim() || undefined,
        toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
        finishReason,
        usage: computeUsage(lastUsage),
      };
      yield { type: 'done', response };
    } finally {
      try {
        await reader.cancel();
      } catch {
        // already closed
      }
    }
  }

  private authHeaders(): Record<string, string> {
    const h: Record<string, string> = {
      Authorization: `Bearer ${this.opts.config.apiKey}`,
    };
    if (this.opts.config.siteUrl) h['HTTP-Referer'] = this.opts.config.siteUrl;
    if (this.opts.config.appName) h['X-Title'] = this.opts.config.appName;
    return h;
  }

  private buildBody(req: ChatRequest, stream: boolean): Record<string, unknown> {
    const body: Record<string, unknown> = {
      model: req.model,
      messages: req.messages.map(toOpenAIMessage),
      stream,
    };
    if (stream) body.stream_options = { include_usage: true };
    if (typeof req.temperature === 'number') body.temperature = req.temperature;
    if (typeof req.maxTokens === 'number') body.max_tokens = req.maxTokens;
    if (req.tools && req.tools.length > 0) body.tools = req.tools.map(toOpenAITool);

    if (req.providerOptions) {
      applyProviderOptions(body, req.providerOptions);
    }
    return body;
  }
}

const SAMPLING_KEYS = new Set([
  'temperature',
  'top_p',
  'top_k',
  'frequency_penalty',
  'presence_penalty',
  'repetition_penalty',
  'min_p',
  'seed',
]);

function applyProviderOptions(
  body: Record<string, unknown>,
  options: Record<string, unknown>,
): void {
  for (const [key, value] of Object.entries(options)) {
    if (value === undefined || value === null || value === '') continue;
    if (key === 'max_tokens') {
      const n = Number(value);
      if (Number.isFinite(n)) body.max_tokens = Math.floor(n);
      continue;
    }
    if (key === 'reasoning_effort') {
      const effort = String(value).toLowerCase();
      if (effort === 'low' || effort === 'medium' || effort === 'high') {
        body.reasoning = { effort };
      }
      continue;
    }
    if (key === 'provider_order') {
      const order = Array.isArray(value)
        ? value.map(String).filter((s) => s.length > 0)
        : [];
      if (order.length > 0) {
        const existing = (body.provider ?? {}) as Record<string, unknown>;
        body.provider = { ...existing, order };
      }
      continue;
    }
    if (key === 'allow_fallbacks') {
      const existing = (body.provider ?? {}) as Record<string, unknown>;
      body.provider = { ...existing, allow_fallbacks: Boolean(value) };
      continue;
    }
    if (key === 'transforms_middle_out') {
      if (Boolean(value)) body.transforms = ['middle-out'];
      continue;
    }
    if (SAMPLING_KEYS.has(key)) {
      body[key] = value;
    }
  }
}

function toOpenAITool(tool: ToolSchema): Record<string, unknown> {
  return {
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters,
    },
  };
}

// OpenRouter follows the OpenAI multimodal content-part envelope:
//   - text                 → { type: 'text', text }
//   - images               → { type: 'image_url', image_url: { url: dataUrl } }
//   - audio (chat models)  → { type: 'input_audio', input_audio: { data, format } }
//   - PDF / video / docs   → { type: 'file', file: { filename, file_data: dataUrl } }
// See https://openrouter.ai/docs/features/multimodal — the `file` envelope is
// the catch-all upstream models (Gemini, Claude, gpt-4o, …) use for non-image,
// non-audio binaries; routing video through it is best-effort for the Gemini
// family. Non-supporting models ignore the part, and the host already inlines
// a text placeholder via `inlineAttachmentsForProvider` so the model still
// knows a video was attached.
type OpenAIContentPart =
  | { type: 'text'; text: string }
  | { type: 'image_url'; image_url: { url: string } }
  | { type: 'input_audio'; input_audio: { data: string; format: string } }
  | { type: 'file'; file: { filename: string; file_data: string } };

function attachmentToContentPart(att: MessageAttachment): OpenAIContentPart | null {
  const mime = (att.mimeType || '').toLowerCase();
  if (!att.data) return null;
  if (mime.startsWith('image/')) {
    return {
      type: 'image_url',
      image_url: { url: `data:${mime};base64,${att.data}` },
    };
  }
  if (mime.startsWith('audio/')) {
    return {
      type: 'input_audio',
      input_audio: { data: att.data, format: audioFormatFromMime(mime) },
    };
  }
  if (mime === 'application/pdf') {
    return {
      type: 'file',
      file: {
        filename: att.name || 'document.pdf',
        file_data: `data:application/pdf;base64,${att.data}`,
      },
    };
  }
  if (mime.startsWith('video/')) {
    return {
      type: 'file',
      file: {
        filename: att.name || 'video',
        file_data: `data:${mime};base64,${att.data}`,
      },
    };
  }
  // Textual + unknown-binary attachments are already inlined into the user
  // message body by the host's `inlineAttachmentsForProvider` — duplicating
  // them as a `file` part would double the byte count and confuse models
  // that don't recognise the envelope.
  return null;
}

function audioFormatFromMime(mime: string): string {
  if (mime.includes('wav')) return 'wav';
  if (mime.includes('mp3') || mime.includes('mpeg')) return 'mp3';
  if (mime.includes('webm')) return 'webm';
  if (mime.includes('ogg')) return 'ogg';
  if (mime.includes('m4a') || mime.includes('mp4')) return 'm4a';
  if (mime.includes('flac')) return 'flac';
  return 'wav';
}

function buildMultimodalContent(msg: Message): string | OpenAIContentPart[] {
  const text = typeof msg.content === 'string' ? msg.content : '';
  const attachments = Array.isArray(msg.attachments) ? msg.attachments : [];
  if (attachments.length > 0) {
    const parts: OpenAIContentPart[] = [];
    if (text) parts.push({ type: 'text', text });
    for (const att of attachments) {
      const part = attachmentToContentPart(att);
      if (part) parts.push(part);
    }
    if (parts.length > 0) return parts;
  }
  // Legacy fallback: messages stored before the `attachments` field landed
  // only carry a bare `images` array of base64 strings with no mime info.
  // PNG is the safe default — most upstream models sniff the actual format
  // from the magic bytes regardless of the data-URL hint.
  if (Array.isArray(msg.images) && msg.images.length > 0) {
    const parts: OpenAIContentPart[] = [];
    if (text) parts.push({ type: 'text', text });
    for (const b64 of msg.images) {
      parts.push({
        type: 'image_url',
        image_url: { url: `data:image/png;base64,${b64}` },
      });
    }
    return parts;
  }
  return text;
}

function toOpenAIMessage(msg: Message): Record<string, unknown> {
  const out: Record<string, unknown> = { role: msg.role };
  if (msg.role === 'tool') {
    out.role = 'tool';
    out.content = msg.content;
    if (msg.toolCallId) out.tool_call_id = msg.toolCallId;
    if (msg.name) out.name = msg.name;
    return out;
  }
  if (msg.role === 'assistant') {
    if (msg.toolCalls && msg.toolCalls.length > 0) {
      out.tool_calls = msg.toolCalls.map((tc) => ({
        id: tc.id,
        type: 'function',
        function: {
          name: tc.name,
          arguments: JSON.stringify(tc.arguments ?? {}),
        },
      }));
    }
    out.content = buildMultimodalContent(msg);
    return out;
  }
  out.content = buildMultimodalContent(msg);
  return out;
}

function parseToolCalls(calls: OpenAIToolCall[] | undefined): ToolCall[] {
  if (!calls || calls.length === 0) return [];
  const out: ToolCall[] = [];
  for (let i = 0; i < calls.length; i++) {
    const c = calls[i];
    const name = c.function?.name;
    if (typeof name !== 'string' || !name) continue;
    const args = tryParseJson(c.function?.arguments ?? '{}') ?? {};
    out.push({ id: c.id || `openrouter-${i}`, name, arguments: args });
  }
  return out;
}

function tryParseJson(s: string): Record<string, unknown> | null {
  try {
    const parsed = JSON.parse(s);
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
  } catch {
    // ignore
  }
  return null;
}

function mapFinishReason(reason: string | null | undefined): ChatResponse['finishReason'] {
  if (!reason) return 'stop';
  if (reason === 'length') return 'length';
  if (reason === 'tool_calls') return 'tool_calls';
  if (reason === 'error') return 'error';
  return 'stop';
}

function computeUsage(u: OpenAIUsage | undefined): ChatResponse['usage'] | undefined {
  if (!u) return undefined;
  if (u.prompt_tokens === undefined && u.completion_tokens === undefined) return undefined;
  return {
    inputTokens: u.prompt_tokens,
    outputTokens: u.completion_tokens,
  };
}

// OpenRouter publishes per-token rates as decimal strings in USD (e.g.
// "0.000003" = $3 / 1M tokens). Convert to the SDK's $/M-tokens convention
// so the UI can render "$3.00 in / $15.00 out" without doing math at render.
function derivePricing(p: OpenRouterModelEntry['pricing']): ModelPricing | undefined {
  if (!p) return undefined;
  const input = parsePerTokenRate(p.prompt);
  const output = parsePerTokenRate(p.completion);
  if (input === undefined && output === undefined) return undefined;
  const pricing: ModelPricing = { currency: 'USD' };
  if (input !== undefined) pricing.inputPerMTokens = input;
  if (output !== undefined) pricing.outputPerMTokens = output;
  return pricing;
}

function parsePerTokenRate(raw: string | undefined): number | undefined {
  if (raw === undefined || raw === null || raw === '') return undefined;
  const n = Number(raw);
  if (!Number.isFinite(n) || n < 0) return undefined;
  return n * 1_000_000;
}

function deriveCapabilities(e: OpenRouterModelEntry): ModelCapabilities {
  const inputs = e.architecture?.input_modalities ?? [];
  const supported = e.supported_parameters ?? [];
  const hasVision = inputs.includes('image');
  const hasAudio = inputs.includes('audio');
  const hasVideo = inputs.includes('video');
  // OpenRouter labels PDF / document support as `file` on the architecture
  // payload — surface it so the host's attachment router can show "PDF
  // viewable" badges in the model picker.
  const hasFiles = inputs.includes('file');
  const hasTools = supported.includes('tools') || supported.includes('tool_choice');
  const hasThinking =
    supported.includes('reasoning') || supported.includes('include_reasoning');
  return {
    tools: hasTools,
    thinking: hasThinking,
    vision: hasVision,
    inputs: {
      image: hasVision || undefined,
      audio: hasAudio || undefined,
      video: hasVideo || undefined,
      files: hasFiles || undefined,
    },
  };
}
