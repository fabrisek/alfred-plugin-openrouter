import type {
  ImageGenerationOptions,
  ImageGenerationResult,
  ImageProvider,
  Logger,
  MediaModelInfo,
} from '@alfred/sdk';

const noopLogger: Logger = {
  debug: () => {},
  info: () => {},
  warn: () => {},
  error: () => {},
};

export type OpenRouterImageConfig = {
  apiKey: string;
  baseUrl: string;
  appName: string;
  siteUrl: string;
  /**
   * Optional fallback model id, used only when the host has not picked one in
   * Settings → Models → Image. Mirrors the {@link OpenRouterSttConfig.sttModel}
   * contract — the modern path is for the host to enumerate
   * {@link OpenRouterImageProvider.listModels} and pass `opts.model` per call.
   */
  imageModel: string;
};

export type OpenRouterImageProviderOptions = {
  config: OpenRouterImageConfig;
  logger?: Logger;
};

type OpenRouterModelEntry = {
  id: string;
  name?: string;
  description?: string;
  context_length?: number;
};

type OpenRouterModelsResponse = {
  data?: OpenRouterModelEntry[];
};

/**
 * One image entry as returned by OpenRouter on a chat-completions response.
 * The upstream shape isn't strictly stable across vendors — some send a plain
 * data URL string, some send the OpenAI `{ image_url: { url } }` envelope,
 * and a few (Gemini family via OpenRouter) send `{ b64_json }` or `{ data }`
 * straight from the SDK. {@link extractImages} normalises all of them into
 * `{ data: Uint8Array, mimeType }`.
 */
type ChatImageEntry =
  | string
  | {
      image_url?: string | { url?: string };
      url?: string;
      b64_json?: string;
      data?: string;
    };

type ChatImageMessage = {
  content?: string | null;
  images?: ChatImageEntry[];
};

type ChatImageResponse = {
  choices?: Array<{ message?: ChatImageMessage }>;
  error?: { message?: string; code?: string | number };
};

/**
 * Image generation backend using OpenRouter's `modalities: ['image', 'text']`
 * extension on `/chat/completions`. See
 * https://openrouter.ai/docs/features/multimodal/image-generation —
 * compatible models (google/gemini-2.5-flash-image-preview,
 * openai/gpt-image-1, …) return generated images on `choice.message.images[]`
 * as data URLs or base64 strings.
 *
 * Model catalog is sourced from `GET /models?output_modalities=image`, the
 * only OpenRouter endpoint that filters down to image-generating models.
 */
export class OpenRouterImageProvider implements ImageProvider {
  readonly id = 'openrouter-image';
  readonly displayName = 'OpenRouter (image generation)';
  readonly kind = 'image' as const;

  private readonly log: Logger;

  constructor(private readonly opts: OpenRouterImageProviderOptions) {
    this.log = opts.logger ?? noopLogger;
  }

  async listModels(): Promise<MediaModelInfo[]> {
    try {
      const url = `${this.opts.config.baseUrl}/models?output_modalities=image`;
      const res = await fetch(url, { headers: this.authHeaders() });
      if (!res.ok) {
        this.log.warn('image catalog fetch failed', { status: res.status });
        return [];
      }
      const data = (await res.json()) as OpenRouterModelsResponse;
      const entries = data.data ?? [];
      return entries.map((e) => ({
        id: e.id,
        providerId: this.id,
        displayName: e.name ?? e.id,
        description: e.description?.trim() || undefined,
        // Pricing intentionally omitted: image-generation rates on OpenRouter
        // are typically per-image, not per-token — the host UI is calibrated
        // to chat $/1M-token rates and would mis-label these models. Users
        // can compare prices on openrouter.ai/models directly.
        capabilities: { inputs: { image: true } },
        contextWindow: e.context_length,
      }));
    } catch (err) {
      this.log.warn('image catalog fetch threw', { error: (err as Error).message });
      return [];
    }
  }

  async healthCheck(): Promise<{ ok: boolean; error?: string }> {
    if (!this.opts.config.apiKey) {
      return { ok: false, error: 'No OpenRouter API key configured.' };
    }
    try {
      const res = await fetch(`${this.opts.config.baseUrl}/auth/key`, {
        headers: this.authHeaders(),
      });
      return res.ok ? { ok: true } : { ok: false, error: `HTTP ${res.status}` };
    } catch (err) {
      return { ok: false, error: (err as Error).message };
    }
  }

  async generateImage(
    prompt: string,
    opts?: ImageGenerationOptions,
  ): Promise<ImageGenerationResult> {
    if (!this.opts.config.apiKey) {
      throw new Error('OpenRouter image: no API key configured.');
    }
    const model = (opts?.model ?? this.opts.config.imageModel ?? '').trim();
    if (!model) {
      throw new Error(
        'OpenRouter image: no model selected. Pick an image model in Settings → Models → Image (e.g. google/gemini-2.5-flash-image-preview, openai/gpt-image-1).',
      );
    }

    const body: Record<string, unknown> = {
      model,
      // The host-side dispatcher reads `modalities` to know the response
      // carries images instead of text. Required by OpenRouter for image-
      // capable chat models — without it gemini-image-preview silently
      // returns a text description of the image it would have drawn.
      modalities: ['image', 'text'],
      messages: [{ role: 'user', content: prompt }],
      stream: false,
    };
    if (typeof opts?.n === 'number' && opts.n > 0) body.n = Math.floor(opts.n);
    // Optional escape hatch: caller can stuff provider-specific knobs (size,
    // quality, response_format, …) under `params` and they're merged verbatim.
    // Keeps the API surface narrow while letting power users tune what they
    // need (the host's image-gen UI exposes a "raw params" textarea for this).
    if (opts?.params && typeof opts.params === 'object') {
      for (const [k, v] of Object.entries(opts.params)) {
        if (v !== undefined && v !== null && v !== '') body[k] = v;
      }
    }

    const startedAt = Date.now();
    this.log.info('image generate', {
      model,
      promptLength: prompt.length,
      width: opts?.width,
      height: opts?.height,
      n: opts?.n,
    });

    let res: Response;
    try {
      res = await fetch(`${this.opts.config.baseUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          ...this.authHeaders(),
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
        signal: opts?.signal,
      });
    } catch (err) {
      if ((err as Error).name === 'AbortError') throw err;
      throw new Error(`OpenRouter image fetch failed: ${(err as Error).message}`);
    }
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      throw new Error(`OpenRouter image failed: ${res.status} ${text.slice(0, 500)}`);
    }
    const data = (await res.json()) as ChatImageResponse;
    if (data.error) {
      throw new Error(`OpenRouter image error: ${data.error.message ?? 'unknown'}`);
    }
    const msg = data.choices?.[0]?.message;
    const images = extractImages(msg?.images);
    if (images.length === 0) {
      throw new Error(
        `OpenRouter image: model '${model}' returned no images. Confirm the model supports image output (try google/gemini-2.5-flash-image-preview).`,
      );
    }
    const revisedPrompt =
      typeof msg?.content === 'string' && msg.content.trim().length > 0
        ? msg.content.trim()
        : undefined;
    return {
      images,
      revisedPrompt,
      durationMs: Date.now() - startedAt,
    };
  }

  private authHeaders(): Record<string, string> {
    const h: Record<string, string> = {
      Authorization: `Bearer ${this.opts.config.apiKey}`,
    };
    if (this.opts.config.siteUrl) h['HTTP-Referer'] = this.opts.config.siteUrl;
    if (this.opts.config.appName) h['X-Title'] = this.opts.config.appName;
    return h;
  }
}

function extractImages(
  images: ChatImageEntry[] | undefined,
): Array<{ data: Uint8Array; mimeType: string }> {
  if (!Array.isArray(images)) return [];
  const out: Array<{ data: Uint8Array; mimeType: string }> = [];
  for (const entry of images) {
    let url: string | undefined;
    let raw: string | undefined;
    if (typeof entry === 'string') {
      url = entry;
    } else if (entry && typeof entry === 'object') {
      if (typeof entry.image_url === 'string') {
        url = entry.image_url;
      } else if (entry.image_url && typeof entry.image_url.url === 'string') {
        url = entry.image_url.url;
      }
      if (!url && typeof entry.url === 'string') url = entry.url;
      if (typeof entry.b64_json === 'string') raw = entry.b64_json;
      if (!raw && typeof entry.data === 'string') raw = entry.data;
    }
    let mimeType = 'image/png';
    let bytes: Uint8Array | undefined;
    if (url) {
      const m = url.match(/^data:([^;,]+);base64,(.*)$/);
      if (m) {
        mimeType = m[1] || mimeType;
        bytes = base64ToBytes(m[2]);
      }
      // Non-data URLs (signed CDN links) aren't downloaded inline — leaving
      // that to the caller keeps this provider stateless and avoids an
      // unexpected outbound fetch from a worker thread.
    } else if (raw) {
      bytes = base64ToBytes(raw);
    }
    if (bytes && bytes.byteLength > 0) {
      out.push({ data: bytes, mimeType });
    }
  }
  return out;
}

function base64ToBytes(b64: string): Uint8Array {
  const g = globalThis as {
    Buffer?: { from(b: string, enc: string): Uint8Array };
  };
  if (g.Buffer) return g.Buffer.from(b64, 'base64');
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}
