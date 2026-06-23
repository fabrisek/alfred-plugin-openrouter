import type {
  Logger,
  MediaModelInfo,
  SttOptions,
  SttProvider,
  SttResult,
} from '@alfred/sdk';
import type { OpenRouterProvider } from './OpenRouterProvider.js';

const noopLogger: Logger = {
  debug: () => {},
  info: () => {},
  warn: () => {},
  error: () => {},
};

export type OpenRouterSttConfig = {
  apiKey: string;
  baseUrl: string;
  appName: string;
  siteUrl: string;
  /**
   * Optional fallback model id, used only when the host has not yet picked one
   * in Settings → Models. The supported path is for the host to enumerate
   * {@link OpenRouterSttProvider.listModels} and pass `opts.model` at transcribe
   * time — this field is kept for backwards compat / installs that pre-date the
   * media model picker.
   */
  sttModel: string;
  /** Free-form instructions appended to the transcription prompt. */
  sttPrompt: string;
};

export type OpenRouterSttOptions = {
  config: OpenRouterSttConfig;
  /** Source for the model catalog. The STT provider reuses the chat provider's
   *  `listModels()` so the cache / filter / API key configured on the LLM side
   *  apply consistently — no second `/models` round-trip. */
  catalog: Pick<OpenRouterProvider, 'listModels'>;
  logger?: Logger;
};

type OpenAIChatResponse = {
  choices?: Array<{
    message?: { content?: string | null; transcript?: string | null };
  }>;
  error?: { message?: string; code?: string | number };
};

const DEFAULT_PROMPT =
  'Transcribe the user audio verbatim. Reply ONLY with the transcript text — no preface, no commentary, no quotes.';

/**
 * Speech-to-text backend that piggybacks on an audio-input-capable OpenRouter
 * model. OpenRouter does not expose `/audio/transcriptions`; instead we send
 * `{ type: 'input_audio', input_audio: { data, format } }` content parts to
 * `/chat/completions`, which Gemini-family and the OpenAI audio models all
 * accept. The model id is configurable per Settings → Plugins → OpenRouter.
 */
export class OpenRouterSttProvider implements SttProvider {
  readonly id = 'openrouter-stt';
  readonly displayName = 'OpenRouter (audio model)';
  readonly kind = 'stt' as const;

  private readonly log: Logger;

  constructor(private readonly opts: OpenRouterSttOptions) {
    this.log = opts.logger ?? noopLogger;
  }

  /**
   * Subset of the chat catalog that can ingest audio — i.e. every model whose
   * OpenRouter `input_modalities` includes `'audio'`. The host renders this in
   * Settings → Models → Voice so users pick from a live, accurate list instead
   * of guessing the model id by hand.
   */
  async listModels(): Promise<MediaModelInfo[]> {
    const all = await this.opts.catalog.listModels();
    return all
      .filter((m) => m.capabilities?.inputs?.audio === true)
      .map((m) => ({
        id: m.id,
        providerId: this.id,
        displayName: m.displayName,
        description: m.description,
        pricing: m.pricing,
        capabilities: m.capabilities,
        contextWindow: m.contextWindow,
      }));
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

  async transcribe(
    audio: Uint8Array,
    mimeType: string,
    opts?: SttOptions,
  ): Promise<SttResult> {
    if (!this.opts.config.apiKey) {
      throw new Error('OpenRouter STT: no API key configured.');
    }
    // Host-resolved model wins (Settings → Models → Voice). The plugin's
    // legacy `sttModel` config only kicks in for installs that pre-date the
    // media model picker — kept as a fallback so we don't break those.
    const model = (opts?.model ?? this.opts.config.sttModel ?? '').trim();
    if (!model) {
      throw new Error(
        'OpenRouter STT: no model selected. Pick an audio-input-capable model in Settings → Models → Voice (e.g. openai/gpt-4o-mini-transcribe, google/gemini-2.0-flash-001).',
      );
    }
    const format = audioFormatFromMime(mimeType);
    const base64 = bytesToBase64(audio);
    const prompt = this.opts.config.sttPrompt.trim() || DEFAULT_PROMPT;
    const languageHint =
      opts?.language && opts.language !== 'auto'
        ? ` Target language: ${opts.language}.`
        : '';

    const startedAt = Date.now();
    const body = {
      model,
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: `${prompt}${languageHint}` },
            { type: 'input_audio', input_audio: { data: base64, format } },
          ],
        },
      ],
      // Transcription is one-shot; a long output is almost always the model
      // narrating instead of transcribing, so clamp.
      max_tokens: 4096,
      stream: false,
    };

    this.log.info('stt transcribe request', {
      model,
      mimeType,
      format,
      bytes: audio.byteLength,
      language: opts?.language,
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
      throw new Error(`OpenRouter STT fetch failed: ${(err as Error).message}`);
    }
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      throw new Error(
        `OpenRouter STT failed: ${res.status} ${text.slice(0, 500)}`,
      );
    }
    const data = (await res.json()) as OpenAIChatResponse;
    if (data.error) {
      throw new Error(`OpenRouter STT error: ${data.error.message ?? 'unknown'}`);
    }
    const choice = data.choices?.[0]?.message;
    const text =
      (typeof choice?.transcript === 'string' && choice.transcript) ||
      (typeof choice?.content === 'string' && choice.content) ||
      '';
    return {
      text: text.trim(),
      language: opts?.language,
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

// Maps the MediaRecorder / file mime types Alfred sees in practice onto the
// `format` token the OpenAI-compatible audio content part expects. Falls back
// to 'wav' (the most universally accepted) when unknown — the upstream model
// usually still copes via container sniffing.
function audioFormatFromMime(mimeType: string): string {
  const m = mimeType.toLowerCase();
  if (m.includes('wav')) return 'wav';
  if (m.includes('mp3') || m.includes('mpeg')) return 'mp3';
  if (m.includes('webm')) return 'webm';
  if (m.includes('ogg')) return 'ogg';
  if (m.includes('m4a') || m.includes('mp4')) return 'm4a';
  if (m.includes('flac')) return 'flac';
  return 'wav';
}

function bytesToBase64(bytes: Uint8Array): string {
  // Node 18+ has Buffer; browsers/edge runtimes use btoa with a chunked
  // string to avoid blowing the stack on multi-MB audio.
  const g = globalThis as { Buffer?: { from(b: Uint8Array): { toString(enc: string): string } } };
  if (g.Buffer) return g.Buffer.from(bytes).toString('base64');
  let binary = '';
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const slice = bytes.subarray(i, Math.min(i + chunkSize, bytes.length));
    binary += String.fromCharCode(...slice);
  }
  return btoa(binary);
}
