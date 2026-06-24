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
  /** Free-form instructions appended to the transcription prompt (chat-mode only). */
  sttPrompt: string;
};

export type OpenRouterSttOptions = {
  config: OpenRouterSttConfig;
  /** Source for the chat catalog. Multimodal audio-input chat models (Gemini,
   *  gpt-audio, …) are kept as a secondary list so users can still fall back to
   *  them, but the primary picks are OpenRouter's dedicated transcription
   *  models — see {@link OpenRouterSttProvider.listModels}. */
  catalog: Pick<OpenRouterProvider, 'listModels'>;
  logger?: Logger;
};

type OpenAIChatResponse = {
  choices?: Array<{
    message?: { content?: string | null; transcript?: string | null };
  }>;
  error?: { message?: string; code?: string | number };
};

type OpenRouterModelEntry = {
  id: string;
  name?: string;
  description?: string;
  context_length?: number;
  pricing?: { prompt?: string; completion?: string };
};

type OpenRouterModelsResponse = {
  data?: OpenRouterModelEntry[];
};

type OpenAITranscriptionResponse = {
  text?: string;
  language?: string;
  error?: { message?: string; code?: string | number };
};

const DEFAULT_PROMPT =
  'Transcribe the user audio verbatim. Reply ONLY with the transcript text — no preface, no commentary, no quotes.';

const CATALOG_TTL_MS = 60 * 60 * 1000;

/**
 * Speech-to-text backend exposing TWO categories of OpenRouter models:
 *
 *  1. **Dedicated transcription models** (Whisper, gpt-4o-transcribe,
 *     voxtral-mini-transcribe, parakeet, chirp, mai-transcribe, qwen3-asr).
 *     OpenRouter only surfaces these via `GET /models?output_modalities=transcription`
 *     — they are *not* part of the default `/models` payload, which is why a
 *     plain audio-input filter missed them. They are dramatically cheaper than
 *     multimodal chat models (Whisper ~$0.006/min vs gpt-audio at chat token
 *     rates) and accuracy-tuned for transcription, so we surface them FIRST.
 *     They speak the OpenAI `/v1/audio/transcriptions` multipart contract,
 *     not chat completions.
 *
 *  2. **Multimodal audio-input chat models** (Gemini 2.5 family, gpt-audio,
 *     Voxtral, …). Kept as a secondary option for users who need custom
 *     transcription prompts / language hints / multi-step audio reasoning.
 *     These go through `/chat/completions` with `input_audio` content parts.
 *
 * The dispatcher in {@link OpenRouterSttProvider.transcribe} picks the right
 * HTTP path automatically based on whether the chosen model id is in the
 * transcription set.
 */
export class OpenRouterSttProvider implements SttProvider {
  readonly id = 'openrouter-stt';
  readonly displayName = 'OpenRouter (audio model)';
  readonly kind = 'stt' as const;

  private readonly log: Logger;
  // Dispatch map populated by listModels(): for every model that appears in
  // the catalog the user can pick from, we record which HTTP contract it
  // speaks. transcribe() reads this — never guessing from the id — so a model
  // we don't recognise hard-fails with a clear message instead of silently
  // hitting the wrong endpoint and surfacing an opaque OpenRouter 500.
  private endpointByModel = new Map<string, 'audio' | 'chat'>();
  private catalogFetchedAt = 0;

  constructor(private readonly opts: OpenRouterSttOptions) {
    this.log = opts.logger ?? noopLogger;
  }

  /**
   * Returns dedicated transcription models FIRST (Whisper, gpt-4o-transcribe,
   * …), then audio-input chat models that aren't already in the transcription
   * set. The host renders this in Settings → Models → Voice.
   */
  async listModels(): Promise<MediaModelInfo[]> {
    const [transcription, chat] = await Promise.all([
      this.fetchTranscriptionModels(),
      this.opts.catalog.listModels().catch((err) => {
        this.log.warn('chat catalog unavailable for STT list', {
          error: (err as Error).message,
        });
        return [];
      }),
    ]);
    const transcriptionIds = new Set(transcription.map((m) => m.id));
    const chatAudio: MediaModelInfo[] = chat
      .filter(
        (m) => m.capabilities?.inputs?.audio === true && !transcriptionIds.has(m.id),
      )
      .map((m) => ({
        id: m.id,
        providerId: this.id,
        displayName: m.displayName,
        description: m.description,
        pricing: m.pricing,
        capabilities: m.capabilities,
        contextWindow: m.contextWindow,
      }));
    // Rebuild dispatch map atomically with the catalog so transcribe() never
    // routes a model the user just picked through stale state.
    const next = new Map<string, 'audio' | 'chat'>();
    for (const m of transcription) next.set(m.id, 'audio');
    for (const m of chatAudio) next.set(m.id, 'chat');
    this.endpointByModel = next;
    this.catalogFetchedAt = Date.now();
    return [...transcription, ...chatAudio];
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
    const model = (opts?.model ?? this.opts.config.sttModel ?? '').trim();
    if (!model) {
      throw new Error(
        'OpenRouter STT: no model selected. Pick a transcription model in Settings → Models → Voice (e.g. openai/whisper-1, mistralai/voxtral-mini-transcribe, openai/gpt-4o-mini-transcribe).',
      );
    }
    const endpoint = await this.ensureEndpoint(model);
    if (!endpoint) {
      throw new Error(
        `OpenRouter STT: model '${model}' is not in the catalog of transcription or audio-input chat models. Pick a different model in Settings → Models → Voice, or check that your OpenRouter key has access to it.`,
      );
    }
    return endpoint === 'audio'
      ? this.transcribeViaAudioEndpoint(audio, mimeType, model, opts)
      : this.transcribeViaChatCompletions(audio, mimeType, model, opts);
  }

  private async transcribeViaAudioEndpoint(
    audio: Uint8Array,
    mimeType: string,
    model: string,
    opts?: SttOptions,
  ): Promise<SttResult> {
    // OpenRouter's /audio/transcriptions endpoint expects a JSON body with
    // the audio inlined as base64 under `input_audio` — it does NOT accept
    // OpenAI's multipart/form-data contract and 400s with
    // "invalid content-type: multipart/form-data; …" when given one.
    // See https://github.com/OpenRouterTeam/skills/tree/main/skills/openrouter-stt
    const format = audioFormatFromMime(mimeType);
    const body: Record<string, unknown> = {
      model,
      input_audio: { data: bytesToBase64(audio), format },
    };
    if (opts?.language && opts.language !== 'auto') {
      body.language = opts.language;
    }

    this.log.info('stt transcribe (audio endpoint)', {
      model,
      mimeType,
      format,
      bytes: audio.byteLength,
      language: opts?.language,
    });

    const startedAt = Date.now();
    let res: Response;
    try {
      res = await fetch(`${this.opts.config.baseUrl}/audio/transcriptions`, {
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
    const data = (await res.json()) as OpenAITranscriptionResponse;
    if (data.error) {
      throw new Error(`OpenRouter STT error: ${data.error.message ?? 'unknown'}`);
    }
    return {
      text: (data.text ?? '').trim(),
      language: data.language ?? opts?.language,
      durationMs: Date.now() - startedAt,
    };
  }

  private async transcribeViaChatCompletions(
    audio: Uint8Array,
    mimeType: string,
    model: string,
    opts?: SttOptions,
  ): Promise<SttResult> {
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

    this.log.info('stt transcribe (chat completions)', {
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

  /**
   * Hit `/models?output_modalities=transcription` for the dedicated transcription
   * catalog. This endpoint is the only way to see Whisper / gpt-4o-transcribe /
   * voxtral-mini-transcribe & friends — the plain `/models` payload omits them
   * because their output modality is `transcription`, not `text`.
   */
  private async fetchTranscriptionModels(): Promise<MediaModelInfo[]> {
    try {
      const url = `${this.opts.config.baseUrl}/models?output_modalities=transcription`;
      const res = await fetch(url, { headers: this.authHeaders() });
      if (!res.ok) {
        this.log.warn('transcription catalog fetch failed', { status: res.status });
        return [];
      }
      const data = (await res.json()) as OpenRouterModelsResponse;
      const entries = data.data ?? [];
      return entries.map((e) => ({
        id: e.id,
        providerId: this.id,
        displayName: e.name ?? e.id,
        description: e.description?.trim() || undefined,
        // Pricing intentionally omitted: OpenRouter mixes per-second (Whisper,
        // Voxtral) and per-token (gpt-4o-transcribe) rates on this endpoint
        // and the host UI is calibrated to chat $/1M-token rates, so any
        // single conversion would mis-label one family or the other. Users
        // can compare costs on openrouter.ai/models directly.
        // Mark as audio-input so the host's ModelPicker (which filters by
        // `inputs.audio`) doesn't drop these models — OpenRouter doesn't
        // expose `inputs` on this endpoint but every entry here is audio-in
        // by definition.
        capabilities: { inputs: { audio: true } },
        contextWindow: e.context_length,
      }));
    } catch (err) {
      this.log.warn('transcription catalog fetch threw', {
        error: (err as Error).message,
      });
      return [];
    }
  }

  /**
   * Resolve the HTTP endpoint to use for `model`, refreshing the catalog if
   * stale or if the model isn't known yet. Returns `undefined` when the model
   * is genuinely absent from OpenRouter's catalog — `transcribe()` turns that
   * into a clear user-facing error rather than silently picking the wrong
   * endpoint and letting OpenRouter 500.
   */
  private async ensureEndpoint(model: string): Promise<'audio' | 'chat' | undefined> {
    const fresh =
      this.endpointByModel.size > 0 &&
      Date.now() - this.catalogFetchedAt < CATALOG_TTL_MS;
    if (!fresh) {
      await this.listModels();
      return this.endpointByModel.get(model);
    }
    const cached = this.endpointByModel.get(model);
    if (cached) return cached;
    // Catalog was fresh but missing the model — maybe added since the last
    // refresh. Force one re-fetch before giving up.
    await this.listModels();
    return this.endpointByModel.get(model);
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

