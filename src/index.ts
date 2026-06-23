import type { AlfredPlugin, PluginContext } from '@alfred/sdk';
import { OpenRouterProvider, type OpenRouterConfig } from './OpenRouterProvider.js';
import {
  OpenRouterSttProvider,
  type OpenRouterSttConfig,
} from './OpenRouterSttProvider.js';

function resolveConfig(raw: Record<string, unknown>): OpenRouterConfig {
  const baseUrl = String(raw.baseUrl ?? '').trim() || 'https://openrouter.ai/api/v1';
  return {
    apiKey: String(raw.apiKey ?? '').trim(),
    baseUrl: baseUrl.replace(/\/+$/, ''),
    appName: String(raw.appName ?? 'Alfred'),
    siteUrl: String(raw.siteUrl ?? ''),
    modelFilter: String(raw.modelFilter ?? ''),
    freeOnly: Boolean(raw.freeOnly),
    modelsCacheMinutes:
      typeof raw.modelsCacheMinutes === 'number' && raw.modelsCacheMinutes >= 0
        ? raw.modelsCacheMinutes
        : 60,
  };
}

function resolveSttConfig(raw: Record<string, unknown>): OpenRouterSttConfig {
  const baseUrl = String(raw.baseUrl ?? '').trim() || 'https://openrouter.ai/api/v1';
  return {
    apiKey: String(raw.apiKey ?? '').trim(),
    baseUrl: baseUrl.replace(/\/+$/, ''),
    appName: String(raw.appName ?? 'Alfred'),
    siteUrl: String(raw.siteUrl ?? ''),
    sttModel: String(raw.sttModel ?? '').trim(),
    sttPrompt: String(raw.sttPrompt ?? ''),
  };
}

const plugin: AlfredPlugin = {
  id: 'openrouter',
  name: 'OpenRouter',
  version: '0.2.0',

  async activate(ctx: PluginContext) {
    const cfg = resolveConfig(ctx.config);
    if (!cfg.apiKey) {
      ctx.logger.warn(
        'OpenRouter plugin loaded without an API key — set it in Settings → Plugins → OpenRouter.',
      );
    }

    const provider = new OpenRouterProvider({ config: cfg, logger: ctx.logger });
    ctx.registerProvider(provider);

    const sttCfg = resolveSttConfig(ctx.config);
    const sttProvider = new OpenRouterSttProvider({
      config: sttCfg,
      logger: ctx.logger,
    });
    ctx.registerMediaProvider(sttProvider);

    ctx.registerAction('testConnection', async () => {
      if (!cfg.apiKey) {
        return { ok: false, error: 'No API key configured.' };
      }
      try {
        const res = await fetch(`${cfg.baseUrl}/auth/key`, {
          headers: { Authorization: `Bearer ${cfg.apiKey}` },
        });
        if (!res.ok) {
          const text = await res.text().catch(() => '');
          return { ok: false, error: `HTTP ${res.status}: ${text.slice(0, 200)}` };
        }
        const data = (await res.json()) as {
          data?: { label?: string; limit?: number; usage?: number };
        };
        return {
          ok: true,
          label: data.data?.label,
          limit: data.data?.limit,
          usage: data.data?.usage,
        };
      } catch (err) {
        return { ok: false, error: (err as Error).message };
      }
    });

    ctx.registerAction('listModelsRaw', async () => {
      try {
        const models = await provider.listModels();
        return { ok: true, count: models.length, models };
      } catch (err) {
        return { ok: false, error: (err as Error).message };
      }
    });

    // Vendor list derived live from the catalog — never hardcoded. The model
    // picker / settings UI can call this to populate a vendor dropdown that
    // reflects exactly what OpenRouter serves today (e.g. 'anthropic/',
    // 'openai/', 'deepseek/', ...), so the list can't go stale.
    ctx.registerAction('listVendors', async () => {
      try {
        const models = await provider.listModels();
        const vendors = Array.from(
          new Set(
            models
              .map((m) => {
                const slash = m.id.indexOf('/');
                return slash > 0 ? m.id.slice(0, slash + 1) : '';
              })
              .filter((v) => v.length > 0),
          ),
        ).sort((a, b) => a.localeCompare(b));
        return { ok: true, count: vendors.length, vendors };
      } catch (err) {
        return { ok: false, error: (err as Error).message };
      }
    });

    ctx.logger.info('OpenRouter plugin activated', {
      baseUrl: cfg.baseUrl,
      modelFilter: cfg.modelFilter || '(none)',
      freeOnly: cfg.freeOnly,
    });
  },
};

export default plugin;
