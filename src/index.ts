import type { AlfredPlugin, PluginContext } from '@alfred/sdk';
import { OpenRouterProvider, type OpenRouterConfig } from './OpenRouterProvider.js';

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

const plugin: AlfredPlugin = {
  id: 'openrouter',
  name: 'OpenRouter',
  version: '0.1.0',

  async activate(ctx: PluginContext) {
    const cfg = resolveConfig(ctx.config);
    if (!cfg.apiKey) {
      ctx.logger.warn(
        'OpenRouter plugin loaded without an API key — set it in Settings → Plugins → OpenRouter.',
      );
    }

    const provider = new OpenRouterProvider({ config: cfg, logger: ctx.logger });
    ctx.registerProvider(provider);

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

    ctx.logger.info('OpenRouter plugin activated', {
      baseUrl: cfg.baseUrl,
      modelFilter: cfg.modelFilter || '(none)',
      freeOnly: cfg.freeOnly,
    });
  },
};

export default plugin;
