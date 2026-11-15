import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicBufferedEvents,
  anthropicError,
  anthropicModelList,
  anthropicToOpenAI,
  openAIBufferedStream,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this baseline reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** Maximum number of dials allowed per prompt. */
const MAX_DIALS = 6;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
let manifestIndex = 0;
/** Per-model trial-level cooldown set for models that refused. */
// Removed trialRefusedModels

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/** Rank eligible models by rating_milli desc, then context_length desc, then brief order. */
function eligibleModelsFor(requires: Requirement, refusedSet: Set<string>): ModelRow[] {
  const allFree = freeModels(brief);
  const eligible = allFree.filter((m) => meetsRequirements(requires, capabilitiesOf(m)));
  
  const sorted = [...eligible].sort((a, b) => {
    if (b.rating_milli !== a.rating_milli) {
      return b.rating_milli - a.rating_milli;
    }
    const ctxA = a.context_length ?? 0;
    const ctxB = b.context_length ?? 0;
    if (ctxB !== ctxA) {
      return ctxB - ctxA;
    }
    return 0;
  });

  return sorted.filter((m) => !refusedSet.has(m.id));
}

function adjustBody(body: Json): Json {
  const out = { ...body };
  const hasMaxTokens = typeof out.max_tokens === 'number' && Number.isFinite(out.max_tokens);
  const hasMaxCompletionTokens = typeof out.max_completion_tokens === 'number' && Number.isFinite(out.max_completion_tokens);
  if (!hasMaxTokens && !hasMaxCompletionTokens) {
    out.max_tokens = 1024;
  } else {
    if (hasMaxTokens) out.max_tokens = Math.min(out.max_tokens as number, 8192);
    if (hasMaxCompletionTokens) out.max_completion_tokens = Math.min(out.max_completion_tokens as number, 8192);
  }
  return out;
}

function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  sendJsonWithHeaders(res, status, body, {});
}

function sendJsonWithHeaders(res: ServerResponse, status: number, body: unknown, extraHeaders: Record<string, string>): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    'content-length': String(Buffer.byteLength(text)),
    ...extraHeaders,
  };
  res.writeHead(status, headers);
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), { method: 'POST', headers, body: JSON.stringify({ ...body, model }) });
}

function refusedWithFailures(
  res: ServerResponse,
  wire: 'openai' | 'anthropic',
  status: number,
  text: string,
  failures: Array<{ status: number; code: string; model: string | null }>,
): void {
  const headers: Record<string, string> = {};
  if (failures.length > 0) {
    headers['x-makefun-upstream-failures'] = JSON.stringify(failures.slice(0, 8));
  }
  if (wire === 'anthropic') {
    const errBody = anthropicError('api_error', `the gateway answered HTTP ${status}${text === '' ? '' : `: ${text.slice(0, 500)}`}`);
    sendJsonWithHeaders(res, 502, errBody, headers);
  } else {
    const errBody = openAIError(`the gateway answered HTTP ${status}${text === '' ? '' : `: ${text.slice(0, 500)}`}`, 'upstream_error', null);
    sendJsonWithHeaders(res, 502, errBody, headers);
  }
}

async function handlePromptRequest(req: IncomingMessage, res: ServerResponse, wire: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, wire, 400, 'invalid_request_error', 'the request body is not one JSON object');
  }

  const requires = requirementsFor(body);
  const adjustedBody = adjustBody(body);

  const failures: Array<{ status: number; code: string; model: string | null }> = [];
  const ladder = eligibleModelsFor(requires, new Set<string>());
  let dialCount = 0;
  let successUpstream: Response | null = null;
  let chosenRoute: ModelRow | null = null;
  let lastStatus = 502;
  let lastText = 'no eligible model succeeded after failover';

  outer: for (const model of ladder) {
    let triedThisModel = false;
    while (true) {
      if (dialCount >= MAX_DIALS) break outer;
      dialCount += 1;

      let upstream: Response;
      let status = 502;
      let text = '';
      let code = 'upstream_unreachable';
      const dialBody = wire === 'anthropic' ? anthropicToOpenAI(adjustedBody, model.id) : { ...adjustedBody, model: model.id };

      try {
        upstream = await dial(model.id, dialBody);
      } catch (err) {
        failures.push({
          status: 502,
          code: 'upstream_unreachable',
          model: model.id,
        });
        lastStatus = 502;
        lastText = 'upstream_unreachable';

        if (!triedThisModel) {
          triedThisModel = true;
          continue;
        }
        break;
      }

      if (upstream.ok && upstream.body !== null) {
        successUpstream = upstream;
        chosenRoute = model;
        break outer;
      }

      status = upstream.status;
      text = await upstream.text();
      try {
        const json = JSON.parse(text);
        if (isRecord(json)) {
          if (isRecord(json.error) && typeof json.error.code === 'string') {
            code = json.error.code;
          } else if (typeof json.code === 'string') {
            code = json.code;
          }
        }
      } catch {}

      failures.push({
        status,
        code,
        model: model.id,
      });
      lastStatus = status;
      lastText = text;

      if (!triedThisModel && (status === 429 || status >= 500)) {
        triedThisModel = true;
        continue;
      }
      break;
    }
  }

  if (successUpstream === null || chosenRoute === null) {
    return refusedWithFailures(res, wire, lastStatus, lastText, failures);
  }

  const headers: Record<string, string> = {};
  if (failures.length > 0) {
    headers['x-makefun-upstream-failures'] = JSON.stringify(failures.slice(0, 8));
  }

  if (wire === 'openai') {
    if (adjustedBody.stream === true) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
      try {
        for await (const frame of openAIBufferedStream(successUpstream.body, chosenRoute.id)) res.write(frame);
      } finally {
        res.end();
      }
      return;
    }
    const answer: unknown = await successUpstream.json();
    sendJsonWithHeaders(res, 200, openAIJsonWithModel(isRecord(answer) ? answer : {}, chosenRoute.id), headers);
  } else {
    if (adjustedBody.stream === true) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
      try {
        for await (const event of anthropicBufferedEvents(successUpstream.body, chosenRoute.id)) res.write(event);
      } finally {
        res.end();
      }
      return;
    }
    const answer: unknown = await successUpstream.json();
    sendJsonWithHeaders(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, chosenRoute.id), headers);
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await handlePromptRequest(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await handlePromptRequest(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // Retry on next tick
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
