# Where the telemetry lands — hosting the receiving endpoint

You asked three things, and the third one decides everything:

> will it work if i turn off my machine?

**No. Not with a tunnel.** `ngrok` and `cloudflared` are *proxies to a process
running on your machine*. If your laptop sleeps, the terminal closes, or the
process dies, the URL dies with it. The Windows run happens at a time neither of
us controls, so a tunnel means: if you are not awake with the tunnel up, the
POST fails and you get nothing.

Tunnels are for interactive testing. They are the wrong tool for "send me the
result whenever the run happens."

---

## What actually survives

| Option | Survives your machine being off | Setup | Cost | Retention |
|---|---|---|---|---|
| **ngrok** (free) | ❌ dies with the process; free URLs rotate on restart | 2 min | free | while running |
| **Cloudflare Tunnel** | ❌ still a tunnel to your machine | 5 min | free | while running |
| **webhook.site** | ✅ | **1 min** | free | ~7 days, then gone |
| **Cloudflare Worker + KV** ⭐ | ✅ | ~10 min | free tier | indefinite |
| Vercel / Deno Deploy function | ✅ | ~10 min | free tier | indefinite if you store it |
| GitHub repo via the API | ✅ | ~10 min | free | indefinite |
| Small VPS | ✅ | ~30 min | ~$5/mo | indefinite |

---

## Recommendation

**If the run is happening in the next day or two → webhook.site.**
Fastest possible path, zero code:

1. Go to `https://webhook.site` — it immediately gives you a unique URL.
2. Put it in `.env` (or let `eval.py --telemetry-setup` prompt you):

```
PROCTOR_TELEMETRY_URL=https://webhook.site/your-unique-id
PROCTOR_TELEMETRY_TOKEN=
```

3. When the run happens, the payload appears in the browser on that page.

Good enough to get the numbers. The free tier keeps history for about a week,
and a third party holds the data — fine for metrics, which contain nothing
personal.

**If you want it to just work, indefinitely → a Cloudflare Worker.**
Deploy once, get a permanent HTTPS URL, and it keeps accepting runs whether your
machine is on, off, asleep or in another country. Free tier is 100k requests/day
— this uses one per run.

---

## The Cloudflare Worker, ready to deploy

Two files. Roughly 30 lines.

**`worker.js`**

```js
// Receives proctoring metrics and stores them in KV.
// The sender never transmits video, frames, filenames or per-frame data, and
// this Worker rejects anything that looks like it carries them anyway.

const MAX_BYTES = 2 * 1024 * 1024;
const SCHEMA = "proctor.metrics/1";

// media signatures: unambiguous wherever they appear
const MEDIA = [".mp4", ".avi", ".mov", ".jpg", ".jpeg", ".png", "data:image", "base64,"];
// exact keys that would indicate per-frame data or provenance
const BAD_KEYS = ["eye_ratio", "head_tilt", "body_tilt", "looking_up",
                  "hand_on_table", "phone_detected", "clip_id", "session_id",
                  "video_path", "raw_rows", "frame_idx"];

function keysDeep(o, out = []) {
  if (Array.isArray(o)) { o.forEach(v => keysDeep(v, out)); }
  else if (o && typeof o === "object") {
    for (const [k, v] of Object.entries(o)) { out.push(k); keysDeep(v, out); }
  }
  return out;
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/health") return new Response("ok");

    const auth = request.headers.get("Authorization") || "";
    if (auth !== `Bearer ${env.INGEST_TOKEN}`) {
      return new Response("unauthorized", { status: 401 });
    }

    if (url.pathname === "/runs" && request.method === "GET") {
      const list = await env.RUNS.list();
      return Response.json({
        count: list.keys.length,
        runs: list.keys.map(k => k.name),
      });
    }

    if (url.pathname !== "/ingest" || request.method !== "POST") {
      return new Response("not found", { status: 404 });
    }

    const body = await request.text();
    if (body.length > MAX_BYTES) return new Response("too large", { status: 413 });

    let payload;
    try { payload = JSON.parse(body); }
    catch { return new Response("bad json", { status: 400 }); }

    if (payload.schema !== SCHEMA) {
      return new Response(`bad schema: ${payload.schema}`, { status: 422 });
    }

    // reject media and per-frame data. Keys are checked exactly, not by
    // substring: the ablation table legitimately contains names like
    // "without_eye_ratio", and the protocol string legitimately mentions
    // "clip_id" as the grouping field.
    const lower = body.toLowerCase();
    for (const sig of MEDIA) {
      if (lower.includes(sig)) {
        return new Response(`rejected: media signature ${sig}`, { status: 422 });
      }
    }
    for (const k of keysDeep(payload)) {
      if (BAD_KEYS.includes(k)) {
        return new Response(`rejected: per-frame key ${k}`, { status: 422 });
      }
    }

    const key = `run:${new Date().toISOString()}:${payload.run_id}`;
    await env.RUNS.put(key, body);
    return Response.json({ ok: true, run_id: payload.run_id, key });
  },
};
```

**`wrangler.toml`**

```toml
name = "proctor-telemetry"
main = "worker.js"
compatibility_date = "2026-01-01"

[[kv_namespaces]]
binding = "RUNS"
id = "REPLACE_WITH_THE_ID_FROM_KV_CREATE"
```

**Deploy**

```bash
npm install -g wrangler
wrangler login
wrangler kv namespace create RUNS        # copy the id into wrangler.toml
wrangler secret put INGEST_TOKEN         # paste a long random string
wrangler deploy
```

`wrangler deploy` prints your permanent URL, e.g.
`https://proctor-telemetry.<your-subdomain>.workers.dev`. Then:

```
PROCTOR_TELEMETRY_URL=https://proctor-telemetry.<you>.workers.dev/ingest
PROCTOR_TELEMETRY_TOKEN=<the string you pasted>
```

**Checking it later**

```bash
curl -H "Authorization: Bearer <token>" \
     https://proctor-telemetry.<you>.workers.dev/runs
```

lists every run received. Add a `GET /run/<key>` route reading from KV if you
want to fetch one directly, or just wire the KV namespace to a tiny read
endpoint of your own.

---

## Testing any of it before the real run

A local receiver ships with the pipeline, so you can prove the whole path
without touching the internet:

```bash
# shell 1
uv run python scripts/telemetry_receiver.py --port 8099 --token secret

# shell 2
PROCTOR_TELEMETRY_URL=http://127.0.0.1:8099/ingest \
PROCTOR_TELEMETRY_TOKEN=secret \
uv run python scripts/eval.py --telemetry-allow-http
```

Or check the payload without sending anything at all:

```bash
uv run python scripts/eval.py --telemetry-dry-run
```

---

## How the sender behaves

| | |
|---|---|
| Enabled by | `--telemetry-url`, or `PROCTOR_TELEMETRY_URL`, or `.env` |
| Default | **off.** Nothing is sent unless configured |
| First run | offers to write `.env` when stdin is a terminal; never blocks an unattended run |
| Transport | **HTTPS only** — plain `http://` is refused unless `--telemetry-allow-http` (local testing) |
| Auth | `Authorization: Bearer <token>` from `PROCTOR_TELEMETRY_TOKEN` |
| Timeout | 12 s |
| Payload cap | 2 MB (a real payload is ~5 KB) |
| On failure | **logs and continues.** A dead endpoint never fails the evaluation |
| Retries | none — a missed run is recoverable, the files still exist on that machine |

**Sent:** metrics, environment metadata, dataset shape.
**Never sent:** video, frames, the per-frame feature table, filenames, paths.

Enforced by `test_telemetry_never_sends_features_or_media`, which walks the
payload and asserts no per-frame key or media signature is present — so a future
field cannot leak by accident.

---

## If you would rather not host anything at all

Perfectly reasonable. The pipeline still writes everything to disk on that
machine, and `RUNBOOK.md` §8 lists exactly what to send back:

```
data/results.json          every number the paper needs
data/dataset_labelled.csv  traceability of every frame
figures/*.png, *.pdf       the three figures
each session's summary.json  hardware, frame rates, guardrail settings
```

Telemetry is a convenience — it removes the round trip, nothing more. If hosting
it is friction, skip it and take the files.
