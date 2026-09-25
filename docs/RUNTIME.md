# Runtime implementation, 25 September 2026

The shipped Python package contains an actual MLX cache writer/reader and chat worker, not only a design checker.
No real model was loaded to validate this implementation; its current verification is model-free.

## Components

| Module | Responsibility |
| --- | --- |
| `cli.py` | Installable command, argument validation, local model paths and worker cleanup |
| `http.py` | Loopback chat/SSE endpoint, authenticated controls, directory advisory lease |
| `service.py` | Serialized admission, process ownership, activation, idle shutdown and recovery |
| `worker.py` | Spawned process and request/event transport; offline model runtime |
| `runtime.py` | Rendering, exact token match, prefix capture, session coverage and stream assembly |
| `learning.py` | Optional capture modes and message boundaries, independent of the model backend |
| `mlx_backend.py` | Local MLX load, materialized prefill, controlled cache codec and sampling |
| `identity.py` | Weight/config/tokenizer/template content hashes and runtime compatibility |
| `recipes.py` | Ordered UTF-8 files and explicit role recipes |
| `store.py` | Atomic profile publication, immutable generations and integrity checking |

The controller imports no neural runtime; only its owned child imports MLX when computation is requested.
Requests serialize through one worker, retaining one completed session.
Snapshot controls wait behind a generation; there is no mid-response checkpoint protocol in this version.
Stopping or abandoning a partial stream terminates that owned worker rather than reusing uncertain state.
A failed pending-tail computation evicts the damaged session; retry cannot process the tail twice.
Failed model switches restore the previous selection and wake that model lazily on its next request.
An unexpected worker exit is reported; responses are never invisibly regenerated.

`use` verifies and materializes the selected state, staging it for the next request to consume.
That cache is never shared mutably across requests; subsequent requests load a fresh copy from disk.
Changing the profile pointer invalidates a staged generation.
A matching prepared cache therefore saves one restore, without doubling cache residency just to clone a baseline.

## Correct state boundary

`advance` calls the model on complete chunks of up to 512 tokens and evaluates logits and layer state before returning.
This avoids MLX generation helpers that leave part of the supplied prompt pending.
The stream tracks both committed token IDs and the count absorbed by attention/recurrent layers.
The final sampled non-EOS token can remain pending; committed export absorbs precisely that tail without invoking a sampler.
Absorbed export saves the shorter token history that the current state actually represents.
EOS is not added to the emitted continuation or advertised as absorbed when it was only sampled.

The adapter serializes `KVCache.state`/`meta_state` and all `ArraysCache` recurrent arrays,
including `left_padding` and `lengths` when present.
Restoration accepts only those exact installed classes, checks layer count and positions,
and materializes arrays on the worker thread before generation.
Rotating, batched and quantized cache classes need separate qualified codecs and currently fail explicitly.

## Rendering and reuse

Recipes render the complete conversation through the model's chat template; blocks are not separately tokenized and concatenated.
Three distinct probe user messages find a conservative shared token prefix.
Every actual request is rendered again and must match every saved token before state is reused.
The probes are only a candidate boundary, never the authority for a cache hit.
A recipe owns its fixed messages and template options; a captured instruction prefix expects a full incoming request.
All requests preserve room for the requested output within the declared context limit.

Learning is off by default and has two explicit modes, with `False` disabling it and the legacy Python value `True`
mapping to `first-turn`.
In `first-turn` mode, only raw requests consisting of leading system/developer messages plus
one final user message can publish a stable instruction prefix.
Earlier user/assistant history disqualifies that request from capture without preventing exact-prefix reuse.
Later eligible fresh requests with changed instructions can replace the selected learned prefix.

In `continuous` mode, the capture candidate contains the supplied messages before the newest
final user message, including earlier user/assistant history.
It excludes the current user message and the newly generated answer.
The renderer probes and exact-prefix check apply to these longer candidates too;
a request without an eligible candidate does not publish a new prefix.
Learning remains configured after successful requests and worker sleep, and explicit activation disables it.
The status response reports `learning_mode` as `off`, `first-turn` or `continuous`.
Automatic capture cannot overwrite computed recipe or continuation profiles.
Each learned profile selects its latest published generation rather than searching old variants,
and superseded immutable artifacts remain until a future garbage-collection facility is added.

Only plain text and boolean `enable_thinking` template configuration are supported.
Sampling supports greedy/default, temperature, top-p, top-k and a seed; unknown options are errors.
The current compute and serve operations reject tools.

## Actual artifact format

The product, Python package and CLI were renamed from Afterglow to Imprint on 25 September 2026.
Persisted format IDs and the safetensors metadata key keep their original names so existing artifacts and recipes remain readable.
The default store is now `~/.cache/imprint`; select an older store explicitly with `--store ~/.cache/afterglow`.
New HTTP headers and control routes use Imprint names; restart an older controller with the renamed CLI before using its controls.

Runtime artifacts use **`afterglow.store.v1`**, with profiles using `afterglow.profile.v1`.

```text
STORE/
  profiles/NAME.json
  artifacts/SHA256/
    manifest.json
    tokens.json
    state.safetensors
  control.json
```

The manifest hashes both payloads and records model location, content identity, mode, token count and optional resolved recipe.
The artifact directory name hashes the canonical manifest.
The MLX safetensors metadata carries `afterglow.mlx.v1` layer descriptors with tensor references and scalar state.
No pickle, code execution, arbitrary class import, or guessed recurrent-state reconstruction is used.
Writes complete and synchronize a staging generation before atomically replacing a profile pointer.
Older generations remain immutable; garbage collection is not implemented.

`control.json` is a private control address/token written only while a server owns the store.
A nonblocking advisory lock on the existing directory prevents competing controllers; there is no lock or guard file.
The chat endpoint is loopback-only and rejects browser Origin headers; control operations additionally require the local bearer token.
Weights, tokenizer assets, cache tensors and recipes stay on disk locally; no publication occurs.

## What the tests establish

The tests run real CLI parsing, filesystem publication, HTTP sockets and Python child processes.
Their backend represents state as token lists and uses deterministic symbolic continuations,
so equality checks establish orchestration and coverage, not neural correctness.
Codec tests execute the production encoder/decoder with structural tensor objects and source-compatible cache stubs;
they do not import MLX or claim a real safetensors round trip.

The remaining release qualification is an authorized real-model run:
save a prefix, compare full fresh and restored continuation outputs at several lengths,
exercise mixed recurrent/attention state, inspect actual child/Metal memory after sleep,
and record measured warm and reload-inclusive TTFT with raw traces.
No current document claims that qualification has occurred.
