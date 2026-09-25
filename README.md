![Imprint: 63× faster time to first token, from 19s to 0.3s; author-reported result with cached context.](assets/imprint-63x.png)

# Imprint

**Hot-swappable memory for local AI.**

Compute your repeated context once, save its model state, and restore it after the model shuts down.
A matching request processes its new suffix instead of recomputing the saved prefix.

**Reported TTFT: 19 seconds → 0.3 seconds, about 63× faster.**
This author-reported reference result has not yet been independently reproduced with the standalone CLI; see [measurement status](docs/EVIDENCE.md).

Imprint is designed to reduce time to first token wherever you repeatedly load the same context:
agentic harnesses, large document collections, codebase context and long-lived project memory.
You can also prepare the same source context for several compatible models and switch between
their saved states, without keeping every model loaded or processing that context again on every switch.

## When to use Imprint

- **Repeated agent prompts:** save fixed instructions and project context so fresh turns can skip their repeated prefill.
- **Massive context:** precompute a large memory file or collection of reference files, then return to it across sessions and model restarts.
- **The same context across models:** build a separate cache for each model from the same files or recipe, then hot-swap the model and its prepared context with `imprint use`.

The source context is shared; each model has its own computed state.
Every selected model must support that context length and fit it in available memory.
Switching still incurs weight-loading and cache-reading time, even when it avoids processing the context again.

**Version 0.1 now contains an installable CLI, an MLX backend and a local chat server.**
The storage, CLI, HTTP and worker lifecycle have automated tests using a symbolic backend;
the MLX adapter has source-reviewed API integration and model-free codec tests.
Standalone real-model equivalence and speed measurements have **not** yet been completed.

## Compute your own blob

Build a blob locally from the context you want available when a session opens:
your harness's instructions, your assistant's memory, or the files your model keeps rereading.
Imprint provides the CLI; you supply the context and a compatible local model.
There is no universal OMP blob to download.

Choose the starting point that matches your setup:

| Your starting point | How to save it |
| --- | --- |
| Memory files, documents or project notes | Use `imprint compute --files` to prepare them before opening a session. |
| Instructions with specific message roles | Use `imprint compute --recipe` to preserve their order and roles. |
| A harness that sends its opening instructions | Use `imprint serve --learn first-turn` to capture a fresh request's instruction prefix. |
| A session already held by the Imprint worker | Use `imprint compute --session` to export its retained state. |

The initial computation still takes time; you do it ahead of use or during the first learning request.
Later matching sessions can restore that work and process only the new suffix.
Reuse requires matching rendered prefix tokens and compatible model, tokenizer and runtime settings.
Changed instructions or a different model can require a new computation.

Keep personal blobs private: they can contain source text, recoverable token IDs, local paths and computed state derived from private context.
Build any public example from reviewed public inputs; removing readable metadata alone does not establish that a private blob is safe to share.

## Install

Use Apple silicon macOS, Python 3.11 or newer, and an already downloaded MLX model directory:

```sh
git clone https://github.com/ashhart/Imprint.git imprint
cd imprint
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[mlx]'
imprint --help
```

The package is named `imprint-cache`; the command is `imprint`.
The backend targets MLX 0.31.2 and MLX LM 0.31.3 with ordinary `KVCache` and hybrid `ArraysCache` layers.
It rejects other cache classes rather than saving incomplete state.
Models need a chat template and an explicit context limit in their local configuration.
No automatic model downloads or remote model code are enabled.

## Example: preload your assistant's memory

Suppose your assistant starts with 300,000 tokens of reference material: your notes, project history and documents.
You want that context ready across new sessions without making the model process the whole collection again each time.
Put the material in your own files and prepare a saved profile:

```sh
imprint compute --files assistant-memory.md project-notes.md \
  --model /absolute/path/to/mlx-model --name assistant-memory
imprint serve --name assistant-memory --idle-unload 5m
```

`compute` performs real model prefill when you run it, saves the blob, then exits its model worker.
The files are copied into an embedded recipe in the supplied order, preserving their text.
For explicit roles and template settings, use `--recipe examples/agent.recipe.json` instead of `--files`.

The 300,000-token example is conditional, not a tested capacity claim:
the model must support the full rendered context plus your new messages and answer reserve,
and your machine needs enough RAM for the model, active cache and working allocations, plus disk space for the saved state.
Saving a blob does not extend the context window or turn the source material into a smaller summary.
Token counts depend on the selected model's tokenizer.

`serve` starts the endpoint; its model worker loads on demand.
To load the model ahead of your first question, run this in a second terminal while the server is running:

```sh
imprint use assistant-memory
```

Requests still restore their own cache state and process new tokens, so disk and generation latency remain.
After idle unloading, the next request also pays model-loading time unless you activate the profile ahead of it.

Send dynamic messages to the running server; it supplies the stored recipe:

```sh
curl http://127.0.0.1:8460/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"assistant-memory","messages":[{"role":"user","content":"What should I work on next?"}],"max_tokens":128}'
```

The response includes `usage.prompt_tokens_details.cached_tokens` and an `X-Imprint-Session-ID` header.
Set `"stream":true` for OpenAI-style server-sent events.
A full request that already begins with the exact recipe messages is also accepted without adding them twice.
A conflicting system prompt is rejected.

## Connect your own harness

Point your harness's OpenAI-compatible chat client at `http://127.0.0.1:8460/v1`
and set its model to the saved profile name, or `imprint` to follow the selected profile.
Run Imprint as the local model server and let the harness send requests to it.
An SDK that requires an API-key value can use a non-secret placeholder such as `local`;
the chat endpoint is loopback-only and is separate from the authenticated management API.

For fixed files, compute a profile first and send only the changing conversation to that profile.
For a harness that already sends its full instructions, use first-turn learning and keep sending the full messages.
For example, after starting `serve --name my-agent --model /path/to/model --learn first-turn`:

```sh
curl http://127.0.0.1:8460/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @examples/agent.hello.json
```

The returned cached-token count lets the harness confirm reuse on later requests.
Current support is plain-text `chat/completions`; tool-bearing harness requests, the Responses API
and Anthropic Messages require adapters that are not included in this release.
Imprint runs its own model worker and does not import state from an unrelated process.
Its store lease prevents a second Imprint worker using the same store; unrelated servers need separate memory planning.

## Example: reuse a fresh harness session's opening context

A fresh agent session often sends the same instructions and project context before your first message.
Precomputing that stable opening block lets later sessions reuse it, so a new conversation does not have to repeat all that prefill.
For a compatible plain-text harness, start first-turn learning:

```sh
imprint serve --model /absolute/path/to/mlx-model --name my-agent --learn first-turn
```

Point the harness at `http://127.0.0.1:8460/v1` with model `my-agent`, then open a fresh session and send `hello`.
The harness must include its normal opening instructions; `hello` by itself cannot capture context it never sends.
That first request computes and saves the instructions, so it still pays the initial prefill cost.
Open another fresh session with the same instructions to reuse them, and check `usage.prompt_tokens_details.cached_tokens` in the response.
If you want the very first interactive session prepared in advance, use `compute --recipe` with those instructions instead.

The OMP reference setup is separate from this standalone CLI.
OMP requests containing tool definitions or tool messages need an adapter that this release does not include;
these steps describe supported plain-text requests, not a drop-in OMP integration.
The reported 0.3-second TTFT is a reference result, not a guarantee for every harness, model or cold start.

Bare `--learn` also means `--learn first-turn`: send a fresh plain-text request containing leading
system/developer instructions and one final user message.
Imprint saves the stable instruction prefix while processing that request and returns the answer.
Follow-up requests reuse it when their rendered tokens match, but their conversation history
does not replace the saved instructions.
A later fresh request with changed instructions computes and selects a new prefix.

For an assistant that sends its full growing conversation each turn, use continuous mode:

```sh
imprint serve --model /absolute/path/to/mlx-model --name assistant --learn continuous
```

Continuous mode saves the supplied context before the newest user message, including earlier
user and assistant messages; it excludes that newest message and the answer being generated.
Each eligible request can advance or replace the selected saved prefix, and every restore still
requires an exact token-prefix match.
Both modes remain enabled after requests and idle sleep; `imprint use NAME` turns learning off.
Omitting `--learn` disables automatic capture.

Use a new name for a learned profile: learning never overwrites a profile created from files,
a recipe or a live-session snapshot.
Each name selects its latest learned prefix; older immutable artifacts remain on disk,
without automatic selection of an older variant or garbage collection.

This first release accepts **plain-text chat only**: tool calls, tool results, images, audio,
custom stop strings and unsupported sampling options are rejected explicitly.

## Save a session already in memory

In another terminal using the same store:

```sh
imprint inspect --sessions
imprint compute --session SESSION_ID --name conversation
```

This contacts the running Imprint worker and exports its retained state without replaying the conversation.
The default `committed` scope may process the one generated token not yet absorbed by the cache;
the result reports `tail_tokens_computed` and does not sample another token.
Use `--scope absorbed` to save only state already computed, with zero additional forward calls.

Version 0.1 retains the latest session only, queues export until a running response finishes,
and evicts that session on the next request, sleep, or worker cancellation.
Missing sessions fail without loading a replacement model or replaying old text.
A conversation snapshot is a continuation profile: callers must supply a full request whose rendered tokens match it;
it is not automatically inserted into unrelated new chats, and a changed chat-template boundary can cause a cache miss.
Export from arbitrary third-party processes is not implemented.

## Sleep and swap

```sh
imprint sleep
imprint use assistant-memory
imprint inspect assistant-memory
```

The server automatically exits its worker after five idle minutes, or the configured timeout.
Saved blobs remain on disk; the next request reloads weights and restores matching state.
A sleeping model still has weight-loading and disk-reading latency.
`use` loads the selected profile ahead of the next request; later requests restore their own private cache state.
It also disables automatic learning for that server.
Same-model changes keep the loaded weights; different-model changes start a replacement worker.

## Hot-swap models with the same large context

Prepare the same files for each compatible model before starting the server:

```sh
imprint compute --files memory.md handbook.md \
  --model /absolute/path/to/model-a --name project-a
imprint compute --files memory.md handbook.md \
  --model /absolute/path/to/model-b --name project-b
imprint serve --name project-a
```

Both profiles contain the same source material, rendered and computed for their respective models.
For explicit fixed instructions and roles, use the same `--recipe` file in both compute commands instead.
The initial preparation runs once per model; later matching requests restore its saved context.

From another terminal, switch the running server's selected model and context:

```sh
imprint use project-b
imprint use project-a
```

Clients can keep using the same local endpoint with `"model":"imprint"` to follow the active profile.
This is useful when comparing models against a large shared reference collection, or choosing
different models for different tasks while retaining the same project background.
Only the selected model needs to remain loaded; its prepared context survives on disk when it sleeps.

Each model needs its own computation: one model's KV tensors cannot be transplanted into another.
Hot-swapping selects the model together with its matching saved state; it does not translate one model's cache into another's.
File/recipe computation is standalone, so stop a server owning that store before building another profile.
Live `--session` export uses its existing server instead.

## Storage and limits

The default store is `~/.cache/imprint`; add `--store /private/path` to any command to select another.
Profiles point to immutable, checksummed artifacts with private permissions.
Tokens, source text and tensors are sensitive local data; artifacts are not encrypted.
`inspect`, CLI help and tests do not load a model.

Context still occupies the model's attention window and RAM after restoration.
The backend checks the configured window and a conservative memory estimate before prefill;
it never truncates a large memory file silently.
The estimate cannot reserve RAM against unrelated applications allocating memory concurrently.
Chat templates must accept the recipe's role sequence; cache formats and templates are not universally interchangeable.

The installed CLI implements `compute`, `serve`, `inspect`, `use` and `sleep`.
Benchmarking, a CUDA backend, custom model kernels, rotating/quantized caches,
concurrent generation and tool parsing are not part of this release.
The [runtime contract](docs/RUNTIME.md) describes the implementation and its storage format.

## How much faster?

The reported reference result is **19 s → 0.3 s TTFT**, approximately **63.3× faster** or **98.4% less waiting**.
The [evidence ledger](docs/EVIDENCE.md) records its provenance and the remaining standalone validation.

For reproducible validation, compare cold prefill and restored state with identical
requests, sampling, outputs and context, measuring model-loaded and model-reloaded cases separately.
A reported speedup must come from those measurements, not from the number of tokens skipped.

## Test without inference

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
python -O -m unittest discover -s tests -v
```

These tests cover persisted artifacts, exact-prefix matching, session tail coverage, failure recovery,
real loopback HTTP, spawned worker shutdown and codec structure using symbolic data.
They do not prove model-level numerical equivalence, Metal memory release or a measured speedup.

See the [CLI contract](docs/CLI.md) and [runtime details](docs/RUNTIME.md) for the implemented interface.
An oMLX integration is planned separately; this CLI does not currently connect to an oMLX server.
