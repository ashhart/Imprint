# Current CLI and HTTP contract

This describes the implemented 0.1 command, with the remaining model-execution qualification recorded in [Runtime](RUNTIME.md).
Only implemented commands are listed here.

| Command | Behavior |
| --- | --- |
| `imprint compute --files notes.md --model /models/a --name work` | Prefill ordered UTF-8 text, persist state, exit worker |
| `imprint compute --recipe recipe.json --model /models/a --name work` | Prefill an explicit text recipe |
| `imprint compute --session ID --name conversation` | Export retained committed state through the running controller |
| `imprint compute --session ID --name partial --scope absorbed` | Save only already-absorbed state, with no forward call |
| `imprint serve --name work` | Start loopback controller, load model on request |
| `imprint serve --model /models/a --name agent --learn` | Learn instructions from fresh plain-text turns, equivalent to `--learn first-turn` |
| `imprint serve --model /models/a --name assistant --learn continuous` | Learn the supplied conversation before the newest user message |
| `imprint use work` | Select a stored profile and load state ahead of the next request |
| `imprint sleep` | Wait for current response, exit worker, evict session |
| `imprint inspect work` | Read saved metadata without model loading |
| `imprint inspect --sessions` | Ask the controller for the latest retained live session |

Supply exactly one compute input; file and recipe computation require an explicit existing local model directory.
Names are 1–64 lowercase letters, digits, underscores or hyphens, starting with a letter.
The only engine is `mlx`; unsupported cache classes fail explicitly.
`--store PATH` and `--json` work before or after the command.
`serve` defaults to `127.0.0.1:8460`, with `--idle-unload 5m`; duration suffixes are `s`, `m`, and `h`.
The standalone compute command cannot create another worker while a server owns the same store.
No verify, benchmark, raw-token endpoint, exact historical checkpoint selection, or third-party extraction command is shipped.

## Automatic learning

`serve --learn` and `serve --learn first-turn` select the same mode.
An eligible fresh request contains one or more leading system/developer messages followed by
exactly one user message, with no earlier user or assistant history.
The saved boundary covers only the stable instruction prefix.
Follow-up requests may reuse it but do not create a replacement from their history;
a later eligible fresh request with changed instructions computes and selects a new prefix.

`serve --learn continuous` saves the supplied context before the final, newest user message,
including earlier user/assistant history.
The newest user message and the answer generated for it are excluded from automatic capture.
Clients still send the complete conversation; this mode does not reconstruct messages omitted by the client.
Requests with no eligible boundary can still run, but do not publish a new prefix.

Both modes require an explicit local `--model` and stay enabled across requests and worker sleep.
Omitting `--learn` disables capture, and `use NAME` disables it when selecting that profile.
Profiles created by `compute --files`, `--recipe` or `--session` remain reusable and are never
overwritten by auto-learning; use a separate name for a learned profile.
The latest learned generation is selected by its profile pointer, with exact token matching on reuse.
There is no automatic search across older generations, and old immutable artifacts are not garbage-collected.

## Requests and responses

`GET /v1/models` advertises the active profile, including a learning profile before its first save.
`POST /v1/chat/completions` accepts that name or the `imprint` alias for the current selection.
An explicitly named inactive profile is rejected: call `use` to switch first.

Messages must contain exactly `role` and string `content`.
Accepted roles are system, developer, user and assistant, subject to the selected model's chat template.
Tools, images/audio, tool results, stop strings, penalties, structured output and unknown fields are errors.
Options are `stream`, `stream_options.include_usage`, `max_tokens` or `max_completion_tokens`,
`temperature`, `top_p`, `top_k`, `seed`, and `chat_template_kwargs.enable_thinking`.
Defaults use greedy sampling and a 256-token maximum; output is capped at 32,768 tokens and the model window.

Recipe profiles prepend fixed messages to dynamic requests unless the exact fixed messages are already present.
A dynamic request cannot introduce another system/developer message or override recipe template options.
Captured and continuation profiles require full messages; only an exact rendered token-prefix match yields a hit.
A mismatch uses cold prefill and reports zero cached tokens; enabled learning can also publish
a new prefix when the request has an eligible boundary.

Both streaming and nonstreaming responses include an `X-Imprint-Session-ID` header and cache status.
Usage contains effective prompt length, completion length and `prompt_tokens_details.cached_tokens`.
Streaming sends each text delta once, optionally sends final usage, and finishes with `[DONE]`.
Failures after headers produce an error event without an automatic retry or leaked request content.
Disconnecting during generation terminates the owned worker and evicts its unfinished session.

## Controls and files

Local controls use authenticated loopback HTTP with a private `STORE/control.json` discovery file.
Endpoints under `/_imprint/` provide status, sessions, snapshot, activate and sleep.
A session export waits for any running response to finish, then reports `tail_tokens_computed`.
The latest session is evicted by another request, sleep, activation or cancellation.
A missing session is an error and never triggers source replay or a replacement model load.
Controls can wait up to ten minutes; a transport timeout does not cancel an operation already admitted.

Private artifacts include text and token history as well as tensors and are not encrypted.
The [runtime format](RUNTIME.md) is distinct from the old synthetic design schema.

Exit codes are 0 success, 2 invalid input, 5 filesystem I/O, 7 worker/control failure and 130 interruption.
`--json` wraps successful command output in `{"version":1,"ok":true,"result":...}` and reported operational errors in an error envelope.
Argparse usage errors retain argparse's standard stderr format and exit code 2.
