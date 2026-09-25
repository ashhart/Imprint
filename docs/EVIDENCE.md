# Evidence and existing systems

## Reported reference result

On 25 September 2026, the project author supplied a before/after latency result of **19 seconds versus 0.3 seconds** for the TTFT example.
That is approximately **63.33× faster**, or a **98.42% reduction** in time to first token.
This is an author-reported result; raw timing traces, complete workload settings and repeat counts are not included here.
It has not been independently reproduced with the standalone Imprint CLI and is not a universal performance guarantee.

The standalone package has model-free tests covering its storage, request handling and worker lifecycle.
Those tests do not prove numerical equivalence, real model memory release or end-to-end latency.
For a reproducible comparison, use identical model, prompt, output allowance and sampling settings, measuring model-resident and model-reloaded cases separately.
Record repeated client TTFT timings, cached-token counts and raw traces; see [runtime](RUNTIME.md) for current implementation limits.

## Related implementations

[MLX LM's prompt-caching documentation](https://github.com/ml-explore/mlx-lm#long-prompts-and-generations) shows a cache creation CLI and later generation using the saved prefix. Its existence means a basic save-and-load command is established functionality; an MLX adapter should reuse suitable upstream facilities.

[vLLM's automatic prefix caching documentation](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) describes reuse for repeated documents and conversations and distinguishes prefill savings from decoding. Imprint retains that distinction in its measurement guidance and README.

[LMCache documentation](https://docs.lmcache.ai/) describes persistent cache storage and integrations with serving engines. It is relevant prior work and a possible later integration, not evidence that Imprint's exact hybrid state can be imported into any engine without qualification.

The proposed workflow is practical product engineering around existing cache concepts. No claim of novel KV-cache theory, universal tensor portability, proven large-memory performance or instant cold-model startup is made.
