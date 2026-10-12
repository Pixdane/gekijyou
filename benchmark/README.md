# benchmark

A non-published workspace package for measuring `gekijyou-computation` through
its public API. Its library owns the scene fixture, independent direct evaluator,
full output oracle, dataset/warmup definitions and shared sample runner. It has no
allocation instrumentation. The two binaries select a wall-clock observer or a
System allocator wrapper; only `allocations` links the latter.

From the workspace root:

```sh
cargo run --release -p benchmark --bin timing -- --help
cargo run --release -p benchmark --bin timing -- --suite memo --samples 9 > timing.tsv
cargo run --release -p benchmark --bin allocations -- --suite memo --samples 3 > allocations.tsv
cargo run --profile perf -p benchmark --bin timing -- --suite context --case repeat --config last_ctx4 --contexts 64 --rounds 0 --samples 9 --iterations 64
```

Run timing and allocation jobs sequentially, on the same machine/toolchain/power
state when comparing versions. Use `--locked` for reproducible dependency builds.
The ordinary release profile and optional `perf` profile are workspace-root
settings; perf inherits release and sets debug=1. A workspace/package migration
can change generated code, so prior reports are historical measurements rather
than a guaranteed performance baseline for these new executables.

## Suites

| Suite | Cases | Configurations |
| --- | --- | --- |
| scene | all_miss | direct / evaluate / end_to_end, all with explicit Transient policy |
| memo | all_miss / high_hit / history4 / graph_edit | transient / last / lru1 / lru4 / lru8 |
| context | repeat / continuous / scrub3 / edit_first / edit_second / unrelated_edit_first / cold_first | last_ctx0 / last_ctx1 / last_ctx4 / last_ctx16 / lru8_ctx0 |

The default matrix uses 32 actors x 16 layers, 1,689 nodes, six requests per
Context, 0/128 compute rounds, 1/64 Context, 9 samples and 32 Batch iterations.
`--suite`, `--case`, `--config`, `--rounds` and `--contexts` select subsets.
Configurations rotate in every sample. All result rows are **raw samples**, in
execution order; there is no second copy of the workloads in the binaries.

Sustained misses preload only a predecessor. History/repeat cases preload their
reusable states; this does not imply a partially filled LRU is full. Continuous
frames increase across the whole sample, with no wrapping. Cold-first creates an
empty evaluator and measures one Batch per sample regardless of --iterations.
To reduce timer noise on short warm operations, increase --iterations and retain
multiple rounds/runs; use at least nine samples for performance comparisons.

## Boundaries and output

Graph/data construction, complete value verification, warmup and final evaluator
destruction occur outside observations. Graph evaluation includes result reads,
checksum and result drop. `evaluate` prepares consumed batches outside observation;
`end_to_end` includes Context/Batch construction. Direct computes the same outputs
and checksum with statically typed values and does not execute the graph. Evaluate
preparation is capped at 4,096 Contexts per sample to bound setup memory.

All edit cases measure replacement validation separately from evaluation.
`edit_second` also excludes the first evaluation after each replacement. Therefore
new `graph_edit` evaluation rows have a different boundary from the historical
Memo example that included replace. Timings are not directly comparable to those
old aggregate rows. No internal containers or temporary-map byte estimates are
exposed through diagnostic hooks.

Both programs use the same TSV schema. Timing fields are `NA` in allocation runs;
allocation fields are `NA` in timing runs. Flow counters and execution stats are
reported per Batch. `live_delta_bytes` is the requested-layout difference at the
scope endpoints, and `peak_extra_bytes` is the increment over the scope's baseline.
`peak_scope=sample` means one whole iteration loop; `peak_scope=batch` means the
maximum of individually observed Batches (edits/cold). Edit live differences sum
only evaluation scopes, excluding replace and edit-second's first evaluation.
These bytes exclude allocator metadata, fragmentation, non-Rust allocations and
RSS. Realloc flow records each successful full new size. No peak is inferred from
cumulative requested bytes. Counts track real allocations/frees outside scopes too,
so releasing preexisting objects is accounted for correctly. Scopes disable
tracking during unwinding; nested allocation scopes are unsupported.

Use executed Memo/Context hits, misses, comparisons, computation_calls and
resolution_visits to check equivalent loads before comparing timing. Ratios and
medians can be derived from the preserved samples; no timing thresholds are
asserted by correctness tests. Scene tests verify full values and duplicate result
ownership; runner tests verify modes, edit boundaries, cold behavior and raw rows.

Historical snapshots, candidate storage implementations, private observers and
one-off investigation scripts remain in the external investigation archive.
