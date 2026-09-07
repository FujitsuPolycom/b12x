# Two recurrent checkpoints in one KDA prefill

The KDA prefill operation, `b12x.sequence.kda_prefill`, saves recurrent state at
two token offsets during one sequence traversal. A serving scheduler can use
those saved states to satisfy checkpoint boundaries without splitting the
model's forward pass at each boundary.

Status: research-only. CPU contract tests pass. GPU execution and throughput
are unqualified. Planning accepts the two-checkpoint mode on NVIDIA GB10,
SM121, 48 SMs. The embedded registry contains no measured KDA-prefill profile.

## Checkpoint contract

A checkpoint is a saved recurrent state after a specified number of request
tokens. The planned checkpoint capacity, `Caps.max_checkpoints`, defaults to
`1`, with vector metadata `[sequence_capacity]`. A capacity of `2` uses
contiguous matrix metadata `[sequence_capacity, 2]` and requires
`checkpoint_export=True` with transactional metadata validation.

State indices use the same `int32` or `int64` dtype as initial/final state
indices. Offsets are `int32`, relative to each request's packed token start.
An active offset must be positive, no larger than the live sequence length,
and divisible by the recurrence tile size of 16 tokens. The tile size is
independent of a serving scheduler's token budget.

Nonpositive offsets disable an entry. A null destination does not own storage;
its positive offset must still be in bounds and aligned. Two active non-null
exports require distinct offsets and globally unique destinations. Offsets
may be unordered. An export at the final token boundary is legal if its
destination differs from the final-state destination.

Initial state may alias its own final slot. Checkpoints may not overwrite an
initial read, another checkpoint or a final write. Invalid device metadata
poisons live output, sets the error bits documented by the operation, and
leaves the recurrent pool unchanged. A request spanning pipeline windows
requires a non-null final slot for its running recurrence.

Activations are BF16. Recurrent state is FP32 with physical layout
`[slot, head, value_dim, key_dim]` and head dimension 128. The recurrence
exports its FP32 accumulator without recomputing a prefix. Checkpoint stores
do not alter prepare arithmetic, recurrence arithmetic or final-state layout.

Each saved state writes `heads * 128 * 128 * 4` bytes: 1 MiB at 16 heads and
2 MiB at 32 heads. The kernel does not select scheduler chunk sizes, allocate
cache pages, export convolution state or publish cache entries. Whole-model
pass removal requires a serving integration that owns those operations.

## Planning and device eligibility

Construct `kda_prefill.Caps` with `max_checkpoints=2` and pass it to
`kda_prefill.plan`. The typed component policy validates the normalized device
identity `nvidia / nvidia gb10 / (12, 1) / 48 SMs` once during planning. Other
devices receive a planning error for two checkpoints. One-checkpoint planning
retains its device eligibility.

The device restriction defines the validation target. There is no TP-count,
ring, HCA or network condition. Enabling another GPU requires a reviewed device
eligibility change and hardware evidence for that device. Live request values
do not enter policy queries or kernel compile/cache keys: checkpoint capacity
is planned, while indices, offsets and live token/request counts are runtime
data. Bind and replay perform no device-eligibility lookup.

Query schema 2 includes `max_checkpoints` for both vector and matrix metadata.
Config schema 1 describes the launch-config fields. A schema-1 KDA profile is
incompatible with the schema-2 query contract and is rejected. AUTO and
HEURISTIC_ONLY resolve through the component heuristic when the registry has
no matching KDA profile. PREPLANNED_ONLY rejects that missing qualification.
Explicit configuration overrides pass the same target validator.

The base source revision, `06b4de7c723e6f166d65abf5909c5b7d0f8acc68`, lacks a
KDA-prefill registration and generator in the planned-op catalog. The catalog
completeness test fails on that source and on this implementation. The
component therefore requires catalog/provider integration, schema-2 coverage,
and measured device profiles, or an explicitly reviewed policy for
unqualified components. GPU qualification and this catalog dependency are
required for promotion beyond research-only status.

## Verification commands and coverage

CPU ownership, reference and policy contracts:

```sh
python -m pytest tests/sequence/test_kda_prefill_two_checkpoints_cpu.py -q
```

The GPU suite exercises public plan/bind/run operations. It compares outputs
and saved states with an independent FP32 oracle; checks in-place initial/final
storage; compares 8192-token exports at 6144 and 7168 with one-checkpoint
full/prefix runs; replays CUDA graphs with changing counts, destinations and
offsets under frozen kernel resolution; checks allocator counters and stable
addresses; and verifies transactional rejection.

```sh
python -m pytest tests/sequence/test_kda_prefill_two_checkpoints_gpu.py -q -rs
B12X_RUN_LARGE_POOL_TESTS=1 python -m pytest \
  tests/sequence/test_kda_prefill_two_checkpoints_gpu.py::test_two_checkpoint_gpu_high_pool_offsets -q -rs
```

The high-offset pool case reserves over 8 GiB, leaves unrelated pages
uninitialized, and places every live slot beyond the signed 32-bit
element-offset boundary. Run it on an explicitly selected idle GB10 with
sufficient free memory. Record a skipped case as missing coverage.

| Geometry | Purpose | GPU evidence |
|---|---|---|
| H1, D128 | FP32 oracle, metadata/replay and high-offset pool checks | Not executed |
| H16, D128 | Per-rank KDA geometry for a 64-head model at TP4 | Not executed |
| H32, D128 | Additional per-rank geometry coverage | Not executed |

The serving validation target is GLM-5.3-Flash on four Sparks at TP4. Its KDA
configuration has 64 heads with dimension 128; the model adapter partitions
those heads over the TP ranks. H32 component coverage does not qualify a
multi-GPU serving configuration.

## Switch-connected Spark qualification

A four-Spark deployment connected through a switch can run the single-GPU
component checks without SparkRing. Whole-model pass removal also requires a
vLLM integration for scheduler checkpoint boundaries, convolution checkpoint
metadata, recurrent-state ownership and cache publication. Keep its collective
backend, switch topology and serving configuration fixed across measurements.

A performance record must identify the source and toolchain, physical device,
correctness gates, warmup and graph state, allocation behavior, raw timings and
ratio direction. Measure checkpoint-export overhead separately from a paired
whole-model coalescing comparison. The CPU results and GPU test collection in
this package establish no throughput result.
