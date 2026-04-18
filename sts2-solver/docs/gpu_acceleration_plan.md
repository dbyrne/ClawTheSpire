# BetaOne GPU Acceleration Plan

## Context

Current gen_time is ~150s with K=32 virtual-loss batched MCTS, sequential was ~540s (3.5× improvement shipped 2026-04-17). All further NN work is still CPU-bound:

- `sts2-engine/src/betaone/inference.rs:29-32` builds `Session::builder()` without `.with_execution_providers(...)` — DirectML feature in `Cargo.toml:15` is dead at runtime.
- `sts2-solver/src/sts2_solver/betaone/selfplay_train.py` has zero `.to(device)` / `autocast` / `GradScaler` / `torch.compile`. PyTorch training is 100% CPU.
- `sts2-engine/src/betaone/selfplay.rs:33-35` uses `thread_local!` per Rayon worker; 8 workers each run independent K=32 ONNX sessions on CPU. No cross-worker batching.

Per-gen time budget (estimates — profile before assuming):

| Phase | Est. time | GPU addressable? |
|---|---|---|
| Self-play NN inference | 60–80s | Yes (if cross-worker batched) |
| Self-play non-NN (clones, enumerate, arena, encode) | 60–80s | No |
| PyTorch training (~388 steps on 138K params) | 3–6s | Yes |
| Replay sampling / tensor marshalling | 2–4s | Partial |
| Misc (ONNX export, I/O) | 1–3s | No |

Theoretical ceiling with NN cost zero: **~70s/gen**. Anything beyond that is a rewrite of the non-NN Rust path.

Hardware: RTX 5080 (16 GB, Blackwell), driver 595.79.

## Lever ordering and sequencing

Ship in this order. Each lever either validates or invalidates the next.

1. **Lever 2** first — CUDA EP flag flip. 1-day test. Tells us whether per-thread batching at K=32 on GPU is viable without the server rewrite.
2. **Lever 4** in parallel — replay buffer cleanup. Independent. Sets up for GPU training.
3. **Lever 3** after #4 lands — GPU training with bfloat16 autocast. Small-win polish.
4. **Lever 1** only if #2 shows per-thread CUDA is not faster than CPU at the current K=32. The 3–5 day inference server is the big commitment.

---

## Lever 2: Register CUDA execution provider (try this first)

**Goal:** verify whether GPU inference beats CPU at K=32 per worker, before committing to a bigger rewrite.

**Changes:**

- `sts2-engine/Cargo.toml:15` — swap `features = ["directml"]` → `features = ["cuda"]`.
- `sts2-engine/src/betaone/inference.rs:29-32` — register CUDA EP:
  ```rust
  use ort::ep::{CUDA, ExecutionProvider};
  let mut builder = Session::builder()?
      .with_optimization_level(GraphOptimizationLevel::Level3)?
      .with_intra_threads(1)?;
  CUDA::default().register(&mut builder)?;
  let session = builder.commit_from_file(model_path)?;
  ```
  Keep a `use_gpu: bool` env flag (`STS2_GPU_INFERENCE`) for easy A/B.
- `sts2-engine/examples/bench_inference.rs` — add a `CUDA intra=1 batch=...` row. Re-run to produce updated bench numbers.
- Document ORT CUDA install (cudnn 9.x, CUDA 12.x DLLs on PATH) in a README note.

**Validation:**

1. `bench_inference` run — CPU vs CUDA at batch={1, 8, 32, 64, 128}. Record us/call and us/sample.
2. `bench_search_batched` with CUDA — sequential, K=4/8/16/32/64. Compare to the published 2× at K=32.
3. End-to-end: run 5 gens on `pomcp-batched-k32-v1` with CUDA inference enabled. Compare gen_time to the 150s baseline.

**Kill criteria:** if 5-gen average gen_time > 140s (i.e., regresses or wins <7%), revert and commit to Lever 1.

**Risks:**

- TF32 auto-promotion on Blackwell changes FP32 outputs by ~1e-4 absolute. Not a correctness bug, but breaks bit-exact comparison across backends. Note in experiment metadata.
- ORT CUDA EP determinism: cuDNN reductions are deterministic for our ops (Linear, softmax, bmm, LayerNorm) by default on recent ORT, but verify with a seeded test.
- Windows install pain: cudnn + CUDA 12 DLLs. Document explicitly.
- 8 workers × 1 GPU contention: CUDA EP defaults to one stream per session. Multi-thread submission may serialize at the driver. This is the most likely failure mode for this lever.

**Effort:** 1 day including bench + 5-gen validation.

**Files touched:** 3 Rust files, 1 Cargo.toml, 1 README.

**Revert:** flip Cargo feature flag back. Trivial.

---

## Lever 4: Pre-stacked replay buffer with fancy indexing

**Goal:** eliminate the triple-copy in `ReplayBuffer.sample_tensors` and set up pinned-memory H2D for Lever 3.

**Current code** (`sts2-solver/src/sts2_solver/betaone/selfplay_train.py:113-128`):

```python
indices = np.random.choice(n, size=min(batch_size, n), replace=False)
return (
    torch.tensor(np.array([self.states[i] for i in indices]), dtype=torch.float32),
    ...
)
```

List comprehension + `np.array(list)` + `torch.tensor` = three copies per field × 7 fields.

**Proposed design — deque-of-ndarray-chunks:**

```python
class ReplayBuffer:
    def __init__(self, max_steps: int = 200_000):
        self.max_steps = max_steps
        self._chunks: deque[dict[str, np.ndarray]] = deque()
        self._total = 0

    def add_generation(self, states, act_feat, act_masks, hand_ids,
                       action_ids, policies, values) -> None:
        chunk = {
            "states": np.asarray(states, dtype=np.float32),
            "act_feat": np.asarray(act_feat, dtype=np.float32),
            "act_masks": np.asarray(act_masks, dtype=bool),
            "hand_ids": np.asarray(hand_ids, dtype=np.int64),
            "action_ids": np.asarray(action_ids, dtype=np.int64),
            "policies": np.asarray(policies, dtype=np.float32),
            "values": np.asarray(values, dtype=np.float32),
        }
        n = len(chunk["states"])
        self._chunks.append(chunk)
        self._total += n
        while self._total > self.max_steps and len(self._chunks) > 1:
            self._total -= len(self._chunks.popleft()["states"])
```

Sampling builds a global-index view lazily and uses fancy indexing:

```python
def sample_tensors(self, batch_size: int):
    concat = {k: np.concatenate([c[k] for c in self._chunks])
              for k in self._chunks[0].keys()}
    n = len(concat["states"])
    idx = np.random.default_rng().choice(n, size=min(batch_size, n), replace=False)
    return tuple(torch.from_numpy(concat[k][idx]) for k in (
        "states", "act_feat", "act_masks", "hand_ids", "action_ids",
        "policies", "values"
    ))
```

Concatenation is O(buffer_size) per sample call. For a 50K-step buffer that's ~20ms — acceptable for one call per training step, but if it bites we cache the concatenation until the next `add_generation`.

**Validation:**

1. Unit test: add 3 generations of known sizes, assert eviction preserves gen boundaries (check `len(_chunks)` shrinks by whole gens).
2. Integration: run 5 gens on any existing experiment. Compare loss curves to prior history file — should be identical within float noise (same RNG seed path).
3. Micro-bench: time `sample_tensors(512)` on a 50K buffer. Target <5ms (current is likely 15–30ms).

**Risks:**

- **Preserve gen-boundary eviction.** The current impl evicts whole generations via `_gen_sizes` deque. If you switch to a fixed ring buffer with sample-level eviction, you change the training distribution stationarity. Don't do that unless intentional.
- `np.random.choice(replace=False)` is O(n) — use `np.random.default_rng().choice(replace=False)` (faster) or `np.random.randint` with rejection sampling for large buffers.
- `torch.from_numpy` shares memory — don't mutate the source arrays after calling.

**Effort:** 4–6 hours including tests.

**Files touched:** `selfplay_train.py` (ReplayBuffer class only), new test in `tests/`.

**Revert:** keep old ReplayBuffer behind an import toggle for one release; delete after training validation confirms parity.

---

## Lever 3: GPU training + bfloat16 autocast

**Goal:** move PyTorch training to GPU with bfloat16 autocast. Small win (~3s/gen) but sets up for larger batches / longer training schedules.

**Changes in `selfplay_train.py`:**

1. Device selection at `train()` entry:
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   network = BetaOneNetwork(...).to(device)
   ```
2. Move sampled tensors to device in the training loop (after Lever 4 lands):
   ```python
   batch = tuple(t.to(device, non_blocking=True) for t in replay.sample_tensors(batch_size))
   ```
   Use `pin_memory()` on the buffer tensors if we're on CUDA.
3. Wrap forward pass in autocast with **bfloat16, not float16**:
   ```python
   with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                      enabled=device.type == "cuda"):
       logits, values = network(...)
   # Loss computation OUTSIDE autocast for numerical stability:
   log_probs = F.log_softmax(logits.float(), dim=1)
   policy_loss = -(target_policies * log_probs).nan_to_num(0.0).sum(dim=1).mean()
   value_loss = F.mse_loss(values.squeeze(-1).float(), target_values)
   ```
4. ONNX export needs network back on CPU (or ONNX tracer handles CUDA export fine in recent torch — test which works):
   ```python
   network.cpu()
   path = export_onnx(network, onnx_dir)
   network.to(device)
   ```
5. Checkpoint save: keep state_dict on CPU for portability:
   ```python
   "model_state_dict": {k: v.cpu() for k, v in network.state_dict().items()},
   ```

**Why bfloat16 not float16:**

- Blackwell supports bfloat16 natively with full throughput.
- bfloat16 has FP32's exponent range → no GradScaler needed, no overflow on small/large values.
- Our 138K-param net with policy loss ~1.3 and value loss ~0.04 has been observed (anecdotally) to struggle under `float16` + `GradScaler` scale convergence. bfloat16 sidesteps this.

**Validation:**

1. Run 5 gens on CPU, seed X. Record losses per gen.
2. Run 5 gens on CUDA+bfloat16, same seed. Losses should track CPU within ~1% (not bit-exact — TF32/bf16 introduces expected drift).
3. Full gen benchmark: if CPU training was 3–6s, GPU+bf16 should be 0.5–1.5s. If not, profiling has surprises.
4. Kill criteria: if value_loss diverges (NaN or stays >2× the CPU run at gen 5), revert. Likely indicates the manual log_softmax is still precision-limited — escalate by using `F.cross_entropy` directly.

**Risks:**

- **FP16 instability** — mitigated by using bfloat16.
- **Manual log_softmax precision** — mitigated by forcing `.float()` on loss inputs.
- **Checkpoint portability** — mitigated by explicit CPU state_dict on save.
- **Bit-exactness lost** — expected. Document in experiment metadata that gen N onward ran on CUDA+bf16.
- **ONNX export from CUDA** — may hit device-placement bugs in older torch.onnx. Keep CPU export path as fallback.

**Effort:** 1 day including 5-gen validation.

**Files touched:** `selfplay_train.py`, minor edits to `network.py` if export_onnx needs device handling.

**Revert:** `device = torch.device("cpu")` flag. Trivial.

---

## Lever 1: Cross-worker inference server

**Goal:** consolidate NN requests from all 8 Rayon workers into one shared batched inference path. This is the biggest lever but also the biggest rewrite.

**Only commit to this if Lever 2 shows per-thread CUDA at K=32 doesn't win.**

**Design:**

New file `sts2-engine/src/betaone/inference_server.rs`:

```rust
pub struct InferenceServer {
    // Session pool — not one session, a vec. Each in-flight batch takes one.
    sessions: Arc<SessionPool>,
    // Request queue: workers push (inputs, oneshot response).
    tx: crossbeam::channel::Sender<Request>,
    // Background thread drains the queue.
    _drainer: JoinHandle<()>,
}

struct Request {
    // Pre-encoded tensors — workers do encoding on their own thread.
    state: Box<[f32; STATE_DIM]>,
    action_features: Box<[f32; MAX_ACTIONS * ACTION_DIM]>,
    action_mask: Box<[bool; MAX_ACTIONS]>,
    hand_ids: Box<[i64; MAX_HAND]>,
    action_ids: Box<[i64; MAX_ACTIONS]>,
    num_valid: usize,
    respond: crossbeam::channel::Sender<(Vec<f32>, f32)>,
}

impl InferenceServer {
    pub fn submit(&self, req: Request) -> Receiver<(Vec<f32>, f32)> { ... }
}
```

Drainer loop:

```rust
loop {
    let mut batch = Vec::with_capacity(max_batch);
    // Wait for at least one request with an unbounded deadline.
    match rx.recv() { Ok(r) => batch.push(r), Err(_) => break }
    // Drain more up to max_batch, with a short deadline.
    let deadline = Instant::now() + Duration::from_micros(500);
    while batch.len() < max_batch {
        match rx.recv_deadline(deadline) {
            Ok(r) => batch.push(r),
            Err(_) => break, // timeout or disconnect
        }
    }
    // Dispatch batch on one Session; send responses.
    let session = sessions.acquire(); // blocking if pool exhausted
    let results = run_batch(&session, &batch);
    sessions.release(session);
    for (req, res) in batch.into_iter().zip(results) {
        let _ = req.respond.send(res);
    }
}
```

Callsite changes:

- `sts2-engine/src/betaone/selfplay.rs:33-35` — replace `thread_local! SELFPLAY_CACHE` with `Arc<InferenceServer>` passed through.
- `sts2-engine/src/betaone/mcts_adapter.rs` — `evaluate`, `evaluate_batch`, `value_only`, `value_only_batch` all become `server.submit(...).recv()` for single, `submit_many(...).zip(recv_all)` for batch.
- `sts2-engine/src/betaone/selfplay.rs:415-470` — the `py.allow_threads` + `into_par_iter` stays; workers just hold an `Arc<InferenceServer>` instead of a `thread_local!` session.
- Add shutdown path on gen boundary (drop server → drainer sees disconnect → exits cleanly).

**Config knobs:**

- `max_batch` — cap at 256 (8 workers × K=32).
- `drain_deadline_us` — start at 500µs, tune. Too small = no aggregation; too large = workers starve.
- `min_batch` — 1 (never block if only one request in flight; always better to return early than stall).
- `num_sessions` — size of the pool. Start at 2 (one draining, one dispatching) — more sessions = more GPU memory but higher throughput if the drain window is short.

**Validation:**

1. Equivalence test: seed both sequential and server paths, assert identical action selections across a full self-play combat. Order of backup changes but output should match within NN FP tolerance.
2. Deadlock test: start 8 workers, end them staggered, assert clean shutdown.
3. Queue depth test: log max queue depth over a gen; should stay bounded.
4. End-to-end: 10-gen run on `pomcp-batched-k32-v1`. Target gen_time 75–95s (vs 150s today). Kill if >120s or if training-eval metrics regress outside gen-to-gen noise (2–4pt WR).

**Risks, in order of severity:**

1. **ORT concurrent `run()` thread-safety.** Documented as session-level safe on CUDA EP and DirectML, but the actual behavior under contention isn't well-published. Mitigation: session *pool* (pick N=2 initially), never share one session across threads.
2. **Drain window tuning.** Too-short window = no aggregation = no speedup. Too-long = workers stall mid-search. Classic Leela/KataGo knob — expect a day of tuning.
3. **Worker backpressure.** If GPU throughput < aggregate submit rate at some awkward batch, queue grows unbounded → OOM. Use `crossbeam::channel::bounded(4096)` and let workers block on push.
4. **Shutdown cleanliness.** Forgetting to drop senders on all workers at gen boundary means the drainer hangs forever. Classic channel bug.
5. **Memory pressure.** 256-row ONNX input tensors: 256 × 427 × 4 + 256 × 30 × 35 × 4 + masks + ids ≈ 2 MB per batch. Negligible. 2 sessions × 2MB fine.

**Semantic correctness concerns:**

- MCTS math is unchanged — each worker runs its own tree with its own virtual-loss. NN is a pure function; batch composition doesn't affect outputs.
- The only indirect effect is that a worker submitting a request may wait slightly longer, which *could* let another worker complete a `backup` that materializes a node state the first worker would have selected into. But the select is already done by the time the request is submitted, so this doesn't affect correctness.
- Dirichlet noise, temperature sampling, RNG usage — all per-worker, untouched.

**Effort:** 3–5 days.

**Files touched:**
- New: `sts2-engine/src/betaone/inference_server.rs`
- Modified: `sts2-engine/src/betaone/selfplay.rs` (selfplay cache → server Arc)
- Modified: `sts2-engine/src/betaone/mcts_adapter.rs` (adapter calls server)
- Modified: `sts2-engine/Cargo.toml` (add `crossbeam = "0.8"`)
- Modified: `sts2-engine/src/betaone/mod.rs` (re-export)
- New tests: `sts2-engine/tests/test_inference_server.rs`
- Bench: `sts2-engine/examples/bench_server_batched.rs`

**Revert:** feature-flag the server path behind `training.mcts.inference_server: bool` in the experiment config. Default off until 10-gen validation passes. Old `thread_local!` code stays behind the flag for one release.

---

## Sequencing and decision points

```
  Week 1
  Day 1: Lever 2 (CUDA EP) — bench + 5-gen test
     |
     v
  Decision point A:
    - CUDA per-thread wins ≥15%? → ship #2, skip #1 for now
    - CUDA regresses or ties?    → keep CPU, commit to #1
  Day 2: Lever 4 (replay refactor)
  Day 3: Lever 3 (GPU training + bf16)
     |
     v
  Decision point B (if #1 is on the table):
  Days 4-8: Lever 1 (inference server)
    - Day 4: inference_server.rs skeleton + tests
    - Day 5: wire adapter + selfplay
    - Day 6: equivalence + deadlock tests
    - Day 7: 10-gen validation
    - Day 8: tuning drain window, landing
```

## Expected combined gain

| Config | gen_time | vs today |
|---|---|---|
| Today (CPU, K=32) | 150s | 1.0× |
| +#2 CUDA EP wins | 110–140s | 1.1–1.4× |
| +#2/#3/#4 (no server) | 105–135s | 1.1–1.4× |
| +#1 server (with CUDA) | 75–95s | 1.6–2.0× |
| Theoretical NN-free ceiling | ~70s | 2.1× |

Any promise beyond 2× without rewriting the non-NN Rust paths (state clone, action enum, arena) is fantasy. If the 2× isn't enough, the next project is either (a) parallelize combats on the GPU side too (combat-level batched MCTS, not just decision-level), which is a 2–4 week rewrite, or (b) reduce sims from 1000 — a knob we already have.

## Open questions

- **Actual CPU/GPU time split within gen_time.** Every number above is estimated. First task should be adding per-phase timing (NN time, non-NN search time, training time) to the per-gen log. `std::time::Instant` around `search_batched` and Python `time.time()` around the training loop. 30 minutes of work. Do this before committing to any lever.
- **Does `pomcp-batched-k32-v1` at ~150s/gen represent the steady state, or are the early gens faster because games end early?** Check gen_time vs win_rate correlation — if wins go up, games get longer, time goes up. Affects the target numbers above.
- **Is there a bound on replay buffer size we should use for GPU training?** Larger batches (2048 vs 512) on GPU amortize transfer cost better. Might want to revisit `batch_size: 512` in the experiment configs once GPU training lands.
