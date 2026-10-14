//! Deterministic DAG differential verification against an independent spec oracle.
//!
//! Every graph is described by an explicit `Op` table first. The oracle is an
//! independent per-item interpreter over that table; the graph closures are built
//! by `GraphEdit::map` with 0 to 4 item tuples and never participate in the oracle.
//! Correctness is established by comparing every requested output, not a checksum.
//!
//! Each fixture runs the full `[S0, S1, S0, S2]` snapshot sequence on one evaluator:
//! a retained `S0` snapshot is bound again after `S1`, the values are re-checked, and
//! the first `S0` results are re-verified at the end of the sequence.

use gekijyou_computation::{
    CachePolicy, ComputeCompletion, ComputeTask, EvalError, EvaluationResults, EvaluationSnapshot,
    Evaluator, FrameIndex, GraphState, Input, Node, NodeCacheConfig, PlannerConfig, PlannerStats,
    RequestBatch, RequestId, SerialExecutor,
};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

/// Fixed generated-graph seeds.
const SEEDS: [u64; 8] = [1, 7, 42, 99, 0x5eed, 0xc0ffee, 0xdeadbeef, 0x12345678];
/// Fixed request frames; order is preserved.
const FRAMES: [u64; 8] = [120, 30, 120, 7, 300, 0, 30, 999];
/// Disconnected appendix of the standard generated graph.
const DISCONNECTED: usize = 12;
/// `(frame_window, max_in_flight, comparison_budget)` planner configurations.
const CONFIGS: [(usize, usize, usize); 3] = [(1, 1, 0), (3, 2, 1), (16, 7, 64)];
/// Input values of snapshots `S0`, `S1`, `S2`; `S0` is bound twice per sequence.
const SNAPSHOTS: [[u64; 3]; 3] = [[10, 20, 30], [11, 21, 31], [7, 19, 23]];

// ---------------------------------------------------------------------------
// Spec-first graph description
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Op {
    Input(usize),
    Frame,
    Constant(u64),
    Fold { deps: Vec<usize>, seed: u64 },
}

/// Closure-side fold helper. The oracle deliberately does not call it.
fn fold_values(seed: u64, values: &[u64]) -> u64 {
    let mut accumulator = seed;
    for &value in values {
        accumulator = accumulator
            .wrapping_mul(33)
            .wrapping_add(value.rotate_left(7));
    }
    accumulator
}

/// Independent spec interpreter: deps are strictly smaller, so one forward pass suffices.
fn spec_values(ops: &[Op], frame: u64, inputs: &[u64; 3]) -> Vec<u64> {
    let mut values = vec![0_u64; ops.len()];
    for (index, op) in ops.iter().enumerate() {
        values[index] = match op {
            Op::Input(position) => inputs[*position],
            Op::Frame => frame,
            Op::Constant(value) => *value,
            Op::Fold { deps, seed } => {
                // Inlined on purpose: the oracle must not share the graph fold helper.
                let mut accumulator = *seed;
                for &dep in deps {
                    accumulator = accumulator
                        .wrapping_mul(33)
                        .wrapping_add(values[dep].rotate_left(7));
                }
                accumulator
            }
        };
    }
    values
}

/// Local LCG. `next()` returns high bits so low-bit modulo does not degenerate topology.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 32
    }
}

/// `spec-first` generator: three `u64` inputs, one frame mapping, 44 derived items,
/// then `disconnected` append-only items that no output references.
fn generate(seed: u64, disconnected: usize) -> (Vec<Op>, Vec<usize>) {
    let mut rng = Lcg::new(seed);
    let mut ops = vec![Op::Input(0), Op::Input(1), Op::Input(2), Op::Frame];
    for _ in 0..44 {
        let index = ops.len();
        if rng.next() % 4 == 0 {
            ops.push(Op::Constant(rng.next()));
        } else {
            let arity = 1 + (rng.next() % 4) as usize;
            let fold_seed = rng.next();
            let mut deps = Vec::with_capacity(arity);
            for _ in 0..arity {
                deps.push((rng.next() % index as u64) as usize);
            }
            // Consume both draws first, then collapse the duplicate edge on even indices.
            if arity == 2 && index % 2 == 0 {
                deps[1] = deps[0];
            }
            ops.push(Op::Fold {
                deps,
                seed: fold_seed,
            });
        }
    }
    let main_len = ops.len();
    let outputs = vec![main_len - 3, main_len - 2, main_len - 1];
    if disconnected > 0 {
        ops.push(Op::Constant(1000));
        for ordinal in 1..disconnected {
            let previous = ops.len() - 1;
            ops.push(Op::Fold {
                deps: vec![previous],
                seed: ordinal as u64,
            });
        }
    }
    (ops, outputs)
}

/// Fixed diamond fixture: `A=Fold([input0],1)`, `B=Fold([A],2)`, `C=Fold([A],3)`,
/// `D=Fold([B,C],4)`, `E=Fold([B,B],5)`, `F=Fold([B,C,A,A],6)`; outputs `D/E/F`.
/// Keeps three inputs for the shared snapshot sequence.
fn diamond() -> (Vec<Op>, Vec<usize>) {
    (
        vec![
            Op::Input(0),
            Op::Input(1),
            Op::Input(2),
            Op::Frame,
            Op::Fold {
                deps: vec![0],
                seed: 1,
            },
            Op::Fold {
                deps: vec![4],
                seed: 2,
            },
            Op::Fold {
                deps: vec![4],
                seed: 3,
            },
            Op::Fold {
                deps: vec![5, 6],
                seed: 4,
            },
            Op::Fold {
                deps: vec![5, 5],
                seed: 5,
            },
            Op::Fold {
                deps: vec![5, 6, 4, 4],
                seed: 6,
            },
        ],
        vec![7, 8, 9],
    )
}

// ---------------------------------------------------------------------------
// Graph construction
// ---------------------------------------------------------------------------

/// Builds the spec with `GraphEdit::map` arities 0 through 4. `eq` registers `u64`
/// equality semantics; without it only Arc identity can ever merge content.
fn build_spec(state: &mut GraphState, ops: &[Op], eq: bool) -> (Vec<Node<u64>>, Vec<Input<u64>>) {
    state
        .edit(|e| {
            if eq {
                e.set_default_eq::<u64>();
            }
            let mut nodes: Vec<Node<u64>> = Vec::with_capacity(ops.len());
            let mut inputs: Vec<Input<u64>> = Vec::new();
            for op in ops {
                let node = match op {
                    Op::Input(position) => {
                        while inputs.len() <= *position {
                            let input: Input<u64> = e.input::<u64>();
                            inputs.push(input);
                        }
                        inputs[*position].node()
                    }
                    Op::Frame => e.map(e.frame(), |frame: &FrameIndex| frame.0)?,
                    Op::Constant(value) => {
                        let value = *value;
                        e.map((), move || value)?
                    }
                    Op::Fold { deps, seed } => {
                        let seed = *seed;
                        match deps.len() {
                            0 => e.map((), move || seed)?,
                            1 => e.map(nodes[deps[0]], {
                                move |a: &u64| fold_values(seed, core::slice::from_ref(a))
                            })?,
                            2 => {
                                e.map((nodes[deps[0]], nodes[deps[1]]), move |a: &u64, b: &u64| {
                                    fold_values(seed, &[*a, *b])
                                })?
                            }
                            3 => e.map(
                                (nodes[deps[0]], nodes[deps[1]], nodes[deps[2]]),
                                move |a: &u64, b: &u64, c: &u64| fold_values(seed, &[*a, *b, *c]),
                            )?,
                            4 => e.map(
                                (
                                    nodes[deps[0]],
                                    nodes[deps[1]],
                                    nodes[deps[2]],
                                    nodes[deps[3]],
                                ),
                                move |a: &u64, b: &u64, c: &u64, d: &u64| {
                                    fold_values(seed, &[*a, *b, *c, *d])
                                },
                            )?,
                            other => unreachable!("generator emits arity 1..=4, got {other}"),
                        }
                    }
                };
                nodes.push(node);
            }
            Ok((nodes, inputs))
        })
        .unwrap()
}

/// Static wide graph: `Constant(7)` then 32 `x + i` branches, `i = 0..31`.
fn build_wide(state: &mut GraphState) -> (Vec<Input<u64>>, Vec<Node<u64>>, Vec<Node<u64>>) {
    state
        .edit(|e| {
            e.set_default_eq::<u64>();
            let input0 = e.input::<u64>();
            let input1 = e.input::<u64>();
            let input2 = e.input::<u64>();
            let base = e.map((), || 7_u64)?;
            let mut branches = Vec::with_capacity(32);
            for offset in 0..32_u64 {
                branches.push(e.map(base, move |x: &u64| x + offset)?);
            }
            let mut computations = vec![base];
            computations.extend(branches.iter().copied());
            Ok((vec![input0, input1, input2], computations, branches))
        })
        .unwrap()
}

/// Linear chain from `input0`, one `+1` layer per step.
fn build_chain(
    state: &mut GraphState,
    layers: usize,
) -> (Vec<Input<u64>>, Vec<Node<u64>>, Node<u64>) {
    state
        .edit(|e| {
            e.set_default_eq::<u64>();
            let input0 = e.input::<u64>();
            let input1 = e.input::<u64>();
            let input2 = e.input::<u64>();
            let mut computations = Vec::with_capacity(layers);
            let mut current = input0.node();
            for _ in 0..layers {
                current = e.map(current, |x: &u64| x.wrapping_add(1))?;
                computations.push(current);
            }
            Ok((vec![input0, input1, input2], computations, current))
        })
        .unwrap()
}

fn set_inputs(state: &mut GraphState, inputs: &[Input<u64>], values: [u64; 3]) {
    state
        .edit(|e| {
            for (&input, &value) in inputs.iter().zip(values.iter()) {
                e.set(input, value)?;
            }
            Ok(())
        })
        .unwrap()
}

/// Publishes `S0`, `S1`, `S2` from one graph; the sequence reuses these snapshots.
fn build_snapshots(state: &mut GraphState, inputs: &[Input<u64>]) -> [EvaluationSnapshot; 3] {
    set_inputs(state, inputs, SNAPSHOTS[0]);
    let s0 = state.snapshot();
    set_inputs(state, inputs, SNAPSHOTS[1]);
    let s1 = state.snapshot();
    set_inputs(state, inputs, SNAPSHOTS[2]);
    let s2 = state.snapshot();
    [s0, s1, s2]
}

// ---------------------------------------------------------------------------
// Cache matrix and planner configuration
// ---------------------------------------------------------------------------

fn policies() -> [CachePolicy; 3] {
    [
        CachePolicy::Off,
        CachePolicy::Last,
        CachePolicy::Lru { capacity: 4 },
    ]
}

/// Full 3x3 memo/context matrix.
fn cache_combos() -> Vec<NodeCacheConfig> {
    let mut combos = Vec::with_capacity(9);
    for memo in policies() {
        for context in policies() {
            combos.push(NodeCacheConfig { memo, context });
        }
    }
    combos
}

fn off_off() -> NodeCacheConfig {
    NodeCacheConfig {
        memo: CachePolicy::Off,
        context: CachePolicy::Off,
    }
}

fn last_last() -> NodeCacheConfig {
    NodeCacheConfig {
        memo: CachePolicy::Last,
        context: CachePolicy::Last,
    }
}

fn lru_lru() -> NodeCacheConfig {
    NodeCacheConfig {
        memo: CachePolicy::Lru { capacity: 4 },
        context: CachePolicy::Lru { capacity: 4 },
    }
}

fn planner_config(config: (usize, usize, usize)) -> PlannerConfig {
    PlannerConfig {
        frame_window: config.0,
        max_in_flight: config.1,
        comparison_budget: config.2,
        ..Default::default()
    }
}

fn configure(evaluator: &mut Evaluator, computations: &[Node<u64>], config: NodeCacheConfig) {
    for &node in computations {
        evaluator.configure_cache(node, config).unwrap();
    }
}

fn assert_peaks(stats: &PlannerStats, config: (usize, usize, usize)) {
    let full = planner_config(config);
    assert!(
        stats.peak_in_flight <= full.max_in_flight,
        "peak_in_flight {} exceeds {}",
        stats.peak_in_flight,
        full.max_in_flight
    );
    assert!(stats.peak_ready_tasks <= full.max_ready_tasks);
    assert!(stats.peak_registry_entries <= full.max_registry_entries);
    assert!(stats.peak_resolution_tasks <= full.max_resolution_tasks);
}

// ---------------------------------------------------------------------------
// Requests, scheduling and comparison
// ---------------------------------------------------------------------------

/// Requests every output at every frame, then repeats the first output once more.
fn build_batch(
    frames: &[u64],
    outputs: &[Node<u64>],
) -> (RequestBatch, Vec<(u64, RequestId<u64>, usize)>) {
    let mut batch = RequestBatch::new();
    let mut ids = Vec::new();
    for &frame in frames {
        for (position, &node) in outputs.iter().enumerate() {
            let id = batch.at(FrameIndex(frame)).request(node);
            ids.push((frame, id, position));
        }
        let repeat = batch.at(FrameIndex(frame)).request(outputs[0]);
        ids.push((frame, repeat, 0));
    }
    (batch, ids)
}

fn check_batch<F: Fn(u64, usize) -> u64>(
    results: &EvaluationResults,
    ids: &[(u64, RequestId<u64>, usize)],
    oracle: F,
) {
    for &(frame, id, position) in ids {
        let value = *results.get(id).unwrap();
        assert_eq!(
            value,
            oracle(frame, position),
            "frame {frame} output index {position}"
        );
    }
}

/// Repeated identical `(frame, node)` requests must return the same Arc identity.
fn check_identity(results: &EvaluationResults, ids: &[(u64, RequestId<u64>, usize)]) {
    let mut seen: Vec<((u64, usize), Arc<u64>)> = Vec::new();
    for &(frame, id, position) in ids {
        let shared = results.get_shared(id).unwrap();
        match seen.iter().find(|(key, _)| *key == (frame, position)) {
            Some((_, first)) => assert!(
                Arc::ptr_eq(first, &shared),
                "duplicate request frame {frame} output {position} returned a distinct Arc"
            ),
            None => seen.push(((frame, position), shared)),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Order {
    /// Whole batch delivered immediately, in issue order.
    Immediate,
    /// Whole batch delivered in reverse issue order.
    Reverse,
    /// Each batch reversed, split once: first half delivered, one no-delivery advance,
    /// then the held half delivered whole in FIFO order.
    ReverseSplitHalf,
}

const ORDERS: [Order; 3] = [Order::Immediate, Order::Reverse, Order::ReverseSplitHalf];

/// Threadless driver over `plan`/`advance_with_budget`. Tasks run for real through
/// `ComputeTask::execute`; no completion is ever fabricated.
///
/// `deliveries` is a queue of whole delivery batches; an empty entry is an explicit
/// no-delivery advance. Split-half mode enqueues `[first half, empty, held half]` once
/// per produced batch and never re-halves an aggregate queue.
fn run_planned(
    evaluator: &mut Evaluator,
    snapshot: EvaluationSnapshot,
    batch: RequestBatch,
    config: PlannerConfig,
    order: Order,
) -> Result<(EvaluationResults, PlannerStats), EvalError> {
    let mut plan = evaluator.plan(snapshot, batch, config)?;
    let mut deliveries: VecDeque<Vec<ComputeCompletion>> = VecDeque::new();
    let mut advances = 0_usize;
    loop {
        let deliver = deliveries.pop_front().unwrap_or_default();
        advances += 1;
        assert!(
            advances <= 100_000,
            "plan did not converge in 100000 advances"
        );
        let tasks = evaluator.advance_with_budget(&mut plan, deliver, usize::MAX)?;
        if plan.is_complete() {
            break;
        }
        if tasks.is_empty() && deliveries.is_empty() {
            plan.cancel();
            return Err(EvalError::PlanIncomplete);
        }
        let mut completions: Vec<ComputeCompletion> =
            tasks.into_iter().map(ComputeTask::execute).collect();
        match order {
            Order::Immediate => {
                if !completions.is_empty() {
                    deliveries.push_back(completions);
                }
            }
            Order::Reverse => {
                if !completions.is_empty() {
                    completions.reverse();
                    deliveries.push_back(completions);
                }
            }
            Order::ReverseSplitHalf => {
                if !completions.is_empty() {
                    completions.reverse();
                    let held = completions.split_off(completions.len().div_ceil(2));
                    deliveries.push_back(completions);
                    if !held.is_empty() {
                        deliveries.push_back(Vec::new());
                        deliveries.push_back(held);
                    }
                }
            }
        }
    }
    assert!(
        deliveries.is_empty(),
        "plan completed without delivering every held completion"
    );
    plan.finish()
}

fn dfs_bind(
    evaluator: &mut Evaluator,
    snapshot: &EvaluationSnapshot,
    outputs: &[Node<u64>],
) -> (EvaluationResults, Vec<(u64, RequestId<u64>, usize)>) {
    let (batch, ids) = build_batch(&FRAMES, outputs);
    let results = evaluator
        .bind(snapshot.clone())
        .unwrap()
        .evaluate(batch)
        .unwrap();
    (results, ids)
}

/// Runs `[S0, S1, S0, S2]` on one evaluator via `run(slot)`, checking values, Arc
/// identity and (for planner runs, inside `run`) capacity peaks for every snapshot,
/// then re-verifies the retained first `S0` results.
fn sequence<F>(evaluator: &mut Evaluator, oracle: &dyn Fn(usize, u64, usize) -> u64, mut run: F)
where
    F: FnMut(&mut Evaluator, usize) -> (EvaluationResults, Vec<(u64, RequestId<u64>, usize)>),
{
    let (retained, retained_ids) = run(evaluator, 0);
    check_batch(&retained, &retained_ids, &|frame, position| {
        oracle(0, frame, position)
    });
    check_identity(&retained, &retained_ids);

    for slot in [1, 0, 2] {
        let (results, ids) = run(evaluator, slot);
        check_batch(&results, &ids, &|frame, position| {
            oracle(slot, frame, position)
        });
        check_identity(&results, &ids);
    }

    // Later bindings, inputs and computations must not disturb the retained results.
    check_batch(&retained, &retained_ids, &|frame, position| {
        oracle(0, frame, position)
    });
}

fn frame_oracle(ops: &[Op], inputs: [u64; 3]) -> HashMap<u64, Vec<u64>> {
    let mut values = HashMap::new();
    for &frame in &FRAMES {
        values
            .entry(frame)
            .or_insert_with(|| spec_values(ops, frame, &inputs));
    }
    values
}

/// Oracle over `(snapshot slot, frame, output position)` for a spec fixture.
fn spec_oracle<'a>(
    ops: &[Op],
    output_indices: &'a [usize],
) -> impl Fn(usize, u64, usize) -> u64 + 'a {
    let values: Vec<HashMap<u64, Vec<u64>>> = SNAPSHOTS
        .iter()
        .map(|&inputs| frame_oracle(ops, inputs))
        .collect();
    move |slot, frame, position| values[slot][&frame][output_indices[position]]
}

// ---------------------------------------------------------------------------
// Step 1 behaviour coverage
// ---------------------------------------------------------------------------

#[test]
fn generated_dags_match_oracle_across_snapshot_cache_config_order_matrix() {
    for &seed in &SEEDS {
        let (ops, output_indices) = generate(seed, DISCONNECTED);
        let mut state = GraphState::new();
        let (nodes, inputs) = build_spec(&mut state, &ops, true);
        let outputs: Vec<Node<u64>> = output_indices.iter().map(|&i| nodes[i]).collect();
        let computations: Vec<Node<u64>> = ops
            .iter()
            .enumerate()
            .filter(|(_, op)| !matches!(op, Op::Input(_)))
            .map(|(index, _)| nodes[index])
            .collect();
        let snapshots = build_snapshots(&mut state, &inputs);
        let oracle = spec_oracle(&ops, &output_indices);

        for combo in cache_combos() {
            let mut evaluator = Evaluator::from_snapshot(&snapshots[0]);
            configure(&mut evaluator, &computations, combo);
            sequence(&mut evaluator, &oracle, |evaluator, slot| {
                dfs_bind(evaluator, &snapshots[slot], &outputs)
            });
        }

        for combo in cache_combos() {
            for &config in &CONFIGS {
                for &order in &ORDERS {
                    let mut evaluator = Evaluator::from_snapshot(&snapshots[0]);
                    configure(&mut evaluator, &computations, combo);
                    sequence(&mut evaluator, &oracle, |evaluator, slot| {
                        let (batch, ids) = build_batch(&FRAMES, &outputs);
                        let (results, stats) = run_planned(
                            evaluator,
                            snapshots[slot].clone(),
                            batch,
                            planner_config(config),
                            order,
                        )
                        .unwrap();
                        assert_peaks(&stats, config);
                        (results, ids)
                    });
                }
            }
        }
    }
}

#[test]
fn fixed_diamond_matches_snapshot_matrix_and_sync_entry() {
    let (ops, output_indices) = diamond();
    for eq in [true, false] {
        let mut state = GraphState::new();
        let (nodes, inputs) = build_spec(&mut state, &ops, eq);
        let outputs: Vec<Node<u64>> = output_indices.iter().map(|&i| nodes[i]).collect();
        let computations: Vec<Node<u64>> = ops
            .iter()
            .enumerate()
            .filter(|(_, op)| !matches!(op, Op::Input(_)))
            .map(|(index, _)| nodes[index])
            .collect();
        let snapshots = build_snapshots(&mut state, &inputs);
        let oracle = spec_oracle(&ops, &output_indices);

        // Identity-only runs take Off/Off; the equality variant walks the full matrix.
        let combos = if eq { cache_combos() } else { vec![off_off()] };
        for combo in combos {
            let mut evaluator = Evaluator::from_snapshot(&snapshots[0]);
            configure(&mut evaluator, &computations, combo);
            sequence(&mut evaluator, &oracle, |evaluator, slot| {
                dfs_bind(evaluator, &snapshots[slot], &outputs)
            });

            for &config in &CONFIGS {
                for &order in &ORDERS {
                    let mut evaluator = Evaluator::from_snapshot(&snapshots[0]);
                    configure(&mut evaluator, &computations, combo);
                    sequence(&mut evaluator, &oracle, |evaluator, slot| {
                        let (batch, ids) = build_batch(&FRAMES, &outputs);
                        let (results, stats) = run_planned(
                            evaluator,
                            snapshots[slot].clone(),
                            batch,
                            planner_config(config),
                            order,
                        )
                        .unwrap();
                        assert_peaks(&stats, config);
                        (results, ids)
                    });
                }
            }
        }

        if eq {
            // Synchronous caller-driven entry point.
            let config = CONFIGS[2];
            let mut evaluator = Evaluator::from_snapshot(&snapshots[0]);
            configure(&mut evaluator, &computations, off_off());
            sequence(&mut evaluator, &oracle, |evaluator, slot| {
                let (batch, ids) = build_batch(&FRAMES, &outputs);
                let (results, stats) = evaluator
                    .bind(snapshots[slot].clone())
                    .unwrap()
                    .evaluate_planned(batch, planner_config(config), &mut SerialExecutor)
                    .unwrap();
                assert_peaks(&stats, config);
                (results, ids)
            });
        }
    }
}

#[test]
fn static_wide_graph_matches_snapshot_configs_and_cache_subset() {
    let mut state = GraphState::new();
    let (inputs, computations, branches) = build_wide(&mut state);
    let snapshots = build_snapshots(&mut state, &inputs);
    assert_eq!(branches.len(), 32);
    // Direct integer oracle: branch `i` is `Constant(7) + i`, independent of frame/snapshot.
    let oracle = |_: usize, _: u64, position: usize| 7 + position as u64;

    for combo in [off_off(), last_last(), lru_lru()] {
        let mut evaluator = Evaluator::from_snapshot(&snapshots[0]);
        configure(&mut evaluator, &computations, combo);
        sequence(&mut evaluator, &oracle, |evaluator, slot| {
            dfs_bind(evaluator, &snapshots[slot], &branches)
        });

        for &config in &CONFIGS {
            for &order in &ORDERS {
                let mut evaluator = Evaluator::from_snapshot(&snapshots[0]);
                configure(&mut evaluator, &computations, combo);
                sequence(&mut evaluator, &oracle, |evaluator, slot| {
                    let (batch, ids) = build_batch(&FRAMES, &branches);
                    let (results, stats) = run_planned(
                        evaluator,
                        snapshots[slot].clone(),
                        batch,
                        planner_config(config),
                        order,
                    )
                    .unwrap();
                    assert_peaks(&stats, config);
                    (results, ids)
                });
            }
        }
    }
}

#[test]
fn chain_depth_boundary_separates_planner_from_dfs_limit() {
    let combo = off_off();

    // 200 layers stay inside the DFS depth limit.
    let mut state = GraphState::new();
    let (inputs, computations, tail) = build_chain(&mut state, 200);
    let snapshots = build_snapshots(&mut state, &inputs);
    let outputs = [tail];
    let oracle = |slot: usize, _: u64, _: usize| SNAPSHOTS[slot][0] + 200;

    let mut evaluator = Evaluator::from_snapshot(&snapshots[0]);
    configure(&mut evaluator, &computations, combo);
    sequence(&mut evaluator, &oracle, |evaluator, slot| {
        dfs_bind(evaluator, &snapshots[slot], &outputs)
    });

    for &config in &CONFIGS {
        for &order in &ORDERS {
            let mut evaluator = Evaluator::from_snapshot(&snapshots[0]);
            configure(&mut evaluator, &computations, combo);
            sequence(&mut evaluator, &oracle, |evaluator, slot| {
                let (batch, ids) = build_batch(&FRAMES, &outputs);
                let (results, stats) = run_planned(
                    evaluator,
                    snapshots[slot].clone(),
                    batch,
                    planner_config(config),
                    order,
                )
                .unwrap();
                assert_peaks(&stats, config);
                (results, ids)
            });
        }
    }

    // 400 layers exceed the unchanged DFS depth limit; the planner succeeds.
    let mut deep = GraphState::new();
    let (inputs, computations, tail) = build_chain(&mut deep, 400);
    let snapshots = build_snapshots(&mut deep, &inputs);
    let outputs = [tail];
    let oracle = |slot: usize, _: u64, _: usize| SNAPSHOTS[slot][0] + 400;

    for &config in &CONFIGS {
        for &order in &ORDERS {
            let mut evaluator = Evaluator::from_snapshot(&snapshots[0]);
            configure(&mut evaluator, &computations, combo);
            sequence(&mut evaluator, &oracle, |evaluator, slot| {
                let (batch, ids) = build_batch(&FRAMES, &outputs);
                let (results, stats) = run_planned(
                    evaluator,
                    snapshots[slot].clone(),
                    batch,
                    planner_config(config),
                    order,
                )
                .unwrap();
                assert_peaks(&stats, config);
                (results, ids)
            });
        }
    }

    let mut evaluator = Evaluator::from_snapshot(&snapshots[0]);
    configure(&mut evaluator, &computations, combo);
    let (batch, _ids) = build_batch(&FRAMES, &outputs);
    let outcome = evaluator
        .bind(snapshots[0].clone())
        .unwrap()
        .evaluate(batch);
    let error = match outcome {
        Ok(_) => panic!("400 layer chain unexpectedly resolved under DFS"),
        Err(error) => error,
    };
    assert_eq!(error, EvalError::DepthLimit { limit: 256 });
}

#[test]
fn disconnected_appendix_is_not_expanded() {
    for combo in [off_off(), last_last()] {
        let base = planner_summary(42, DISCONNECTED, combo);
        let large = planner_summary(42, DISCONNECTED + 1024, combo);
        assert_eq!(base.0, large.0, "appendix changed requested outputs");
        assert_eq!(
            base.1.resolution_tasks, large.1.resolution_tasks,
            "appendix expanded resolution work"
        );
        assert_eq!(
            base.1.compute_tasks, large.1.compute_tasks,
            "appendix expanded computations"
        );
        assert_eq!(
            base.1.windows, large.1.windows,
            "appendix changed window scheduling"
        );
    }
}

/// Planner summary of seed 42 with `disconnected` appended items: `(outputs, stats)`.
fn planner_summary(
    seed: u64,
    disconnected: usize,
    combo: NodeCacheConfig,
) -> (Vec<u64>, PlannerStats) {
    let (ops, output_indices) = generate(seed, disconnected);
    let mut state = GraphState::new();
    let (nodes, inputs) = build_spec(&mut state, &ops, true);
    set_inputs(&mut state, &inputs, SNAPSHOTS[0]);
    let snapshot = state.snapshot();
    let outputs: Vec<Node<u64>> = output_indices.iter().map(|&i| nodes[i]).collect();
    let computations: Vec<Node<u64>> = ops
        .iter()
        .enumerate()
        .filter(|(_, op)| !matches!(op, Op::Input(_)))
        .map(|(index, _)| nodes[index])
        .collect();
    let values = frame_oracle(&ops, SNAPSHOTS[0]);
    let oracle = |frame: u64, position: usize| values[&frame][output_indices[position]];

    let mut evaluator = Evaluator::from_snapshot(&snapshot);
    configure(&mut evaluator, &computations, combo);
    let (batch, ids) = build_batch(&FRAMES, &outputs);
    let (results, stats) = run_planned(
        &mut evaluator,
        snapshot,
        batch,
        planner_config(CONFIGS[1]),
        Order::Immediate,
    )
    .unwrap();
    check_batch(&results, &ids, &oracle);
    assert_peaks(&stats, CONFIGS[1]);
    let collected = ids
        .iter()
        .map(|&(_, id, _)| *results.get(id).unwrap())
        .collect();
    (collected, stats)
}
