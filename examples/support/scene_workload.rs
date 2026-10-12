//! Shared fixture for correctness tests and the release performance example.
//! This is an integer scene-state workload, not rendering or GPU work.
use gekijyou::computation::{
    BatchResults, CachePolicy, ComputationInputs, Context, Evaluator, FrameIndex, Graph, Node,
    RequestBatch, RequestId,
};
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub actors: usize,
    pub stages: usize,
    pub rounds: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Viewport {
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct Input {
    pub frame: FrameIndex,
    pub viewport: Viewport,
    pub seed: u64,
}

// Deliberately non-Clone, shared by all actors in both implementations.
struct Style {
    palette: Vec<u64>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Camera {
    x: i64,
    y: i64,
    zoom: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Actor {
    x: i64,
    y: i64,
    width: u32,
    opacity: u32,
    deformation: u64,
    color: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Caption {
    visible_chars: u32,
    anchor_x: i64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Aggregate {
    signature: u64,
    visible: u32,
    left: i64,
    right: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scene {
    signature: u64,
    visible: u32,
    left: i64,
    right: i64,
    viewport_area: u64,
    caption: Caption,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Metrics {
    signature: u64,
    span: u64,
    exposure: u64,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Snapshot {
    scene: Scene,
    first_actor: Actor,
    metrics: Metrics,
    last_actor: Actor,
    caption: Caption,
    duplicate_scene: Scene,
}

struct Requests {
    scene: RequestId<Scene>,
    first_actor: RequestId<Actor>,
    metrics: RequestId<Metrics>,
    last_actor: RequestId<Actor>,
    caption: RequestId<Caption>,
    duplicate_scene: RequestId<Scene>,
}

pub struct Prepared {
    batch: RequestBatch,
    requests: Vec<Requests>,
}

pub struct Evaluated {
    results: BatchResults,
    requests: Vec<Requests>,
}

pub struct DirectOutput {
    scene: Arc<Scene>,
    first_actor: Arc<Actor>,
    metrics: Arc<Metrics>,
    last_actor: Arc<Actor>,
    caption: Arc<Caption>,
    duplicate_scene: Arc<Scene>,
}

pub struct Workload {
    graph: Graph,
    viewport: Node<Viewport>,
    #[allow(dead_code)]
    phase: Node<u64>,
    #[allow(dead_code)]
    phase_input: Node<u64>,
    seed: Node<u64>,
    scene: Node<Scene>,
    first_actor: Node<Actor>,
    last_actor: Node<Actor>,
    metrics: Node<Metrics>,
    caption: Node<Caption>,
    #[allow(dead_code)]
    unrelated: Node<u64>,
    style: Style,
    config: Config,
    node_count: usize,
}

const BASE: u64 = 0x9e37_79b9_7f4a_7c15;

fn phase(frame: &FrameIndex) -> u64 {
    (frame.0 / 4).wrapping_mul(3)
}

fn camera(phase: &u64, viewport: &Viewport) -> Camera {
    Camera {
        x: (*phase % 801) as i64 - 400,
        y: (*phase % 241) as i64 - 120,
        zoom: 1000 + ((*phase ^ u64::from(viewport.width)) % 101),
    }
}

fn exposure(phase: &u64) -> u64 {
    512 + phase % 513
}

fn caption(phase: &u64, camera: &Camera) -> Caption {
    Caption {
        visible_chars: (phase % 81) as u32,
        anchor_x: 960 - camera.x,
    }
}

fn actor_seed(phase: &u64, seed: &u64, style: &Style, base: &u64, actor: usize) -> u64 {
    phase.wrapping_add(*seed).wrapping_add(*base)
        ^ style.palette[actor % style.palette.len()]
        ^ (actor as u64).wrapping_mul(BASE)
}

fn position(phase: &u64, camera: &Camera, actor: usize) -> i64 {
    (actor as i64 * 53 + (*phase % 997) as i64 - camera.x) * camera.zoom as i64 / 1000
}

fn kernel(mut value: u64, rounds: usize) -> u64 {
    for _ in 0..rounds {
        value ^= value >> 27;
        value = value.wrapping_mul(0x3c79_ac49_2ba7_b653).rotate_left(17);
        value ^= value >> 33;
        value = value.wrapping_add(BASE);
    }
    value
}

fn left(state: &u64, camera: &Camera, stage: usize, rounds: usize) -> u64 {
    kernel(
        state
            .wrapping_add(camera.zoom)
            .wrapping_add(camera.y as u64)
            .wrapping_add(stage as u64),
        rounds,
    )
}

fn right(state: &u64, exposure: &u64, stage: usize, rounds: usize) -> u64 {
    kernel(state.rotate_left(11) ^ exposure ^ stage as u64, rounds)
}

fn merge(a: &u64, b: &u64, previous: &u64, repeated_a: &u64) -> u64 {
    a.wrapping_add(b.rotate_left(7)) ^ previous.rotate_left(19) ^ repeated_a.rotate_right(5)
}

fn actor(state: &u64, x: &i64, exposure: &u64, style: &Style) -> Actor {
    Actor {
        x: *x,
        y: ((*state >> 17) % 720) as i64,
        width: 64 + (*state % 128) as u32,
        opacity: ((*state % 1025) * exposure / 1024) as u32,
        deformation: *state,
        color: style.palette[(*state % style.palette.len() as u64) as usize],
    }
}

fn aggregate_actor(actor: &Actor) -> Aggregate {
    Aggregate {
        signature: actor.deformation
            ^ actor.color
            ^ (actor.x as u64).rotate_left(13)
            ^ (actor.y as u64).rotate_left(29)
            ^ u64::from(actor.opacity),
        visible: u32::from(actor.opacity > 0),
        left: actor.x,
        right: actor.x + i64::from(actor.width),
    }
}

fn combine(a: Aggregate, b: Aggregate) -> Aggregate {
    Aggregate {
        signature: a.signature.wrapping_add(b.signature),
        visible: a.visible + b.visible,
        left: a.left.min(b.left),
        right: a.right.max(b.right),
    }
}

fn combine_four(a: &Aggregate, b: &Aggregate, c: &Aggregate, d: &Aggregate) -> Aggregate {
    combine(combine(*a, *b), combine(*c, *d))
}

fn scene(aggregate: &Aggregate, caption: &Caption, viewport: &Viewport) -> Scene {
    Scene {
        signature: aggregate.signature,
        visible: aggregate.visible,
        left: aggregate.left,
        right: aggregate.right,
        viewport_area: u64::from(viewport.width) * u64::from(viewport.height),
        caption: *caption,
    }
}

fn metrics(aggregate: &Aggregate, exposure: &u64) -> Metrics {
    Metrics {
        signature: aggregate.signature,
        span: (aggregate.right - aggregate.left) as u64,
        exposure: *exposure,
    }
}

fn checksum(snapshot: &Snapshot) -> u64 {
    // Observe every requested result, including the separately issued duplicate.
    snapshot.scene.signature
        ^ snapshot.scene.viewport_area
        ^ snapshot.scene.visible as u64
        ^ (snapshot.scene.left as u64).rotate_left(7)
        ^ (snapshot.scene.right as u64).rotate_left(11)
        ^ u64::from(snapshot.scene.caption.visible_chars)
        ^ (snapshot.scene.caption.anchor_x as u64).rotate_left(3)
        ^ aggregate_actor(&snapshot.first_actor)
            .signature
            .rotate_left(17)
        ^ aggregate_actor(&snapshot.last_actor)
            .signature
            .rotate_left(31)
        ^ snapshot.metrics.signature.rotate_left(5)
        ^ snapshot.metrics.span
        ^ snapshot.metrics.exposure
        ^ u64::from(snapshot.caption.visible_chars).rotate_left(23)
        ^ (snapshot.caption.anchor_x as u64)
        ^ snapshot.duplicate_scene.signature.rotate_left(43)
}

/// How newly created scene values are compared as downstream memo inputs.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ValueComparison {
    #[allow(dead_code)] // The cache timing example uses only content comparison.
    Identity,
    Content,
}

// Fixture-only helper: apply the selected policy as each typed node is created.
trait MapWithPolicy {
    fn map_with_policy<I, F, O>(&mut self, policy: CachePolicy, inputs: I, f: F) -> Node<O>
    where
        I: ComputationInputs<F, O>,
        O: 'static;
}

impl MapWithPolicy for Graph {
    fn map_with_policy<I, F, O>(&mut self, policy: CachePolicy, inputs: I, f: F) -> Node<O>
    where
        I: ComputationInputs<F, O>,
        O: 'static,
    {
        let node = self.map(inputs, f);
        self.set_cache_policy(node, policy).unwrap();
        node
    }
}

impl Workload {
    #[allow(dead_code)] // The cache timing example always supplies explicit options.
    pub fn new(config: Config) -> Self {
        Self::configured(config, CachePolicy::Last, ValueComparison::Identity)
    }

    /// Same scene and closures; optional exact content semantics for P2 reuse loads.
    pub fn configured(config: Config, policy: CachePolicy, comparison: ValueComparison) -> Self {
        assert!(config.actors > 0);
        let style = || Style {
            palette: (0..16)
                .map(|i: u64| i.wrapping_mul(BASE) ^ 0xaabb_ccdd)
                .collect(),
        };
        let mut graph = Graph::new();
        if comparison == ValueComparison::Content {
            graph.set_default_eq::<u64>();
            graph.set_default_eq::<i64>();
            graph.set_default_eq::<Viewport>();
            graph.set_default_eq::<Camera>();
            graph.set_default_eq::<Actor>();
            graph.set_default_eq::<Caption>();
            graph.set_default_eq::<Aggregate>();
        }
        let viewport = graph.input::<Viewport>();
        let seed = graph.input::<u64>();
        let style_node = graph.constant(style());
        graph.set_cache_policy(style_node, policy).unwrap();
        let base = graph.map_with_policy(policy, (), || BASE);
        // The replacement connects an early node to a later node, so graph storage
        // order cannot be used as an execution schedule.
        let phase_node = graph.map_with_policy(policy, (), || 0_u64);
        let camera_node = graph.map_with_policy(policy, (phase_node, viewport), camera);
        let exposure_node = graph.map_with_policy(policy, phase_node, exposure);
        let caption_node = graph.map_with_policy(policy, (phase_node, camera_node), caption);
        let later_phase =
            graph.map_with_policy(policy, graph.frame(), |frame: &FrameIndex| frame.0 / 4);
        graph
            .replace(phase_node, later_phase, |n: &u64| n.wrapping_mul(3))
            .unwrap();
        let mut actors = Vec::with_capacity(config.actors);
        // Ten shared nodes above, including the built-in FrameIndex.
        let mut node_count = 10;
        for index in 0..config.actors {
            let mut state = graph.map_with_policy(
                policy,
                (phase_node, seed, style_node, base),
                move |p: &u64, s: &u64, style: &Style, base: &u64| {
                    actor_seed(p, s, style, base, index)
                },
            );
            let x = graph.map_with_policy(
                policy,
                (phase_node, camera_node),
                move |p: &u64, c: &Camera| position(p, c, index),
            );
            for stage in 0..config.stages {
                let a = graph.map_with_policy(
                    policy,
                    (state, camera_node),
                    move |s: &u64, c: &Camera| left(s, c, stage, config.rounds),
                );
                let b = graph.map_with_policy(
                    policy,
                    (state, exposure_node),
                    move |s: &u64, e: &u64| right(s, e, stage, config.rounds),
                );
                state = graph.map_with_policy(policy, (a, b, state, a), merge);
            }
            actors.push(graph.map_with_policy(
                policy,
                (state, x, exposure_node, style_node),
                actor,
            ));
            node_count += 3 + config.stages * 3;
        }
        let first_actor = actors[0];
        let last_actor = actors[actors.len() - 1];
        let mut aggregates: Vec<_> = actors
            .into_iter()
            .map(|a| graph.map_with_policy(policy, a, aggregate_actor))
            .collect();
        node_count += aggregates.len();
        while aggregates.len() > 1 {
            aggregates = aggregates
                .chunks(4)
                .map(|chunk| {
                    node_count += 1;
                    match chunk {
                        [a, b, c, d] => {
                            graph.map_with_policy(policy, (*a, *b, *c, *d), combine_four)
                        }
                        [a, b, c] => graph.map_with_policy(
                            policy,
                            (*a, *b, *c),
                            |a: &Aggregate, b: &Aggregate, c: &Aggregate| {
                                combine(combine(*a, *b), *c)
                            },
                        ),
                        [a, b] => graph.map_with_policy(
                            policy,
                            (*a, *b),
                            |a: &Aggregate, b: &Aggregate| combine(*a, *b),
                        ),
                        [a] => graph.map_with_policy(policy, *a, |a: &Aggregate| *a),
                        _ => unreachable!(),
                    }
                })
                .collect();
        }
        let scene_node =
            graph.map_with_policy(policy, (aggregates[0], caption_node, viewport), scene);
        let metrics_node = graph.map_with_policy(policy, (aggregates[0], exposure_node), metrics);
        let unused = graph.input::<u64>();
        let unrelated = graph.map_with_policy(policy, unused, |_: &u64| -> u64 {
            panic!("Unrequested branch must not execute")
        });
        node_count += 4;
        graph.validate().unwrap();
        Self {
            graph,
            viewport,
            phase: phase_node,
            phase_input: later_phase,
            seed,
            scene: scene_node,
            first_actor,
            last_actor,
            metrics: metrics_node,
            caption: caption_node,
            unrelated,
            style: style(),
            config,
            node_count,
        }
    }

    /// Install a fresh phase definition with equivalent output. Graph validation
    /// is part of this edit; unchanged downstream history must remain usable.
    #[allow(dead_code)] // Shared fixture; the legacy timing example does not edit.
    pub fn replace_phase(&mut self) {
        self.graph
            .replace(self.phase, self.phase_input, |n: &u64| n.wrapping_mul(3))
            .unwrap();
    }

    /// Enable projection caching only at the five requested output boundaries.
    /// The sixth request duplicates Scene. Internal cheap nodes retain P2 policy.
    #[allow(dead_code)]
    pub fn set_context_capacity(&mut self, capacity: usize) {
        self.graph
            .set_context_cache_capacity(self.scene, capacity)
            .unwrap();
        self.graph
            .set_context_cache_capacity(self.first_actor, capacity)
            .unwrap();
        self.graph
            .set_context_cache_capacity(self.last_actor, capacity)
            .unwrap();
        self.graph
            .set_context_cache_capacity(self.metrics, capacity)
            .unwrap();
        self.graph
            .set_context_cache_capacity(self.caption, capacity)
            .unwrap();
    }

    #[allow(dead_code)]
    pub fn replace_unrelated(&mut self) {
        self.graph.replace(self.unrelated, (), || 0_u64).unwrap();
    }

    pub fn node_count(&self) -> usize {
        self.node_count
    }

    pub fn evaluator(&self) -> Evaluator {
        Evaluator::new(&self.graph)
    }

    pub fn prepare(&self, inputs: &[Input]) -> Prepared {
        let mut batch = RequestBatch::new(&self.graph);
        let requests = inputs
            .iter()
            .map(|input| {
                let mut context = Context::new(&self.graph);
                context
                    .set(&self.graph, self.graph.frame(), input.frame)
                    .unwrap();
                context
                    .set(&self.graph, self.viewport, input.viewport)
                    .unwrap();
                context.set(&self.graph, self.seed, input.seed).unwrap();
                let id = batch.add_context(context).unwrap();
                // Scene first traverses the entire actor DAG. Later requests share its
                // upstream work; duplicate scene still receives its own RequestId.
                Requests {
                    scene: batch.request(id, self.scene).unwrap(),
                    first_actor: batch.request(id, self.first_actor).unwrap(),
                    metrics: batch.request(id, self.metrics).unwrap(),
                    last_actor: batch.request(id, self.last_actor).unwrap(),
                    caption: batch.request(id, self.caption).unwrap(),
                    duplicate_scene: batch.request(id, self.scene).unwrap(),
                }
            })
            .collect();
        Prepared { batch, requests }
    }

    pub fn evaluate(&self, evaluator: &mut Evaluator, prepared: Prepared) -> Evaluated {
        Evaluated {
            results: evaluator
                .evaluate_batch(&self.graph, prepared.batch)
                .unwrap(),
            requests: prepared.requests,
        }
    }

    pub fn direct(&self, inputs: &[Input]) -> Vec<DirectOutput> {
        inputs
            .iter()
            .map(|input| {
                // Shared globals and actor states are computed once per context, just
                // like the graph. The direct baseline uses statically typed locals.
                let phase = phase(&input.frame);
                let camera = camera(&phase, &input.viewport);
                let exposure = exposure(&phase);
                let caption = caption(&phase, &camera);
                let actors: Vec<_> = (0..self.config.actors)
                    .map(|index| {
                        let mut state = actor_seed(&phase, &input.seed, &self.style, &BASE, index);
                        let x = position(&phase, &camera, index);
                        for stage in 0..self.config.stages {
                            let a = left(&state, &camera, stage, self.config.rounds);
                            let b = right(&state, &exposure, stage, self.config.rounds);
                            state = merge(&a, &b, &state, &a);
                        }
                        actor(&state, &x, &exposure, &self.style)
                    })
                    .collect();
                let mut aggregates: Vec<_> = actors.iter().map(aggregate_actor).collect();
                while aggregates.len() > 1 {
                    aggregates = aggregates
                        .chunks(4)
                        .map(|chunk| chunk.iter().copied().reduce(combine).unwrap())
                        .collect();
                }
                let scene = Arc::new(scene(&aggregates[0], &caption, &input.viewport));
                DirectOutput {
                    duplicate_scene: Arc::clone(&scene),
                    scene,
                    first_actor: Arc::new(actors[0]),
                    last_actor: Arc::new(actors[actors.len() - 1]),
                    metrics: Arc::new(metrics(&aggregates[0], &exposure)),
                    caption: Arc::new(caption),
                }
            })
            .collect()
    }
}

impl Evaluated {
    fn snapshot(&self, request: &Requests) -> Snapshot {
        Snapshot {
            scene: *self.results.get(request.scene).unwrap(),
            first_actor: *self.results.get(request.first_actor).unwrap(),
            metrics: *self.results.get(request.metrics).unwrap(),
            last_actor: *self.results.get(request.last_actor).unwrap(),
            caption: *self.results.get(request.caption).unwrap(),
            duplicate_scene: *self.results.get(request.duplicate_scene).unwrap(),
        }
    }

    pub fn snapshots(&self) -> Vec<Snapshot> {
        self.requests.iter().map(|r| self.snapshot(r)).collect()
    }

    pub fn checksum(&self) -> u64 {
        self.requests.iter().fold(0_u64, |sum, r| {
            sum.wrapping_add(checksum(&self.snapshot(r)))
        })
    }

    pub fn duplicates_share_storage(&self) -> bool {
        self.requests.iter().all(|r| {
            std::ptr::eq(
                self.results.get(r.scene).unwrap(),
                self.results.get(r.duplicate_scene).unwrap(),
            )
        })
    }
}

impl DirectOutput {
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            scene: *self.scene,
            first_actor: *self.first_actor,
            metrics: *self.metrics,
            last_actor: *self.last_actor,
            caption: *self.caption,
            duplicate_scene: *self.duplicate_scene,
        }
    }
}

pub fn direct_checksum(outputs: &[DirectOutput]) -> u64 {
    outputs.iter().fold(0_u64, |sum, output| {
        sum.wrapping_add(checksum(&output.snapshot()))
    })
}

#[allow(dead_code)] // P3 datasets use explicit frame sequences instead.
pub fn inputs(contexts: usize, variant: u64) -> Vec<Input> {
    (0..contexts)
        .map(|i| Input {
            frame: FrameIndex(variant.wrapping_mul(97).wrapping_add(i as u64 * 37)),
            viewport: Viewport {
                width: 1280 + (i as u32 % 5) * 160,
                height: 720 + (variant as u32 % 3) * 180,
            },
            seed: variant.wrapping_mul(BASE) ^ (i as u64).wrapping_mul(0xdead_beef),
        })
        .collect()
}
