use super::{
    error::GraphError,
    inputs::{ComputationDefinition, ComputationInputs},
    node::{CachePolicy, Frame, GraphId, Node, NodeId, NodeInput, NodeKind, NodeRecord},
    value::{Computation, SharedValue, ValueSemantics},
};
use slotmap::{SecondaryMap, SlotMap};
use std::{
    any::TypeId,
    collections::{HashMap, VecDeque},
    sync::Arc,
};

/// A graph storing heterogeneous nodes, dependencies, and computation closures.
///
/// Obtain the initial input with [`Graph::frame`] and add computations with
/// [`Graph::map`]. New nodes depend on existing nodes;
/// [`Graph::replace`] may reconnect
/// computations to any existing nodes as long as the graph remains acyclic.
///
/// Graph identities validate handle ownership at runtime; they are not stable
/// object identifiers across project rebuilds.
pub struct Graph {
    id: GraphId,
    nodes: SlotMap<NodeId, NodeRecord>,
    frame: Node<Frame>,
    value_semantics: HashMap<TypeId, ValueSemantics>,
}

impl Graph {
    /// Creates a graph containing only its frame input node.
    ///
    /// Each call allocates a distinct graph identity. [`Default::default`] has
    /// the same behavior.
    ///
    /// # Panics
    ///
    /// Panics if the process-local graph identity counter is exhausted, rather
    /// than reusing an existing identity.
    pub fn new() -> Self {
        let id = GraphId::new();
        let mut nodes = SlotMap::with_key();
        let frame_semantics = ValueSemantics::by_eq::<Frame>();
        let value_semantics = HashMap::from([(TypeId::of::<Frame>(), frame_semantics.clone())]);
        let frame = Node::new(
            id,
            nodes.insert(NodeRecord {
                type_id: TypeId::of::<Frame>(),
                kind: NodeKind::Input,
                semantics: frame_semantics,
                cache_policy: CachePolicy::default(),
            }),
        );

        Self {
            id,
            nodes,
            frame,
            value_semantics,
        }
    }

    /// Returns a handle to the frame input node.
    ///
    /// Returns the handle by value without retaining a graph borrow, allowing
    /// `graph.map(graph.frame(), |f: &Frame| f.0)`.
    /// Does not set or read an actual frame input value.
    pub fn frame(&self) -> Node<Frame> {
        self.frame
    }

    /// Declares an external input whose value will be supplied by a future context.
    pub fn input<T: Send + Sync + 'static>(&mut self) -> Node<T> {
        let record = self.record::<T>(NodeKind::Input);
        let id = self.nodes.insert(record);
        Node::new(self.id, id)
    }

    /// Registers dependencies and a closure, returning the typed output handle.
    ///
    /// Built-in inputs support `()`, a single node, and tuples of one to four nodes.
    /// The closure receives shared references in input order and returns `O`.
    /// Repeated input nodes are legal. This does not invoke `f`; input handles
    /// remain available for other branches.
    ///
    /// Built-in adapters require input types to be `'static` and the output and
    /// closure to be `Send + Sync + 'static`. Use `move` to capture owned data;
    /// this does not extend the lifetime of references it contains. Copying node
    /// handles does not require input values to be copyable.
    ///
    /// `Fn + Send + Sync` does not guarantee purity or determinism. Model external
    /// inputs explicitly and uphold the module's computation and value contracts.
    ///
    /// # Panics
    ///
    /// Panics if an input belongs to another graph, is missing, or has the wrong
    /// type. Invalid dependencies are rejected before inserting the node.
    pub fn map<I, F, O>(&mut self, inputs: I, f: F) -> Node<O>
    where
        I: ComputationInputs<F, O>,
        O: 'static,
    {
        self.insert_computation(inputs.build_computation(f))
    }

    /// Stores a constant as a zero-input computation, without requiring `T: Clone`.
    ///
    /// Its closure returns the same shared value on each invocation. The graph
    /// retains the value even when no evaluator caches its result.
    pub fn constant<T: Send + Sync + 'static>(&mut self, value: T) -> Node<T> {
        let value: SharedValue = Arc::new(value);
        let computation: Box<Computation> = Box::new(move |args| {
            assert!(args.is_empty(), "Invalid input count");
            Arc::clone(&value)
        });
        self.insert_computation(ComputationDefinition::<T>::new(Box::new([]), computation))
    }

    /// Uses `T::eq` for nodes of type `T` created after this call.
    ///
    /// Existing nodes, including the built-in frame input, retain their semantics.
    /// `Eq` alone does not establish that every downstream closure respects it;
    /// callers must uphold the module's substitution and immutability contracts.
    pub fn register_eq<T: Eq + 'static>(&mut self) {
        self.value_semantics
            .insert(TypeId::of::<T>(), ValueSemantics::by_eq::<T>());
    }

    /// Replaces a computation's closure and inputs, preserving its typed handle.
    ///
    /// `inputs` supports `()`, a single node, or tuples of one to four nodes.
    /// The closure is registered without being invoked. Output type, value
    /// semantics, and cache policy are preserved. Input nodes cannot be replaced.
    ///
    /// Returns an error for invalid handles, dependencies, or cycles, leaving the
    /// graph unchanged. Validation takes O(nodes + edges) time and space.
    pub fn replace<I, F, O>(&mut self, target: Node<O>, inputs: I, f: F) -> Result<(), GraphError>
    where
        I: ComputationInputs<F, O>,
        O: 'static,
    {
        self.replace_computation(target, inputs.build_computation(f))
    }

    /// Checks frame identity, dependency ownership, existence, types, and acyclicity.
    ///
    /// Checks the entire graph, including disconnected nodes, without executing
    /// closures. Takes O(nodes + edges) time and space; repeated edges are legal.
    pub fn validate(&self) -> Result<(), GraphError> {
        self.validate_with_replacement(None)
    }

    // Structural roots include every external input and zero-input computation.
    // Derive them from node definitions so graph edits need no separate root state.
    fn roots(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes.iter().filter_map(|(id, record)| {
            let is_root = match &record.kind {
                NodeKind::Input => true,
                NodeKind::Computation { inputs, .. } => inputs.is_empty(),
            };
            is_root.then_some(id)
        })
    }

    fn record<T: 'static>(&self, kind: NodeKind) -> NodeRecord {
        NodeRecord {
            type_id: TypeId::of::<T>(),
            kind,
            semantics: self
                .value_semantics
                .get(&TypeId::of::<T>())
                .cloned()
                .unwrap_or_default(),
            cache_policy: CachePolicy::default(),
        }
    }

    fn check_node(&self, graph_id: GraphId, id: NodeId, type_id: TypeId) -> Result<(), GraphError> {
        if graph_id != self.id {
            return Err(GraphError::ForeignNode { node: id });
        }
        let record = self
            .nodes
            .get(id)
            .ok_or(GraphError::MissingNode { node: id })?;
        if record.type_id != type_id {
            return Err(GraphError::TypeMismatch { node: id });
        }
        Ok(())
    }

    fn check_inputs(&self, inputs: &[NodeInput]) -> Result<(), GraphError> {
        for input in inputs {
            self.check_node(input.graph_id, input.source, input.type_id)?;
        }
        Ok(())
    }

    // The typed definition preserves output identity across the erasure boundary.
    // Check every edge before insertion so rejection is non-mutating.
    fn insert_computation<T: 'static>(&mut self, definition: ComputationDefinition<T>) -> Node<T> {
        let (inputs, computation) = definition.into_parts();
        self.check_inputs(&inputs)
            .unwrap_or_else(|error| panic!("{error}"));
        let record = self.record::<T>(NodeKind::Computation {
            inputs,
            computation,
        });
        let id = self.nodes.insert(record);
        Node::new(self.id, id)
    }

    fn replace_computation<T: 'static>(
        &mut self,
        target: Node<T>,
        definition: ComputationDefinition<T>,
    ) -> Result<(), GraphError> {
        self.check_node(target.graph_id, target.id, TypeId::of::<T>())?;
        if matches!(self.nodes[target.id].kind, NodeKind::Input) {
            return Err(GraphError::InputNodeReplacement { node: target.id });
        }
        let (inputs, computation) = definition.into_parts();
        self.validate_with_replacement(Some((target.id, &inputs)))?;
        // All fallible checks precede the commit. Existing semantics and policy
        // belong to the node's output contract, not to its replacement closure.
        self.nodes[target.id].kind = NodeKind::Computation {
            inputs,
            computation,
        };
        Ok(())
    }

    fn validate_with_replacement(
        &self,
        replacement: Option<(NodeId, &[NodeInput])>,
    ) -> Result<(), GraphError> {
        self.check_node(self.frame.graph_id, self.frame.id, TypeId::of::<Frame>())?;
        if !matches!(self.nodes[self.frame.id].kind, NodeKind::Input) {
            return Err(GraphError::InvalidFrame);
        }

        let mut pending: SecondaryMap<NodeId, usize> = SecondaryMap::new();
        let mut dependents: SecondaryMap<NodeId, Vec<NodeId>> = SecondaryMap::new();
        let mut ready: VecDeque<_> = self
            .roots()
            .filter(|&id| replacement.is_none_or(|(target, _)| id != target))
            .collect();
        // The target's current root status is excluded above. Seed it only when
        // the candidate definition has zero inputs, before committing the edit.
        if let Some((target, inputs)) = replacement
            && inputs.is_empty()
        {
            ready.push_back(target);
        }
        for (id, record) in &self.nodes {
            let inputs = match replacement {
                Some((target, inputs)) if id == target => inputs,
                _ => match &record.kind {
                    NodeKind::Input => &[],
                    NodeKind::Computation { inputs, .. } => inputs.as_ref(),
                },
            };
            self.check_inputs(inputs)?;
            pending.insert(id, inputs.len());
            for input in inputs {
                // Count and notify per edge, preserving duplicate dependencies.
                dependents
                    .entry(input.source)
                    .unwrap()
                    .or_default()
                    .push(id);
            }
        }

        let mut visited = 0;
        while let Some(id) = ready.pop_front() {
            visited += 1;
            if let Some(children) = dependents.get(id) {
                for &child in children {
                    pending[child] -= 1;
                    if pending[child] == 0 {
                        ready.push_back(child);
                    }
                }
            }
        }
        if visited != self.nodes.len() {
            return Err(GraphError::Cycle);
        }
        Ok(())
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::Graph;
    use crate::computation::{
        Frame, GraphError, Node, NodeId,
        node::{NodeInput, NodeKind},
        value::SharedValue,
    };
    use slotmap::SecondaryMap;
    use std::{
        any::{Any, TypeId},
        collections::HashSet,
        panic::{AssertUnwindSafe, catch_unwind},
        sync::{
            Arc,
            atomic::{AtomicU64, Ordering},
        },
    };

    // Deliberately not Clone: both actor branches must borrow the shared constant.
    #[derive(Debug, PartialEq, Eq)]
    struct ActorStyle {
        width: i64,
        palette: String,
    }

    impl ActorStyle {
        fn warm() -> Self {
            Self {
                width: 128,
                palette: "warm".into(),
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Expression {
        Neutral,
        Smile,
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Actor {
        x: i64,
        width: i64,
        palette: String,
        opacity: u64,
        blink: bool,
        expression: Expression,
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Subtitle {
        text: String,
        canvas_width: i64,
        anchor_x: i64,
    }

    #[derive(Debug, PartialEq, Eq)]
    struct Scene {
        left: Actor,
        right: Actor,
        subtitle: Subtitle,
        watermark: String,
    }

    struct DialogueGraph {
        graph: Graph,
        frame: Node<Frame>,
        canvas_width: Node<i64>,
        origin: Node<i64>,
        dialogue: Node<String>,
        step: Node<i64>,
        gap: Node<i64>,
        style: Node<ActorStyle>,
        watermark: Node<String>,
        phase: Node<u64>,
        left_world_x: Node<i64>,
        right_world_x: Node<i64>,
        left_screen_x: Node<i64>,
        right_screen_x: Node<i64>,
        bubble_x: Node<i64>,
        scene: Node<Scene>,
        execution_order: Vec<NodeId>,
    }

    impl DialogueGraph {
        fn new() -> Self {
            let mut graph = Graph::new();
            graph.register_eq::<i64>();
            graph.register_eq::<u64>();
            graph.register_eq::<Scene>();

            let frame = graph.frame();
            let canvas_width = graph.input::<i64>();
            let origin = graph.input::<i64>();
            let dialogue = graph.input::<String>();
            // Dialogue keeps identity semantics; later String nodes get content equality.
            graph.register_eq::<String>();

            let step = graph.constant(20_i64);
            let gap = graph.constant(160_i64);
            let style = graph.constant(ActorStyle::warm());
            let watermark = graph.map((), || "Gekijyou".to_string());
            let watermark_layout = graph.map((watermark,), |text: &String| format!("©{text}"));

            let phase = graph.map(frame, |f: &Frame| f.0 / 30);
            let camera_x = graph.map(frame, |f: &Frame| (f.0 / 15) as i64);
            let opacity = graph.map(frame, |f: &Frame| (f.0 / 15).min(10));
            let blink = graph.map(frame, |f: &Frame| f.0 % 60 < 3);
            let expression = graph.map(frame, |f: &Frame| {
                if (f.0 / 45) % 2 == 0 {
                    Expression::Neutral
                } else {
                    Expression::Smile
                }
            });
            let reveal_count = graph.map(frame, |f: &Frame| f.0 / 15);

            let left_world_x = graph.map(
                (origin, phase, step),
                |origin: &i64, phase: &u64, step: &i64| origin + *phase as i64 * step,
            );
            let right_world_x = graph.map(
                (canvas_width, left_world_x, gap),
                |width: &i64, left: &i64, gap: &i64| width - left - gap,
            );
            let left_screen_x =
                graph.map((left_world_x, camera_x), |x: &i64, camera: &i64| x - camera);
            let right_screen_x = graph.map((right_world_x, camera_x), |x: &i64, camera: &i64| {
                x - camera
            });
            let bubble_x = graph.map((left_screen_x, left_screen_x), |a: &i64, b: &i64| {
                (a + b) / 2
            });

            let visible_text = graph.map((dialogue, reveal_count), |text: &String, count: &u64| {
                text.chars().take(*count as usize).collect::<String>()
            });
            let subtitle = graph.map(
                (visible_text, canvas_width, bubble_x),
                |text: &String, width: &i64, anchor: &i64| Subtitle {
                    text: text.clone(),
                    canvas_width: *width,
                    anchor_x: *anchor,
                },
            );
            let left_actor = graph.map(
                (left_screen_x, style, blink, opacity),
                |x: &i64, style: &ActorStyle, blink: &bool, opacity: &u64| Actor {
                    x: *x,
                    width: style.width,
                    palette: style.palette.clone(),
                    opacity: *opacity,
                    blink: *blink,
                    expression: Expression::Neutral,
                },
            );
            let right_actor = graph.map(
                (right_screen_x, style, expression, opacity),
                |x: &i64, style: &ActorStyle, expression: &Expression, opacity: &u64| Actor {
                    x: *x,
                    width: style.width,
                    palette: style.palette.clone(),
                    opacity: *opacity,
                    blink: false,
                    expression: *expression,
                },
            );
            let scene = graph.map(
                (left_actor, right_actor, subtitle, watermark_layout),
                |left: &Actor, right: &Actor, subtitle: &Subtitle, watermark: &String| Scene {
                    left: left.clone(),
                    right: right.clone(),
                    subtitle: subtitle.clone(),
                    watermark: watermark.clone(),
                },
            );

            // A fixed scenario schedule, not a general evaluator. Phase precedes step
            // so later edits can make the originally constant step depend on phase.
            let execution_order = vec![
                phase.id,
                camera_x.id,
                opacity.id,
                blink.id,
                expression.id,
                reveal_count.id,
                step.id,
                gap.id,
                style.id,
                watermark.id,
                watermark_layout.id,
                left_world_x.id,
                right_world_x.id,
                left_screen_x.id,
                right_screen_x.id,
                bubble_x.id,
                visible_text.id,
                subtitle.id,
                left_actor.id,
                right_actor.id,
                scene.id,
            ];
            Self {
                graph,
                frame,
                canvas_width,
                origin,
                dialogue,
                step,
                gap,
                style,
                watermark,
                phase,
                left_world_x,
                right_world_x,
                left_screen_x,
                right_screen_x,
                bubble_x,
                scene,
                execution_order,
            }
        }

        fn sample(&self, frame: u64) -> SecondaryMap<NodeId, SharedValue> {
            let mut values: SecondaryMap<NodeId, SharedValue> = SecondaryMap::new();
            values.insert(self.frame.id, Arc::new(Frame(frame)));
            values.insert(self.canvas_width.id, Arc::new(1920_i64));
            values.insert(self.origin.id, Arc::new(100_i64));
            values.insert(self.dialogue.id, Arc::new("今天一起出发吧".to_string()));
            for &id in &self.execution_order {
                execute(&self.graph, id, &mut values);
            }
            values
        }

        fn roots(&self) -> HashSet<NodeId> {
            self.graph.roots().collect()
        }

        fn assert_scene(&self, frame: u64, expected: Scene) {
            assert_eq!(self.graph.validate(), Ok(()));
            let values = self.sample(frame);
            assert_eq!(value(&values, self.scene), &expected);
        }
    }

    // Only invokes the actual stored closure with its actual ordered dependency values.
    // It performs no traversal, scheduling, caching, or independent scene calculations.
    fn execute(graph: &Graph, id: NodeId, values: &mut SecondaryMap<NodeId, SharedValue>) {
        let record = &graph.nodes[id];
        let NodeKind::Computation {
            inputs,
            computation,
        } = &record.kind
        else {
            panic!("The fixed execution schedule must only contain computations");
        };
        let args: Vec<&dyn Any> = inputs
            .iter()
            .map(|input| {
                values
                    .get(input.source)
                    .expect("Scenario dependency must already be available")
                    .as_ref() as &dyn Any
            })
            .collect();
        let output = computation(&args);
        assert_eq!(output.as_ref().type_id(), record.type_id);
        values.insert(id, output);
    }

    fn value<T: 'static>(values: &SecondaryMap<NodeId, SharedValue>, node: Node<T>) -> &T {
        values[node.id]
            .downcast_ref::<T>()
            .expect("Scenario output type mismatch")
    }

    fn expected_scene(
        left_x: i64,
        right_x: i64,
        opacity: u64,
        blink: bool,
        expression: Expression,
        text: &str,
        anchor_x: i64,
    ) -> Scene {
        Scene {
            left: Actor {
                x: left_x,
                width: 128,
                palette: "warm".into(),
                opacity,
                blink,
                expression: Expression::Neutral,
            },
            right: Actor {
                x: right_x,
                width: 128,
                palette: "warm".into(),
                opacity,
                blink: false,
                expression,
            },
            subtitle: Subtitle {
                text: text.into(),
                canvas_width: 1920,
                anchor_x,
            },
            watermark: "©Gekijyou".into(),
        }
    }

    #[test]
    fn dialogue_graph_equality_respects_content_identity_and_registration_time() {
        let mut fixture = DialogueGraph::new();
        let first = fixture.sample(75);
        let repeated = fixture.sample(75);
        let later = fixture.sample(120);
        for id in [
            fixture.frame.id,
            fixture.phase.id,
            fixture.scene.id,
            fixture.watermark.id,
        ] {
            let semantics = &fixture.graph.nodes[id].semantics;
            assert!(!Arc::ptr_eq(&first[id], &repeated[id]));
            assert!(semantics.equivalent(&first[id], &repeated[id]));
        }
        for id in [fixture.frame.id, fixture.phase.id, fixture.scene.id] {
            assert!(
                !fixture.graph.nodes[id]
                    .semantics
                    .equivalent(&first[id], &later[id])
            );
        }

        // The input was created before String registration; later registration cannot
        // reinterpret it, even though these two samples supplied identical text.
        let dialogue_semantics = &fixture.graph.nodes[fixture.dialogue.id].semantics;
        assert!(
            dialogue_semantics.equivalent(&first[fixture.dialogue.id], &first[fixture.dialogue.id])
        );
        assert!(
            !dialogue_semantics
                .equivalent(&first[fixture.dialogue.id], &repeated[fixture.dialogue.id])
        );

        // Both samples retain the same non-Clone constant object.
        assert!(Arc::ptr_eq(
            &first[fixture.style.id],
            &later[fixture.style.id]
        ));
        fixture.graph.register_eq::<ActorStyle>();
        let new_style = fixture.graph.constant(ActorStyle::warm());
        let mut values = fixture.sample(75);
        execute(&fixture.graph, new_style.id, &mut values);
        assert!(!Arc::ptr_eq(
            &values[fixture.style.id],
            &values[new_style.id]
        ));
        assert!(
            !fixture.graph.nodes[fixture.style.id]
                .semantics
                .equivalent(&values[fixture.style.id], &values[new_style.id])
        );
        assert!(
            fixture.graph.nodes[new_style.id]
                .semantics
                .equivalent(&values[fixture.style.id], &values[new_style.id])
        );
        // Replacing the old style after registration must retain identity semantics,
        // even though the new closure allocates identical content on each invocation.
        fixture
            .graph
            .replace(fixture.style, (), ActorStyle::warm)
            .unwrap();
        let replaced = fixture.sample(75);
        let replaced_again = fixture.sample(75);
        assert!(!Arc::ptr_eq(
            &replaced[fixture.style.id],
            &replaced_again[fixture.style.id]
        ));
        assert!(!fixture.graph.nodes[fixture.style.id].semantics.equivalent(
            &replaced[fixture.style.id],
            &replaced_again[fixture.style.id]
        ));
        fixture.assert_scene(
            75,
            expected_scene(135, 1615, 5, false, Expression::Smile, "今天一起出", 135),
        );
    }

    #[test]
    fn dialogue_graph_adds_an_overlay_without_changing_existing_scene() {
        let mut fixture = DialogueGraph::new();
        fixture.assert_scene(
            75,
            expected_scene(135, 1615, 5, false, Expression::Smile, "今天一起出", 135),
        );
        fixture.assert_scene(
            120,
            expected_scene(
                172,
                1572,
                8,
                true,
                Expression::Neutral,
                "今天一起出发吧",
                172,
            ),
        );
        let original_roots = fixture.roots();
        assert_eq!(
            original_roots,
            HashSet::from([
                fixture.frame.id,
                fixture.canvas_width.id,
                fixture.origin.id,
                fixture.dialogue.id,
                fixture.step.id,
                fixture.gap.id,
                fixture.style.id,
                fixture.watermark.id,
            ])
        );
        let original_count = fixture.graph.nodes.len();

        let enabled = fixture.graph.input::<bool>();
        let margin = fixture.graph.constant(10_i64);
        let calls = Arc::new(AtomicU64::new(0));
        let label_calls = Arc::clone(&calls);
        let label = fixture.graph.map((), move || {
            label_calls.fetch_add(1, Ordering::Relaxed);
            "speaker".to_string()
        });
        // These new roots are valid even before any consumer is connected.
        assert_eq!(fixture.graph.validate(), Ok(()));
        let overlay = fixture.graph.map(
            (fixture.scene, margin, enabled, label),
            |scene: &Scene, margin: &i64, enabled: &bool, label: &String| {
                if *enabled {
                    format!(
                        "{label}:{}@{}",
                        scene.subtitle.text,
                        scene.subtitle.anchor_x + margin
                    )
                } else {
                    String::new()
                }
            },
        );
        assert_eq!(fixture.graph.validate(), Ok(()));
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert_eq!(fixture.graph.nodes.len(), original_count + 4);
        let mut expected_roots = original_roots;
        expected_roots.extend([enabled.id, margin.id, label.id]);
        assert_eq!(fixture.roots(), expected_roots);

        let mut values = fixture.sample(75);
        values.insert(enabled.id, Arc::new(true));
        for id in [margin.id, label.id, overlay.id] {
            execute(&fixture.graph, id, &mut values);
        }
        assert_eq!(value(&values, overlay), "speaker:今天一起出@145");
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(
            value(&values, fixture.scene),
            &expected_scene(135, 1615, 5, false, Expression::Smile, "今天一起出", 135)
        );
    }

    #[test]
    fn dialogue_graph_replaces_time_branches_and_reconnects_existing_handles() {
        let mut fixture = DialogueGraph::new();
        let original_count = fixture.graph.nodes.len();
        let mut expected_roots = fixture.roots();
        let phase_semantics = fixture.graph.nodes[fixture.phase.id].semantics.clone();
        let before = fixture.sample(75);

        fixture.graph.replace(fixture.phase, (), || 2_u64).unwrap();
        expected_roots.insert(fixture.phase.id);
        assert_eq!(fixture.roots(), expected_roots);
        fixture.assert_scene(
            75,
            expected_scene(135, 1615, 5, false, Expression::Smile, "今天一起出", 135),
        );
        fixture.assert_scene(
            120,
            expected_scene(
                132,
                1612,
                8,
                true,
                Expression::Neutral,
                "今天一起出发吧",
                132,
            ),
        );
        let frozen = fixture.sample(75);
        assert!(phase_semantics.equivalent(&before[fixture.phase.id], &frozen[fixture.phase.id]));
        assert!(
            fixture.graph.nodes[fixture.phase.id]
                .semantics
                .equivalent(&before[fixture.phase.id], &frozen[fixture.phase.id])
        );

        fixture
            .graph
            .replace(fixture.phase, fixture.frame, |f: &Frame| f.0 / 15)
            .unwrap();
        expected_roots.remove(&fixture.phase.id);
        assert_eq!(fixture.roots(), expected_roots);
        fixture.assert_scene(
            75,
            expected_scene(195, 1555, 5, false, Expression::Smile, "今天一起出", 195),
        );

        // Step was inserted before phase, so this is a legal forward reference after editing.
        fixture
            .graph
            .replace(fixture.step, fixture.phase, |p: &u64| 20 + *p as i64)
            .unwrap();
        expected_roots.remove(&fixture.step.id);
        assert_eq!(fixture.roots(), expected_roots);
        fixture.assert_scene(
            75,
            expected_scene(220, 1530, 5, false, Expression::Smile, "今天一起出", 220),
        );

        fixture
            .graph
            .replace(fixture.gap, fixture.frame, |f: &Frame| {
                160 + (f.0 / 30) as i64 * 10
            })
            .unwrap();
        expected_roots.remove(&fixture.gap.id);
        assert_eq!(fixture.roots(), expected_roots);
        fixture.assert_scene(
            75,
            expected_scene(220, 1510, 5, false, Expression::Smile, "今天一起出", 220),
        );

        fixture
            .graph
            .replace(
                fixture.bubble_x,
                (fixture.left_screen_x, fixture.right_screen_x),
                |a: &i64, b: &i64| (a + b) / 2,
            )
            .unwrap();
        assert_eq!(fixture.roots(), expected_roots);
        assert_eq!(fixture.graph.nodes.len(), original_count);
        fixture.assert_scene(
            75,
            expected_scene(220, 1510, 5, false, Expression::Smile, "今天一起出", 865),
        );

        let error = fixture
            .graph
            .replace(fixture.canvas_width, (), || 1280_i64)
            .unwrap_err();
        assert_eq!(
            error,
            GraphError::InputNodeReplacement {
                node: fixture.canvas_width.id
            }
        );
        assert_eq!(fixture.roots(), expected_roots);
        fixture.assert_scene(
            75,
            expected_scene(220, 1510, 5, false, Expression::Smile, "今天一起出", 865),
        );
    }

    #[test]
    fn dialogue_graph_rejects_foreign_handles_even_when_storage_keys_match() {
        let mut fixture = DialogueGraph::new();
        let foreign = DialogueGraph::new();
        assert_eq!(fixture.frame.id, foreign.frame.id);
        assert_eq!(fixture.phase.id, foreign.phase.id);
        assert_ne!(fixture.graph.id, foreign.graph.id);
        let original_count = fixture.graph.nodes.len();
        let original_roots = fixture.roots();

        let insertion = catch_unwind(AssertUnwindSafe(|| {
            fixture.graph.map(foreign.frame, |f: &Frame| f.0 / 30);
        }));
        assert!(insertion.is_err());
        let insertion = catch_unwind(AssertUnwindSafe(|| {
            fixture
                .graph
                .map((fixture.phase, foreign.step), |p: &u64, s: &i64| {
                    *p as i64 * s
                });
        }));
        assert!(insertion.is_err());

        assert_eq!(
            fixture.graph.replace(foreign.phase, (), || 2_u64),
            Err(GraphError::ForeignNode {
                node: foreign.phase.id
            })
        );
        assert_eq!(
            fixture
                .graph
                .replace(fixture.phase, foreign.frame, |f: &Frame| f.0 / 15),
            Err(GraphError::ForeignNode {
                node: foreign.frame.id
            })
        );
        assert_eq!(
            fixture.graph.replace(
                fixture.left_world_x,
                (fixture.origin, foreign.phase, fixture.step),
                |o: &i64, p: &u64, s: &i64| o + *p as i64 * s
            ),
            Err(GraphError::ForeignNode {
                node: foreign.phase.id
            })
        );
        assert_eq!(fixture.graph.nodes.len(), original_count);
        assert_eq!(fixture.roots(), original_roots);
        fixture.assert_scene(
            75,
            expected_scene(135, 1615, 5, false, Expression::Smile, "今天一起出", 135),
        );
    }

    #[test]
    fn dialogue_graph_rejects_cycles_without_losing_the_last_valid_scene() {
        let mut fixture = DialogueGraph::new();
        let original_count = fixture.graph.nodes.len();
        let original_roots = fixture.roots();
        assert_eq!(
            fixture
                .graph
                .replace(fixture.left_world_x, fixture.left_world_x, |x: &i64| x + 1),
            Err(GraphError::Cycle)
        );
        assert_eq!(
            fixture.graph.replace(
                fixture.left_world_x,
                (fixture.right_world_x, fixture.gap),
                |right: &i64, gap: &i64| right - gap
            ),
            Err(GraphError::Cycle)
        );
        assert_eq!(fixture.graph.nodes.len(), original_count);
        assert_eq!(fixture.roots(), original_roots);
        fixture.assert_scene(
            75,
            expected_scene(135, 1615, 5, false, Expression::Smile, "今天一起出", 135),
        );

        fixture.graph.replace(fixture.phase, (), || 2_u64).unwrap();
        let frozen_roots = fixture.roots();
        // The candidate must stop being a root; seeding the old root could hide this cycle.
        assert_eq!(
            fixture
                .graph
                .replace(fixture.phase, fixture.phase, |p: &u64| p + 1),
            Err(GraphError::Cycle)
        );
        assert_eq!(fixture.roots(), frozen_roots);
        fixture.assert_scene(
            120,
            expected_scene(
                132,
                1612,
                8,
                true,
                Expression::Neutral,
                "今天一起出发吧",
                132,
            ),
        );

        // Public edits cannot create an invalid graph. Inject a cycle into an unused
        // archived-title branch to exercise validate() independently of replace().
        let archived_title = fixture.graph.map((), || "第一幕".to_string());
        let archived_layout = fixture
            .graph
            .map(archived_title, |title: &String| title.clone());
        assert_eq!(fixture.graph.validate(), Ok(()));
        let NodeKind::Computation { inputs, .. } = &mut fixture.graph.nodes[archived_title.id].kind
        else {
            unreachable!()
        };
        *inputs = Box::new([NodeInput {
            graph_id: fixture.graph.id,
            source: archived_layout.id,
            type_id: TypeId::of::<String>(),
        }]);
        assert_eq!(fixture.graph.validate(), Err(GraphError::Cycle));
    }
}
