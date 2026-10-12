use gekijyou::computation::{
    BatchResults, Context, Evaluator, FrameIndex, Graph, Node, RequestBatch, RequestId,
};

// Deliberately not Clone: both actor branches must borrow the shared constant.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct ActorStyle {
    width: i64,
    palette: String,
}

impl ActorStyle {
    pub(super) fn warm() -> Self {
        Self {
            width: 128,
            palette: "warm".into(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Expression {
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
pub(super) struct Subtitle {
    pub(super) text: String,
    canvas_width: i64,
    pub(super) anchor_x: i64,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct Scene {
    left: Actor,
    right: Actor,
    pub(super) subtitle: Subtitle,
    watermark: String,
}

pub(super) struct DialogueGraph {
    pub(super) graph: Graph,
    pub(super) frame: Node<FrameIndex>,
    pub(super) canvas_width: Node<i64>,
    pub(super) origin: Node<i64>,
    pub(super) dialogue: Node<String>,
    pub(super) step: Node<i64>,
    pub(super) gap: Node<i64>,
    pub(super) style: Node<ActorStyle>,
    pub(super) phase: Node<u64>,
    pub(super) left_world_x: Node<i64>,
    pub(super) right_world_x: Node<i64>,
    pub(super) left_screen_x: Node<i64>,
    pub(super) right_screen_x: Node<i64>,
    pub(super) bubble_x: Node<i64>,
    pub(super) scene: Node<Scene>,
}

impl DialogueGraph {
    pub(super) fn new() -> Self {
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
        let watermark = graph.map((watermark,), |text: &String| format!("©{text}"));

        let phase = graph.map(frame, |f: &FrameIndex| f.0 / 30);
        let camera_x = graph.map(frame, |f: &FrameIndex| (f.0 / 15) as i64);
        let opacity = graph.map(frame, |f: &FrameIndex| (f.0 / 15).min(10));
        let blink = graph.map(frame, |f: &FrameIndex| f.0 % 60 < 3);
        let expression = graph.map(frame, |f: &FrameIndex| {
            if (f.0 / 45).is_multiple_of(2) {
                Expression::Neutral
            } else {
                Expression::Smile
            }
        });
        let reveal_count = graph.map(frame, |f: &FrameIndex| f.0 / 15);

        let left_world_x = graph.map(
            (origin, phase, step),
            |origin: &i64, phase: &u64, step: &i64| origin + *phase as i64 * step,
        );
        let right_world_x = graph.map(
            (canvas_width, left_world_x, gap),
            |width: &i64, left: &i64, gap: &i64| width - left - gap,
        );
        let left_screen_x = graph.map((left_world_x, camera_x), |x: &i64, camera: &i64| x - camera);
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
            (left_actor, right_actor, subtitle, watermark),
            |left: &Actor, right: &Actor, subtitle: &Subtitle, watermark: &String| Scene {
                left: left.clone(),
                right: right.clone(),
                subtitle: subtitle.clone(),
                watermark: watermark.clone(),
            },
        );

        Self {
            graph,
            frame,
            canvas_width,
            origin,
            dialogue,
            step,
            gap,
            style,
            phase,
            left_world_x,
            right_world_x,
            left_screen_x,
            right_screen_x,
            bubble_x,
            scene,
        }
    }

    pub(super) fn context(&self, frame: u64) -> Context {
        let mut context = Context::new(&self.graph);
        context
            .set(&self.graph, self.frame, FrameIndex(frame))
            .unwrap();
        context
            .set(&self.graph, self.canvas_width, 1920_i64)
            .unwrap();
        context.set(&self.graph, self.origin, 100_i64).unwrap();
        context
            .set(&self.graph, self.dialogue, String::from("今天一起出发吧"))
            .unwrap();
        context
    }

    pub(super) fn evaluate<T: 'static>(
        &self,
        context: Context,
        node: Node<T>,
    ) -> (BatchResults, RequestId<T>) {
        let mut batch = RequestBatch::new(&self.graph);
        let context = batch.add_context(context).unwrap();
        let request = batch.request(context, node).unwrap();
        let results = Evaluator::new(&self.graph)
            .evaluate_batch(&self.graph, batch)
            .unwrap();
        (results, request)
    }

    pub(super) fn assert_scene(&self, frame: u64, expected: Scene) {
        assert_eq!(self.graph.validate(), Ok(()));
        let (results, request) = self.evaluate(self.context(frame), self.scene);
        assert_eq!(results.get(request).unwrap(), &expected);
    }
}

pub(super) fn expected_scene(
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
