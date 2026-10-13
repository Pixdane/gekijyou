//! Sample a simple animation at two frame indices.
//! Run with `cargo run --example evaluate_scene`.
use gekijyou_computation::p35::{Context, EvalError, Evaluator, FrameIndex, Graph, RequestBatch};

#[derive(Debug)]
struct Scene {
    x: u64,
    opacity: f64,
}

fn main() -> Result<(), EvalError> {
    let mut graph = Graph::new();
    let position = graph.map(graph.frame(), |frame: &FrameIndex| frame.0 * 2);
    let opacity = graph.map(graph.frame(), |frame: &FrameIndex| {
        (frame.0 as f64 / 30.0).min(1.0)
    });
    let scene = graph.map((position, opacity), |x: &u64, opacity: &f64| Scene {
        x: *x,
        opacity: *opacity,
    });

    let mut batch = RequestBatch::new(&graph);
    let mut requests = Vec::new();
    for frame in [12, 30] {
        let mut context = Context::new(&graph);
        context.set(&graph, graph.frame(), FrameIndex(frame))?;
        let context = batch.add_context(context)?;
        requests.push((frame, batch.request(context, scene)?));
    }
    let results = Evaluator::new(&graph).evaluate_batch(&graph, batch)?;
    for (frame, request) in requests {
        let scene = results.get(request)?;
        println!("frame {frame}: x={}, opacity={:.2}", scene.x, scene.opacity);
    }
    Ok(())
}
