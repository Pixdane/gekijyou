//! Sample independent animation frames from a single immutable input snapshot.
use gekijyou_computation::{EvalError, Evaluator, FrameIndex, GraphState, RequestBatch};
#[derive(Debug)]
struct Scene {
    x: u64,
    opacity: f64,
}
fn main() -> Result<(), EvalError> {
    let mut state = GraphState::new();
    let scene = state.edit(|e| {
        let position = e.map(e.frame(), |f: &FrameIndex| f.0 * 2)?;
        let opacity = e.map(e.frame(), |f: &FrameIndex| (f.0 as f64 / 30.0).min(1.0))?;
        e.map((position, opacity), |x: &u64, opacity: &f64| Scene {
            x: *x,
            opacity: *opacity,
        })
    })?;
    let mut batch = RequestBatch::new();
    let requests: Vec<_> = [12, 30, 12]
        .map(|f| (f, batch.at(FrameIndex(f)).request(scene)))
        .into();
    let results = Evaluator::new(&state)
        .bind(state.snapshot())?
        .evaluate(batch)?;
    for (frame, id) in requests {
        let scene = results.get(id)?;
        println!("frame {frame}: x={}, opacity={:.2}", scene.x, scene.opacity);
    }
    Ok(())
}
