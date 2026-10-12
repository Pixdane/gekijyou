mod fixture;

use fixture::{ActorStyle, DialogueGraph, Expression, Scene, expected_scene};
use gekijyou::computation::{FrameIndex, GraphError};
use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

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
    let mut context = fixture.context(75);
    context.set(&fixture.graph, enabled, true).unwrap();
    let (results, request) = fixture.evaluate(context, overlay);
    assert_eq!(results.get(request).unwrap(), "speaker:今天一起出@145");
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    fixture.assert_scene(
        75,
        expected_scene(135, 1615, 5, false, Expression::Smile, "今天一起出", 135),
    );
}

#[test]
fn dialogue_graph_replaces_time_branches_and_reconnects_existing_handles() {
    let mut fixture = DialogueGraph::new();

    fixture.graph.replace(fixture.phase, (), || 2_u64).unwrap();
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
    fixture
        .graph
        .replace(fixture.phase, fixture.frame, |f: &FrameIndex| f.0 / 15)
        .unwrap();
    fixture.assert_scene(
        75,
        expected_scene(195, 1555, 5, false, Expression::Smile, "今天一起出", 195),
    );

    // Step was inserted before phase, so this is a legal forward reference after editing.
    fixture
        .graph
        .replace(fixture.step, fixture.phase, |p: &u64| 20 + *p as i64)
        .unwrap();
    fixture.assert_scene(
        75,
        expected_scene(220, 1530, 5, false, Expression::Smile, "今天一起出", 220),
    );

    fixture
        .graph
        .replace(fixture.gap, fixture.frame, |f: &FrameIndex| {
            160 + (f.0 / 30) as i64 * 10
        })
        .unwrap();
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
    fixture.assert_scene(
        75,
        expected_scene(220, 1510, 5, false, Expression::Smile, "今天一起出", 865),
    );

    let error = fixture
        .graph
        .replace(fixture.canvas_width, (), || 1280_i64)
        .unwrap_err();
    assert!(matches!(error, GraphError::InputNodeReplacement { .. }));
    fixture.assert_scene(
        75,
        expected_scene(220, 1510, 5, false, Expression::Smile, "今天一起出", 865),
    );
}

#[test]
fn dialogue_graph_rejects_foreign_handles_without_changing_the_scene() {
    let mut fixture = DialogueGraph::new();
    let foreign = DialogueGraph::new();
    let insertion = catch_unwind(AssertUnwindSafe(|| {
        fixture.graph.map(foreign.frame, |f: &FrameIndex| f.0 / 30);
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
    assert!(matches!(
        fixture.graph.replace(foreign.phase, (), || 2_u64),
        Err(GraphError::ForeignNode { .. })
    ));
    assert!(matches!(
        fixture
            .graph
            .replace(fixture.phase, foreign.frame, |f: &FrameIndex| f.0 / 15),
        Err(GraphError::ForeignNode { .. })
    ));
    assert!(matches!(
        fixture.graph.replace(
            fixture.left_world_x,
            (fixture.origin, foreign.phase, fixture.step),
            |o: &i64, p: &u64, s: &i64| o + *p as i64 * s,
        ),
        Err(GraphError::ForeignNode { .. })
    ));
    fixture.assert_scene(
        75,
        expected_scene(135, 1615, 5, false, Expression::Smile, "今天一起出", 135),
    );
}

#[test]
fn dialogue_graph_rejects_cycles_without_losing_the_last_valid_scene() {
    let mut fixture = DialogueGraph::new();
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
    fixture.assert_scene(
        75,
        expected_scene(135, 1615, 5, false, Expression::Smile, "今天一起出", 135),
    );

    fixture.graph.replace(fixture.phase, (), || 2_u64).unwrap();
    // The candidate must stop being a root; seeding the old root could hide this cycle.
    assert_eq!(
        fixture
            .graph
            .replace(fixture.phase, fixture.phase, |p: &u64| p + 1),
        Err(GraphError::Cycle)
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
}

#[test]
fn non_clone_constant_shares_storage_across_batches_and_replacement_publishes_new_values() {
    let mut fixture = DialogueGraph::new();
    let (first, first_request) = fixture.evaluate(fixture.context(75), fixture.style);
    let (second, second_request) = fixture.evaluate(fixture.context(120), fixture.style);
    assert!(std::ptr::eq(
        first.get(first_request).unwrap(),
        second.get(second_request).unwrap()
    ));
    fixture
        .graph
        .replace(fixture.style, (), ActorStyle::warm)
        .unwrap();
    let (third, third_request) = fixture.evaluate(fixture.context(75), fixture.style);
    let (fourth, fourth_request) = fixture.evaluate(fixture.context(75), fixture.style);
    assert!(!std::ptr::eq(
        third.get(third_request).unwrap(),
        fourth.get(fourth_request).unwrap()
    ));
    assert_eq!(
        third.get(third_request).unwrap(),
        first.get(first_request).unwrap()
    );
    fixture.assert_scene(
        75,
        expected_scene(135, 1615, 5, false, Expression::Smile, "今天一起出", 135),
    );
}
