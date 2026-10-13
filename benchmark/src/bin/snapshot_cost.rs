//! Edit and stable-snapshot costs measured separately. Large payloads and closures
//! are shared; input-only edits must not clone graph metadata. Run with --release.
use gekijyou_computation::{FrameIndex, GraphState};
use std::{hint::black_box, sync::Arc, time::Instant};
fn main() {
    println!("sample\tnodes\tpath\titerations\tns_per_operation");
    for nodes in [16, 1690] {
        let mut state = GraphState::new();
        let input = state
            .edit(|e| {
                let i = e.input::<Vec<u8>>();
                e.set_shared(i, Arc::new(vec![0; 1024 * 1024]))?;
                for _ in 2..nodes {
                    e.map(e.frame(), |f: &FrameIndex| f.0)?;
                }
                Ok(i)
            })
            .unwrap();
        let payload = state.get_shared(input).unwrap().unwrap();
        for sample in 0..9 {
            let start = Instant::now();
            for _ in 0..10000 {
                black_box(state.snapshot());
            }
            println!(
                "{sample}\t{nodes}\tsnapshot\t10000\t{}",
                start.elapsed().as_nanos() / 10000
            );
            let a = payload.clone();
            let b = Arc::new(vec![1; 1024 * 1024]);
            let start = Instant::now();
            for n in 0..1000 {
                state
                    .edit(|e| e.set_shared(input, if n % 2 == 0 { a.clone() } else { b.clone() }))
                    .unwrap();
                black_box(state.snapshot());
            }
            println!(
                "{sample}\t{nodes}\tinput_edit\t1000\t{}",
                start.elapsed().as_nanos() / 1000
            );
            let start = Instant::now();
            for _ in 0..100 {
                // Keep an old snapshot alive to force graph metadata COW.
                let old = state.snapshot();
                state
                    .edit(|e| {
                        let temporary = e.constant(1_u64);
                        e.remove(temporary)
                    })
                    .unwrap();
                black_box(old);
            }
            println!(
                "{sample}\t{nodes}\tdefinition_edit\t100\t{}",
                start.elapsed().as_nanos() / 100
            );
        }
    }
}
