use super::*;
use crate::Evaluator;

#[test]
fn result_reads_check_index_type_and_batch_before_access() {
    let mut graph = Graph::new();
    let target = graph.constant(7_u64);
    let mut batch = RequestBatch::new(&graph);
    let context = batch.add_context(Context::new(&graph)).unwrap();
    let request = batch.request(context, target).unwrap();
    let results = Evaluator::new(&graph)
        .evaluate_batch(&graph, batch)
        .unwrap();
    let wrong_type: RequestId<String> = RequestId {
        batch_id: request.batch_id,
        index: request.index,
        _marker: PhantomData,
    };
    assert_eq!(
        results.get(wrong_type),
        Err(EvalError::ResultTypeMismatch { index: 0 })
    );
    let invalid = RequestId::<u64> {
        index: 1,
        ..request
    };
    assert_eq!(
        results.get(invalid),
        Err(EvalError::InvalidRequest { index: 1 })
    );
    let foreign = RequestId::<u64> {
        batch_id: BatchId::new(),
        ..invalid
    };
    assert_eq!(results.get(foreign), Err(EvalError::ForeignRequest));
}

#[test]
fn invalid_context_index_is_rejected_without_adding_request() {
    let graph = Graph::new();
    let mut batch = RequestBatch::new(&graph);
    let valid = batch.add_context(Context::new(&graph)).unwrap();
    let invalid = ContextId { index: 1, ..valid };
    assert_eq!(
        batch.request(invalid, graph.frame()),
        Err(EvalError::InvalidContext { index: 1 })
    );
    assert!(batch.requests.is_empty());
    let request = batch.request(valid, graph.frame()).unwrap();
    assert_eq!(request.index, 0);
}
