// K197: jix-py's GIL decision in `PyArray::numpy` (jix-py/src/array.rs:631-639) keeps a read
// attached when the spec says the storage reads straight from memory (ZeroCopy or Fusable) and
// the read is cheap. Every jix-py array is an `ArrayAny`, and `AnyMeta::new`
// (jix/src/storage/any.rs:105-108) demotes Fusable to SinglePass, so the Fusable arm never
// matches: a tiny fused op over numpy memory (`jix.asarray(x) + 1`) always detaches.
//
// The test mirrors jix-py's construction (numpy -> Plain -> into_any; op over into_typed leaves;
// result -> into_any) and its `should_detach` expression verbatim. It asserts the intended
// outcome (stay attached), so it fails while the bug exists.
use jix::__private::ReadTiling;
use jix::{Array, ArrayAny, ArrayStorage};
use ndarray::array;

/// Copy of jix-py/src/array.rs:631-639.
fn should_detach(arr: &ArrayAny) -> bool {
    let nitems = arr.shape().iter().product::<u64>() as usize;
    let itemsize = arr.dtype().itemsize() as usize;
    let spec = arr.storage().spec();
    let from_memory = matches!(
        spec.flags().read_tiling(),
        ReadTiling::ZeroCopy | ReadTiling::Fusable
    );
    let read_cost = (nitems * itemsize) as f32 * spec.element_cost().max(1.0);
    !from_memory || read_cost > 4096.0
}

/// What jix-py does for a numpy input (`plain_from_numpy`, jix-py/src/ops/common/operand.rs:520).
fn py_input() -> ArrayAny {
    Array::plain_ndarray(array![1.0f32, 2.0, 3.0, 4.0]).unwrap().into_any()
}

#[test]
fn small_fused_read_over_memory_stays_attached() {
    // Control: the plain input itself (ZeroCopy) stays attached.
    let plain = py_input();
    assert!(!should_detach(&plain), "control: plain input detaches");

    // The same op, typed: the core spec says Fusable.
    let typed = py_input().into_typed::<f32>().unwrap() + py_input().into_typed::<f32>().unwrap();
    let t_typed = typed.storage().spec().flags().read_tiling();

    // What jix-py holds after `jix.asarray(x) + jix.asarray(x)`.
    let op = typed.into_any();
    let t_any = op.storage().spec().flags().read_tiling();
    println!(
        "typed op: {t_typed:?}, as jix-py holds it (ArrayAny): {t_any:?}, \
         element_cost {}, should_detach {}",
        op.storage().spec().element_cost(),
        should_detach(&op)
    );
    assert_eq!(t_typed, ReadTiling::Fusable);
    assert!(
        !should_detach(&op),
        "a 16-byte fused read over numpy memory detaches: jix-py sees {t_any:?}, not Fusable"
    );
}
