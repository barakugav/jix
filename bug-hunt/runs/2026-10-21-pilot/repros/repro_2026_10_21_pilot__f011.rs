// F011: ArrayVec::extend_from_iter drops each ZST element while also counting it in len.
//
// `ArrayVec` lives in a pub(crate) module, but the public `Dimension` trait exposes it:
// `<DimDyn as Dimension>::Vec<T> = DimArray<T> = ArrayVec<T, 8>`, and the public
// `Dimension::vec` hands out a value of that type for any `T`. Its std trait impls
// (`Extend`, `FromIterator`, `Clone::clone_from`, `IntoIterator` -> `IntoIter: Clone`)
// are then callable from safe code outside the crate.
use std::sync::atomic::{AtomicUsize, Ordering};

use jix::{DimDyn, Dimension};

static DROPS_EXTEND: AtomicUsize = AtomicUsize::new(0);
static DROPS_COLLECT: AtomicUsize = AtomicUsize::new(0);
static DROPS_CLONE_FROM: AtomicUsize = AtomicUsize::new(0);
static DROPS_ITER_CLONE: AtomicUsize = AtomicUsize::new(0);
static DROPS_CLONE: AtomicUsize = AtomicUsize::new(0);

macro_rules! zst {
    ($name:ident, $counter:ident) => {
        struct $name;
        impl Clone for $name {
            fn clone(&self) -> Self {
                $name
            }
        }
        impl Drop for $name {
            fn drop(&mut self) {
                $counter.fetch_add(1, Ordering::SeqCst);
            }
        }
    };
}
zst!(ZExtend, DROPS_EXTEND);
zst!(ZCollect, DROPS_COLLECT);
zst!(ZCloneFrom, DROPS_CLONE_FROM);
zst!(ZIterClone, DROPS_ITER_CLONE);
zst!(ZClone, DROPS_CLONE);

#[test]
fn extend_double_drops_zst() {
    // Empty container obtained through the public Dimension trait.
    let mut v = <DimDyn as Dimension>::vec(0, |_| ZExtend);
    v.extend([ZExtend, ZExtend, ZExtend]);
    let after_extend = DROPS_EXTEND.load(Ordering::SeqCst);
    assert_eq!(v.as_ref().len(), 3);
    drop(v);
    let total = DROPS_EXTEND.load(Ordering::SeqCst);
    println!("extend: 3 values moved in; drops during extend = {after_extend}, total = {total}");
    assert_eq!(total, 3, "each of the 3 ZST values must be dropped exactly once");
}

#[test]
#[allow(unused_assignments)] // the first value only pins the (unnameable) target type of collect()
fn collect_double_drops_zst() {
    let mut v = <DimDyn as Dimension>::vec(0, |_| ZCollect);
    v = (0..3).map(|_| ZCollect).collect();
    let after_collect = DROPS_COLLECT.load(Ordering::SeqCst);
    drop(v);
    let total = DROPS_COLLECT.load(Ordering::SeqCst);
    println!("collect: 3 values created; drops during collect = {after_collect}, total = {total}");
    assert_eq!(total, 3, "each of the 3 ZST values must be dropped exactly once");
}

#[test]
fn clone_from_double_drops_zst() {
    let src = <DimDyn as Dimension>::vec(2, |_| ZCloneFrom); // 2 created
    let mut dst = <DimDyn as Dimension>::vec(0, |_| ZCloneFrom);
    dst.clone_from(&src); // 2 clones via extend_from_slice
    drop(dst);
    drop(src);
    let total = DROPS_CLONE_FROM.load(Ordering::SeqCst);
    println!("clone_from: 2 created + 2 clones; total drops = {total} (expected 4)");
    assert_eq!(total, 4);
}

#[test]
fn into_iter_clone_double_drops_zst() {
    let v = <DimDyn as Dimension>::vec(2, |_| ZIterClone); // 2 created
    let mut it = v.into_iter();
    let first = it.next(); // moved out, 1 left in `it`
    let it2 = it.clone(); // 1 clone via extend_from_slice
    drop(it2);
    drop(it);
    drop(first);
    let total = DROPS_ITER_CLONE.load(Ordering::SeqCst);
    println!("IntoIter::clone: 2 created + 1 clone; total drops = {total} (expected 3)");
    assert_eq!(total, 3);
}

#[test]
fn clone_is_correct_control() {
    // Control: ArrayVec::clone forgets the ZST explicitly and is correct.
    let v = <DimDyn as Dimension>::vec(2, |_| ZClone); // 2 created
    let w = <_ as Clone>::clone(&v); // 2 clones
    drop(w);
    drop(v);
    let total = DROPS_CLONE.load(Ordering::SeqCst);
    println!("clone (control): 2 created + 2 clones; total drops = {total} (expected 4)");
    assert_eq!(total, 4);
}

// Consequence in otherwise-sound safe user code: a ZST handle that uniquely owns a heap
// allocation (the classic "token" pattern; sound because the only constructor hands out one
// handle per allocation). The double drop turns into a double free. Ignored by default because
// glibc aborts the whole test process; run with `-- --ignored double_free`.
mod token {
    use std::sync::atomic::{AtomicPtr, Ordering};
    static SLOT: AtomicPtr<u64> = AtomicPtr::new(std::ptr::null_mut());
    pub struct Token(());
    impl Token {
        pub fn new() -> Option<Token> {
            let p = Box::into_raw(Box::new(42u64));
            if SLOT.compare_exchange(std::ptr::null_mut(), p, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                Some(Token(()))
            } else {
                drop(unsafe { Box::from_raw(p) });
                None
            }
        }
    }
    impl Drop for Token {
        fn drop(&mut self) {
            // Sound as long as each Token is dropped once: the Token owns SLOT's allocation.
            let p = SLOT.load(Ordering::SeqCst);
            println!("Token::drop frees {p:?}");
            drop(unsafe { Box::from_raw(p) });
        }
    }
}

#[test]
#[ignore]
fn double_free_via_extend() {
    let t = token::Token::new().unwrap();
    let mut v = <DimDyn as Dimension>::vec::<token::Token>(0, |_| unreachable!());
    v.extend([t]); // Token dropped here (frees the box) ...
    drop(v); // ... and again here: double free
}
