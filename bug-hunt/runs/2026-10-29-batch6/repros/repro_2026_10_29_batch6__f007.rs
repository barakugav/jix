// F007 repro: Op2::shape (jix/src/ops/op2.rs:184-188) is #[inline(always)] and in debug builds
// calls the left child's shape() twice (`debug_assert_eq!(self.a.shape(), self.b.shape());
// self.a.shape()`). Every wrapper (Add, Mul, ...) forwards shape() with #[inline(always)] too, so
// a left-deep chain of k binary ops inlines ~2^k copies of the subtree into each caller of
// shape(), and at -O0 every copy keeps its own stack slots.
//
// THE TWO NON-IGNORED TESTS ABORT THE PROCESS (stack overflow, SIGABRT) in a debug build while the
// bug exists. They assert the correct behavior: an ordinary 12-term sum (or a degree-6 Horner
// polynomial) reads fine on a default (2 MiB) std::thread, as the right-deep chain of the same
// length does. Run them one at a time by name, since an abort kills the whole binary.
//
// `stack_probe` is an #[ignore]d helper used to measure the minimum stack size by bisection:
//   F007_KIND = left | right | shape_left | shape_right, F007_N = number of terms, F007_STACK = bytes
use jix::Array;
use ndarray::{ArrayD, IxDyn};

fn nds(n: usize) -> Vec<ArrayD<i64>> {
    (0..n)
        .map(|k| {
            ArrayD::from_shape_fn(IxDyn(&[3, 4]), |ix| (ix[0] * 4 + ix[1]) as i64 + k as i64)
        })
        .collect()
}

fn reference_sum(nds: &[ArrayD<i64>]) -> ArrayD<i64> {
    let mut r = nds[0].clone();
    for nd in &nds[1..] {
        r = r + nd;
    }
    r
}

/// a[0] + a[1] + ... + a[11]: Rust's `+` is left-associative, so this is the left-deep chain.
#[test]
fn left_deep_sum_of_12_terms_on_default_thread() {
    std::thread::spawn(|| {
        let nds = nds(12);
        let a: Vec<_> = nds.iter().map(|nd| Array::compact_ndarray(nd).unwrap()).collect();
        let v = a[0].view()
            + a[1].view()
            + a[2].view()
            + a[3].view()
            + a[4].view()
            + a[5].view()
            + a[6].view()
            + a[7].view()
            + a[8].view()
            + a[9].view()
            + a[10].view()
            + a[11].view();
        assert_eq!(v.to_ndarray::<IxDyn>().unwrap(), reference_sum(&nds));
    })
    .join()
    .unwrap();
}

/// Degree-6 polynomial in Horner form: ((((((c6*x + c5)*x + c4)*x + c3)*x + c2)*x + c1)*x + c0),
/// 12 Op2 levels (alternating Mul / Add), left-deep.
#[test]
fn horner_degree_6_on_default_thread() {
    std::thread::spawn(|| {
        let xs = ArrayD::from_shape_fn(IxDyn(&[3, 4]), |ix| (ix[0] * 4 + ix[1]) as i64 - 5);
        let cs: Vec<ArrayD<i64>> = (0..7)
            .map(|k| ArrayD::from_elem(IxDyn(&[3, 4]), k as i64 + 1))
            .collect();
        let x = Array::compact_ndarray(&xs).unwrap();
        let c: Vec<_> = cs.iter().map(|nd| Array::compact_ndarray(nd).unwrap()).collect();
        let p = (((((c[6].view() * x.view() + c[5].view()) * x.view() + c[4].view())
            * x.view()
            + c[3].view())
            * x.view()
            + c[2].view())
            * x.view()
            + c[1].view())
            * x.view()
            + c[0].view();
        let expected = xs.mapv(|x| {
            let mut acc = 7i64;
            for k in (0..6).rev() {
                acc = acc * x + (k as i64 + 1);
            }
            acc
        });
        assert_eq!(p.to_ndarray::<IxDyn>().unwrap(), expected);
    })
    .join()
    .unwrap();
}

fn probe(kind: &str, n: usize) {
    let nds = nds(12);
    let a: Vec<_> = nds.iter().map(|nd| Array::compact_ndarray(nd).unwrap()).collect();
    macro_rules! lchain {
        ($($i:literal)+) => { a[0].view() $(+ a[$i].view())+ };
    }
    macro_rules! rchain {
        ($i:literal) => { a[$i].view() };
        ($i:literal $($rest:literal)+) => { a[$i].view() + (rchain!($($rest)+)) };
    }
    // Each arm runs in its own closure (a separate, non-inlined function in a debug build), so the
    // measured stack is that of one chain, not the sum of every arm's stack slots.
    macro_rules! run {
        ($v:expr, shape) => {
            (|| {
                let v = $v;
                assert_eq!(std::hint::black_box(v.shape()), &[3, 4]);
            })()
        };
        ($v:expr, read) => {
            (|| {
                let v = $v;
                assert_eq!(v.to_ndarray::<IxDyn>().unwrap(), reference_sum(&nds[..n]));
            })()
        };
    }
    macro_rules! dispatch {
        ($what:ident, $chain:ident) => {
            match n {
                2 => run!($chain!(0 1), $what),
                4 => run!($chain!(0 1 2 3), $what),
                6 => run!($chain!(0 1 2 3 4 5), $what),
                8 => run!($chain!(0 1 2 3 4 5 6 7), $what),
                10 => run!($chain!(0 1 2 3 4 5 6 7 8 9), $what),
                12 => run!($chain!(0 1 2 3 4 5 6 7 8 9 10 11), $what),
                _ => panic!("unsupported F007_N"),
            }
        };
    }
    macro_rules! lchain0 {
        (0 $($i:literal)+) => { lchain!($($i)+) };
    }
    match kind {
        "left" => dispatch!(read, lchain0),
        "right" => dispatch!(read, rchain),
        "shape_left" => dispatch!(shape, lchain0),
        "shape_right" => dispatch!(shape, rchain),
        _ => panic!("unsupported F007_KIND"),
    }
}

#[test]
#[ignore]
fn stack_probe() {
    let kind = std::env::var("F007_KIND").unwrap_or_else(|_| "left".into());
    let n: usize = std::env::var("F007_N").map(|s| s.parse().unwrap()).unwrap_or(12);
    let stack: usize = std::env::var("F007_STACK").map(|s| s.parse().unwrap()).unwrap_or(2 << 20);
    std::thread::Builder::new()
        .stack_size(stack)
        .spawn(move || probe(&kind, n))
        .unwrap()
        .join()
        .unwrap();
}
