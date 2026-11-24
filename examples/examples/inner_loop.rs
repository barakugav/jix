//! Successive versions of the inner loop of an element-wise `neg`, from a naive n-d loop to an
//! explicit SIMD one, for comparing their codegen (e.g. with llvm-mca) version to version.
//!
//! All strides are in bytes. `main` checks every version against a reference implementation:
//!
//!     cargo run --release --example inner_loop

use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Neg;

/// Version 1: one n-d loop over all the elements, recomputing the offsets from the n-d index.
#[inline(never)]
unsafe fn negative_v1<T>(
    dst: *mut T,
    src: *const T,
    shape: &[usize],
    dst_strides: &[usize],
    src_strides: &[usize],
) where
    T: Copy + std::ops::Neg<Output = T>,
{
    assert!(!shape.contains(&0), "TODO");
    let ndim = shape.len();
    let mut nd_index = vec![0; ndim];
    loop {
        // compute element offsets
        let (mut dst_offset, mut src_offset) = (0usize, 0usize);
        for i in 0..ndim {
            dst_offset += nd_index[i] * dst_strides[i];
            src_offset += nd_index[i] * src_strides[i];
        }

        unsafe {
            let elm = src.byte_add(src_offset).read();
            let neg = -elm;
            dst.byte_add(dst_offset).write(neg);
        }

        // advance the multi-index
        let mut i = ndim;
        loop {
            if i == 0 {
                return; // done
            }
            i -= 1;
            nd_index[i] += 1;
            if nd_index[i] < shape[i] {
                break;
            }
            nd_index[i] = 0;
        }
    }
}

/// Version 2: an n-d loop over the outer dims, calling a 1-d inner loop over the last dim.
unsafe fn negative_v2<T>(
    dst: *mut T,
    src: *const T,
    shape: &[usize],
    dst_strides: &[usize],
    src_strides: &[usize],
) where
    T: Copy + std::ops::Neg<Output = T>,
{
    assert!(!shape.contains(&0), "TODO");
    assert!(!shape.is_empty(), "TODO");

    let inner_dim = *shape.last().unwrap();
    let inner_dst_stride = *dst_strides.last().unwrap();
    let inner_src_stride = *src_strides.last().unwrap();

    let outer_ndim = shape.len() - 1;
    let mut nd_index = vec![0; outer_ndim];
    loop {
        // compute element offsets for outer dimensions
        let (mut dst_offset, mut src_offset) = (0usize, 0usize);
        for i in 0..outer_ndim {
            dst_offset += nd_index[i] * dst_strides[i];
            src_offset += nd_index[i] * src_strides[i];
        }

        unsafe {
            negative_v2_inner_loop(
                dst.byte_add(dst_offset),
                src.byte_add(src_offset),
                inner_dim,
                inner_dst_stride,
                inner_src_stride,
            )
        };

        // advance the multi-index for outer dimensions
        let mut i = outer_ndim;
        loop {
            if i == 0 {
                return; // done
            }
            i -= 1;
            nd_index[i] += 1;
            if nd_index[i] < shape[i] {
                break;
            }
            nd_index[i] = 0;
        }
    }
}
#[inline(never)]
unsafe fn negative_v2_inner_loop<T>(
    dst: *mut T,
    src: *const T,
    len: usize,
    dst_stride: usize,
    src_stride: usize,
) where
    T: Copy + std::ops::Neg<Output = T>,
{
    for i in 0..len {
        unsafe {
            let elm = src.byte_add(i * src_stride).read();
            let neg = -elm;
            dst.byte_add(i * dst_stride).write(neg);
        }
    }
}

/// Version 3: the inner loop specialized at compile time for contiguous operands.
#[inline(never)]
unsafe fn negative_v3_inner_loop<T, const IN_CONTIGUOUS: bool, const OUT_CONTIGUOUS: bool>(
    dst: *mut T,
    src: *const T,
    len: usize,
    dst_stride: usize,
    src_stride: usize,
) where
    T: Copy + std::ops::Neg<Output = T>,
{
    for i in 0..len {
        unsafe {
            let elm = if IN_CONTIGUOUS {
                src.add(i).read()
            } else {
                src.byte_add(i * src_stride).read()
            };
            let neg = -elm;
            if OUT_CONTIGUOUS {
                dst.add(i).write(neg);
            } else {
                dst.byte_add(i * dst_stride).write(neg);
            }
        }
    }
}

/// A pointer to `span` bytes that nothing writes to while it lives: a `&[u8]`, so the function
/// taking it as an argument gets LLVM's `noalias`.
struct PtrNoalias<'a, T>(&'a [u8], PhantomData<T>);
impl<'a, T> PtrNoalias<'a, T> {
    unsafe fn new(ptr: *const u8, span: usize) -> Self {
        unsafe { Self(std::slice::from_raw_parts(ptr, span), PhantomData) }
    }
    fn as_ptr(&self) -> *const T {
        self.0.as_ptr().cast::<T>()
    }
}
/// A pointer to `span` bytes that nothing else reads or writes while it lives: a `&mut [u8]`.
struct PtrMutNoalias<'a, T>(&'a mut [u8], PhantomData<T>);
impl<'a, T> PtrMutNoalias<'a, T> {
    unsafe fn new(ptr: *mut u8, span: usize) -> Self {
        unsafe { Self(std::slice::from_raw_parts_mut(ptr, span), PhantomData) }
    }
    fn as_mut_ptr(self) -> *mut T {
        self.0.as_mut_ptr().cast::<T>()
    }
}

/// Version 4: version 3, with the input marked as not aliasing the output.
#[inline(never)]
unsafe fn negative_v4_inner_loop<T, const IN_CONTIGUOUS: bool, const OUT_CONTIGUOUS: bool>(
    dst: PtrMutNoalias<'_, T>,
    src: PtrNoalias<'_, T>,
    len: usize,
    dst_stride: usize,
    src_stride: usize,
) where
    T: Copy + std::ops::Neg<Output = T>,
{
    let dst = dst.as_mut_ptr();
    let src = src.as_ptr();
    for i in 0..len {
        unsafe {
            let elm = if IN_CONTIGUOUS {
                src.add(i).read()
            } else {
                src.byte_add(i * src_stride).read()
            };
            let neg = -elm;
            if OUT_CONTIGUOUS {
                dst.add(i).write(neg);
            } else {
                dst.byte_add(i * dst_stride).write(neg);
            }
        }
    }
}

/// Version 5: version 4, processing `LANES` elements per iteration as arrays.
#[inline(never)]
unsafe fn negative_v5_inner_loop<
    T,
    const LANES: usize,
    const IN_CONTIGUOUS: bool,
    const OUT_CONTIGUOUS: bool,
>(
    dst: PtrMutNoalias<'_, T>,
    src: PtrNoalias<'_, T>,
    len: usize,
    dst_stride: usize,
    src_stride: usize,
) where
    T: Copy + std::ops::Neg<Output = T>,
{
    let dst = dst.as_mut_ptr();
    let src = src.as_ptr();

    let body_limit = len - len % LANES;
    let mut i = 0;
    while i < body_limit {
        unsafe {
            let elms = if IN_CONTIGUOUS {
                src.add(i).cast::<[T; LANES]>().read()
            } else {
                std::array::from_fn(|j| src.byte_add((i + j) * src_stride).read())
            };
            let neg = std::array::from_fn(|j| -elms[j]);
            if OUT_CONTIGUOUS {
                dst.add(i).cast::<[T; LANES]>().write(neg);
            } else {
                for j in 0..LANES {
                    dst.byte_add((i + j) * dst_stride).write(neg[j]);
                }
            }
        }
        i += LANES;
    }
    while i < len {
        unsafe {
            let elm = if IN_CONTIGUOUS {
                src.add(i).read()
            } else {
                src.byte_add(i * src_stride).read()
            };
            let neg = -elm;
            if OUT_CONTIGUOUS {
                dst.add(i).write(neg);
            } else {
                dst.byte_add(i * dst_stride).write(neg);
            }
        }
        i += 1;
    }
}

/// The SIMD vector of `Self` at the SIMD level `S`.
trait SimdElem: Copy + std::ops::Neg<Output = Self> + Default {
    type Vector<S: fearless_simd::Simd>: fearless_simd::SimdBase<S, Element = Self>
        + std::ops::Neg<Output = Self::Vector<S>>;
}
impl SimdElem for f32 {
    type Vector<S: fearless_simd::Simd> = S::f32s;
}
impl SimdElem for i32 {
    type Vector<S: fearless_simd::Simd> = S::i32s;
}
impl SimdElem for f64 {
    type Vector<S: fearless_simd::Simd> = S::f64s;
}

/// Version 6: version 5, negating the arrays with explicit SIMD vectors, compiled once per SIMD
/// level with the CPU's level picked at runtime. `dispatch!` runs its body inside a function with
/// the level's target features enabled, so everything it calls must be inlined into it.
#[inline(never)]
unsafe fn negative_v6_inner_loop<
    T: SimdElem,
    const LANES: usize,
    const IN_CONTIGUOUS: bool,
    const OUT_CONTIGUOUS: bool,
>(
    dst: PtrMutNoalias<'_, T>,
    src: PtrNoalias<'_, T>,
    len: usize,
    dst_stride: usize,
    src_stride: usize,
) {
    fearless_simd::dispatch!(fearless_simd::Level::new(), simd => unsafe {
        negative_v6_inner_loop_per_level::<T, T::Vector<_>, LANES, IN_CONTIGUOUS, OUT_CONTIGUOUS, _>(
            dst, src, len, dst_stride, src_stride, simd,
        )
    })
}
#[inline(always)]
unsafe fn negative_v6_inner_loop_per_level<
    T,
    TS,
    const LANES: usize,
    const IN_CONTIGUOUS: bool,
    const OUT_CONTIGUOUS: bool,
    S: fearless_simd::Simd,
>(
    dst: PtrMutNoalias<'_, T>,
    src: PtrNoalias<'_, T>,
    len: usize,
    dst_stride: usize,
    src_stride: usize,
    simd: S,
) where
    T: Copy + std::ops::Neg<Output = T> + Default,
    TS: fearless_simd::SimdBase<S, Element = T> + std::ops::Neg<Output = TS>,
    S: fearless_simd::Simd,
{
    let dst = dst.as_mut_ptr();
    let src = src.as_ptr();

    let body_limit = len - len % LANES;
    let mut i = 0;
    while i < body_limit {
        unsafe {
            let elms = if IN_CONTIGUOUS {
                src.add(i).cast::<[T; LANES]>().read()
            } else {
                std::array::from_fn(|j| src.byte_add((i + j) * src_stride).read())
            };
            let neg = negative_v6_kernel::<T, TS, LANES, S>(elms, simd);
            if OUT_CONTIGUOUS {
                dst.add(i).cast::<[T; LANES]>().write(neg);
            } else {
                for j in 0..LANES {
                    dst.byte_add((i + j) * dst_stride).write(neg[j]);
                }
            }
        }
        i += LANES;
    }
    while i < len {
        unsafe {
            let elm = if IN_CONTIGUOUS {
                src.add(i).read()
            } else {
                src.byte_add(i * src_stride).read()
            };
            let neg = -elm;
            if OUT_CONTIGUOUS {
                dst.add(i).write(neg);
            } else {
                dst.byte_add(i * dst_stride).write(neg);
            }
        }
        i += 1;
    }
}
#[inline(always)]
fn negative_v6_kernel<T, TS, const N: usize, S: fearless_simd::Simd>(a: [T; N], simd: S) -> [T; N]
where
    T: Copy + std::ops::Neg<Output = T> + Default,
    TS: fearless_simd::SimdBase<S, Element = T> + std::ops::Neg<Output = TS>,
    S: fearless_simd::Simd,
{
    let lanes = TS::LEN;
    if !N.is_multiple_of(lanes) {
        return std::array::from_fn(|i| -a[i]);
    }
    let mut result = [T::default(); N];
    for i in 0..(N / lanes) {
        let vec = TS::from_slice(simd, &a[i * lanes..][..lanes]);
        (-vec).store_slice(&mut result[i * lanes..][..lanes]);
    }
    result
}

// ---------------------------------------------------------------------------------------------
// Checks against a reference
// ---------------------------------------------------------------------------------------------

trait Elem: Copy + Neg<Output = Self> + PartialEq + Debug + Default + From<i16> {}
impl<T: Copy + Neg<Output = T> + PartialEq + Debug + Default + From<i16>> Elem for T {}

/// The value of the output's untouched elements, which no negated input equals.
const SENTINEL: i16 = i16::MIN;

/// The bytes an operand of `len` elements `stride` bytes apart spans.
fn span<T>(len: usize, stride: usize) -> usize {
    if len == 0 {
        0
    } else {
        (len - 1) * stride + size_of::<T>()
    }
}

/// The operands of a neg: the output filled with `SENTINEL`, the input with distinct nonzero
/// values (a negated zero cannot tell `-x` from `0 - x` apart for floats).
struct Operands<T> {
    dst: Vec<T>,
    src: Vec<T>,
}
impl<T: Elem> Operands<T> {
    fn new(dst_len: usize, src_len: usize) -> Self {
        // a little padding past each operand's last element, to catch reads/writes past the end
        let pad = 4;
        Self {
            dst: vec![T::from(SENTINEL); dst_len + pad],
            src: (0..src_len + pad).map(|k| T::from(k as i16 + 1)).collect(),
        }
    }

    /// Asserts that `dst` holds `-src` at the element offsets of each index, and is untouched
    /// elsewhere.
    fn check(&self, name: &str, offsets: impl Iterator<Item = [usize; 2]>) {
        let mut expected = vec![T::from(SENTINEL); self.dst.len()];
        for [d, s] in offsets {
            expected[d] = -self.src[s];
        }
        assert_eq!(self.dst, expected, "{name}");
    }
}

/// The element strides to try for an operand: contiguous only, or any including broadcast.
fn steps(contiguous: bool, broadcast: bool) -> &'static [usize] {
    match (contiguous, broadcast) {
        (true, _) => &[1],
        (false, false) => &[1, 2, 3],
        (false, true) => &[0, 1, 2, 3],
    }
}

/// A version of the 1-d inner loop, run with raw pointers.
trait InnerLoop<T> {
    unsafe fn run<const IN_CONTIGUOUS: bool, const OUT_CONTIGUOUS: bool>(
        &self,
        dst: *mut T,
        src: *const T,
        len: usize,
        dst_stride: usize,
        src_stride: usize,
    );
}

/// Checks an inner loop on every length up to 40 and strides that `IN_CONTIGUOUS` /
/// `OUT_CONTIGUOUS` allow.
fn check_inner<T: Elem, const IN_CONTIGUOUS: bool, const OUT_CONTIGUOUS: bool>(
    name: &str,
    f: &impl InnerLoop<T>,
) {
    let name = format!(
        "{name}<{}, {IN_CONTIGUOUS}, {OUT_CONTIGUOUS}>",
        std::any::type_name::<T>()
    );
    let size = size_of::<T>();
    for len in 0..=40 {
        for &ds in steps(OUT_CONTIGUOUS, false) {
            for &ss in steps(IN_CONTIGUOUS, true) {
                let mut ops = Operands::<T>::new(len * ds, len * ss);
                let (d, s) = (ops.dst.as_mut_ptr(), ops.src.as_ptr());
                unsafe { f.run::<IN_CONTIGUOUS, OUT_CONTIGUOUS>(d, s, len, ds * size, ss * size) };
                let name = format!("{name} len={len} steps=({ds}, {ss})");
                ops.check(&name, (0..len).map(|i| [i * ds, i * ss]));
            }
        }
    }
}

/// `check_inner` for every combination of the contiguity flags.
fn check_inner_all<T: Elem>(name: &str, f: &impl InnerLoop<T>) {
    check_inner::<T, false, false>(name, f);
    check_inner::<T, false, true>(name, f);
    check_inner::<T, true, false>(name, f);
    check_inner::<T, true, true>(name, f);
}

/// The noalias pointers of versions 4-6 over the raw pointers of a check.
unsafe fn noalias<'a, T>(
    dst: *mut T,
    src: *const T,
    len: usize,
    dst_stride: usize,
    src_stride: usize,
) -> (PtrMutNoalias<'a, T>, PtrNoalias<'a, T>) {
    unsafe {
        (
            PtrMutNoalias::new(dst.cast(), span::<T>(len, dst_stride)),
            PtrNoalias::new(src.cast(), span::<T>(len, src_stride)),
        )
    }
}

struct V2;
impl<T: Elem> InnerLoop<T> for V2 {
    unsafe fn run<const I: bool, const O: bool>(
        &self,
        d: *mut T,
        s: *const T,
        len: usize,
        ds: usize,
        ss: usize,
    ) {
        unsafe { negative_v2_inner_loop(d, s, len, ds, ss) }
    }
}

struct V3;
impl<T: Elem> InnerLoop<T> for V3 {
    unsafe fn run<const I: bool, const O: bool>(
        &self,
        d: *mut T,
        s: *const T,
        len: usize,
        ds: usize,
        ss: usize,
    ) {
        unsafe { negative_v3_inner_loop::<T, I, O>(d, s, len, ds, ss) }
    }
}

struct V4;
impl<T: Elem> InnerLoop<T> for V4 {
    unsafe fn run<const I: bool, const O: bool>(
        &self,
        d: *mut T,
        s: *const T,
        len: usize,
        ds: usize,
        ss: usize,
    ) {
        unsafe {
            let (dn, sn) = noalias(d, s, len, ds, ss);
            negative_v4_inner_loop::<T, I, O>(dn, sn, len, ds, ss)
        }
    }
}

struct V5<const LANES: usize>;
impl<T: Elem, const LANES: usize> InnerLoop<T> for V5<LANES> {
    unsafe fn run<const I: bool, const O: bool>(
        &self,
        d: *mut T,
        s: *const T,
        len: usize,
        ds: usize,
        ss: usize,
    ) {
        unsafe {
            let (dn, sn) = noalias(d, s, len, ds, ss);
            negative_v5_inner_loop::<T, LANES, I, O>(dn, sn, len, ds, ss)
        }
    }
}

struct V6<const LANES: usize>;
impl<T: Elem + SimdElem, const LANES: usize> InnerLoop<T> for V6<LANES> {
    unsafe fn run<const I: bool, const O: bool>(
        &self,
        d: *mut T,
        s: *const T,
        len: usize,
        ds: usize,
        ss: usize,
    ) {
        unsafe {
            let (dn, sn) = noalias(d, s, len, ds, ss);
            negative_v6_inner_loop::<T, LANES, I, O>(dn, sn, len, ds, ss)
        }
    }
}

/// An n-d layout of an operand: element strides from a shape.
#[derive(Clone, Copy, Debug)]
enum Layout {
    RowMajor,
    ColMajor,
    /// Row-major, with the last dim 2 elements apart.
    RowMajorPadded,
    /// Row-major, broadcast along the first dim (stride 0).
    BroadcastFirst,
}
impl Layout {
    /// The element strides, and the number of elements the operand's buffer needs.
    fn steps(self, shape: &[usize]) -> (Vec<usize>, usize) {
        let ndim = shape.len();
        let mut steps = vec![0; ndim];
        let mut order: Vec<usize> = (0..ndim).rev().collect(); // innermost first
        if let Layout::ColMajor = self {
            order.reverse();
        }
        let mut step = if let Layout::RowMajorPadded = self {
            2
        } else {
            1
        };
        for d in order {
            steps[d] = step;
            step *= shape[d];
        }
        if let (Layout::BroadcastFirst, Some(s)) = (self, steps.first_mut()) {
            *s = 0;
        }
        let len = 1 + shape
            .iter()
            .zip(&steps)
            .map(|(n, s)| (n - 1) * s)
            .sum::<usize>();
        (steps, len)
    }
}

/// An n-d neg: `f(dst, src, shape, dst_strides, src_strides)`.
type NdNeg<T> = unsafe fn(*mut T, *const T, &[usize], &[usize], &[usize]);

/// Checks an n-d neg on every shape of up to 3 dims (at least `min_ndim`) and every combination
/// of the operand layouts.
fn check_nd<T: Elem>(name: &str, min_ndim: usize, f: NdNeg<T>) {
    let name = format!("{name}<{}>", std::any::type_name::<T>());
    let size = size_of::<T>();
    let dims = [1, 2, 3, 5];
    let layouts = [
        Layout::RowMajor,
        Layout::ColMajor,
        Layout::RowMajorPadded,
        Layout::BroadcastFirst,
    ];
    for ndim in min_ndim..=3 {
        for k in 0..dims.len().pow(ndim as u32) {
            let shape: Vec<usize> = (0..ndim)
                .map(|d| dims[k / dims.len().pow(d as u32) % dims.len()])
                .collect();
            for dst_layout in &layouts[..3] {
                for src_layout in layouts {
                    let (ds, d_len) = dst_layout.steps(&shape);
                    let (ss, s_len) = src_layout.steps(&shape);
                    let mut ops = Operands::<T>::new(d_len, s_len);
                    let bytes = |s: &[usize]| s.iter().map(|s| s * size).collect::<Vec<_>>();
                    unsafe {
                        f(
                            ops.dst.as_mut_ptr(),
                            ops.src.as_ptr(),
                            &shape,
                            &bytes(&ds),
                            &bytes(&ss),
                        )
                    };
                    // the offsets of each flat index, unraveled row-major
                    let n: usize = shape.iter().product();
                    let offsets = (0..n).map(|mut flat| {
                        let mut off = [0; 2];
                        for d in (0..ndim).rev() {
                            let idx = flat % shape[d];
                            flat /= shape[d];
                            off[0] += idx * ds[d];
                            off[1] += idx * ss[d];
                        }
                        off
                    });
                    let name =
                        format!("{name} shape={shape:?} layouts=({dst_layout:?}, {src_layout:?})");
                    ops.check(&name, offsets);
                }
            }
        }
    }
}

/// Runs the checks of every version for `T`.
fn check_all<T: Elem + SimdElem>() {
    check_nd::<T>("negative_v1", 0, negative_v1::<T>);
    check_nd::<T>("negative_v2", 1, negative_v2::<T>);
    check_inner_all::<T>("negative_v2_inner_loop", &V2);
    check_inner_all::<T>("negative_v3_inner_loop", &V3);
    check_inner_all::<T>("negative_v4_inner_loop", &V4);
    check_inner_all::<T>("negative_v5_inner_loop<3>", &V5::<3>);
    check_inner_all::<T>("negative_v5_inner_loop<8>", &V5::<8>);
    check_inner_all::<T>("negative_v5_inner_loop<16>", &V5::<16>);
    // 3 is not a multiple of any vector length: the kernel falls back to scalar negs
    check_inner_all::<T>("negative_v6_inner_loop<3>", &V6::<3>);
    check_inner_all::<T>("negative_v6_inner_loop<8>", &V6::<8>);
    check_inner_all::<T>("negative_v6_inner_loop<16>", &V6::<16>);
    check_inner_all::<T>("negative_v6_inner_loop<32>", &V6::<32>);
}

fn main() {
    check_all::<i32>();
    check_all::<f32>();
    check_all::<f64>();
    println!(
        "all versions match the reference ({:?})",
        fearless_simd::Level::new()
    );
}
