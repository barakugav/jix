use std::hint::unreachable_unchecked;
use std::mem::MaybeUninit;

#[inline(always)]
pub(crate) fn array_from_fn_inline<T, const N: usize>(mut f: impl FnMut(usize) -> T) -> [T; N] {
    array_try_from_fn_inline(|i| Ok(f(i)))
        .unwrap_or_else(|_: ()| unsafe { unreachable_unchecked() })
}
#[inline(always)]
pub(crate) fn array_try_from_fn_inline<T, E, const N: usize>(
    mut f: impl FnMut(usize) -> Result<T, E>,
) -> Result<[T; N], E> {
    struct Guard<'a, T, const N: usize> {
        data: &'a mut [MaybeUninit<T>; N],
        initialized: usize,
    }
    impl<T, const N: usize> Drop for Guard<'_, T, N> {
        #[inline(always)]
        fn drop(&mut self) {
            unsafe { std::hint::assert_unchecked(self.initialized <= N) };
            for i in 0..self.initialized {
                unsafe { std::ptr::drop_in_place(self.data[i].as_mut_ptr()) };
            }
        }
    }
    let mut data = MaybeUninit::<[T; N]>::uninit();
    let mut guard = Guard {
        data: unsafe { &mut *(&mut data as *mut MaybeUninit<[T; N]> as *mut [MaybeUninit<T>; N]) },
        initialized: 0,
    };
    for i in 0..N {
        guard.data[i].write(f(i)?);
        guard.initialized += 1;
    }
    std::mem::forget(guard);
    Ok(unsafe { data.assume_init() })
}

pub(crate) trait ArrayExt<T, const N: usize> {
    fn map_inline<U>(self, f: impl FnMut(T) -> U) -> [U; N]
    where
        Self: Sized;

    fn try_map_inline<U, E>(self, f: impl FnMut(T) -> Result<U, E>) -> Result<[U; N], E>
    where
        Self: Sized;

    fn map_inline_ref<U>(&self, f: impl FnMut(&T) -> U) -> [U; N]
    where
        Self: Sized;

    fn try_map_inline_ref<U, E>(&self, f: impl FnMut(&T) -> Result<U, E>) -> Result<[U; N], E>
    where
        Self: Sized;
}
impl<T, const N: usize> ArrayExt<T, N> for [T; N] {
    #[inline(always)]
    fn map_inline<U>(self, mut f: impl FnMut(T) -> U) -> [U; N]
    where
        Self: Sized,
    {
        self.try_map_inline(|x| Ok(f(x)))
            .unwrap_or_else(|_: ()| unsafe { unreachable_unchecked() })
    }

    #[inline(always)]
    fn try_map_inline<U, E>(self, mut f: impl FnMut(T) -> Result<U, E>) -> Result<[U; N], E>
    where
        Self: Sized,
    {
        let mut data = self.into_iter();
        array_try_from_fn_inline(|_| f(unsafe { data.next().unwrap_unchecked() }))
    }

    #[inline(always)]
    fn map_inline_ref<U>(&self, mut f: impl FnMut(&T) -> U) -> [U; N]
    where
        Self: Sized,
    {
        self.try_map_inline_ref(|x| Ok(f(x)))
            .unwrap_or_else(|_: ()| unsafe { unreachable_unchecked() })
    }

    #[inline(always)]
    fn try_map_inline_ref<U, E>(&self, mut f: impl FnMut(&T) -> Result<U, E>) -> Result<[U; N], E>
    where
        Self: Sized,
    {
        let mut data = self.iter();
        array_try_from_fn_inline(|_| f(unsafe { data.next().unwrap_unchecked() }))
    }
}

// pub(crate) fn array_map2_inline<T, U, V, const N: usize>(
//     a: [T; N],
//     b: [U; N],
//     mut f: impl FnMut(T, U) -> V,
// ) -> [V; N] {
//     let mut data_a = a.into_iter();
//     let mut data_b = b.into_iter();
//     array_from_fn_inline(|_| {
//         let x = unsafe { data_a.next().unwrap_unchecked() };
//         let y = unsafe { data_b.next().unwrap_unchecked() };
//         f(x, y)
//     })
// }

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::rc::Rc;

    // ---------------------------------------------------------------------
    // Drop-tracking harness.
    //
    // Every `Tracked` records the `id`s of *closure invocations* and *drops*
    // into a shared `Log`, so a test can assert exactly which elements were
    // visited and that every value was dropped exactly once (no leak, no
    // double-drop). Each `Tracked` also owns a `Box`, so Miri independently
    // flags a double-free (double-drop) or a leaked allocation even if the
    // hand-written counting somehow missed it.
    // ---------------------------------------------------------------------

    #[derive(Default)]
    struct Log {
        /// `id`s the mapping closure was invoked with, in call order.
        called: Vec<u32>,
        /// `id`s of every `Tracked` that was dropped, in drop order.
        dropped: Vec<u32>,
    }
    type LogRef = Rc<RefCell<Log>>;

    /// A non-`Copy`, `Drop`-tracking element carrying a heap allocation.
    struct Tracked {
        id: u32,
        log: LogRef,
        _heap: Box<u32>,
    }
    impl Tracked {
        fn new(id: u32, log: &LogRef) -> Self {
            Tracked {
                id,
                log: Rc::clone(log),
                _heap: Box::new(id),
            }
        }
    }
    impl Drop for Tracked {
        fn drop(&mut self) {
            self.log.borrow_mut().dropped.push(self.id);
        }
    }

    /// `[Tracked; N]` with ids `0..N`.
    fn tracked_array<const N: usize>(log: &LogRef) -> [Tracked; N] {
        std::array::from_fn(|i| Tracked::new(i as u32, log))
    }

    /// `dropped` sorted ascending - convenient for multiset comparison. Since
    /// each id is unique, equality against a strictly-increasing expected list
    /// simultaneously proves "nothing leaked" (all present) and "nothing
    /// double-dropped" (no duplicates).
    fn sorted_dropped(log: &LogRef) -> Vec<u32> {
        let mut v = log.borrow().dropped.clone();
        v.sort_unstable();
        v
    }

    // =====================================================================
    // array_from_fn_inline / array_try_from_fn_inline - correctness
    //
    // These free functions hold the actual unsafe Guard; map_inline and
    // try_map_inline are thin wrappers over them, so they get their own
    // direct tests in addition to the wrapper tests further down.
    // =====================================================================

    #[test]
    fn array_from_fn_inline_builds_from_index() {
        let out: [u32; 5] = array_from_fn_inline(|i| (i as u32) * 2);
        assert_eq!(out, [0, 2, 4, 6, 8]);
    }

    #[test]
    fn array_from_fn_inline_empty_never_calls() {
        let called = Cell::new(0u32);
        let out: [u32; 0] = array_from_fn_inline(|i| {
            called.set(called.get() + 1);
            i as u32
        });
        assert_eq!(out, [] as [u32; 0]);
        assert_eq!(called.get(), 0);
    }

    #[test]
    fn array_from_fn_inline_visits_indices_in_order() {
        let order = RefCell::new(Vec::new());
        let out: [usize; 4] = array_from_fn_inline(|i| {
            order.borrow_mut().push(i);
            i
        });
        assert_eq!(*order.borrow(), vec![0, 1, 2, 3]);
        assert_eq!(out, [0, 1, 2, 3]);
    }

    #[test]
    fn array_from_fn_inline_large() {
        const N: usize = 257;
        let out: [u32; N] = array_from_fn_inline(|i| i as u32 * 3 + 1);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, i as u32 * 3 + 1, "mismatch at index {i}");
        }
    }

    #[test]
    fn array_try_from_fn_inline_all_ok() {
        let out: Result<[i32; 4], ()> = array_try_from_fn_inline(|i| Ok(i as i32 + 10));
        assert_eq!(out, Ok([10, 11, 12, 13]));
    }

    #[test]
    fn array_try_from_fn_inline_empty_ok_never_calls() {
        let called = Cell::new(0u32);
        let out: Result<[u32; 0], ()> = array_try_from_fn_inline(|i| {
            called.set(called.get() + 1);
            Ok(i as u32)
        });
        assert_eq!(out, Ok([] as [u32; 0]));
        assert_eq!(called.get(), 0);
    }

    #[test]
    fn array_try_from_fn_inline_short_circuits_on_err() {
        let calls = Cell::new(0u32);
        let out: Result<[usize; 6], usize> = array_try_from_fn_inline(|i| {
            calls.set(calls.get() + 1);
            if i == 3 {
                Err(i)
            } else {
                Ok(i)
            }
        });
        assert_eq!(out, Err(3));
        // Called on indices 0, 1, 2, 3 only - never 4 or 5.
        assert_eq!(calls.get(), 4);
    }

    #[test]
    fn array_try_from_fn_inline_ok_drops_only_when_output_dropped() {
        let log: LogRef = Default::default();
        let out: [Tracked; 4] =
            array_try_from_fn_inline(|i| Ok::<_, ()>(Tracked::new(i as u32, &log))).unwrap();
        // Nothing dropped while the produced array is still alive.
        assert_eq!(sorted_dropped(&log), Vec::<u32>::new());
        drop(out);
        assert_eq!(sorted_dropped(&log), vec![0, 1, 2, 3]);
    }

    #[test]
    fn array_try_from_fn_inline_err_frees_produced_outputs_exactly_once() {
        let log: LogRef = Default::default();
        let result: Result<[Tracked; 5], u32> = array_try_from_fn_inline(|i| {
            log.borrow_mut().called.push(i as u32);
            if i == 3 {
                Err(i as u32)
            } else {
                Ok(Tracked::new(i as u32, &log))
            }
        });
        assert_eq!(result.err(), Some(3));
        // Generator invoked on 0..=3 (short-circuit), producing outputs 0, 1, 2.
        assert_eq!(log.borrow().called, vec![0, 1, 2, 3]);
        // The Guard frees the three produced outputs - each exactly once, none leaked.
        assert_eq!(sorted_dropped(&log), vec![0, 1, 2]);
    }

    #[test]
    fn array_try_from_fn_inline_panic_frees_produced_outputs() {
        let log: LogRef = Default::default();
        let log_in = Rc::clone(&log);
        let caught = catch_unwind(AssertUnwindSafe(move || {
            let _: Result<[Tracked; 5], ()> = array_try_from_fn_inline(|i| {
                log_in.borrow_mut().called.push(i as u32);
                if i == 3 {
                    panic!("boom");
                }
                Ok(Tracked::new(i as u32, &log_in))
            });
            unreachable!("generator above always panics");
        }));
        assert!(caught.is_err());
        assert_eq!(log.borrow().called, vec![0, 1, 2, 3]);
        // Outputs 0, 1, 2 produced before the panic are freed by the Guard during unwind.
        assert_eq!(sorted_dropped(&log), vec![0, 1, 2]);
    }

    #[test]
    fn array_from_fn_inline_panic_frees_produced_outputs() {
        let log: LogRef = Default::default();
        let log_in = Rc::clone(&log);
        let caught = catch_unwind(AssertUnwindSafe(move || {
            let _out: [Tracked; 4] = array_from_fn_inline(|i| {
                log_in.borrow_mut().called.push(i as u32);
                if i == 2 {
                    panic!("boom");
                }
                Tracked::new(i as u32, &log_in)
            });
            unreachable!("generator above always panics");
        }));
        assert!(caught.is_err());
        assert_eq!(log.borrow().called, vec![0, 1, 2]);
        assert_eq!(sorted_dropped(&log), vec![0, 1]);
    }

    // =====================================================================
    // map_inline - correctness
    // =====================================================================

    #[test]
    fn map_inline_maps_each_element() {
        let out = [1, 2, 3, 4].map_inline(|x| x * 2);
        assert_eq!(out, [2, 4, 6, 8]);
    }

    #[test]
    fn map_inline_identity() {
        let out = [10, 20, 30].map_inline(|x| x);
        assert_eq!(out, [10, 20, 30]);
    }

    #[test]
    fn map_inline_changes_type() {
        let out: [String; 3] = [1, 2, 3].map_inline(|x| x.to_string());
        assert_eq!(out, ["1".to_string(), "2".to_string(), "3".to_string()]);
    }

    #[test]
    fn map_inline_empty_array() {
        let called = Cell::new(0u32);
        let out: [u64; 0] = ([] as [u32; 0]).map_inline(|x| {
            called.set(called.get() + 1);
            x as u64
        });
        assert_eq!(out, [] as [u64; 0]);
        assert_eq!(called.get(), 0, "closure must not run for an empty array");
    }

    #[test]
    fn map_inline_single_element() {
        let out = [42].map_inline(|x| x + 1);
        assert_eq!(out, [43]);
    }

    #[test]
    fn map_inline_large_array_preserves_positions() {
        const N: usize = 257;
        let input: [u32; N] = std::array::from_fn(|i| i as u32);
        let out = input.map_inline(|x| x * 3 + 1);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, (i as u32) * 3 + 1, "mismatch at index {i}");
        }
    }

    #[test]
    fn map_inline_visits_elements_in_order() {
        let order = RefCell::new(Vec::new());
        let out = [5, 6, 7, 8].map_inline(|x| {
            order.borrow_mut().push(x);
            x
        });
        assert_eq!(*order.borrow(), vec![5, 6, 7, 8]);
        assert_eq!(out, [5, 6, 7, 8]);
    }

    // =====================================================================
    // try_map_inline - correctness (Ok path)
    // =====================================================================

    #[test]
    fn try_map_inline_all_ok() {
        let out: Result<[i32; 4], ()> = [1, 2, 3, 4].try_map_inline(|x| Ok(x * 10));
        assert_eq!(out, Ok([10, 20, 30, 40]));
    }

    #[test]
    fn try_map_inline_empty_is_ok_without_calling() {
        let called = Cell::new(0u32);
        let out: Result<[u32; 0], ()> = ([] as [u32; 0]).try_map_inline(|x| {
            called.set(called.get() + 1);
            Ok(x)
        });
        assert_eq!(out, Ok([] as [u32; 0]));
        assert_eq!(called.get(), 0);
    }

    #[test]
    fn try_map_inline_single_ok() {
        let out: Result<[i32; 1], ()> = [7].try_map_inline(|x| Ok(x - 1));
        assert_eq!(out, Ok([6]));
    }

    // =====================================================================
    // try_map_inline - short-circuit on Err
    // =====================================================================

    #[test]
    fn try_map_inline_short_circuits_on_first_err() {
        let calls = Cell::new(0u32);
        let out: Result<[i32; 5], &str> = [1, 2, 3, 4, 5].try_map_inline(|x| {
            calls.set(calls.get() + 1);
            if x == 3 {
                Err("stop")
            } else {
                Ok(x * 10)
            }
        });
        assert_eq!(out, Err("stop"));
        // Called on 1, 2, 3 only - never on 4 or 5.
        assert_eq!(calls.get(), 3);
    }

    #[test]
    fn try_map_inline_err_on_first_element() {
        let calls = Cell::new(0u32);
        let out: Result<[i32; 4], i32> = [1, 2, 3, 4].try_map_inline(|x| {
            calls.set(calls.get() + 1);
            Err(x)
        });
        assert_eq!(out, Err(1));
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn try_map_inline_err_on_last_element() {
        let calls = Cell::new(0u32);
        let out: Result<[i32; 4], &str> = [1, 2, 3, 4].try_map_inline(|x| {
            calls.set(calls.get() + 1);
            if x == 4 {
                Err("last")
            } else {
                Ok(x)
            }
        });
        assert_eq!(out, Err("last"));
        assert_eq!(calls.get(), 4);
    }

    #[test]
    fn try_map_inline_returns_first_error_encountered() {
        // Elements 2, 3, 4 would all fail; short-circuiting means only the
        // first (value 2) is ever produced.
        let out: Result<[i32; 4], i32> =
            [1, 2, 3, 4].try_map_inline(|x| if x >= 2 { Err(x) } else { Ok(x) });
        assert_eq!(out, Err(2));
    }

    // =====================================================================
    // Drop safety - Ok path
    // =====================================================================

    #[test]
    fn try_map_inline_ok_drops_inputs_not_outputs_until_dropped() {
        let log: LogRef = Default::default();
        let input: [Tracked; 4] = tracked_array(&log);

        let out = input
            .try_map_inline(|t: Tracked| Ok::<_, ()>(Tracked::new(100 + t.id, &log)))
            .unwrap();

        // Inputs are consumed by the closure and dropped there; outputs are
        // still alive inside `out` and must NOT have been dropped yet.
        assert_eq!(sorted_dropped(&log), vec![0, 1, 2, 3]);

        drop(out);
        // Now every input (0..4) and every output (100..104) has dropped once.
        assert_eq!(sorted_dropped(&log), vec![0, 1, 2, 3, 100, 101, 102, 103]);
    }

    #[test]
    fn map_inline_ok_drops_inputs_and_returns_outputs() {
        let log: LogRef = Default::default();
        let input: [Tracked; 3] = tracked_array(&log);

        let out = input.map_inline(|t: Tracked| Tracked::new(100 + t.id, &log));

        assert_eq!(sorted_dropped(&log), vec![0, 1, 2]);
        let ids: Vec<u32> = out.iter().map(|t| t.id).collect();
        assert_eq!(ids, vec![100, 101, 102]);

        drop(out);
        assert_eq!(sorted_dropped(&log), vec![0, 1, 2, 100, 101, 102]);
    }

    // =====================================================================
    // Drop safety - Err path (the crux of the unsafe Guard)
    // =====================================================================

    #[test]
    fn try_map_inline_err_middle_drops_everything_exactly_once() {
        let log: LogRef = Default::default();
        #[derive(Debug, PartialEq)]
        struct MyErr(u32);

        let input: [Tracked; 5] = tracked_array(&log);
        let result: Result<[Tracked; 5], MyErr> = input.try_map_inline(|t: Tracked| {
            log.borrow_mut().called.push(t.id);
            if t.id == 2 {
                Err(MyErr(t.id)) // input `t` dropped here; no output produced
            } else {
                // output produced; input dropped at end of closure
                Ok(Tracked::new(100 + t.id, &log))
            }
        });

        assert_eq!(result.err(), Some(MyErr(2)));

        // Short-circuit: closure invoked on 0, 1, 2 only.
        assert_eq!(log.borrow().called, vec![0, 1, 2]);

        // Drops, exactly once each:
        //   inputs  0..5      (0,1 in closure; 2 in the Err branch; 3,4 by the array IntoIter)
        //   outputs 100, 101  (produced for 0 and 1, then freed by the Guard on unwind)
        assert_eq!(sorted_dropped(&log), vec![0, 1, 2, 3, 4, 100, 101]);
    }

    #[test]
    fn try_map_inline_err_first_drops_all_inputs_no_outputs() {
        let log: LogRef = Default::default();
        let input: [Tracked; 4] = tracked_array(&log);
        let result: Result<[Tracked; 4], u32> = input.try_map_inline(|t: Tracked| {
            log.borrow_mut().called.push(t.id);
            Err(t.id)
        });
        assert_eq!(result.err(), Some(0));
        assert_eq!(log.borrow().called, vec![0]);
        // No output ever created; all four inputs dropped once (0 in the closure, 1..4 by IntoIter).
        assert_eq!(sorted_dropped(&log), vec![0, 1, 2, 3]);
    }

    #[test]
    fn try_map_inline_err_last_drops_all_outputs_and_inputs() {
        let log: LogRef = Default::default();
        let input: [Tracked; 4] = tracked_array(&log);
        let result: Result<[Tracked; 4], u32> = input.try_map_inline(|t: Tracked| {
            log.borrow_mut().called.push(t.id);
            if t.id == 3 {
                Err(t.id)
            } else {
                Ok(Tracked::new(100 + t.id, &log))
            }
        });
        assert_eq!(result.err(), Some(3));
        assert_eq!(log.borrow().called, vec![0, 1, 2, 3]);
        // Inputs 0..4 once; outputs 100,101,102 (created for 0,1,2) freed by the Guard.
        assert_eq!(sorted_dropped(&log), vec![0, 1, 2, 3, 100, 101, 102]);
    }

    #[test]
    fn try_map_inline_err_drops_are_unique_via_box() {
        // Redundant with the counting above, but stated as an explicit
        // uniqueness invariant: no id is dropped twice. Under Miri the owned
        // `Box` in each `Tracked` turns a double-drop into a hard double-free.
        let log: LogRef = Default::default();
        let input: [Tracked; 6] = tracked_array(&log);
        let _: Result<[Tracked; 6], u32> = input.try_map_inline(|t: Tracked| {
            if t.id == 4 {
                Err(t.id)
            } else {
                Ok(Tracked::new(100 + t.id, &log))
            }
        });
        let dropped = log.borrow().dropped.clone();
        let mut unique = dropped.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(
            dropped.len(),
            unique.len(),
            "some element dropped more than once: {dropped:?}"
        );
    }

    // =====================================================================
    // Drop safety - panic path (Guard must run during unwind)
    // =====================================================================

    #[test]
    fn try_map_inline_panic_frees_outputs_and_inputs() {
        let log: LogRef = Default::default();
        let input: [Tracked; 5] = tracked_array(&log);

        let log_in = Rc::clone(&log);
        let caught = catch_unwind(AssertUnwindSafe(move || {
            let _: Result<[Tracked; 5], ()> = input.try_map_inline(|t: Tracked| {
                log_in.borrow_mut().called.push(t.id);
                if t.id == 3 {
                    panic!("boom"); // input `t` dropped during unwind of the closure frame
                }
                Ok(Tracked::new(100 + t.id, &log_in))
            });
            unreachable!("closure above always panics");
        }));

        assert!(
            caught.is_err(),
            "the panic must propagate out of try_map_inline"
        );
        assert_eq!(log.borrow().called, vec![0, 1, 2, 3]);
        // Inputs 0..5 dropped once (0,1,2 in closure; 3 unwinding the closure; 4 by IntoIter),
        // outputs 100,101,102 freed by the Guard during unwind. `mem::forget` was never reached.
        assert_eq!(sorted_dropped(&log), vec![0, 1, 2, 3, 4, 100, 101, 102]);
    }

    #[test]
    fn map_inline_panic_frees_outputs_and_inputs() {
        let log: LogRef = Default::default();
        let input: [Tracked; 4] = tracked_array(&log);

        let log_in = Rc::clone(&log);
        let caught = catch_unwind(AssertUnwindSafe(move || {
            let _out: [Tracked; 4] = input.map_inline(|t: Tracked| {
                log_in.borrow_mut().called.push(t.id);
                if t.id == 2 {
                    panic!("boom");
                }
                Tracked::new(100 + t.id, &log_in)
            });
            unreachable!("closure above always panics");
        }));

        assert!(caught.is_err());
        assert_eq!(log.borrow().called, vec![0, 1, 2]);
        // Inputs 0..4 once; outputs 100,101 (for 0,1) freed by the Guard.
        assert_eq!(sorted_dropped(&log), vec![0, 1, 2, 3, 100, 101]);
    }

    #[test]
    fn try_map_inline_panic_on_first_element_frees_all_inputs() {
        let log: LogRef = Default::default();
        let input: [Tracked; 3] = tracked_array(&log);

        let log_in = Rc::clone(&log);
        let caught = catch_unwind(AssertUnwindSafe(move || {
            let _: Result<[Tracked; 3], ()> = input.try_map_inline(|t: Tracked| {
                log_in.borrow_mut().called.push(t.id);
                panic!("boom on {}", t.id);
            });
            unreachable!();
        }));

        assert!(caught.is_err());
        assert_eq!(log.borrow().called, vec![0]);
        // No output produced; all inputs dropped once (0 unwinding the closure, 1,2 by IntoIter).
        assert_eq!(sorted_dropped(&log), vec![0, 1, 2]);
    }
}
