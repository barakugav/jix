use core::ptr::{null_mut, NonNull};
use std::alloc::{alloc, alloc_zeroed, dealloc, handle_alloc_error, realloc, Layout};

use crate::util::aligned_vec::RuntimeAlign;

pub struct RawAlignedBytes {
    pub ptr: NonNull<u8>,
    pub capacity: usize,
    pub align: RuntimeAlign,
}

impl Drop for RawAlignedBytes {
    #[inline]
    fn drop(&mut self) {
        if self.capacity > 0 {
            // SAFETY: memory was allocated with alloc::alloc::alloc
            unsafe {
                dealloc(
                    self.ptr.as_ptr() as *mut u8,
                    Layout::from_size_align_unchecked(self.capacity, self.align.alignment()),
                )
            }
        }
    }
}

pub fn capacity_overflow() -> ! {
    panic!("capacity overflow")
}

impl RawAlignedBytes {
    const MIN_NON_ZERO_CAP: usize = 8;

    /// # Safety
    ///
    /// `align` must be a power of two.
    /// `align` must be greater than or equal to `core::mem::align_of::<u8>()`.
    #[inline]
    pub unsafe fn new_unchecked(align: usize) -> Self {
        unsafe { Self::from_raw_parts(null_mut::<u8>().wrapping_add(align), 0, align) }
    }

    /// # Safety
    ///
    /// `align` must be a power of two.
    /// `align` must be greater than or equal to `core::mem::align_of::<u8>()`.
    #[inline]
    pub unsafe fn with_capacity_unchecked(capacity: usize, align: usize) -> Self {
        if capacity == 0 {
            unsafe { Self::new_unchecked(align) }
        } else {
            Self {
                ptr: unsafe { NonNull::new_unchecked(with_capacity_unchecked(capacity, align)) },
                capacity,
                align: RuntimeAlign::new(align),
            }
        }
    }

    /// # Safety
    ///
    /// `align` must be a power of two.
    /// `align` must be greater than or equal to `core::mem::align_of::<u8>()`.
    #[inline]
    pub unsafe fn with_capacity_unchecked_zeroed(capacity: usize, align: usize) -> Self {
        if capacity == 0 {
            unsafe { Self::new_unchecked(align) }
        } else {
            Self {
                ptr: unsafe {
                    NonNull::new_unchecked(with_capacity_unchecked_zeroed(capacity, align))
                },
                capacity,
                align: RuntimeAlign::new(align),
            }
        }
    }

    pub unsafe fn grow_amortized(&mut self, len: usize, additional: usize) {
        debug_assert!(additional > 0);
        if self.capacity == 0 {
            *self = unsafe {
                Self::with_capacity_unchecked(
                    additional.max(Self::MIN_NON_ZERO_CAP),
                    self.align.alignment(),
                )
            };
            return;
        }

        let new_cap = match len.checked_add(additional) {
            Some(cap) => cap,
            None => capacity_overflow(),
        };

        // self.cap * 2 can't overflow because it's less than isize::MAX
        let new_cap = new_cap.max(self.capacity * 2);
        let new_cap = new_cap.max(Self::MIN_NON_ZERO_CAP);

        let ptr = unsafe {
            grow_unchecked(
                self.as_mut_ptr(),
                self.capacity,
                new_cap,
                self.align.alignment(),
            )
        };

        self.capacity = new_cap;
        self.ptr = unsafe { NonNull::new_unchecked(ptr) };
    }

    pub unsafe fn grow_exact(&mut self, len: usize, additional: usize) {
        debug_assert!(additional > 0);

        if self.capacity == 0 {
            *self = unsafe { Self::with_capacity_unchecked(additional, self.align.alignment()) };
            return;
        }

        let new_cap = match len.checked_add(additional) {
            Some(cap) => cap,
            None => capacity_overflow(),
        };

        let ptr = unsafe {
            grow_unchecked(
                self.as_mut_ptr(),
                self.capacity,
                new_cap,
                self.align.alignment(),
            )
        };

        self.capacity = new_cap;
        self.ptr = unsafe { NonNull::new_unchecked(ptr) };
    }

    pub unsafe fn shrink_to(&mut self, len: usize) {
        debug_assert!(len < self.capacity());
        let old_capacity = self.capacity;
        let align = self.align;
        let old_ptr = self.ptr.as_ptr() as *mut u8;

        // this cannot overflow or exceed isize::MAX bytes since len < cap and the same was true
        // for cap
        let old_layout =
            unsafe { Layout::from_size_align_unchecked(old_capacity, align.alignment()) };

        let ptr = unsafe { realloc(old_ptr, old_layout, len) };
        let ptr = ptr as *mut u8;
        self.capacity = len;
        self.ptr = unsafe { NonNull::new_unchecked(ptr) };
    }

    #[inline]
    pub unsafe fn from_raw_parts(ptr: *mut u8, capacity: usize, align: usize) -> Self {
        Self {
            ptr: unsafe { NonNull::new_unchecked(ptr) },
            capacity,
            align: RuntimeAlign::new(align),
        }
    }

    /// Returns the capacity of the vector.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline]
    pub fn align(&self) -> usize {
        self.align.alignment()
    }

    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

pub unsafe fn with_capacity_unchecked(capacity: usize, align: usize) -> *mut u8 {
    debug_assert!(capacity > 0);
    if !unsafe { is_valid_alloc(capacity, align) } {
        capacity_overflow();
    }

    let layout = unsafe { Layout::from_size_align_unchecked(capacity, align) };
    let ptr = unsafe { alloc(layout) };
    if ptr.is_null() {
        handle_alloc_error(layout);
    }
    ptr
}

pub unsafe fn with_capacity_unchecked_zeroed(capacity: usize, align: usize) -> *mut u8 {
    debug_assert!(capacity > 0);
    if !unsafe { is_valid_alloc(capacity, align) } {
        capacity_overflow();
    }

    let layout = unsafe { Layout::from_size_align_unchecked(capacity, align) };
    let ptr = unsafe { alloc_zeroed(layout) };
    if ptr.is_null() {
        handle_alloc_error(layout);
    }
    ptr
}

unsafe fn grow_unchecked(
    old_ptr: *mut u8,
    old_capacity: usize,
    new_capacity: usize,
    align: usize,
) -> *mut u8 {
    if !unsafe { is_valid_alloc(new_capacity, align) } {
        capacity_overflow();
    }

    // can't overflow because we already allocated this much
    let old_size_bytes = old_capacity;
    let old_layout = unsafe { Layout::from_size_align_unchecked(old_size_bytes, align) };

    let ptr = unsafe { realloc(old_ptr, old_layout, new_capacity) };

    if ptr.is_null() {
        let new_layout = unsafe { Layout::from_size_align_unchecked(new_capacity, align) };
        handle_alloc_error(new_layout);
    }

    ptr
}

#[inline]
unsafe fn is_valid_alloc(alloc_size: usize, align: usize) -> bool {
    debug_assert!(align.is_power_of_two());
    // "size, when rounded up to the nearest multiple of align, must not overflow isize"
    let max = (isize::MAX as usize) - (align - 1);
    alloc_size <= max
}
