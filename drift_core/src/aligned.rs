use std::alloc::{Layout, alloc, dealloc, handle_alloc_error};
use std::ops::{Index, IndexMut};
use std::ptr::NonNull;
use std::slice;

/// A contiguous byte buffer guaranteed to be 64-byte aligned.
/// Critical for SIMD (AVX-512) operations which segfault on unaligned access.
pub struct AlignedBytes {
    ptr: NonNull<u8>,
    capacity: usize,
    len: usize,
}

impl AlignedBytes {
    const ALIGNMENT: usize = 64;

    pub fn new(capacity: usize) -> Self {
        // Ensure strictly positive capacity to avoid edge cases with alloc
        let capacity = capacity.max(64);
        let layout = Layout::from_size_align(capacity, Self::ALIGNMENT).unwrap();

        let ptr = unsafe {
            let p = alloc(layout);
            if p.is_null() {
                handle_alloc_error(layout);
            }
            NonNull::new_unchecked(p)
        };

        Self {
            ptr,
            capacity,
            len: 0,
        }
    }

    pub fn push(&mut self, byte: u8) {
        if self.len == self.capacity {
            self.grow();
        }

        unsafe {
            // Write to offset
            *self.ptr.as_ptr().add(self.len) = byte;
        }
        self.len += 1;
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    // --- The System Engineering Core: Safe Reallocation ---
    fn grow(&mut self) {
        let new_capacity = self.capacity * 2;
        let new_layout = Layout::from_size_align(new_capacity, Self::ALIGNMENT).unwrap();

        // We cannot use standard `realloc` safely because it doesn't guarantee
        // the *new* pointer maintains the specific 64-byte alignment we need
        // (standard realloc only guarantees alignment to max_align_t or similar).
        // To be safe and portable for SIMD, we alloc-copy-dealloc.

        unsafe {
            let new_ptr = alloc(new_layout);
            if new_ptr.is_null() {
                handle_alloc_error(new_layout);
            }

            // Copy old data
            std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_ptr, self.len);

            // Dealloc old
            let old_layout = Layout::from_size_align(self.capacity, Self::ALIGNMENT).unwrap();
            dealloc(self.ptr.as_ptr(), old_layout);

            self.ptr = NonNull::new_unchecked(new_ptr);
            self.capacity = new_capacity;
        }
    }

    /// SAFETY: Caller must ensure `new_len <= self.capacity` and that
    /// the memory up to `new_len` is initialized (or will be immediately).
    pub unsafe fn set_len(&mut self, new_len: usize) {
        assert!(new_len <= self.capacity);
        self.len = new_len;
    }

    pub fn as_mut_slice_full(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.capacity) }
    }
}

impl Drop for AlignedBytes {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.capacity, Self::ALIGNMENT).unwrap();
        unsafe {
            dealloc(self.ptr.as_ptr(), layout);
        }
    }
}

unsafe impl Send for AlignedBytes {}
unsafe impl Sync for AlignedBytes {}

impl Clone for AlignedBytes {
    fn clone(&self) -> Self {
        // 1. Allocate new aligned memory (Deep Copy)
        // We use Self::new to handle the complex layout/alloc logic safety
        let mut new_buffer = Self::new(self.capacity);

        // 2. Memcpy the data
        // Safety: Both pointers are valid, aligned, and we copy exactly 'len' bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_buffer.ptr.as_ptr(), self.len);
        }

        // 3. Sync metadata
        new_buffer.len = self.len;
        new_buffer
    }
}

// Optional: Debug is helpful for tests too
impl std::fmt::Debug for AlignedBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlignedBytes")
            .field("len", &self.len)
            .field("capacity", &self.capacity)
            .finish()
    }
}

impl Index<usize> for AlignedBytes {
    type Output = u8;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        // Panics if out of bounds, just like Vec
        &self.as_slice()[index]
    }
}

impl IndexMut<usize> for AlignedBytes {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_slice()[index]
    }
}
