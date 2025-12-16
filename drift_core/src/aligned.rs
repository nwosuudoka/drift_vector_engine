use std::alloc::{Layout, alloc, dealloc};
use std::ops::Deref;
use std::ptr::NonNull;

/// A Byte Buffer guaranteed to be aligned to 64 bytes.
/// Essential for AVX-512 / SIMD operations.
pub struct AlignedBytes {
    ptr: NonNull<u8>,
    layout: Layout,
    len: usize,
    cap: usize,
}

impl AlignedBytes {
    pub fn new(capacity: usize) -> Self {
        // Ensure alignment is 64 bytes for AVX-512
        let layout = Layout::from_size_align(capacity, 64).unwrap();
        let ptr = unsafe {
            let p = alloc(layout);
            if p.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            NonNull::new_unchecked(p)
        };

        Self {
            ptr,
            layout,
            len: 0,
            cap: capacity,
        }
    }

    pub fn push(&mut self, byte: u8) {
        if self.len == self.cap {
            panic!("Resizing AlignedVec not implemented for safety in V1");
        }
        unsafe {
            *self.ptr.as_ptr().add(self.len) = byte;
            self.len += 1;
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for AlignedBytes {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

// Allow it to act like a slice
impl Deref for AlignedBytes {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

unsafe impl Send for AlignedBytes {}
unsafe impl Sync for AlignedBytes {}
