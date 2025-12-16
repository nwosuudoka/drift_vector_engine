use std::alloc::{Layout, alloc, dealloc};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::slice;

/// A 4KB-aligned memory block.
/// Essential for O_DIRECT I/O and ALP Vectorization (1024 floats = 4KB).
///
/// ALP naturally processes vectors of 1024 values[cite: 91].
/// 1024 * 4 bytes (f32) = 4096 bytes.
/// This matches the OS Page Size perfectly.
pub struct PageBlock {
    ptr: NonNull<u8>,
    layout: Layout,
    size: usize,
}

impl PageBlock {
    // pub const SIZE: usize = 4096; // Standard Page Size
    pub const DEFAULT_SIZE: usize = 16 * 1024; // 16KB
    pub const ALIGNMENT: usize = 4096; // 4KB (Required for O_DIRECT)

    /// Allocate a default 16KB block.
    pub fn new() -> Self {
        Self::with_size(Self::DEFAULT_SIZE)
    }

    /// Allocate a custom-sized block (must be multiple of 4KB).
    pub fn with_size(size: usize) -> Self {
        assert!(
            size > 0 && size % Self::ALIGNMENT == 0,
            "Block size must be multiple of 4KB"
        );

        let layout =
            Layout::from_size_align(size, Self::ALIGNMENT).expect("Failed to create page layout");

        let ptr = unsafe {
            let p = alloc(layout);
            if p.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            // Zero-initialize (Security + Determinism)
            std::ptr::write_bytes(p, 0, size);
            NonNull::new_unchecked(p)
        };

        Self { ptr, layout, size }
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
    }

    pub fn load(&mut self, src: &[u8]) {
        assert!(src.len() <= self.size, "Source data exceeds Block size");
        self.as_mut_slice()[..src.len()].copy_from_slice(src);
        if src.len() < self.size {
            self.as_mut_slice()[src.len()..].fill(0);
        }
    }
}

impl Drop for PageBlock {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

impl Deref for PageBlock {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
    }
}

impl DerefMut for PageBlock {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
    }
}

unsafe impl Send for PageBlock {}
unsafe impl Sync for PageBlock {}

#[cfg(test)]
mod tests {
    use crate::block::PageBlock;

    #[test]
    fn test_large_block_alignment() {
        // Test 16KB (Default)
        let block = PageBlock::new();
        let addr = block.as_ptr() as usize;

        println!("16KB Block Address: 0x{:x}", addr);
        assert_eq!(addr % 4096, 0, "16KB Block is NOT 4KB aligned!");
        assert_eq!(block.len(), 16384);

        // Test 64KB (Jumbo Frame)
        let jumbo = PageBlock::with_size(64 * 1024);
        let j_addr = jumbo.as_ptr() as usize;
        assert_eq!(j_addr % 4096, 0, "64KB Block is NOT 4KB aligned!");
        assert_eq!(jumbo.len(), 65536);
    }

    #[test]
    #[should_panic]
    fn test_invalid_alignment_panics() {
        // This should fail because 100 bytes is not a multiple of 4KB
        PageBlock::with_size(100);
    }
}
