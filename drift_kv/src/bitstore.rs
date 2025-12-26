use crc32fast::Hasher as CrcHasher;
use parking_lot::RwLock;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use xxhash_rust::xxh3::xxh3_64;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;

const PAGE_SIZE: usize = 4096;
const PAGE_HEADER_SIZE: usize = 24;
const SLOT_SIZE: usize = 16;
const SLOTS_PER_PAGE: usize = (PAGE_SIZE - PAGE_HEADER_SIZE) / SLOT_SIZE;
const TOMBSTONE: u64 = u64::MAX;
pub const LOG_HEADER_SIZE: usize = 12;

#[repr(C)]
#[derive(Debug, Clone, Copy, IntoBytes, FromBytes, Immutable, KnownLayout, PartialEq)]
pub struct IndexSlot {
    pub data_offset: u64,
    pub key_fp: u32,
    pub data_len: u32,
}

impl IndexSlot {
    pub fn is_empty(&self) -> bool {
        self.data_len == 0
    }
    pub fn is_tombstone(&self) -> bool {
        self.data_offset == TOMBSTONE
    }
}

#[repr(C, align(4096))]
#[derive(Debug, Clone, Copy, IntoBytes, FromBytes, Immutable, KnownLayout)]
pub struct DiskPage {
    // --- Header (24 bytes) ---
    pub next_page_offset: u64, // Offset 0..8
    pub magic: u32,            // Offset 8..12
    pub checksum: u32,         // Offset 12..16 [NEW]
    pub count: u16,            // Offset 16..18
    pub _padding_1: u16,       // Offset 18..20
    pub _padding_2: u32,       // Offset 20..24 (Aligns body to 8 bytes)

    // --- Body (4064 bytes) ---
    // 254 * 16 = 4064 bytes
    pub slots: [IndexSlot; SLOTS_PER_PAGE],

    // --- Tail (8 bytes) ---
    // 4096 - 24 (Header) - 4064 (Slots) = 8 bytes remaining
    pub _padding_3: [u8; 8],
}

impl Default for DiskPage {
    fn default() -> Self {
        Self {
            magic: 0xDB5708E,
            next_page_offset: 0,
            checksum: 0,
            count: 0,
            _padding_1: 0,
            _padding_2: 0,
            slots: [IndexSlot {
                key_fp: 0,
                data_len: 0,
                data_offset: 0,
            }; SLOTS_PER_PAGE],
            _padding_3: [0; 8],
        }
    }
}

#[repr(C, align(4096))]
#[derive(Debug, Clone, Copy, IntoBytes, FromBytes, Immutable, KnownLayout)]
pub struct MetaPage {
    pub n_buckets: u64,       // 8
    pub split_ptr: u64,       // 8
    pub total_items: u64,     // 8
    pub first_free_page: u64, // 8
    pub magic: u32,           // 4
    pub version: u32,         // 4
    // Used: 40 bytes.
    // Padding: 4096 - 40 = 4056.
    pub _padding: [u8; 4056],
}

impl Default for MetaPage {
    fn default() -> Self {
        Self {
            magic: 0xDB5708E,
            version: 1,
            n_buckets: 4, // Start small (4 buckets)
            split_ptr: 0,
            first_free_page: 0,
            total_items: 0,
            _padding: [0; 4056],
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum StoreError {
    #[error("IO Error: {0}")]
    Io(#[from] io::Error),
    #[error("Data Corruption Detected")]
    CorruptedData,
    #[error("Index Page Corrupted")]
    CorruptedIndex,
}

pub struct BitStore {
    pub(crate) index_file: RwLock<Option<File>>,
    data_file: RwLock<Option<File>>,
    base_path: PathBuf,
    pub(crate) meta: RwLock<MetaPage>,
}

impl BitStore {
    pub fn new<P: AsRef<Path>>(base_path: P) -> Result<Self, StoreError> {
        let base_path = base_path.as_ref().to_path_buf();
        let (index_file, data_file, meta) = Self::open_files(&base_path)?;

        Ok(Self {
            index_file: RwLock::new(Some(index_file)),
            data_file: RwLock::new(Some(data_file)),
            base_path,
            meta: RwLock::new(meta),
            // num_buckets: buckets,
        })
    }

    fn open_files(base_path: &Path) -> Result<(File, File, MetaPage), StoreError> {
        let index_path = base_path.with_extension("idx");
        let data_path = base_path.with_extension("dat");

        let mut index_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(index_path)?;

        let data_file = OpenOptions::new()
            .read(true)
            .write(true)
            .truncate(false)
            .create(true)
            .open(data_path)?;

        // Check if new database
        let file_len = index_file.metadata()?.len();
        let meta = if file_len == 0 {
            // 1. Initialize empty MetaPage
            let meta = MetaPage::default();

            // 2. Write MetaPage to Offset 0
            index_file.write_all(meta.as_bytes())?;

            // 3. Pre-allocate the initial N buckets immediately after Page 0
            // We start with 4 buckets. Each bucket is 1 page.
            // Offset starts at PAGE_SIZE (4096).
            let initial_pages = meta.n_buckets;
            let mut empty_page = DiskPage::default();
            empty_page.checksum = Self::calculate_page_checksum(&empty_page);
            for _ in 0..initial_pages {
                index_file.write_all(empty_page.as_bytes())?;
            }
            index_file.flush()?;
            meta
        } else {
            // Load existing MetaPage
            let mut buf = [0u8; PAGE_SIZE];
            index_file.seek(SeekFrom::Start(0))?;
            index_file.read_exact(&mut buf)?;

            let meta = MetaPage::read_from_bytes(&buf).map_err(|_e| StoreError::CorruptedIndex)?;

            if meta.magic != 0xDB5708E {
                return Err(StoreError::CorruptedIndex);
            }
            meta
        };

        Ok((index_file, data_file, meta))
    }

    /// Calculates the physical file offset for a bucket based on the Linear Hashing formula.
    pub fn get_bucket_offset(&self, hash: u64) -> u64 {
        let meta = self.meta.read();

        // 1. Calculate Initial Bucket (Level 0)
        let mut bucket_idx = hash % meta.n_buckets; // h0(k) = k mod N

        // 2. Linear Hashing Check
        // If the result points to a bucket that has ALREADY been split (it is "below" the split pointer),
        // we must use the next level hash function to distinguish if it belongs
        // in the original slot or the new split slot.
        if bucket_idx < meta.split_ptr {
            bucket_idx = hash % (meta.n_buckets * 2); // h1(k) = k mod 2N
        }

        // 3. Convert Bucket Index to File Offset
        // Page 0 is Metadata. Actual buckets start at Page 1.
        // Formula: (Bucket Index + 1) * PAGE_SIZE
        (bucket_idx + 1) * (PAGE_SIZE as u64)
    }

    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, StoreError> {
        let hash = xxh3_64(key);
        let fingerprint = (hash >> 32) as u32;

        // [NEW] Use Linear Hashing Offset
        let mut current_offset = self.get_bucket_offset(hash);

        let i_lock = self.index_file.read();
        let file = i_lock.as_ref().ok_or(io::Error::other("Store Closed"))?;

        loop {
            let page = Self::read_page(file, current_offset)?;
            for slot in page.slots.iter() {
                if !slot.is_empty() && !slot.is_tombstone() && slot.key_fp == fingerprint {
                    // Check Data File
                    let d_lock = self.data_file.read();
                    let d_file = d_lock.as_ref().ok_or(io::Error::other("Store Closed"))?;

                    let mut buf = vec![0u8; slot.data_len as usize];
                    #[cfg(unix)]
                    d_file.read_exact_at(&mut buf, slot.data_offset)?;

                    // Verify CRC
                    let stored_crc = u32::from_le_bytes(buf[0..4].try_into().unwrap());
                    let k_len = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
                    let v_len = u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;

                    let mut hasher = CrcHasher::new();
                    hasher.update(&buf[12..12 + k_len]);
                    hasher.update(&buf[12 + k_len..12 + k_len + v_len]);

                    if hasher.finalize() != stored_crc {
                        return Err(StoreError::CorruptedData);
                    }

                    if &buf[12..12 + k_len] == key {
                        return Ok(Some(buf[12 + k_len..].to_vec()));
                    }
                }
            }

            if page.next_page_offset == 0 {
                return Ok(None);
            }
            current_offset = page.next_page_offset;
        }
    }

    pub fn remove(&self, key: &[u8]) -> Result<bool, StoreError> {
        let hash = xxh3_64(key);
        let fingerprint = (hash >> 32) as u32;
        let mut current_offset = self.get_bucket_offset(hash);

        let mut i_lock = self.index_file.write();
        let file = i_lock.as_mut().ok_or(io::Error::other("Store Closed"))?;

        loop {
            let mut page = Self::read_page(file, current_offset)?;

            for slot in page.slots.iter_mut() {
                if !slot.is_empty()
                    && !slot.is_tombstone()
                    && slot.key_fp == fingerprint
                    && self.verify_key_on_disk(slot, key)?
                {
                    slot.data_offset = TOMBSTONE;
                    // Decrement count so this page looks "emptier" to future inserts
                    if page.count > 0 {
                        page.count -= 1;
                    }
                    Self::write_page(file, current_offset, &page)?;
                    return Ok(true);
                }
            }

            if page.next_page_offset == 0 {
                return Ok(false);
            }
            current_offset = page.next_page_offset;
        }
    }

    fn recover_full_hash(&self, slot: &IndexSlot) -> Result<u64, StoreError> {
        let d_lock = self.data_file.read();
        let file = d_lock.as_ref().ok_or(StoreError::CorruptedData)?;

        // Optimization: Read Header (12 bytes) + a reasonable Key prefix (e.g., 64 bytes)
        // in one syscall to avoid double-dipping for small keys.
        // However, since keys are variable length, the safest robust method is:

        // 1. Read just the Key Length (4 bytes at offset + 4)
        let mut len_buf = [0u8; 4];
        #[cfg(unix)]
        file.read_exact_at(&mut len_buf, slot.data_offset + 4)?;
        let k_len = u32::from_le_bytes(len_buf) as usize;

        // 2. Allocate exact buffer
        let mut key_buf = vec![0u8; k_len];
        #[cfg(unix)]
        file.read_exact_at(&mut key_buf, slot.data_offset + 12)?;

        Ok(xxhash_rust::xxh3::xxh3_64(&key_buf))
    }

    fn split(&self) -> Result<(), StoreError> {
        let mut meta = self.meta.write();
        let mut i_lock = self.index_file.write();
        let file = i_lock.as_mut().ok_or(StoreError::CorruptedIndex)?;

        let old_idx = meta.split_ptr;
        let new_idx = meta.n_buckets + meta.split_ptr;

        // 1. Collect ALL slots
        let mut collected = Vec::new();
        let mut current_offset = (old_idx + 1) * (PAGE_SIZE as u64);

        // [NEW] Track if we have an overflow chain to free
        // We read the primary page first to see its 'next' pointer
        let primary_page = Self::read_page(file, current_offset)?;
        let chain_to_free = primary_page.next_page_offset;

        loop {
            let page = Self::read_page(file, current_offset)?;
            for slot in page.slots.iter() {
                if !slot.is_empty() && !slot.is_tombstone() {
                    collected.push(*slot);
                }
            }
            if page.next_page_offset == 0 {
                break;
            }
            current_offset = page.next_page_offset;
        }

        // 2. [NEW] Free the old overflow chain
        // We do this BEFORE writing new chains so we can potentially reuse these pages immediately!
        if chain_to_free != 0 {
            self.free_chain(file, &mut meta, chain_to_free)?;
        }

        // 3. Reset Old Bucket (Wipe Primary Page)
        // We do this explicitly to clear the primary page
        let empty_page = DiskPage::default();
        let old_bucket_offset = (old_idx + 1) * (PAGE_SIZE as u64);
        Self::write_page(file, old_bucket_offset, &empty_page)?;

        // 4. Partition
        let mut old_slots = Vec::new();
        let mut new_slots = Vec::new();

        for slot in collected {
            if let Ok(full_hash) = self.recover_full_hash(&slot) {
                let target = full_hash % (meta.n_buckets * 2);
                if target == old_idx {
                    old_slots.push(slot);
                } else {
                    new_slots.push(slot);
                }
            }
        }

        // 5. Rewrite Chains (using Allocator)
        // Note: We pass &mut meta because allocate_page needs to update free_list
        self.write_bucket_chain(file, &mut meta, old_idx, &old_slots)?;
        self.write_bucket_chain(file, &mut meta, new_idx, &new_slots)?;

        // 6. Update Meta
        meta.split_ptr += 1;
        if meta.split_ptr == meta.n_buckets {
            meta.n_buckets *= 2;
            meta.split_ptr = 0;
        }

        file.seek(SeekFrom::Start(0))?;
        file.write_all(meta.as_bytes())?;
        file.flush()?;

        Ok(())
    }

    fn read_page(file: &File, offset: u64) -> Result<DiskPage, StoreError> {
        let mut buf = [0u8; PAGE_SIZE];
        #[cfg(unix)]
        file.read_exact_at(&mut buf, offset)?;

        let page = DiskPage::read_from_bytes(&buf).map_err(|_| StoreError::CorruptedIndex)?;

        // 1. Basic Magic Check
        if page.magic == 0 {
            // Empty/Zeroed page is valid (count=0)
            return Ok(DiskPage::default());
        }
        if page.magic != 0xDB5708E {
            return Err(StoreError::CorruptedIndex);
        }

        // 2. [NEW] Integrity Check
        let expected = Self::calculate_page_checksum(&page);
        if page.checksum != expected {
            eprintln!(
                "Checksum Mismatch at offset {}: Expected {:x}, Found {:x}",
                offset, expected, page.checksum
            );
            return Err(StoreError::CorruptedIndex);
        }

        Ok(page)
    }

    fn write_page(file: &File, offset: u64, page: &DiskPage) -> Result<(), StoreError> {
        // 1. [NEW] Calculate and set checksum before writing
        // We need a mutable copy since the input reference is immutable
        let mut page_copy = *page; // DiskPage is Copy
        page_copy.checksum = Self::calculate_page_checksum(&page_copy);

        #[cfg(unix)]
        file.write_all_at(page_copy.as_bytes(), offset)?;
        Ok(())
    }

    // Updated Helper for O(N) Split
    fn write_bucket_chain(
        &self, // [CHANGED] Now requires self to call allocate_page
        file: &mut File,
        meta: &mut MetaPage, // [CHANGED] Needs meta to allocate
        bucket_idx: u64,
        slots: &[IndexSlot],
    ) -> Result<(), StoreError> {
        let primary_offset = (bucket_idx + 1) * (PAGE_SIZE as u64);

        let total_slots = slots.len();
        let pages_needed = if total_slots == 0 {
            1
        } else {
            (total_slots + SLOTS_PER_PAGE - 1) / SLOTS_PER_PAGE
        };

        // 1. Allocate offsets for all pages
        let mut offsets = Vec::with_capacity(pages_needed);
        offsets.push(primary_offset); // Primary page is always fixed

        // Allocate overflow pages (reusing free list if available)
        for _ in 1..pages_needed {
            let off = self.allocate_page(file, meta)?;
            offsets.push(off);
        }

        // 2. Write the pages
        for i in 0..pages_needed {
            let mut page = DiskPage::default();

            // Link to next page if one exists
            if i < pages_needed - 1 {
                page.next_page_offset = offsets[i + 1];
            } else {
                page.next_page_offset = 0; // Terminate
            }

            // Fill slots
            let start = i * SLOTS_PER_PAGE;
            let end = usize::min(start + SLOTS_PER_PAGE, total_slots);
            let chunk = &slots[start..end];

            for (j, slot) in chunk.iter().enumerate() {
                page.slots[j] = *slot;
            }
            page.count = chunk.len() as u16;

            Self::write_page(file, offsets[i], &page)?;
        }

        Ok(())
    }

    fn verify_key_on_disk(&self, slot: &IndexSlot, key: &[u8]) -> Result<bool, StoreError> {
        let d_lock = self.data_file.read();
        let file = d_lock.as_ref().unwrap();

        // Read K_LEN (4 bytes) and V_LEN (4 bytes)
        let mut head = [0u8; 8];
        #[cfg(unix)]
        file.read_exact_at(&mut head, slot.data_offset + 4)?;

        let k_len = u32::from_le_bytes(head[0..4].try_into().unwrap()) as usize;

        // Only check for length mismatch. k_len == 0 is valid!
        if k_len != key.len() {
            return Ok(false);
        }

        // Optimization: If key is empty, we don't need to read bytes to compare.
        if k_len == 0 {
            return Ok(true);
        }

        let mut stored_key = vec![0u8; k_len];
        #[cfg(unix)]
        file.read_exact_at(&mut stored_key, slot.data_offset + 12)?;

        Ok(stored_key == key)
    }

    pub fn sync(&self) -> Result<(), StoreError> {
        if let Some(f) = self.data_file.read().as_ref() {
            f.sync_data()?;
        }

        if let Some(f) = self.index_file.read().as_ref() {
            f.sync_data()?;
        }

        Ok(())
    }

    pub fn put(&self, key: &[u8], value: &[u8]) -> Result<(), StoreError> {
        let hash = xxh3_64(key);
        let fingerprint = (hash >> 32) as u32;

        // 1. Prepare Data
        let key_len = key.len() as u32;
        let val_len = value.len() as u32;
        let total_entry_len = (LOG_HEADER_SIZE as u32) + key_len + val_len;

        let mut crc = CrcHasher::new();
        crc.update(key);
        crc.update(value);
        let checksum = crc.finalize();

        // 2. Write Data
        let data_offset = {
            let mut d_lock = self.data_file.write();
            let file = d_lock.as_mut().ok_or(io::Error::other("Store Closed"))?;

            let offset = file.seek(SeekFrom::End(0))?;

            let mut buf = Vec::with_capacity(total_entry_len as usize);
            buf.extend_from_slice(&checksum.to_le_bytes());
            buf.extend_from_slice(&key_len.to_le_bytes());
            buf.extend_from_slice(&val_len.to_le_bytes());
            buf.extend_from_slice(key);
            buf.extend_from_slice(value);

            file.write_all(&buf)?;
            // file.flush()?; // Optimization: Rely on OS page cache
            offset
        };

        // 3. Update Index
        let start_offset = self.get_bucket_offset(hash);
        let mut i_lock = self.index_file.write();
        let file = i_lock.as_mut().ok_or(io::Error::other("Store Closed"))?;

        let mut current_offset = start_offset;
        let mut inserted = false;
        let mut loop_safety = 0;

        loop {
            // Safety: Detect cycles in corrupt files
            loop_safety += 1;
            if loop_safety > 10_000 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Cycle detected in bucket chain",
                )
                .into());
            }

            let mut page = Self::read_page(file, current_offset)?;
            let mut free_slot_idx = None;

            for i in 0..SLOTS_PER_PAGE {
                let slot = &mut page.slots[i];

                if free_slot_idx.is_none() && (slot.is_empty() || slot.is_tombstone()) {
                    free_slot_idx = Some(i);
                }

                if !slot.is_empty()
                    && !slot.is_tombstone()
                    && slot.key_fp == fingerprint
                    && self.verify_key_on_disk(slot, key)?
                {
                    slot.data_offset = data_offset;
                    slot.data_len = total_entry_len;
                    Self::write_page(file, current_offset, &page)?;
                    inserted = true;
                    break;
                }
            }
            if inserted {
                break;
            }

            // Case B: Insert into found free slot
            if let Some(idx) = free_slot_idx {
                let slot = &mut page.slots[idx];
                let was_empty = slot.is_empty();

                *slot = IndexSlot {
                    key_fp: fingerprint,
                    data_offset,
                    data_len: total_entry_len,
                };

                if was_empty {
                    page.count += 1;
                }

                Self::write_page(file, current_offset, &page)?;
                self.meta.write().total_items += 1;
                inserted = true;
                break;
            }

            // Case C: Overflow (Page Full) - Use Allocator!
            if page.next_page_offset == 0 {
                // [CHANGED] Use allocate_page instead of raw seek(End)
                // We lock meta briefly to allocate and increment item count atomically
                let new_offset = {
                    let mut meta = self.meta.write();
                    let off = self.allocate_page(file, &mut meta)?;
                    meta.total_items += 1;
                    off
                };

                // Link current -> new
                page.next_page_offset = new_offset;
                Self::write_page(file, current_offset, &page)?;

                // Initialize new page
                let mut new_page = DiskPage::default();
                new_page.slots[0] = IndexSlot {
                    key_fp: fingerprint,
                    data_offset,
                    data_len: total_entry_len,
                };
                new_page.count = 1;
                Self::write_page(file, new_offset, &new_page)?;

                inserted = true;
                break;
            }

            current_offset = page.next_page_offset;
        }

        // 4. Check Load Factor & Split
        drop(i_lock); // Drop lock before check

        if inserted {
            let should_split = {
                let meta = self.meta.read();
                let capacity = meta.n_buckets as f64 * SLOTS_PER_PAGE as f64;
                (meta.total_items as f64 / capacity) > 0.75
            };

            if should_split {
                let _ = self.split();
            }
        }

        Ok(())
    }

    fn allocate_page(&self, file: &mut File, meta: &mut MetaPage) -> Result<u64, StoreError> {
        if meta.first_free_page != 0 {
            // 1. Reuse: Pop from head of Free List
            let free_offset = meta.first_free_page;

            // Read the page to find the *next* free page
            // (We reuse the `next_page_offset` field of the free page as the list pointer)
            let page = Self::read_page(file, free_offset)?;

            // Update Head to point to the next one
            meta.first_free_page = page.next_page_offset;

            return Ok(free_offset);
        }

        // 2. Extend: No free pages, append to end of file
        let new_offset = file.seek(SeekFrom::End(0))?;

        // Physically reserve the space immediately to prevent offset overlap bugs
        file.set_len(new_offset + PAGE_SIZE as u64)?;

        Ok(new_offset)
    }

    /// Takes a chain of overflow pages (starting at `start_offset`) and pushes them
    /// onto the Free List.
    fn free_chain(
        &self,
        file: &mut File,
        meta: &mut MetaPage,
        start_offset: u64,
    ) -> Result<(), StoreError> {
        if start_offset == 0 {
            return Ok(());
        }

        // 1. Find the tail of the chain we are freeing
        let mut tail_offset = start_offset;
        loop {
            let page = Self::read_page(file, tail_offset)?;
            if page.next_page_offset == 0 {
                break;
            }
            tail_offset = page.next_page_offset;
        }

        // 2. Stitch: Tail points to Old Head
        // We read the tail page again to update it
        let mut tail_page = Self::read_page(file, tail_offset)?;
        tail_page.next_page_offset = meta.first_free_page;
        Self::write_page(file, tail_offset, &tail_page)?;

        // 3. Update Head: New Head is the start of this chain
        meta.first_free_page = start_offset;

        Ok(())
    }

    fn calculate_page_checksum(page: &DiskPage) -> u32 {
        let mut hasher = CrcHasher::new();
        // 1. Hash Header Fields (Skip checksum field)
        hasher.update(&page.next_page_offset.to_le_bytes());
        hasher.update(&page.magic.to_le_bytes());
        // Skip page.checksum!
        hasher.update(&page.count.to_le_bytes());
        hasher.update(&page._padding_1.to_le_bytes());
        hasher.update(&page._padding_2.to_le_bytes());

        // 2. Hash Slots
        // Since IndexSlot is IntoBytes, we can cast the whole array
        hasher.update(page.slots.as_bytes());

        // 3. Hash Tail
        hasher.update(&page._padding_3);

        hasher.finalize()
    }

    // Returns an iterator over all keys in the store
    pub fn iter(&self) -> BitStoreIterator<'_> {
        BitStoreIterator::new(self)
    }

    pub fn compact(&mut self) -> Result<(), StoreError> {
        // Make a NEW base name, not an extension.
        let mut compact_base = self.base_path.clone();
        let stem = compact_base
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("db");
        compact_base.set_file_name(format!("{stem}_compact"));

        let new_store = BitStore::new(&compact_base)?;

        // Copy live K/Vs via iterator
        for item in self.iter() {
            let (k, v) = item?;
            new_store.put(&k, &v)?;
        }

        // Close temp store (Windows rename needs closed handles)
        drop(new_store);

        // Close current handles
        *self.index_file.write() = None;
        *self.data_file.write() = None;

        // Compute paths
        let idx_path = self.base_path.with_extension("idx");
        let dat_path = self.base_path.with_extension("dat");
        let new_idx = compact_base.with_extension("idx");
        let new_dat = compact_base.with_extension("dat");

        // On Windows, rename-to-existing fails; remove first.
        let _ = std::fs::remove_file(&idx_path);
        let _ = std::fs::remove_file(&dat_path);

        std::fs::rename(&new_idx, &idx_path)?;
        std::fs::rename(&new_dat, &dat_path)?;

        // Reopen
        let (index_file, data_file, meta) = Self::open_files(&self.base_path)?;
        *self.index_file.write() = Some(index_file);
        *self.data_file.write() = Some(data_file);
        *self.meta.write() = meta;

        Ok(())
    }
}

// --- ITERATOR IMPLEMENTATION ---

pub struct BitStoreIterator<'a> {
    store: &'a BitStore,      // Holds a reference to the store
    meta: MetaPage,           // Snapshot of meta to know bounds
    current_bucket: u64,      // Which bucket index we are on (0..N+S)
    current_page_offset: u64, // File offset of current page
    current_page: DiskPage,   // Cache of current page
    current_slot: usize,      // Index within page (0..SLOTS_PER_PAGE)
    loaded_page: bool,        // Have we loaded the first page yet?
}

impl<'a> BitStoreIterator<'a> {
    // Helper to create the iterator
    fn new(store: &'a BitStore) -> Self {
        let meta = *store.meta.read();
        Self {
            store,
            meta,
            current_bucket: 0,
            current_page_offset: 0,
            current_page: DiskPage::default(),
            current_slot: 0,
            loaded_page: false,
        }
    }

    // Helper to advance to the next valid slot
    fn advance(&mut self) -> Result<bool, StoreError> {
        let file_lock = self.store.index_file.read();
        let file = file_lock.as_ref().ok_or(io::Error::other("Store Closed"))?;

        loop {
            // 1. Load Page if needed
            if !self.loaded_page {
                let total_buckets = self.meta.n_buckets + self.meta.split_ptr;
                if self.current_bucket >= total_buckets {
                    return Ok(false); // End of iteration
                }

                // Calculate offset for current bucket's primary page
                if self.current_page_offset == 0 {
                    self.current_page_offset = (self.current_bucket + 1) * (PAGE_SIZE as u64);
                }

                // Use the static read_page (we can't call self.store.read_page easily due to private method visibility
                // unless we made read_page pub(crate) or similar.
                // Assuming read_page is an associated function on BitStore:)
                self.current_page = BitStore::read_page(file, self.current_page_offset)?;
                self.loaded_page = true;
                self.current_slot = 0;
            }

            // 2. Scan Slots
            while self.current_slot < SLOTS_PER_PAGE {
                let slot = &self.current_page.slots[self.current_slot];
                if !slot.is_empty() && !slot.is_tombstone() {
                    // Found a valid item!
                    return Ok(true);
                }
                self.current_slot += 1;
            }

            // 3. Page Finished. Move to Next Page or Next Bucket
            if self.current_page.next_page_offset != 0 {
                self.current_page_offset = self.current_page.next_page_offset;
                self.loaded_page = false; // Trigger reload
            } else {
                self.current_bucket += 1;
                self.current_page_offset = 0;
                self.loaded_page = false;
            }
        }
    }
}

impl<'a> Iterator for BitStoreIterator<'a> {
    type Item = Result<(Vec<u8>, Vec<u8>), StoreError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.advance() {
            Ok(true) => {
                let slot = self.current_page.slots[self.current_slot];
                self.current_slot += 1;

                let d_lock = self.store.data_file.read();
                let d_file = match d_lock.as_ref() {
                    Some(f) => f,
                    None => return Some(Err(io::Error::other("Store Closed").into())),
                };

                // Read Key/Value
                let mut head = [0u8; 12];
                if let Err(e) = d_file.read_exact_at(&mut head, slot.data_offset) {
                    return Some(Err(e.into()));
                }

                let k_len = u32::from_le_bytes(head[4..8].try_into().unwrap()) as usize;
                let v_len = u32::from_le_bytes(head[8..12].try_into().unwrap()) as usize;

                let mut body = vec![0u8; k_len + v_len];
                if let Err(e) = d_file.read_exact_at(&mut body, slot.data_offset + 12) {
                    return Some(Err(e.into()));
                }

                let key = body[0..k_len].to_vec();
                let val = body[k_len..].to_vec();

                Some(Ok((key, val)))
            }
            Ok(false) => None, // Done
            Err(e) => Some(Err(e)),
        }
    }
}
