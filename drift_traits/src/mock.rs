use crate::TombstoneView;

#[derive(Debug, Clone)]
pub struct NoTombstones;

impl TombstoneView for NoTombstones {
    fn contains(&self, _id: u64) -> bool {
        false
    }
    fn len(&self) -> usize {
        0
    }
}
