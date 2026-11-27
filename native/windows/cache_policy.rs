use std::collections::VecDeque;

mod layer_loop;

// Layer quotas fitted on four coding requests at 75,264 MiB. The native caller
// checks every cached layer before enabling this profile.
const QUOTAS: [i32; 40] = [
    192, 218, 192, 192, 192, 193, 175, 214, 186, 172, 168, 174, 189, 209, 172, 209, 175, 209, 208,
    217, 176, 188, 188, 220, 188, 191, 191, 195, 191, 190, 190, 190, 190, 193, 209, 194, 193, 187,
    193, 193,
];

#[no_mangle]
pub extern "C" fn n1x_seed_quota(layer: usize, slots: i32, layers: usize, stride: usize) -> i32 {
    let geometry = match layer {
        0 => (192, 1, 8_478_720),
        1 | 5..=9 => (1158, 6, 8_017_920),
        2..=4 => (578, 3, 7_649_280),
        10..=19 => (1930, 10, 9_861_120),
        20..=24 => (964, 5, 10_598_400),
        25..=34 => (1930, 10, 12_072_960),
        35..=39 => (964, 5, 11_335_680),
        _ => return 0,
    };
    if (slots, layers, stride) == geometry {
        QUOTAS[layer]
    } else {
        0
    }
}

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SnapshotClass {
    Recent,
    Frequent,
}

struct Ghost {
    key: [u8; 32],
    bytes: usize,
    class: SnapshotClass,
}

#[derive(Default)]
pub struct SnapshotHistory {
    ghosts: VecDeque<Ghost>,
    recent_target: usize,
}

impl SnapshotHistory {
    fn record(&mut self, key: [u8; 32], bytes: usize, class: SnapshotClass, budget: usize) {
        self.ghosts.retain(|g| g.key != key);
        if bytes > 0 && bytes <= budget {
            self.ghosts.push_back(Ghost { key, bytes, class });
        }
        self.recent_target = self.recent_target.min(budget);
        let mut total: u128 = self.ghosts.iter().map(|g| g.bytes as u128).sum();
        while total > budget as u128 || self.ghosts.len() > 256 {
            total -= self.ghosts.pop_front().unwrap().bytes as u128;
        }
    }

    fn revisit(&mut self, key: &[u8; 32], budget: usize) -> bool {
        let Some(i) = self.ghosts.iter().position(|g| g.key == *key) else {
            return false;
        };
        let hit = &self.ghosts[i];
        let mut bytes = [0u128; 2];
        for g in &self.ghosts {
            bytes[g.class as usize] += g.bytes as u128;
        }
        let class = hit.class as usize;
        let step = (hit.bytes as u128 * bytes[1 - class] / bytes[class])
            .max(hit.bytes as u128)
            .min(budget as u128) as usize;
        self.recent_target = if hit.class == SnapshotClass::Frequent {
            self.recent_target.saturating_sub(step)
        } else {
            self.recent_target.saturating_add(step).min(budget)
        };
        self.ghosts.remove(i);
        true
    }
}

#[no_mangle]
pub extern "C" fn n1x_history_new() -> *mut SnapshotHistory {
    Box::into_raw(Box::default())
}

// The server owns one history per RAM cache. Calls run on its slot scheduler
// thread. A key points to a complete SHA-256 digest for the duration of a call.
#[no_mangle]
pub unsafe extern "C" fn n1x_history_free(history: *mut SnapshotHistory) {
    if !history.is_null() {
        drop(unsafe { Box::from_raw(history) });
    }
}

#[no_mangle]
pub extern "C" fn n1x_victim_class(
    history: &SnapshotHistory,
    recent_bytes: usize,
    budget: usize,
) -> SnapshotClass {
    if recent_bytes <= history.recent_target.min(budget) {
        SnapshotClass::Frequent
    } else {
        SnapshotClass::Recent
    }
}

#[no_mangle]
pub extern "C" fn n1x_record_eviction(
    history: &mut SnapshotHistory,
    key: &[u8; 32],
    bytes: usize,
    class: SnapshotClass,
    budget: usize,
) {
    history.record(*key, bytes, class, budget);
}

#[no_mangle]
pub extern "C" fn n1x_revisit(
    history: &mut SnapshotHistory,
    key: &[u8; 32],
    budget: usize,
) -> bool {
    history.revisit(key, budget)
}

#[cfg(test)]
mod tests {
    use super::*;
    use SnapshotClass::{Frequent, Recent};

    #[test]
    fn quotas_preserve_each_bank_budget() {
        for (layers, slots) in [
            (vec![0], 192),
            (vec![1, 5, 6, 7, 8, 9], 1158),
            ((2..5).collect(), 578),
            ((10..20).collect(), 1930),
            ((20..25).collect(), 964),
            ((25..35).collect(), 1930),
            ((35..40).collect(), 964),
        ] {
            assert_eq!(
                layers.iter().map(|&i| QUOTAS[i]).sum::<i32>(),
                slots / layers.len() as i32 * layers.len() as i32
            );
        }
        assert_eq!(n1x_seed_quota(1, 1158, 6, 8_017_920), 218);
        assert_eq!(n1x_seed_quota(1, 1157, 6, 8_017_920), 0);
        assert_eq!(n1x_seed_quota(40, 192, 1, 8_478_720), 0);
    }

    #[test]
    fn shrink_and_cap_ghost_history() {
        let mut h = SnapshotHistory::default();
        for i in 0..300u32 {
            let mut key = [0; 32];
            key[..4].copy_from_slice(&i.to_le_bytes());
            h.record(key, 1, Recent, 1024);
        }
        assert_eq!(h.ghosts.len(), 256);
        h.record([0; 32], 1024, Recent, 3);
        assert_eq!(h.ghosts.len(), 3);
    }

    #[test]
    fn ghost_hits_adapt_in_both_directions_and_expire() {
        let mut h = SnapshotHistory::default();
        h.record([1; 32], 2, Recent, 12);
        h.record([2; 32], 8, Frequent, 12);
        assert!(h.revisit(&[1; 32], 12));
        assert_eq!(h.recent_target, 8);
        assert!(matches!(n1x_victim_class(&h, 8, 12), Frequent));
        assert!(matches!(n1x_victim_class(&h, 9, 12), Recent));
        assert!(!h.revisit(&[1; 32], 12));
        assert!(h.revisit(&[2; 32], 12));
        assert_eq!(h.recent_target, 0);
        h.record([3; 32], 9, Recent, 12);
        h.record([4; 32], 9, Frequent, 12);
        assert!(!h.revisit(&[3; 32], 12));
        h.record([4; 32], 6, Recent, 12);
        assert_eq!(h.ghosts.len(), 1);
        assert!(h.revisit(&[4; 32], 12));
        assert_eq!(h.recent_target, 6);
    }
}
