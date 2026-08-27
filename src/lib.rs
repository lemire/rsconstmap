//! A static map from strings to `u64` values using binary fuse filter construction.
//!
//! Given a set of (key, value) pairs, it builds a compact array such that for any key
//! in the set, `data[h0(key)] XOR data[h1(key)] XOR data[h2(key)] == value`.
//!
//! Lookup is extremely fast: one xxhash call plus three array accesses and two XORs.
//! The data structure is immutable after construction.
//!
//! Two map types are provided: [`ConstMap`], which returns an undefined value for a
//! key that was not in the original set, and [`VerifiedConstMap`], which stores a
//! fingerprint per key and returns [`NOT_FOUND`] instead, for twice the memory.
//!
//! Both support batched lookups (`map_many` / `map_many_into`), which overlap the
//! memory accesses of several keys and are faster than a loop over `map`, and both
//! serialize to a checksummed binary format (`write_to` / `read_from`,
//! `save_to_file` / `load_from_file`).

use std::collections::HashMap;

use xxhash_rust::xxh64::xxh64;

mod mapmany;
#[cfg(test)]
mod report;
mod serialize;

/// Sentinel value returned by [`VerifiedConstMap::map`] when the key was not
/// in the original set.
pub const NOT_FOUND: u64 = u64::MAX;

// ---------- hash helpers ----------

#[inline]
fn murmur64(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;
    h
}

#[inline]
pub(crate) fn mixsplit(key: u64, seed: u64) -> u64 {
    murmur64(key.wrapping_add(seed))
}

#[inline]
fn splitmix64(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[inline]
pub(crate) fn fingerprint(hash: u64) -> u64 {
    hash ^ (hash >> 32)
}

// ---------- parameter calculations ----------

fn calculate_segment_length(size: u32) -> u32 {
    if size == 0 {
        return 4;
    }
    let exp = ((size as f64).ln() / (3.33_f64).ln() + 2.25).floor() as u32;
    1u32 << exp
}

fn calculate_size_factor(size: u32) -> f64 {
    f64::max(
        1.125,
        0.875 + 0.25 * (1_000_000_f64).ln() / (size as f64).ln(),
    )
}

// ---------- ConstMap ----------

/// An immutable map from strings to `u64` values.
///
/// Lookup of keys not in the original set returns an undefined value.
pub struct ConstMap {
    pub(crate) seed: u64,
    pub(crate) segment_length: u32,
    pub(crate) segment_length_mask: u32,
    pub(crate) segment_count: u32,
    pub(crate) segment_count_length: u32,
    pub(crate) data: Vec<u64>,
}

impl ConstMap {
    #[inline]
    pub(crate) fn get_hash_from_hash(&self, hash: u64) -> (u32, u32, u32) {
        let hi = ((hash as u128 * self.segment_count_length as u128) >> 64) as u32;
        let h0 = hi;
        let mut h1 = h0.wrapping_add(self.segment_length);
        let mut h2 = h1.wrapping_add(self.segment_length);
        h1 ^= (hash >> 18) as u32 & self.segment_length_mask;
        h2 ^= hash as u32 & self.segment_length_mask;
        (h0, h1, h2)
    }

    fn initialize_parameters(&mut self, size: u32) {
        self.segment_length = calculate_segment_length(size);
        if self.segment_length > 262144 {
            self.segment_length = 262144;
        }
        self.segment_length_mask = self.segment_length - 1;
        let capacity = if size > 1 {
            let size_factor = calculate_size_factor(size);
            (size as f64 * size_factor).round() as u32
        } else {
            0
        };
        let mut total_segment_count =
            (capacity + self.segment_length - 1) / self.segment_length;
        if total_segment_count < 3 {
            total_segment_count = 3;
        }
        self.segment_count = total_segment_count - 2;
        self.segment_count_length = self.segment_count * self.segment_length;
        self.data = vec![0u64; (total_segment_count * self.segment_length) as usize];
    }

    /// Build a `ConstMap` from a set of string keys and their associated `u64` values.
    ///
    /// The two slices must have equal length. Keys must be unique.
    pub fn new(keys: &[&str], values: &[u64]) -> Result<Self, String> {
        if keys.len() != values.len() {
            return Err("constmap: keys and values must have equal length".into());
        }
        let n = keys.len();
        if n == 0 {
            return Ok(ConstMap {
                seed: 0,
                segment_length: 0,
                segment_length_mask: 0,
                segment_count: 0,
                segment_count_length: 0,
                data: Vec::new(),
            });
        }

        let hashed: Vec<u64> = keys.iter().map(|k| xxh64(k.as_bytes(), 0)).collect();

        let mut hash_to_value: HashMap<u64, u64> = HashMap::with_capacity(n);

        let size = n as u32;
        let mut cm = ConstMap {
            seed: 0,
            segment_length: 0,
            segment_length_mask: 0,
            segment_count: 0,
            segment_count_length: 0,
            data: Vec::new(),
        };
        cm.initialize_parameters(size);
        let capacity = cm.data.len() as u32;

        let mut rngcounter: u64 = 1;
        cm.seed = splitmix64(&mut rngcounter);

        let mut alone = vec![0u32; capacity as usize];
        let mut t2count = vec![0u8; capacity as usize];
        let mut t2hash = vec![0u64; capacity as usize];
        let mut reverse_h = vec![0u8; size as usize];
        let mut reverse_order = vec![0u64; (size + 1) as usize];
        reverse_order[size as usize] = 1;

        let max_iterations = 100;

        for iterations in 0..=max_iterations {
            if iterations > max_iterations {
                return Err(
                    "constmap: failed to construct map, possible duplicate keys".into(),
                );
            }

            if size > 4 && size < 1_000_000 {
                match iterations % 4 {
                    2 => {
                        cm.segment_length /= 2;
                        cm.segment_length_mask = cm.segment_length - 1;
                        cm.segment_count = cm.segment_count * 2 + 2;
                        cm.segment_count_length =
                            cm.segment_count * cm.segment_length;
                    }
                    3 => {
                        cm.segment_length *= 2;
                        cm.segment_length_mask = cm.segment_length - 1;
                        cm.segment_count = cm.segment_count / 2 - 1;
                        cm.segment_count_length =
                            cm.segment_count * cm.segment_length;
                    }
                    _ => {}
                }
            }

            let mut block_bits: u32 = 1;
            while (1u32 << block_bits) < cm.segment_count {
                block_bits += 1;
            }
            let mut start_pos = vec![0u32; 1 << block_bits];
            for i in 0..start_pos.len() {
                start_pos[i] = ((i as u64 * size as u64) >> block_bits) as u32;
            }
            for &key in &hashed {
                let hash = mixsplit(key, cm.seed);
                let mut segment_index = hash >> (64 - block_bits);
                while reverse_order[start_pos[segment_index as usize] as usize] != 0 {
                    segment_index += 1;
                    segment_index &= (1u64 << block_bits) - 1;
                }
                reverse_order[start_pos[segment_index as usize] as usize] = hash;
                start_pos[segment_index as usize] += 1;
            }

            let mut has_error = false;
            for i in 0..size {
                let hash = reverse_order[i as usize];
                let (index1, index2, index3) = cm.get_hash_from_hash(hash);
                t2count[index1 as usize] += 4;
                t2hash[index1 as usize] ^= hash;
                t2count[index2 as usize] += 4;
                t2count[index2 as usize] ^= 1;
                t2hash[index2 as usize] ^= hash;
                t2count[index3 as usize] += 4;
                t2count[index3 as usize] ^= 2;
                t2hash[index3 as usize] ^= hash;

                if t2hash[index1 as usize]
                    & t2hash[index2 as usize]
                    & t2hash[index3 as usize]
                    == 0
                {
                    if (t2hash[index1 as usize] == 0 && t2count[index1 as usize] == 8)
                        || (t2hash[index2 as usize] == 0
                            && t2count[index2 as usize] == 8)
                        || (t2hash[index3 as usize] == 0
                            && t2count[index3 as usize] == 8)
                    {
                        return Err("constmap: duplicate key hash detected".into());
                    }
                }
                if t2count[index1 as usize] < 4
                    || t2count[index2 as usize] < 4
                    || t2count[index3 as usize] < 4
                {
                    has_error = true;
                }
            }

            if has_error {
                for i in 0..size as usize {
                    reverse_order[i] = 0;
                }
                for i in 0..capacity as usize {
                    t2count[i] = 0;
                    t2hash[i] = 0;
                }
                cm.seed = splitmix64(&mut rngcounter);
                continue;
            }

            // Peeling: find singletons and peel them off.
            let mut qsize: usize = 0;
            for i in 0..capacity {
                alone[qsize] = i;
                if (t2count[i as usize] >> 2) == 1 {
                    qsize += 1;
                }
            }

            let mut stacksize: u32 = 0;
            let seg_len = cm.segment_length;
            let seg_len_to_minus_seg_len_x2 = seg_len ^ (0u32.wrapping_sub(2u32.wrapping_mul(seg_len)));
            while qsize > 0 {
                qsize -= 1;
                let index = alone[qsize];
                if (t2count[index as usize] >> 2) == 1 {
                    let hash = t2hash[index as usize];
                    let found = t2count[index as usize] & 3;
                    reverse_h[stacksize as usize] = found;
                    reverse_order[stacksize as usize] = hash;
                    stacksize += 1;

                    let h01 = (hash >> 18) as u32 & cm.segment_length_mask;
                    let h02 = hash as u32 & cm.segment_length_mask;

                    // Branchless index computation using bitmasks.
                    // found is 0, 1, or 2; these masks select the right offsets.
                    let is0 = 0u32.wrapping_sub(((found.wrapping_sub(1)) >> 7) as u32);
                    let is1 = 0u32.wrapping_sub((found & 1) as u32);
                    let is2 = 0u32.wrapping_sub((found >> 1) as u32);

                    let mut other_index1 =
                        index.wrapping_add(seg_len ^ (seg_len_to_minus_seg_len_x2 & is2));
                    let mut other_index2 =
                        index.wrapping_sub(seg_len ^ (seg_len_to_minus_seg_len_x2 & is0));

                    other_index1 ^= (h01 & !is2) ^ (h02 & !is0);
                    other_index2 ^= (h01 & !is0) ^ (h02 & !is1);

                    let f1 = (is0 & 1 | is1 & 2) as u8;
                    let f2 = (is0 & 2 | is2 & 1) as u8;

                    alone[qsize] = other_index1;
                    if (t2count[other_index1 as usize] >> 2) == 2 {
                        qsize += 1;
                    }
                    t2count[other_index1 as usize] -= 4;
                    t2count[other_index1 as usize] ^= f1;
                    t2hash[other_index1 as usize] ^= hash;

                    alone[qsize] = other_index2;
                    if (t2count[other_index2 as usize] >> 2) == 2 {
                        qsize += 1;
                    }
                    t2count[other_index2 as usize] -= 4;
                    t2count[other_index2 as usize] ^= f2;
                    t2hash[other_index2 as usize] ^= hash;
                }
            }

            if stacksize == size {
                hash_to_value.clear();
                for i in 0..n {
                    hash_to_value.insert(mixsplit(hashed[i], cm.seed), values[i]);
                }
                break;
            }

            for i in 0..size as usize {
                reverse_order[i] = 0;
            }
            for i in 0..capacity as usize {
                t2count[i] = 0;
                t2hash[i] = 0;
            }
            cm.seed = splitmix64(&mut rngcounter);
        }

        // Assignment phase: walk the stack in reverse, assigning values.
        let mut h012 = [0u32; 5];
        for i in (0..size as usize).rev() {
            let hash = reverse_order[i];
            let val = hash_to_value[&hash];
            let (index1, index2, index3) = cm.get_hash_from_hash(hash);
            let found = reverse_h[i] as usize;
            h012[0] = index1;
            h012[1] = index2;
            h012[2] = index3;
            h012[3] = h012[0];
            h012[4] = h012[1];
            cm.data[h012[found] as usize] = val
                ^ cm.data[h012[found + 1] as usize]
                ^ cm.data[h012[found + 2] as usize];
        }

        Ok(cm)
    }

    /// Return the `u64` value associated with the given key.
    ///
    /// The key must have been present in the original set passed to [`ConstMap::new`].
    /// If the key was not in the original set, the return value is undefined.
    #[inline]
    pub fn map(&self, key: &str) -> u64 {
        let hash = mixsplit(xxh64(key.as_bytes(), 0), self.seed);
        let (h0, h1, h2) = self.get_hash_from_hash(hash);
        self.data[h0 as usize] ^ self.data[h1 as usize] ^ self.data[h2 as usize]
    }
}


impl std::fmt::Debug for ConstMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConstMap")
            .field("seed", &self.seed)
            .field("segment_length", &self.segment_length)
            .field("segment_count", &self.segment_count)
            .field("data_len", &self.data.len())
            .finish()
    }
}

// ---------- VerifiedConstMap ----------

/// Like [`ConstMap`] but additionally stores a verification fingerprint for each key.
///
/// On lookup, the fingerprint is checked; if it does not match, [`NOT_FOUND`] is
/// returned. This detects (with high probability) lookups of keys that were not in
/// the original set, at the cost of doubling memory usage (~18 bytes/key instead of ~9).
pub struct VerifiedConstMap {
    pub(crate) seed: u64,
    pub(crate) segment_length: u32,
    pub(crate) segment_length_mask: u32,
    pub(crate) segment_count: u32,
    pub(crate) segment_count_length: u32,
    pub(crate) data: Vec<u64>,
    pub(crate) checks: Vec<u64>,
}

impl VerifiedConstMap {
    #[inline]
    pub(crate) fn get_hash_from_hash(&self, hash: u64) -> (u32, u32, u32) {
        let hi = ((hash as u128 * self.segment_count_length as u128) >> 64) as u32;
        let h0 = hi;
        let mut h1 = h0.wrapping_add(self.segment_length);
        let mut h2 = h1.wrapping_add(self.segment_length);
        h1 ^= (hash >> 18) as u32 & self.segment_length_mask;
        h2 ^= hash as u32 & self.segment_length_mask;
        (h0, h1, h2)
    }

    /// Build a `VerifiedConstMap` from a set of string keys and their associated `u64` values.
    ///
    /// The two slices must have equal length. Keys must be unique.
    /// Values equal to [`NOT_FOUND`] are allowed but cannot be distinguished from a
    /// missing key on lookup.
    pub fn new(keys: &[&str], values: &[u64]) -> Result<Self, String> {
        if keys.len() != values.len() {
            return Err("constmap: keys and values must have equal length".into());
        }
        let n = keys.len();
        if n == 0 {
            return Ok(VerifiedConstMap {
                seed: 0,
                segment_length: 0,
                segment_length_mask: 0,
                segment_count: 0,
                segment_count_length: 0,
                data: Vec::new(),
                checks: Vec::new(),
            });
        }

        let hashed: Vec<u64> = keys.iter().map(|k| xxh64(k.as_bytes(), 0)).collect();
        let mut hash_to_value: HashMap<u64, u64> = HashMap::with_capacity(n);

        let size = n as u32;
        let mut vm = VerifiedConstMap {
            seed: 0,
            segment_length: 0,
            segment_length_mask: 0,
            segment_count: 0,
            segment_count_length: 0,
            data: Vec::new(),
            checks: Vec::new(),
        };

        // Initialize parameters.
        vm.segment_length = calculate_segment_length(size);
        if vm.segment_length > 262144 {
            vm.segment_length = 262144;
        }
        vm.segment_length_mask = vm.segment_length - 1;
        let cap0 = if size > 1 {
            let size_factor = calculate_size_factor(size);
            (size as f64 * size_factor).round() as u32
        } else {
            0
        };
        let mut total_segment_count =
            (cap0 + vm.segment_length - 1) / vm.segment_length;
        if total_segment_count < 3 {
            total_segment_count = 3;
        }
        vm.segment_count = total_segment_count - 2;
        vm.segment_count_length = vm.segment_count * vm.segment_length;
        let array_len = (total_segment_count * vm.segment_length) as usize;
        vm.data = vec![0u64; array_len];
        vm.checks = vec![0u64; array_len];

        let capacity = array_len as u32;

        let mut rngcounter: u64 = 1;
        vm.seed = splitmix64(&mut rngcounter);

        let mut alone = vec![0u32; capacity as usize];
        let mut t2count = vec![0u8; capacity as usize];
        let mut t2hash = vec![0u64; capacity as usize];
        let mut reverse_h = vec![0u8; size as usize];
        let mut reverse_order = vec![0u64; (size + 1) as usize];
        reverse_order[size as usize] = 1;

        let max_iterations = 100;

        for iterations in 0..=max_iterations {
            if iterations > max_iterations {
                return Err(
                    "constmap: failed to construct map, possible duplicate keys".into(),
                );
            }

            if size > 4 && size < 1_000_000 {
                match iterations % 4 {
                    2 => {
                        vm.segment_length /= 2;
                        vm.segment_length_mask = vm.segment_length - 1;
                        vm.segment_count = vm.segment_count * 2 + 2;
                        vm.segment_count_length =
                            vm.segment_count * vm.segment_length;
                    }
                    3 => {
                        vm.segment_length *= 2;
                        vm.segment_length_mask = vm.segment_length - 1;
                        vm.segment_count = vm.segment_count / 2 - 1;
                        vm.segment_count_length =
                            vm.segment_count * vm.segment_length;
                    }
                    _ => {}
                }
            }

            let mut block_bits: u32 = 1;
            while (1u32 << block_bits) < vm.segment_count {
                block_bits += 1;
            }
            let mut start_pos = vec![0u32; 1usize << block_bits];
            for i in 0..start_pos.len() {
                start_pos[i] = ((i as u64 * size as u64) >> block_bits) as u32;
            }
            for &key in &hashed {
                let hash = mixsplit(key, vm.seed);
                let mut segment_index = hash >> (64 - block_bits);
                while reverse_order[start_pos[segment_index as usize] as usize] != 0 {
                    segment_index += 1;
                    segment_index &= (1u64 << block_bits) - 1;
                }
                reverse_order[start_pos[segment_index as usize] as usize] = hash;
                start_pos[segment_index as usize] += 1;
            }

            let mut has_error = false;
            for i in 0..size {
                let hash = reverse_order[i as usize];
                let (index1, index2, index3) = vm.get_hash_from_hash(hash);
                t2count[index1 as usize] += 4;
                t2hash[index1 as usize] ^= hash;
                t2count[index2 as usize] += 4;
                t2count[index2 as usize] ^= 1;
                t2hash[index2 as usize] ^= hash;
                t2count[index3 as usize] += 4;
                t2count[index3 as usize] ^= 2;
                t2hash[index3 as usize] ^= hash;

                if t2hash[index1 as usize]
                    & t2hash[index2 as usize]
                    & t2hash[index3 as usize]
                    == 0
                {
                    if (t2hash[index1 as usize] == 0
                        && t2count[index1 as usize] == 8)
                        || (t2hash[index2 as usize] == 0
                            && t2count[index2 as usize] == 8)
                        || (t2hash[index3 as usize] == 0
                            && t2count[index3 as usize] == 8)
                    {
                        return Err("constmap: duplicate key hash detected".into());
                    }
                }
                if t2count[index1 as usize] < 4
                    || t2count[index2 as usize] < 4
                    || t2count[index3 as usize] < 4
                {
                    has_error = true;
                }
            }

            if has_error {
                for i in 0..size as usize {
                    reverse_order[i] = 0;
                }
                for i in 0..capacity as usize {
                    t2count[i] = 0;
                    t2hash[i] = 0;
                }
                vm.seed = splitmix64(&mut rngcounter);
                continue;
            }

            let mut qsize: usize = 0;
            for i in 0..capacity {
                alone[qsize] = i;
                if (t2count[i as usize] >> 2) == 1 {
                    qsize += 1;
                }
            }

            let mut stacksize: u32 = 0;
            let seg_len = vm.segment_length;
            let seg_len_to_minus_seg_len_x2 =
                seg_len ^ (0u32.wrapping_sub(2u32.wrapping_mul(seg_len)));
            while qsize > 0 {
                qsize -= 1;
                let index = alone[qsize];
                if (t2count[index as usize] >> 2) == 1 {
                    let hash = t2hash[index as usize];
                    let found = t2count[index as usize] & 3;
                    reverse_h[stacksize as usize] = found;
                    reverse_order[stacksize as usize] = hash;
                    stacksize += 1;

                    let h01 = (hash >> 18) as u32 & vm.segment_length_mask;
                    let h02 = hash as u32 & vm.segment_length_mask;

                    let is0 = 0u32.wrapping_sub(((found.wrapping_sub(1)) >> 7) as u32);
                    let is1 = 0u32.wrapping_sub(found as u32 & 1);
                    let is2 = 0u32.wrapping_sub(found as u32 >> 1);

                    let mut other_index1 = index
                        .wrapping_add(seg_len ^ (seg_len_to_minus_seg_len_x2 & is2));
                    let mut other_index2 = index
                        .wrapping_sub(seg_len ^ (seg_len_to_minus_seg_len_x2 & is0));

                    other_index1 ^= (h01 & !is2) ^ (h02 & !is0);
                    other_index2 ^= (h01 & !is0) ^ (h02 & !is1);

                    let f1 = (is0 & 1 | is1 & 2) as u8;
                    let f2 = (is0 & 2 | is2 & 1) as u8;

                    alone[qsize] = other_index1;
                    if (t2count[other_index1 as usize] >> 2) == 2 {
                        qsize += 1;
                    }
                    t2count[other_index1 as usize] -= 4;
                    t2count[other_index1 as usize] ^= f1;
                    t2hash[other_index1 as usize] ^= hash;

                    alone[qsize] = other_index2;
                    if (t2count[other_index2 as usize] >> 2) == 2 {
                        qsize += 1;
                    }
                    t2count[other_index2 as usize] -= 4;
                    t2count[other_index2 as usize] ^= f2;
                    t2hash[other_index2 as usize] ^= hash;
                }
            }

            if stacksize == size {
                hash_to_value.clear();
                for i in 0..n {
                    hash_to_value.insert(mixsplit(hashed[i], vm.seed), values[i]);
                }
                break;
            }

            for i in 0..size as usize {
                reverse_order[i] = 0;
            }
            for i in 0..capacity as usize {
                t2count[i] = 0;
                t2hash[i] = 0;
            }
            vm.seed = splitmix64(&mut rngcounter);
        }

        // Assignment phase.
        let mut h012 = [0u32; 5];
        for i in (0..size as usize).rev() {
            let hash = reverse_order[i];
            let val = hash_to_value[&hash];
            let fp = fingerprint(hash);
            let (index1, index2, index3) = vm.get_hash_from_hash(hash);
            let found = reverse_h[i] as usize;
            h012[0] = index1;
            h012[1] = index2;
            h012[2] = index3;
            h012[3] = h012[0];
            h012[4] = h012[1];
            vm.data[h012[found] as usize] = val
                ^ vm.data[h012[found + 1] as usize]
                ^ vm.data[h012[found + 2] as usize];
            vm.checks[h012[found] as usize] = fp
                ^ vm.checks[h012[found + 1] as usize]
                ^ vm.checks[h012[found + 2] as usize];
        }

        Ok(vm)
    }

    /// Return the `u64` value associated with the given key.
    ///
    /// If the key was not in the original set, [`NOT_FOUND`] is returned.
    #[inline]
    pub fn map(&self, key: &str) -> u64 {
        if self.data.is_empty() {
            return NOT_FOUND;
        }
        let hash = mixsplit(xxh64(key.as_bytes(), 0), self.seed);
        let (h0, h1, h2) = self.get_hash_from_hash(hash);
        let fp = self.checks[h0 as usize]
            ^ self.checks[h1 as usize]
            ^ self.checks[h2 as usize];
        if fp != fingerprint(hash) {
            return NOT_FOUND;
        }
        self.data[h0 as usize] ^ self.data[h1 as usize] ^ self.data[h2 as usize]
    }
}

impl std::fmt::Debug for VerifiedConstMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VerifiedConstMap")
            .field("seed", &self.seed)
            .field("segment_length", &self.segment_length)
            .field("segment_count", &self.segment_count)
            .field("data_len", &self.data.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let keys = vec!["apple", "banana", "cherry", "date", "elderberry"];
        let values = vec![100u64, 200, 300, 400, 500];

        let cm = ConstMap::new(&keys, &values).unwrap();

        for (i, &k) in keys.iter().enumerate() {
            assert_eq!(cm.map(k), values[i], "Map({}) mismatch", k);
        }
    }

    #[test]
    fn test_empty() {
        let cm = ConstMap::new(&[], &[]).unwrap();
        assert!(cm.data.is_empty());
    }

    #[test]
    fn test_large() {
        let n = 100_000;
        let keys: Vec<String> = (0..n).map(|i| format!("key-{}", i)).collect();
        let key_refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
        let values: Vec<u64> = (0..n).map(|i| (i * 7) as u64).collect();

        let cm = ConstMap::new(&key_refs, &values).unwrap();

        for (i, k) in keys.iter().enumerate() {
            assert_eq!(cm.map(k), values[i], "Map({}) mismatch", k);
        }
    }

    #[test]
    fn test_mismatched_lengths() {
        let result = ConstMap::new(&["a"], &[1, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_random_values() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let n = 50_000;
        let keys: Vec<String> = (0..n)
            .map(|i| format!("random-key-{}-{}", i, rng.gen::<u64>()))
            .collect();
        let key_refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
        let values: Vec<u64> = (0..n).map(|_| rng.gen()).collect();

        let cm = ConstMap::new(&key_refs, &values).unwrap();

        for (i, k) in keys.iter().enumerate() {
            assert_eq!(cm.map(k), values[i], "Map({}) mismatch", k);
        }
    }

    #[test]
    fn test_verified_basic() {
        let keys = vec!["apple", "banana", "cherry", "date", "elderberry"];
        let values = vec![100u64, 200, 300, 400, 500];

        let vm = VerifiedConstMap::new(&keys, &values).unwrap();

        for (i, &k) in keys.iter().enumerate() {
            assert_eq!(vm.map(k), values[i], "Map({}) mismatch", k);
        }
    }

    #[test]
    fn test_verified_missing() {
        let keys = vec!["apple", "banana", "cherry"];
        let values = vec![100u64, 200, 300];

        let vm = VerifiedConstMap::new(&keys, &values).unwrap();

        let missing = ["grape", "kiwi", "mango", "pear", "plum"];
        for &k in &missing {
            assert_eq!(vm.map(k), NOT_FOUND, "Map({}) should be NOT_FOUND", k);
        }
    }

    #[test]
    fn test_verified_large() {
        let n = 100_000;
        let keys: Vec<String> = (0..n).map(|i| format!("key-{}", i)).collect();
        let key_refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
        let values: Vec<u64> = (0..n).map(|i| (i * 7) as u64).collect();

        let vm = VerifiedConstMap::new(&key_refs, &values).unwrap();

        for (i, k) in keys.iter().enumerate() {
            assert_eq!(vm.map(k), values[i], "Map({}) mismatch", k);
        }

        for i in 0..10_000 {
            let k = format!("missing-{}", i);
            assert_eq!(vm.map(&k), NOT_FOUND, "Map({}) should be NOT_FOUND", k);
        }
    }

    #[test]
    fn test_verified_empty() {
        let vm = VerifiedConstMap::new(&[], &[]).unwrap();
        assert_eq!(vm.map("anything"), NOT_FOUND);
    }
}
