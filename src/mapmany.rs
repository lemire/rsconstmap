//! Batched lookups: resolving a slice of keys at once is faster than a loop
//! over [`ConstMap::map`].

use xxhash_rust::xxh64::xxh64;

use crate::{fingerprint, mixsplit, ConstMap, VerifiedConstMap, NOT_FOUND};

/// How many keys the batched lookups hash before gathering any values. Hashing
/// a whole block first lets the array accesses of every key in the block be in
/// flight at once, instead of each key's loads waiting behind the hashing of
/// the one before it. That memory-level parallelism is the only reason a batch
/// beats a loop over `map`.
///
/// Eight is a compromise between two machines that disagree. On an Apple M4
/// Max it was the best of 4, 8 and 16 in both the cold and the hot regime of
/// `benches/constmap_bench.rs`. On an Intel Xeon Gold 6548N, 16 was about 17%
/// better than 8 on a cold `VerifiedConstMap` batch but about 8% worse on a hot
/// one, while 16 was clearly worse than 8 hot on the M4 Max. Eight wins or
/// nearly wins everywhere; 16 wins one case and loses another.
const BATCH_BLOCK: usize = 8;

impl ConstMap {
    /// Look up every key in `keys` and return the values in a newly allocated
    /// `Vec` of the same length, where `result[i]` corresponds to `keys[i]`.
    ///
    /// This is equivalent to calling [`ConstMap::map`] on each key in turn, and
    /// carries the same caveat: a key that was not in the original set yields
    /// an undefined value. Use [`VerifiedConstMap::map_many`] if you need
    /// missing keys reported.
    ///
    /// Batching is faster than the equivalent loop because it overlaps the
    /// memory accesses of several keys. How much it wins depends on how much
    /// memory latency there is to hide, so it varies with the machine and with
    /// how much of the map fits in cache; on a map small enough to sit in a
    /// large last-level cache it can come out roughly even.
    pub fn map_many<K: AsRef<str>>(&self, keys: &[K]) -> Vec<u64> {
        let mut dst = vec![0u64; keys.len()];
        self.map_many_into(&mut dst, keys);
        dst
    }

    /// [`ConstMap::map_many`] writing into a caller-provided slice, so that a
    /// repeated batch need not allocate. It fills `dst[..keys.len()]` and
    /// leaves the rest of `dst` alone.
    ///
    /// # Panics
    ///
    /// Panics if `dst` is shorter than `keys`.
    pub fn map_many_into<K: AsRef<str>>(&self, dst: &mut [u64], keys: &[K]) {
        assert!(
            dst.len() >= keys.len(),
            "constmap: map_many_into destination is shorter than keys"
        );

        let mut h0 = [0u32; BATCH_BLOCK];
        let mut h1 = [0u32; BATCH_BLOCK];
        let mut h2 = [0u32; BATCH_BLOCK];

        let blocks = keys.len() / BATCH_BLOCK * BATCH_BLOCK;
        let mut i = 0;
        while i < blocks {
            let block = &keys[i..i + BATCH_BLOCK];
            for (j, key) in block.iter().enumerate() {
                let hash = mixsplit(xxh64(key.as_ref().as_bytes(), 0), self.seed);
                let (a, b, c) = self.get_hash_from_hash(hash);
                h0[j] = a;
                h1[j] = b;
                h2[j] = c;
            }
            let out = &mut dst[i..i + BATCH_BLOCK];
            for j in 0..BATCH_BLOCK {
                out[j] = self.data[h0[j] as usize]
                    ^ self.data[h1[j] as usize]
                    ^ self.data[h2[j] as usize];
            }
            i += BATCH_BLOCK;
        }
        // Tail: fewer than BATCH_BLOCK keys left.
        for (slot, key) in dst[blocks..keys.len()].iter_mut().zip(&keys[blocks..]) {
            *slot = self.map(key.as_ref());
        }
    }
}

impl VerifiedConstMap {
    /// Look up every key in `keys` and return the values in a newly allocated
    /// `Vec` of the same length, where `result[i]` corresponds to `keys[i]`.
    /// Keys that were not in the original set yield [`NOT_FOUND`], exactly as
    /// [`VerifiedConstMap::map`] does.
    ///
    /// Each lookup touches two arrays rather than one, so there is more memory
    /// latency for batching to hide.
    pub fn map_many<K: AsRef<str>>(&self, keys: &[K]) -> Vec<u64> {
        let mut dst = vec![0u64; keys.len()];
        self.map_many_into(&mut dst, keys);
        dst
    }

    /// [`VerifiedConstMap::map_many`] writing into a caller-provided slice, so
    /// that a repeated batch need not allocate. It fills `dst[..keys.len()]`
    /// and leaves the rest of `dst` alone.
    ///
    /// # Panics
    ///
    /// Panics if `dst` is shorter than `keys`.
    pub fn map_many_into<K: AsRef<str>>(&self, dst: &mut [u64], keys: &[K]) {
        assert!(
            dst.len() >= keys.len(),
            "constmap: map_many_into destination is shorter than keys"
        );
        if self.data.is_empty() {
            dst[..keys.len()].fill(NOT_FOUND);
            return;
        }

        let mut h0 = [0u32; BATCH_BLOCK];
        let mut h1 = [0u32; BATCH_BLOCK];
        let mut h2 = [0u32; BATCH_BLOCK];
        let mut hashes = [0u64; BATCH_BLOCK];

        let blocks = keys.len() / BATCH_BLOCK * BATCH_BLOCK;
        let mut i = 0;
        while i < blocks {
            let block = &keys[i..i + BATCH_BLOCK];
            for (j, key) in block.iter().enumerate() {
                let hash = mixsplit(xxh64(key.as_ref().as_bytes(), 0), self.seed);
                let (a, b, c) = self.get_hash_from_hash(hash);
                hashes[j] = hash;
                h0[j] = a;
                h1[j] = b;
                h2[j] = c;
            }
            let out = &mut dst[i..i + BATCH_BLOCK];
            for j in 0..BATCH_BLOCK {
                let fp = self.checks[h0[j] as usize]
                    ^ self.checks[h1[j] as usize]
                    ^ self.checks[h2[j] as usize];
                out[j] = if fp == fingerprint(hashes[j]) {
                    self.data[h0[j] as usize]
                        ^ self.data[h1[j] as usize]
                        ^ self.data[h2[j] as usize]
                } else {
                    NOT_FOUND
                };
            }
            i += BATCH_BLOCK;
        }
        // Tail: fewer than BATCH_BLOCK keys left.
        for (slot, key) in dst[blocks..keys.len()].iter_mut().zip(&keys[blocks..]) {
            *slot = self.map(key.as_ref());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_data(n: usize) -> (Vec<String>, Vec<u64>) {
        let keys: Vec<String> = (0..n).map(|i| format!("key-{}", i)).collect();
        let values: Vec<u64> = (0..n).map(|i| (i * 7) as u64).collect();
        (keys, values)
    }

    fn refs(keys: &[String]) -> Vec<&str> {
        keys.iter().map(|s| s.as_str()).collect()
    }

    /// Sizes that straddle the block boundary, so the tail path is exercised.
    const SIZES: [usize; 6] = [0, 1, 7, 8, 9, 1000];

    #[test]
    fn test_map_many_matches_map() {
        let (keys, values) = build_data(20_000);
        let cm = ConstMap::new(&refs(&keys), &values).unwrap();

        for n in SIZES {
            let batch = &keys[..n];
            let got = cm.map_many(batch);
            assert_eq!(got.len(), n);
            for (i, k) in batch.iter().enumerate() {
                assert_eq!(got[i], cm.map(k), "n={}: map_many({}) mismatch", n, k);
            }
        }
    }

    #[test]
    fn test_map_many_values() {
        let keys = vec!["apple", "banana", "cherry"];
        let values = vec![100u64, 200, 300];
        let cm = ConstMap::new(&keys, &values).unwrap();
        assert_eq!(cm.map_many(&keys), values);
    }

    #[test]
    fn test_verified_map_many_matches_map() {
        let (keys, values) = build_data(20_000);
        let vm = VerifiedConstMap::new(&refs(&keys), &values).unwrap();

        for n in SIZES {
            let batch = &keys[..n];
            let got = vm.map_many(batch);
            assert_eq!(got.len(), n);
            for (i, k) in batch.iter().enumerate() {
                assert_eq!(got[i], vm.map(k), "n={}: map_many({}) mismatch", n, k);
            }
        }
    }

    #[test]
    fn test_verified_map_many_reports_missing() {
        let (keys, values) = build_data(1000);
        let vm = VerifiedConstMap::new(&refs(&keys), &values).unwrap();

        // Interleave present and absent keys, across and within blocks.
        let mut batch: Vec<String> = Vec::new();
        let mut want: Vec<u64> = Vec::new();
        for i in 0..100 {
            batch.push(keys[i].clone());
            want.push(values[i]);
            batch.push(format!("missing-{}", i));
            want.push(NOT_FOUND);
        }

        assert_eq!(vm.map_many(&batch), want);
    }

    #[test]
    fn test_verified_map_many_empty_map() {
        let vm = VerifiedConstMap::new(&[], &[]).unwrap();
        let batch: Vec<&str> = (0..20).map(|_| "anything").collect();
        assert_eq!(vm.map_many(&batch), vec![NOT_FOUND; 20]);
    }

    #[test]
    fn test_map_many_no_keys() {
        let (keys, values) = build_data(100);
        let cm = ConstMap::new(&refs(&keys), &values).unwrap();
        let vm = VerifiedConstMap::new(&refs(&keys), &values).unwrap();

        let none: [&str; 0] = [];
        assert!(cm.map_many(&none).is_empty());
        assert!(vm.map_many(&none).is_empty());

        // An empty batch must not touch a destination of any size.
        let mut dst = [7u64; 4];
        cm.map_many_into(&mut dst, &none);
        vm.map_many_into(&mut dst, &none);
        assert_eq!(dst, [7u64; 4]);
    }

    #[test]
    fn test_map_many_into_leaves_tail_alone() {
        let (keys, values) = build_data(1000);
        let cm = ConstMap::new(&refs(&keys), &values).unwrap();
        let vm = VerifiedConstMap::new(&refs(&keys), &values).unwrap();

        let batch = &keys[..10];

        let mut dst = vec![0xdeadu64; 20];
        cm.map_many_into(&mut dst, batch);
        assert_eq!(&dst[..10], &cm.map_many(batch)[..]);
        assert!(dst[10..].iter().all(|&v| v == 0xdead), "tail was clobbered");

        let mut dst = vec![0xdeadu64; 20];
        vm.map_many_into(&mut dst, batch);
        assert_eq!(&dst[..10], &vm.map_many(batch)[..]);
        assert!(dst[10..].iter().all(|&v| v == 0xdead), "tail was clobbered");
    }

    #[test]
    #[should_panic(expected = "shorter than keys")]
    fn test_map_many_into_short_destination() {
        let (keys, values) = build_data(100);
        let cm = ConstMap::new(&refs(&keys), &values).unwrap();
        let mut dst = [0u64; 3];
        cm.map_many_into(&mut dst, &keys[..4]);
    }

    #[test]
    #[should_panic(expected = "shorter than keys")]
    fn test_verified_map_many_into_short_destination() {
        let (keys, values) = build_data(100);
        let vm = VerifiedConstMap::new(&refs(&keys), &values).unwrap();
        let mut dst = [0u64; 3];
        vm.map_many_into(&mut dst, &keys[..4]);
    }
}
