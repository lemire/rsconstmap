//! A reporting test that measures lookup time and memory usage across map
//! sizes. It is `#[ignore]`d because the largest size builds three
//! ten-million-key maps; run it with
//!
//! ```text
//! cargo test --release -- --ignored --nocapture scaling_table
//! ```
//!
//! Memory is measured with a counting global allocator rather than estimated
//! from capacities, so what it reports is what the process actually asked for.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static FREED: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        FREED.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATED.fetch_add(new_size, Ordering::Relaxed);
        FREED.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Runs `f` and returns its value along with the number of bytes it allocated
/// and did not free -- that is, the live heap footprint of what it built.
fn measure_alloc<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let a0 = ALLOCATED.load(Ordering::Relaxed);
    let f0 = FREED.load(Ordering::Relaxed);
    let v = f();
    let net =
        (ALLOCATED.load(Ordering::Relaxed) - a0).saturating_sub(FREED.load(Ordering::Relaxed) - f0);
    (v, net)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ConstMap, VerifiedConstMap};
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    use std::collections::HashMap;
    use std::hint::black_box;
    use std::time::Instant;

    /// Every key exactly once, in random order, each body re-allocated in that
    /// order. See the note on `make_query_order` in `benches/constmap_bench.rs`
    /// for why both halves of that matter.
    fn make_query_order(keys: &[String], seed: u64) -> Vec<String> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut perm: Vec<usize> = (0..keys.len()).collect();
        perm.shuffle(&mut rng);
        perm.into_iter().map(|j| keys[j].clone()).collect()
    }

    /// Times `iterations` lookups after a shorter warmup pass, so that the
    /// first structure measured in the process is not charged for cold code
    /// paths the later ones find warm.
    fn ns_per_lookup(iterations: usize, mut lookup: impl FnMut(usize) -> u64) -> f64 {
        let mut warm = 0u64;
        for i in 0..iterations / 10 {
            warm ^= lookup(i);
        }
        black_box(warm);

        let start = Instant::now();
        let mut sink = 0u64;
        for i in 0..iterations {
            sink ^= lookup(i);
        }
        black_box(sink);
        start.elapsed().as_nanos() as f64 / iterations as f64
    }

    #[test]
    #[ignore = "reporting test; run with --ignored --nocapture"]
    fn scaling_table() {
        let sizes = [10_000usize, 100_000, 1_000_000, 10_000_000];
        let iterations = 1_000_000usize;

        println!(
            "| keys | ConstMap | VerifiedConstMap | HashMap | ConstMap bytes/key | Verified bytes/key | HashMap bytes/key |"
        );
        println!("|---|---|---|---|---|---|---|");

        for n in sizes {
            let keys: Vec<String> = (0..n).map(|i| format!("key-{}", i)).collect();
            let values: Vec<u64> = (0..n).map(|i| i as u64).collect();
            let key_refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();

            let (cm, cm_bytes) = measure_alloc(|| ConstMap::new(&key_refs, &values).unwrap());
            let (vm, vm_bytes) =
                measure_alloc(|| VerifiedConstMap::new(&key_refs, &values).unwrap());
            // Borrowed keys, so the measurement is the table itself and not a
            // second copy of the key text. A ConstMap needs no key text at all;
            // a HashMap needs it kept alive somewhere either way.
            let (hm, hm_bytes) = measure_alloc(|| {
                let mut m: HashMap<&str, u64> = HashMap::with_capacity(n);
                for (i, &k) in key_refs.iter().enumerate() {
                    m.insert(k, values[i]);
                }
                m
            });

            let queries = make_query_order(&keys, 1);

            let cm_ns = ns_per_lookup(iterations, |i| cm.map(&queries[i % n]));
            let vm_ns = ns_per_lookup(iterations, |i| vm.map(&queries[i % n]));
            let hm_ns = ns_per_lookup(iterations, |i| {
                hm.get(queries[i % n].as_str()).copied().unwrap_or(0)
            });

            let per_key = |b: usize| b as f64 / n as f64;
            println!(
                "| {} | {:.1} ns | {:.1} ns | {:.1} ns | {:.1} | {:.1} | {:.1} |",
                n,
                cm_ns,
                vm_ns,
                hm_ns,
                per_key(cm_bytes),
                per_key(vm_bytes),
                per_key(hm_bytes),
            );
        }
    }
}
