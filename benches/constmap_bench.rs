//! Benchmarks for `constmap`.
//!
//! Three groups:
//!
//! * `lookup` -- single-key lookup throughput against `std::collections::HashMap`,
//!   querying every key exactly once in random order from a buffer built in that
//!   order (see `make_query_order`).
//! * `batch` -- `map_many_into` against the loop over `map` it replaces, in a cold
//!   and a hot regime.
//! * `serialize` -- `save_to_file` / `load_from_file` for both map types.

use std::collections::HashMap;
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use constmap::{ConstMap, VerifiedConstMap};

const BENCH_N: usize = 1_000_000;

/// Keys per batch in the `batch` group.
const BATCH_SIZE: usize = 2000;

/// How many distinct random batches the cold benchmarks rotate through, so that
/// repeated iterations do not just replay the same 2000 keys (which would go
/// cache-resident and understate the cost of a genuinely fresh batch against a
/// 9 MB table).
const NUM_BATCH_POOLS: usize = 64;

fn make_bench_data(n: usize) -> (Vec<String>, Vec<u64>) {
    let keys: Vec<String> = (0..n).map(|i| format!("key-{}", i)).collect();
    let values: Vec<u64> = (0..n).map(|i| i as u64).collect();
    (keys, values)
}

/// Returns every key exactly once, in random order, with each string body
/// re-allocated in that order.
///
/// Both halves of that matter, and they are easy to conflate.
///
/// Random order is what exercises the map: walking keys in the order they were
/// built is a pattern no caller has, and it lets the hardware prefetcher hide
/// work that a real lookup has to do.
///
/// Re-allocating the bodies is what keeps the benchmark honest about whose cost
/// it measures. `make_bench_data` allocates the key text in index order, so
/// merely permuting the slice would leave every body where it was and add a
/// scattered, dependency-carrying load per lookup -- you cannot hash a key
/// before reading its bytes. That cost is real, but it belongs to whatever
/// produced the keys, not to the map. Cloning in read order gives the query
/// buffer the compact layout a caller with a batch of keys in hand would have.
fn make_query_order(keys: &[String], seed: u64) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut perm: Vec<usize> = (0..keys.len()).collect();
    perm.shuffle(&mut rng);
    perm.into_iter().map(|j| keys[j].clone()).collect()
}

/// Builds `pools` independent random samples of `BATCH_SIZE` keys drawn from
/// `all_keys`. The keys are cloned in draw order rather than aliased, for the
/// same reason `make_query_order` clones.
fn make_query_batches(all_keys: &[String], seed: u64, pools: usize) -> Vec<Vec<String>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut perm: Vec<usize> = (0..all_keys.len()).collect();
    (0..pools)
        .map(|_| {
            perm.shuffle(&mut rng);
            perm[..BATCH_SIZE]
                .iter()
                .map(|&j| all_keys[j].clone())
                .collect()
        })
        .collect()
}

fn bench_lookup(
    c: &mut Criterion,
    keys: &[String],
    values: &[u64],
    cm: &ConstMap,
    vm: &VerifiedConstMap,
) {
    let queries = make_query_order(keys, 1);
    let hm: HashMap<String, u64> = keys.iter().cloned().zip(values.iter().copied()).collect();

    let mut group = c.benchmark_group("lookup");
    group.throughput(Throughput::Elements(1));

    group.bench_function("ConstMap", |b| {
        let mut i = 0usize;
        b.iter(|| {
            let v = cm.map(black_box(&queries[i]));
            i = (i + 1) % queries.len();
            v
        });
    });

    group.bench_function("VerifiedConstMap", |b| {
        let mut i = 0usize;
        b.iter(|| {
            let v = vm.map(black_box(&queries[i]));
            i = (i + 1) % queries.len();
            v
        });
    });

    group.bench_function("HashMap", |b| {
        let mut i = 0usize;
        b.iter(|| {
            let v = hm.get(black_box(&queries[i])).copied();
            i = (i + 1) % queries.len();
            v
        });
    });

    group.finish();
}

/// The benchmarks below pair a cold and a hot regime for each map type. Cold
/// rotates through `NUM_BATCH_POOLS` distinct random batches, so the touched
/// cache lines are not already resident and the measurement is dominated by
/// memory latency. Hot replays one batch, so the touched region goes
/// cache-resident and hashing dominates instead. Batching helps in both, but
/// for different reasons, and a single regime would hide one of them.
fn bench_batch(c: &mut Criterion, keys: &[String], cm: &ConstMap, vm: &VerifiedConstMap) {
    let batches = make_query_batches(keys, 1, NUM_BATCH_POOLS);
    let hot = &batches[0];
    let mut out = vec![0u64; BATCH_SIZE];

    let mut group = c.benchmark_group("batch");
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    group.bench_function("naive_cold", |b| {
        let mut p = 0usize;
        b.iter(|| {
            let q = &batches[p];
            p = (p + 1) % NUM_BATCH_POOLS;
            for (slot, k) in out.iter_mut().zip(q) {
                *slot = cm.map(k);
            }
            black_box(out[0])
        });
    });

    group.bench_function("map_many_cold", |b| {
        let mut p = 0usize;
        b.iter(|| {
            let q = &batches[p];
            p = (p + 1) % NUM_BATCH_POOLS;
            cm.map_many_into(&mut out, q);
            black_box(out[0])
        });
    });

    group.bench_function("naive_hot", |b| {
        b.iter(|| {
            for (slot, k) in out.iter_mut().zip(hot) {
                *slot = cm.map(k);
            }
            black_box(out[0])
        });
    });

    group.bench_function("map_many_hot", |b| {
        b.iter(|| {
            cm.map_many_into(&mut out, hot);
            black_box(out[0])
        });
    });

    group.bench_function("verified_naive_cold", |b| {
        let mut p = 0usize;
        b.iter(|| {
            let q = &batches[p];
            p = (p + 1) % NUM_BATCH_POOLS;
            for (slot, k) in out.iter_mut().zip(q) {
                *slot = vm.map(k);
            }
            black_box(out[0])
        });
    });

    group.bench_function("verified_map_many_cold", |b| {
        let mut p = 0usize;
        b.iter(|| {
            let q = &batches[p];
            p = (p + 1) % NUM_BATCH_POOLS;
            vm.map_many_into(&mut out, q);
            black_box(out[0])
        });
    });

    group.bench_function("verified_naive_hot", |b| {
        b.iter(|| {
            for (slot, k) in out.iter_mut().zip(hot) {
                *slot = vm.map(k);
            }
            black_box(out[0])
        });
    });

    group.bench_function("verified_map_many_hot", |b| {
        b.iter(|| {
            vm.map_many_into(&mut out, hot);
            black_box(out[0])
        });
    });

    group.finish();
}

fn bench_serialize(c: &mut Criterion, cm: &ConstMap, vm: &VerifiedConstMap) {
    let dir = tempfile::tempdir().unwrap();
    let cpath = dir.path().join("bench.cmap");
    let cpath = cpath.to_str().unwrap();
    let vpath = dir.path().join("bench.vmap");
    let vpath = vpath.to_str().unwrap();

    cm.save_to_file(cpath).unwrap();
    vm.save_to_file(vpath).unwrap();

    let mut group = c.benchmark_group("serialize");
    group.sample_size(20);

    group.bench_function("save_to_file", |b| {
        b.iter(|| cm.save_to_file(cpath).unwrap());
    });
    group.bench_function("load_from_file", |b| {
        b.iter(|| black_box(ConstMap::load_from_file(cpath).unwrap()));
    });
    group.bench_function("verified_save_to_file", |b| {
        b.iter(|| vm.save_to_file(vpath).unwrap());
    });
    group.bench_function("verified_load_from_file", |b| {
        b.iter(|| black_box(VerifiedConstMap::load_from_file(vpath).unwrap()));
    });

    group.finish();
}

fn benches(c: &mut Criterion) {
    let (keys, values) = make_bench_data(BENCH_N);
    let key_refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
    let cm = ConstMap::new(&key_refs, &values).unwrap();
    let vm = VerifiedConstMap::new(&key_refs, &values).unwrap();

    bench_lookup(c, &keys, &values, &cm, &vm);
    bench_batch(c, &keys, &cm, &vm);
    bench_serialize(c, &cm, &vm);
}

criterion_group!(all, benches);
criterion_main!(all);
