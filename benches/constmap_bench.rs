use std::collections::HashMap;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use constmap::{ConstMap, VerifiedConstMap};

const BENCH_N: usize = 1_000_000;

fn make_bench_data(n: usize) -> (Vec<String>, Vec<u64>) {
    let keys: Vec<String> = (0..n).map(|i| format!("key-{}", i)).collect();
    let values: Vec<u64> = (0..n).map(|i| i as u64).collect();
    (keys, values)
}

fn bench_constmap(c: &mut Criterion) {
    let (keys, values) = make_bench_data(BENCH_N);
    let key_refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
    let cm = ConstMap::new(&key_refs, &values).unwrap();

    c.bench_function("ConstMap lookup", |b| {
        let mut i = 0usize;
        b.iter(|| {
            let v = cm.map(black_box(&keys[i % BENCH_N]));
            black_box(v);
            i += 1;
        });
    });
}

fn bench_verified_constmap(c: &mut Criterion) {
    let (keys, values) = make_bench_data(BENCH_N);
    let key_refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
    let vm = VerifiedConstMap::new(&key_refs, &values).unwrap();

    c.bench_function("VerifiedConstMap lookup", |b| {
        let mut i = 0usize;
        b.iter(|| {
            let v = vm.map(black_box(&keys[i % BENCH_N]));
            black_box(v);
            i += 1;
        });
    });
}

fn bench_hashmap(c: &mut Criterion) {
    let (keys, values) = make_bench_data(BENCH_N);
    let m: HashMap<String, u64> = keys
        .iter()
        .zip(values.iter())
        .map(|(k, &v)| (k.clone(), v))
        .collect();

    c.bench_function("HashMap lookup", |b| {
        let mut i = 0usize;
        b.iter(|| {
            let v = m.get(black_box(&keys[i % BENCH_N]));
            black_box(v);
            i += 1;
        });
    });
}

criterion_group!(benches, bench_constmap, bench_verified_constmap, bench_hashmap);
criterion_main!(benches);
