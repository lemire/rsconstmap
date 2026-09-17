//! The files under tests/interop were written by the other two
//! implementations from the same input as `data()`: constmap (Go) and
//! fastconstmap (C/Python), plus the two formats fastconstmap 0.9 wrote with a
//! different key hash. The three implementations are meant to read each
//! other's files, so these tests are the guard on that promise: a change to
//! the hash, the mixing, the seed sequence, the layout or the checksum shows
//! up here as a failure to load or a wrong value.
//!
//! To regenerate: build a map from `data()` in each implementation and save
//! it (see the README of each repository).

use std::path::PathBuf;

use constmap::{ConstMap, PairedVerifiedConstMap, VerifiedConstMap, NOT_FOUND};

fn data() -> (Vec<String>, Vec<u64>) {
    let keys: Vec<String> = (0..200).map(|i| format!("key-{}", i)).collect();
    let values: Vec<u64> = (0..200u64).map(|i| 7 * i).collect();
    (keys, values)
}

fn path(name: &str) -> String {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("interop");
    p.push(name);
    p.to_str().unwrap().to_string()
}

fn check(name: &str, lookup: impl Fn(&str) -> u64, verified: bool) {
    let (keys, values) = data();
    for (k, &v) in keys.iter().zip(&values) {
        assert_eq!(lookup(k), v, "{}: map({}) mismatch", name, k);
    }
    if verified {
        for i in 0..1000 {
            let k = format!("absent-{}", i);
            assert_eq!(lookup(&k), NOT_FOUND, "{}: map({}) should be NOT_FOUND", name, k);
        }
    }
}

#[test]
fn reads_other_implementations() {
    for src in ["constmap", "fastconstmap"] {
        let cm = ConstMap::load_from_file(&path(&format!("{}.cmap", src))).unwrap();
        check(&format!("{}.cmap", src), |k| cm.map(k), false);
        let vm = VerifiedConstMap::load_from_file(&path(&format!("{}.vmap", src))).unwrap();
        check(&format!("{}.vmap", src), |k| vm.map(k), true);
        let pm = PairedVerifiedConstMap::load_from_file(&path(&format!("{}.pmap", src))).unwrap();
        check(&format!("{}.pmap", src), |k| pm.map(k), true);
    }
}

/// The stronger property that holds between this crate and the Go package:
/// for the same input they write the same bytes, since they share the hash,
/// the seed sequence and the layout.
#[test]
fn writes_identical_bytes_to_go() {
    let (keys, values) = data();
    let refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
    let cm = ConstMap::new(&refs, &values).unwrap();
    let vm = VerifiedConstMap::new(&refs, &values).unwrap();
    let pm = vm.paired();

    let mut got = Vec::new();
    cm.write_to(&mut got).unwrap();
    assert_eq!(got, std::fs::read(path("constmap.cmap")).unwrap(), "cmap differs from Go");
    got.clear();
    vm.write_to(&mut got).unwrap();
    assert_eq!(got, std::fs::read(path("constmap.vmap")).unwrap(), "vmap differs from Go");
    got.clear();
    pm.write_to(&mut got).unwrap();
    assert_eq!(got, std::fs::read(path("constmap.pmap")).unwrap(), "pmap differs from Go");
}

/// A file from fastconstmap 0.9 or earlier, which used a different key hash,
/// is refused with an error that says so rather than "invalid magic".
#[test]
fn legacy_fastconstmap_is_named() {
    let err = ConstMap::load_from_file(&path("fastconstmap-0.9.cmap")).unwrap_err();
    assert!(err.to_string().contains("fastconstmap 0.9"), "unexpected error: {}", err);
    let err = VerifiedConstMap::load_from_file(&path("fastconstmap-0.9.vmap")).unwrap_err();
    assert!(err.to_string().contains("fastconstmap 0.9"), "unexpected error: {}", err);
    let err = PairedVerifiedConstMap::load_from_file(&path("fastconstmap-0.9.vmap")).unwrap_err();
    assert!(err.to_string().contains("fastconstmap 0.9"), "unexpected error: {}", err);
}
