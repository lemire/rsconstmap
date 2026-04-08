# constmap

A static map from strings to `u64` values using binary fuse filter construction. It can be several times smaller and faster than the equivalent HashMap.


The data structure is ideal when you have a known set of string keys at construction time and need fast, memory-efficient lookups afterward.

## Reference

This implementation is based on the binary fuse filter algorithm described in:

> Thomas Mueller Graf and Daniel Lemire, [Binary Fuse Filters: Fast and Smaller Than Xor Filters](https://arxiv.org/abs/2201.01174), *ACM Journal of Experimental Algorithmics*, Volume 27, 2022. DOI: [10.1145/3510449](https://doi.org/10.1145/3510449)

See also the earlier xor filter paper:

> Thomas Mueller Graf and Daniel Lemire, [Xor Filters: Faster and Smaller Than Bloom and Cuckoo Filters](https://arxiv.org/abs/1912.08258), *ACM Journal of Experimental Algorithmics*, Volume 25, 2020. DOI: [10.1145/3376122](https://doi.org/10.1145/3376122)

Given a set of (key, value) pairs, `constmap` builds a compact array such that for any key in the set, `data[h0(key)] XOR data[h1(key)] XOR data[h2(key)] == value`. Lookup is extremely fast: one xxhash call plus three array accesses and two XORs. The data structure is immutable after construction.

### Basic example

```rust
use constmap::ConstMap;

let keys = vec!["apple", "banana", "cherry"];
let values = vec![100u64, 200, 300];

let map = ConstMap::new(&keys, &values).unwrap();
assert_eq!(map.map("apple"), 100);
assert_eq!(map.map("banana"), 200);
```

**Note:** Looking up a key that was not in the original set returns an undefined value.

### Verified lookups

`VerifiedConstMap` stores an additional fingerprint per key so that lookups of unknown keys return `NOT_FOUND` instead of garbage, at the cost of roughly doubling memory usage (~18 bytes/key instead of ~9).

```rust
use constmap::{VerifiedConstMap, NOT_FOUND};

let keys = vec!["apple", "banana", "cherry"];
let values = vec![100u64, 200, 300];

let map = VerifiedConstMap::new(&keys, &values).unwrap();
assert_eq!(map.map("apple"), 100);
assert_eq!(map.map("unknown"), NOT_FOUND);
```

### Serialization

Both `ConstMap` and `VerifiedConstMap` can be serialized to and from files or any `Read`/`Write` stream. A FNV-1a checksum is used for integrity verification.

```rust
// Save to file
map.save_to_file("map.bin")?;

// Load from file
let map = ConstMap::load_from_file("map.bin")?;
```

## Benchmarks

```sh
cargo bench
```

Possible result:

```
ConstMap lookup         time:   [6.8145 ns 6.8957 ns 6.9838 ns]
VerifiedConstMap lookup time:   [13.321 ns 13.453 ns 13.588 ns]
HashMap lookup          time:   [36.590 ns 37.117 ns 37.701 ns]
```



## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
