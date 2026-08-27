# constmap 
[![CI](https://github.com/lemire/rsconstmap/actions/workflows/ci.yml/badge.svg)](https://github.com/lemire/rsconstmap/actions/workflows/ci.yml)

A static map from strings to `u64` values using binary fuse filter construction. It can be several times smaller and faster than the equivalent HashMap.


The data structure is ideal when you have a known set of string keys at construction time and need fast, memory-efficient lookups afterward.

This is a Rust port of [lemire/constmap](https://github.com/lemire/constmap), and tracks its v1.1.0 release.

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

## Batched lookups

If you have many keys to resolve at once, `map_many` takes a slice of keys and returns
a `Vec` of values, where `result[i]` corresponds to `keys[i]`:

```rust
let values = cm.map_many(&["apple", "banana", "cherry"]); // [100, 200, 300]
```

`VerifiedConstMap` has the same method, and still reports absent keys as `NOT_FOUND`:

```rust
let values = vm.map_many(&["banana", "grape"]); // [200, NOT_FOUND]
```

If you resolve batch after batch, `map_many_into` writes into a buffer you own instead
of allocating a new one each time. It fills `dst[..keys.len()]`, leaves the rest of
`dst` untouched, and panics if `dst` is shorter than `keys`:

```rust
let mut dst = vec![0u64; 4096];
for batch in batches {
    cm.map_many_into(&mut dst, batch);
    use_values(&dst[..batch.len()]);
}
```

Both are exactly equivalent to calling `map` on each key in turn. They are faster
because a batch hashes a block of eight keys before gathering any values, which lets
the three array accesses of all eight keys be in flight at once instead of each key's
loads waiting behind the previous key's hashing. A lookup is memory-latency-bound
whenever the map is larger than last-level cache, so that overlap is where the gain
comes from.

Both key types accept anything that is `AsRef<str>`, so `&[&str]` and `&[String]` work
without conversion.

### Measured

A 2000-key batch against a 1,000,000-key map, ns/key. *Cold* rotates through 64
distinct random batches so the touched cache lines are not already resident; *hot*
replays one batch, so the touched region of the array stays cache-resident and hashing
dominates instead.

| | | loop over `map` | `map_many_into` | |
|---|---|---|---|---|
| **Apple M4 Max** | `ConstMap` cold | 5.1 | **4.0** | 20% |
| | `ConstMap` hot | 4.8 | **3.6** | 25% |
| | `VerifiedConstMap` cold | 11.4 | **8.3** | 27% |
| | `VerifiedConstMap` hot | 7.1 | **5.2** | 27% |
| **Xeon Gold 6548N** | `ConstMap` cold | 15.8 | **12.5** | 21% |
| | `ConstMap` hot | 9.0 | **7.6** | 16% |
| | `VerifiedConstMap` cold | 20.3 | 21.1 | none |
| | `VerifiedConstMap` hot | 11.8 | **9.6** | 19% |

The one place batching does not pay is `VerifiedConstMap` in the cold regime on the
Xeon, where repeated runs put the difference at anywhere from 2% faster to 4% slower --
that is, inside the run-to-run spread. That map probes two arrays per lookup, and on a
fresh batch it is bound by memory latency the batching cannot hide. The same case on
the M4 Max still gains 27%. This matches what the Go implementation measures on the
same two machines.

Raising the block size from 8 to 16 recovers that case on the Xeon (about 17% faster
cold) but costs about 8% hot there, and is clearly worse hot on the M4 Max. Eight wins
or nearly wins everywhere, so eight is what the block size is.

Unlike the Go implementation, there is no hand-written batched hashing routine. Go
needs one because a per-key call to `xxhash.Sum64String` reloads the five XXH64 primes
and pays a call for every key; in Rust `xxh64` inlines directly into the block loop, so
that half of the Go version's gain is already there and there is nothing left to
hoist.

## Serialization

Both map types can be serialized to disk and loaded back later, avoiding the cost of
reconstruction. Each binary format includes a FNV-1a checksum to detect corruption.

```rust
// Save to file.
cm.save_to_file("mymap.cmap")?;

// Load from file.
let cm = ConstMap::load_from_file("mymap.cmap")?;

// Same for VerifiedConstMap, in its own format.
vm.save_to_file("myverifiedmap.vmap")?;
let vm = VerifiedConstMap::load_from_file("myverifiedmap.vmap")?;
```

For streaming use, `write_to` and `read_from` work with any `Write` / `Read`:

```rust
let mut buf = Vec::new();
cm.write_to(&mut buf)?;
let cm = ConstMap::read_from(&mut &buf[..])?;
```

The two formats carry different magic bytes, so handing a file of one kind to the
other kind's reader is reported rather than silently misinterpreted.

`write_to` and `read_from` move the data array in 64 KiB chunks rather than one `u64`
at a time. That matters most when the underlying writer or reader is an unbuffered
`File`, as it is here: a call per word means a syscall per word. For 1,000,000 keys
(a 9.04 MB file, 18.1 MB for the verified map):

| Operation | Apple M4 Max | Xeon Gold 6548N |
|---|---|---|
| `ConstMap::save_to_file` | 1020 ms -> 11.1 ms | 497 ms -> 12.2 ms |
| `ConstMap::load_from_file` | 361 ms -> 10.1 ms | 307 ms -> 11.5 ms |
| `VerifiedConstMap::save_to_file` | 2049 ms -> 20.7 ms | 1080 ms -> 24.3 ms |
| `VerifiedConstMap::load_from_file` | 727 ms -> 19.0 ms | 611 ms -> 23.0 ms |

Wrapping the file in a `BufReader` does not help: the chunking already batches the
syscalls, and the extra copy costs slightly more than it saves. On the write side a
`BufWriter` measured about the same, occasionally a little faster; either is fine.

The verified format pads its header to 32 bytes so that both `u64` arrays begin on a
64-bit boundary: `data` at offset 32, and `checks` at `32 + 8*data.len()`, which is a
multiple of eight because the first array is a whole number of words. A reader that
maps or otherwise aliases the file can treat either array as a `[u64]` without a
misaligned access.

## Performance gains

Construction time is higher (as expected for any compact data structure), but lookups
are optimized for speed. Against `std::collections::HashMap<String, u64>` with
1,000,000 keys:

| Data structure | Apple M4 Max | Xeon Gold 6548N |
|---|---|---|
| `ConstMap` | 7.0 ns | 18.8 ns |
| `VerifiedConstMap` | 12.3 ns | 26.4 ns |
| `HashMap` | 54.6 ns | 94.6 ns |

The speed varies depending on your system, the size of your dataset, the keys, the
order of the lookups and so forth. If `constmap` can reside in CPU cache while the
`HashMap` cannot, then it will be significantly faster.

The memory usage should always be significantly better with `ConstMap` as long as you
have many thousands of keys -- and unlike a `HashMap`, it does not store the keys at
all, so the key text can be dropped after construction.

### How these are measured

Lookup benchmarks query every key exactly once, in random order, from a buffer built in
that order (`make_query_order` in `benches/constmap_bench.rs`). Both halves of that are
deliberate, and both change the answer a lot.

*Random order* is what exercises the map. Walking the keys in the order they were
constructed is a pattern no caller has, and it lets the hardware prefetcher hide work a
real lookup must do.

*A buffer built in query order* is what keeps the measurement about the map. The key
text is allocated in index order, so merely permuting the slice would leave every
string body where it was and add a scattered, dependency-carrying load to each lookup --
you cannot hash a key before reading its bytes. That cost is real, but it belongs to
whatever produced the keys, not to the map.

Querying the whole key set, rather than a small sample, is deliberate too. A small
sample keeps the key text in cache, but it also leaves most of the map untouched and
cache-resident, which flatters every implementation and hides exactly the compactness
that is the point.

### Scaling

One pass over the whole key set, so every lookup is a first touch. Memory is counted by
a global allocator installed for the test binary, so the figures are bytes the process
actually asked for, not an estimate from capacities; the `HashMap` column is a
`HashMap<&str, u64>`, measuring the table alone rather than a second copy of the key
text. Run it with:

```sh
cargo test --release -- --ignored --nocapture scaling_table
```

Apple M4 Max:

| keys | `ConstMap` | `VerifiedConstMap` | `HashMap` | `ConstMap` bytes/key | `VerifiedConstMap` bytes/key | `HashMap` bytes/key |
|---|---|---|---|---|---|---|
| 10,000 | 6.8 ns | 9.2 ns | 14.7 ns | 10.2 | 20.5 | 41.0 |
| 100,000 | 6.5 ns | 8.3 ns | 16.9 ns | 9.5 | 19.0 | 32.8 |
| 1,000,000 | 9.2 ns | 15.0 ns | 53.4 ns | 9.0 | 18.1 | 52.4 |
| 10,000,000 | 23.5 ns | 31.1 ns | 100.8 ns | 9.0 | 18.0 | 41.9 |

Intel Xeon Gold 6548N:

| keys | `ConstMap` | `VerifiedConstMap` | `HashMap` |
|---|---|---|---|
| 10,000 | 10.0 ns | 12.8 ns | 21.4 ns |
| 100,000 | 12.3 ns | 20.2 ns | 43.2 ns |
| 1,000,000 | 22.6 ns | 26.7 ns | 110.6 ns |
| 10,000,000 | 50.1 ns | 78.2 ns | 166.3 ns |

The margin grows with the key count, which is the expected shape: at ten thousand keys
everything fits in cache and the compactness buys little, while at a million and beyond
it is most of the story. The `HashMap` bytes/key figure wobbles because the table grows
in powers of two.

## Benchmarks

```sh
cargo bench
```

Three groups:

- **`lookup`** -- single-key throughput for `ConstMap`, `VerifiedConstMap` and `HashMap`
- **`batch`** -- `map_many_into` against the loop over `map` it replaces, cold and hot,
  for both map types
- **`serialize`** -- `save_to_file` / `load_from_file` for both map types

## Tests

```sh
cargo test
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
