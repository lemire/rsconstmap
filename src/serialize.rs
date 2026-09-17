//! Binary serialization for [`ConstMap`] and [`VerifiedConstMap`].
//!
//! Both formats are little-endian and end with a FNV-1a 64-bit checksum over
//! every preceding byte. They carry different magic bytes, so handing a file of
//! one kind to the other kind's reader is reported rather than silently
//! misinterpreted.

use std::io::{self, Read, Write};

use crate::{ConstMap, PairedVerifiedConstMap, VerifiedConstMap};

// ---------- FNV-1a 64-bit ----------

struct Fnv64a(u64);

impl Fnv64a {
    fn new() -> Self {
        Fnv64a(0xcbf29ce484222325)
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

/// A writer that feeds everything it forwards through a FNV-1a hash, so the
/// checksum is accumulated without a second pass over the data.
struct HashWriter<'a, W: Write> {
    inner: &'a mut W,
    hasher: Fnv64a,
}

impl<'a, W: Write> HashWriter<'a, W> {
    fn new(inner: &'a mut W) -> Self {
        HashWriter {
            inner,
            hasher: Fnv64a::new(),
        }
    }
}

impl<W: Write> Write for HashWriter<'_, W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.hasher.write(&buf[..n]);
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// A reader that feeds everything it forwards through a FNV-1a hash.
struct HashReader<'a, R: Read> {
    inner: &'a mut R,
    hasher: Fnv64a,
}

impl<'a, R: Read> HashReader<'a, R> {
    fn new(inner: &'a mut R) -> Self {
        HashReader {
            inner,
            hasher: Fnv64a::new(),
        }
    }
}

impl<R: Read> Read for HashReader<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.hasher.write(&buf[..n]);
        Ok(n)
    }
}

// ---------- chunked word I/O ----------

/// How many `u64`s [`ConstMap::write_to`] and [`ConstMap::read_from`] move per
/// call to the underlying writer or reader. Moving one word at a time costs a
/// call per word -- and, when the reader is a bare `File` as it is in
/// [`ConstMap::load_from_file`], a syscall per word, which dominates everything
/// else. 8192 words is 64 KiB.
const IO_CHUNK_WORDS: usize = 8192;

/// Allocates a scratch buffer for moving an array of `n` words, big enough for
/// one chunk but never larger than the array itself.
fn chunk_buffer(n: usize) -> Vec<u8> {
    vec![0u8; n.min(IO_CHUNK_WORDS) * 8]
}

/// Encodes `words` as little-endian `u64`s and writes them a chunk at a time,
/// reusing `chunk` as scratch space.
fn write_words<W: Write>(w: &mut W, words: &[u64], chunk: &mut [u8]) -> io::Result<()> {
    for block in words.chunks(IO_CHUNK_WORDS) {
        let b = &mut chunk[..block.len() * 8];
        for (slot, &v) in b.chunks_exact_mut(8).zip(block) {
            slot.copy_from_slice(&v.to_le_bytes());
        }
        w.write_all(b)?;
    }
    Ok(())
}

/// Fills `words` from `r` a chunk at a time, decoding in place. `name`
/// identifies the array in error messages.
fn read_words<R: Read>(
    r: &mut R,
    words: &mut [u64],
    chunk: &mut [u8],
    name: &str,
) -> io::Result<()> {
    let mut i = 0;
    while i < words.len() {
        let n = (words.len() - i).min(IO_CHUNK_WORDS);
        let b = &mut chunk[..n * 8];
        r.read_exact(b).map_err(|e| {
            io::Error::new(
                e.kind(),
                format!("constmap: reading {}[{}:{}]: {}", name, i, i + n, e),
            )
        })?;
        for (slot, src) in words[i..i + n].iter_mut().zip(b.chunks_exact(8)) {
            *slot = u64::from_le_bytes(src.try_into().unwrap());
        }
        i += n;
    }
    Ok(())
}

fn invalid_data<E: Into<Box<dyn std::error::Error + Send + Sync>>>(msg: E) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

fn checksum_mismatch(got: u64, expected: u64) -> io::Error {
    invalid_data(format!(
        "constmap: checksum mismatch (got {:016x}, expected {:016x})",
        got, expected
    ))
}

// ---------- ConstMap ----------

// Binary format for ConstMap (all little-endian):
//   [8] magic "CMAP0001"
//   [8] seed
//   [4] segment_length
//   [4] segment_count
//   [4] data.len()
//   [8*data.len()] data
//   [8] FNV-1a 64-bit checksum of all preceding bytes
//
// The formats of this crate are shared with constmap (Go) and fastconstmap
// (C/Python): a file written by any of the three loads in the other two, on a
// little-endian host. fastconstmap writes its ConstMaps with magic "CMAP0003",
// which is this format with a fourth 4-byte header field holding the original
// key count (so that its len() survives a save and load); read_from accepts
// it and ignores the count. It also recognises the "CMAP0002" files
// fastconstmap 0.9 and earlier wrote, which hashed keys with XXH3 rather than
// XXH64 and so cannot be used here, and names them so the user knows to
// rebuild the map with a current fastconstmap.

const MAGIC: &[u8; 8] = b"CMAP0001";

/// fastconstmap's ConstMap magic: CMAP0001 plus a key count field after the
/// data length.
const MAGIC_COUNTED: &[u8; 8] = b"CMAP0003";

/// Magics of files written by fastconstmap 0.9 and earlier, which used a
/// different key hash. They cannot be loaded, only recognised.
const LEGACY_FAST_MAGIC: &[u8; 8] = b"CMAP0002";
const LEGACY_FAST_VERIFIED_MAGIC: &[u8; 8] = b"VCMP0002";

fn legacy_fastconstmap() -> io::Error {
    invalid_data(
        "constmap: this file was written by fastconstmap 0.9 or earlier with a different key hash; \
         rebuild it from its keys with fastconstmap 0.10 or later, which writes the shared format",
    )
}

impl ConstMap {
    /// Serialize the `ConstMap` to a writer in a portable binary format.
    ///
    /// A FNV-1a checksum is appended for integrity verification. Returns the
    /// number of bytes written.
    pub fn write_to<W: Write>(&self, w: &mut W) -> io::Result<u64> {
        let sum = {
            let mut hw = HashWriter::new(w);

            hw.write_all(MAGIC)?;
            hw.write_all(&self.seed.to_le_bytes())?;
            hw.write_all(&self.segment_length.to_le_bytes())?;
            hw.write_all(&self.segment_count.to_le_bytes())?;
            hw.write_all(&(self.data.len() as u32).to_le_bytes())?;

            // Data, encoded and written a chunk at a time.
            let mut chunk = chunk_buffer(self.data.len());
            write_words(&mut hw, &self.data, &mut chunk)?;

            hw.hasher.finish()
        };

        // Checksum (written to w only, not fed back into the hash).
        w.write_all(&sum.to_le_bytes())?;

        Ok(8 + 8 + 4 + 4 + 4 + 8 * self.data.len() as u64 + 8)
    }

    /// Deserialize a `ConstMap` from a reader.
    ///
    /// Verifies the trailing checksum and returns an error if the data is
    /// corrupted.
    pub fn read_from<R: Read>(r: &mut R) -> io::Result<Self> {
        let mut buf8 = [0u8; 8];
        let mut buf4 = [0u8; 4];

        let (cm, expected_sum) = {
            let mut hr = HashReader::new(r);

            hr.read_exact(&mut buf8)?;
            let counted = &buf8 == MAGIC_COUNTED;
            if &buf8 != MAGIC && !counted {
                if &buf8 == VERIFIED_MAGIC {
                    return Err(invalid_data(
                        "constmap: this is a VerifiedConstMap file, use VerifiedConstMap::load_from_file",
                    ));
                }
                if &buf8 == PAIRED_MAGIC {
                    return Err(invalid_data(
                        "constmap: this is a PairedVerifiedConstMap file, use PairedVerifiedConstMap::load_from_file",
                    ));
                }
                if &buf8 == LEGACY_FAST_MAGIC || &buf8 == LEGACY_FAST_VERIFIED_MAGIC {
                    return Err(legacy_fastconstmap());
                }
                return Err(invalid_data("constmap: invalid magic bytes"));
            }

            hr.read_exact(&mut buf8)?;
            let seed = u64::from_le_bytes(buf8);

            hr.read_exact(&mut buf4)?;
            let segment_length = u32::from_le_bytes(buf4);

            hr.read_exact(&mut buf4)?;
            let segment_count = u32::from_le_bytes(buf4);

            hr.read_exact(&mut buf4)?;
            let data_len = u32::from_le_bytes(buf4) as usize;

            // Key count, present only in fastconstmap's CMAP0003; not kept.
            if counted {
                hr.read_exact(&mut buf4)?;
            }

            // Data, read a chunk at a time and decoded in place.
            let mut data = vec![0u64; data_len];
            let mut chunk = chunk_buffer(data_len);
            read_words(&mut hr, &mut data, &mut chunk, "data")?;

            let cm = ConstMap {
                seed,
                segment_length,
                segment_length_mask: segment_length.wrapping_sub(1),
                segment_count,
                segment_count_length: segment_count.wrapping_mul(segment_length),
                data,
            };
            (cm, hr.hasher.finish())
        };

        // Checksum: read from r directly, not through the hashing reader.
        r.read_exact(&mut buf8)?;
        let got_sum = u64::from_le_bytes(buf8);
        if got_sum != expected_sum {
            return Err(checksum_mismatch(got_sum, expected_sum));
        }

        Ok(cm)
    }

    /// Serialize the `ConstMap` to a file at the given path.
    pub fn save_to_file(&self, path: &str) -> io::Result<()> {
        let mut f = std::fs::File::create(path)?;
        self.write_to(&mut f)?;
        Ok(())
    }

    /// Deserialize a `ConstMap` from a file at the given path.
    pub fn load_from_file(path: &str) -> io::Result<Self> {
        let mut f = std::fs::File::open(path)?;
        Self::read_from(&mut f)
    }
}

// ---------- VerifiedConstMap ----------

// Binary format for VerifiedConstMap (all little-endian):
//   [8] magic "VMAP0001"
//   [8] seed
//   [4] segment_length
//   [4] segment_count
//   [4] data.len(), which is also checks.len()
//   [4] zero padding; fastconstmap stores its original key count here, and
//       every reader ignores the field
//   [8*data.len()] data
//   [8*data.len()] checks
//   [8] FNV-1a 64-bit checksum of all preceding bytes
//
// The padding brings the header to 32 bytes so that both `u64` arrays start on
// a 64-bit boundary: data at offset 32, and checks at 32+8*data.len(), which is
// a multiple of eight because the first array is a whole number of words. A
// reader that maps or otherwise aliases the file can therefore treat either
// array as a `[u64]` without a misaligned access.

const VERIFIED_MAGIC: &[u8; 8] = b"VMAP0001";

/// The number of bytes preceding the data array, including the padding that
/// aligns it.
const VERIFIED_HEADER_SIZE: u64 = 32;

impl VerifiedConstMap {
    /// Serialize the `VerifiedConstMap` to a writer in a portable binary format.
    ///
    /// A FNV-1a checksum is appended for integrity verification. Returns the
    /// number of bytes written.
    pub fn write_to<W: Write>(&self, w: &mut W) -> io::Result<u64> {
        if self.checks.len() != self.data.len() {
            return Err(invalid_data(format!(
                "constmap: {} data words but {} check words",
                self.data.len(),
                self.checks.len()
            )));
        }

        let sum = {
            let mut hw = HashWriter::new(w);

            hw.write_all(VERIFIED_MAGIC)?;
            hw.write_all(&self.seed.to_le_bytes())?;
            hw.write_all(&self.segment_length.to_le_bytes())?;
            hw.write_all(&self.segment_count.to_le_bytes())?;
            // Length, shared by both arrays.
            hw.write_all(&(self.data.len() as u32).to_le_bytes())?;
            // Padding, so that both arrays begin on a 64-bit boundary.
            hw.write_all(&0u32.to_le_bytes())?;

            // Both arrays, encoded and written a chunk at a time.
            let mut chunk = chunk_buffer(self.data.len());
            write_words(&mut hw, &self.data, &mut chunk)?;
            write_words(&mut hw, &self.checks, &mut chunk)?;

            hw.hasher.finish()
        };

        // Checksum (written to w only, not fed back into the hash).
        w.write_all(&sum.to_le_bytes())?;

        Ok(VERIFIED_HEADER_SIZE + 16 * self.data.len() as u64 + 8)
    }

    /// Deserialize a `VerifiedConstMap` from a reader.
    ///
    /// Verifies the trailing checksum and returns an error if the data is
    /// corrupted.
    pub fn read_from<R: Read>(r: &mut R) -> io::Result<Self> {
        let mut buf8 = [0u8; 8];
        let mut buf4 = [0u8; 4];

        let (vm, expected_sum) = {
            let mut hr = HashReader::new(r);

            hr.read_exact(&mut buf8)?;
            if &buf8 != VERIFIED_MAGIC {
                if &buf8 == MAGIC {
                    return Err(invalid_data(
                        "constmap: this is a ConstMap file, use ConstMap::load_from_file",
                    ));
                }
                if &buf8 == PAIRED_MAGIC {
                    return Err(invalid_data(
                        "constmap: this is a PairedVerifiedConstMap file, use PairedVerifiedConstMap::load_from_file",
                    ));
                }
                if &buf8 == LEGACY_FAST_MAGIC || &buf8 == LEGACY_FAST_VERIFIED_MAGIC {
                    return Err(legacy_fastconstmap());
                }
                return Err(invalid_data("constmap: invalid magic bytes"));
            }

            hr.read_exact(&mut buf8)?;
            let seed = u64::from_le_bytes(buf8);

            hr.read_exact(&mut buf4)?;
            let segment_length = u32::from_le_bytes(buf4);

            hr.read_exact(&mut buf4)?;
            let segment_count = u32::from_le_bytes(buf4);

            // Length, shared by both arrays.
            hr.read_exact(&mut buf4)?;
            let data_len = u32::from_le_bytes(buf4) as usize;

            // Padding.
            hr.read_exact(&mut buf4)?;

            // Both arrays, read a chunk at a time and decoded in place.
            let mut chunk = chunk_buffer(data_len);
            let mut data = vec![0u64; data_len];
            read_words(&mut hr, &mut data, &mut chunk, "data")?;
            let mut checks = vec![0u64; data_len];
            read_words(&mut hr, &mut checks, &mut chunk, "checks")?;

            let vm = VerifiedConstMap {
                seed,
                segment_length,
                segment_length_mask: segment_length.wrapping_sub(1),
                segment_count,
                segment_count_length: segment_count.wrapping_mul(segment_length),
                data,
                checks,
            };
            (vm, hr.hasher.finish())
        };

        // Checksum: read from r directly, not through the hashing reader.
        r.read_exact(&mut buf8)?;
        let got_sum = u64::from_le_bytes(buf8);
        if got_sum != expected_sum {
            return Err(checksum_mismatch(got_sum, expected_sum));
        }

        Ok(vm)
    }

    /// Serialize the `VerifiedConstMap` to a file at the given path.
    pub fn save_to_file(&self, path: &str) -> io::Result<()> {
        let mut f = std::fs::File::create(path)?;
        self.write_to(&mut f)?;
        Ok(())
    }

    /// Deserialize a `VerifiedConstMap` from a file at the given path.
    pub fn load_from_file(path: &str) -> io::Result<Self> {
        let mut f = std::fs::File::open(path)?;
        Self::read_from(&mut f)
    }
}

// ---------- PairedVerifiedConstMap ----------

// Binary format for PairedVerifiedConstMap (all little-endian):
//   [8] magic "PMAP0001"
//   [8] seed
//   [4] segment_length
//   [4] segment_count
//   [4] slots.len()
//   [4] zero padding; fastconstmap stores its original key count here, and
//       every reader ignores the field
//   [16*slots.len()] slots, each a value word followed by its fingerprint word
//   [8] FNV-1a 64-bit checksum of all preceding bytes
//
// This is the VerifiedConstMap format with its two arrays zipped together,
// under its own magic so that a file of either kind fed to the other's reader
// is reported rather than silently misinterpreted. The padding keeps the slot
// array on a 64-bit boundary, as in the other two formats.

const PAIRED_MAGIC: &[u8; 8] = b"PMAP0001";

impl PairedVerifiedConstMap {
    /// Serialize the `PairedVerifiedConstMap` to a writer in a portable binary
    /// format.
    ///
    /// A FNV-1a checksum is appended for integrity verification. Returns the
    /// number of bytes written.
    pub fn write_to<W: Write>(&self, w: &mut W) -> io::Result<u64> {
        let sum = {
            let mut hw = HashWriter::new(w);

            hw.write_all(PAIRED_MAGIC)?;
            hw.write_all(&self.seed.to_le_bytes())?;
            hw.write_all(&self.segment_length.to_le_bytes())?;
            hw.write_all(&self.segment_count.to_le_bytes())?;
            hw.write_all(&(self.slots.len() as u32).to_le_bytes())?;
            // Padding, so that the slot array begins on a 64-bit boundary.
            hw.write_all(&0u32.to_le_bytes())?;

            // The slot array is already in file order: value, fingerprint,
            // value, ...
            let words = self.slots.as_flattened();
            write_words(&mut hw, words, &mut chunk_buffer(words.len()))?;

            hw.hasher.finish()
        };

        // Checksum (written to w only, not fed back into the hash).
        w.write_all(&sum.to_le_bytes())?;

        Ok(VERIFIED_HEADER_SIZE + 16 * self.slots.len() as u64 + 8)
    }

    /// Deserialize a `PairedVerifiedConstMap` from a reader.
    ///
    /// Verifies the trailing checksum and returns an error if the data is
    /// corrupted.
    pub fn read_from<R: Read>(r: &mut R) -> io::Result<Self> {
        let mut buf8 = [0u8; 8];
        let mut buf4 = [0u8; 4];

        let (pm, expected_sum) = {
            let mut hr = HashReader::new(r);

            hr.read_exact(&mut buf8)?;
            if &buf8 != PAIRED_MAGIC {
                if &buf8 == MAGIC {
                    return Err(invalid_data(
                        "constmap: this is a ConstMap file, use ConstMap::load_from_file",
                    ));
                }
                if &buf8 == VERIFIED_MAGIC {
                    return Err(invalid_data(
                        "constmap: this is a VerifiedConstMap file, use VerifiedConstMap::load_from_file",
                    ));
                }
                if &buf8 == LEGACY_FAST_MAGIC || &buf8 == LEGACY_FAST_VERIFIED_MAGIC {
                    return Err(legacy_fastconstmap());
                }
                return Err(invalid_data("constmap: invalid magic bytes"));
            }

            hr.read_exact(&mut buf8)?;
            let seed = u64::from_le_bytes(buf8);

            hr.read_exact(&mut buf4)?;
            let segment_length = u32::from_le_bytes(buf4);

            hr.read_exact(&mut buf4)?;
            let segment_count = u32::from_le_bytes(buf4);

            hr.read_exact(&mut buf4)?;
            let slot_count = u32::from_le_bytes(buf4) as usize;

            // Padding.
            hr.read_exact(&mut buf4)?;

            // The parameters must describe exactly this many slots, so that
            // every position a lookup derives from them is in range: h0 is
            // below segment_count * segment_length, and h1 and h2 each one
            // segment further, so h2 < (segment_count + 2) * segment_length.
            // That needs segment_length to be a power of two (h1 and h2 are
            // formed by XORing bits below it) and segment_count to be at
            // least one (with zero, h0 is always 0 and h2 lands in a third
            // segment that does not exist). The checksum below catches
            // accidental corruption; this catches a file that is consistent
            // but not ours.
            if slot_count != 0 {
                if segment_length == 0 || !segment_length.is_power_of_two() {
                    return Err(invalid_data(format!(
                        "constmap: segment length {} is not a power of two",
                        segment_length
                    )));
                }
                if segment_count == 0 {
                    return Err(invalid_data("constmap: segment count is zero"));
                }
                let want = (segment_count as u64 + 2) * segment_length as u64;
                if want != slot_count as u64 {
                    return Err(invalid_data(format!(
                        "constmap: {} slots but the segment parameters describe {}",
                        slot_count, want
                    )));
                }
            }

            let mut slots = vec![[0u64; 2]; slot_count];
            let words = slots.as_flattened_mut();
            read_words(&mut hr, words, &mut chunk_buffer(words.len()), "slots")?;

            let pm = PairedVerifiedConstMap {
                seed,
                segment_length,
                segment_length_mask: segment_length.wrapping_sub(1),
                segment_count,
                segment_count_length: segment_count.wrapping_mul(segment_length),
                slots,
            };
            (pm, hr.hasher.finish())
        };

        // Checksum: read from r directly, not through the hashing reader.
        r.read_exact(&mut buf8)?;
        let got_sum = u64::from_le_bytes(buf8);
        if got_sum != expected_sum {
            return Err(checksum_mismatch(got_sum, expected_sum));
        }

        Ok(pm)
    }

    /// Serialize the `PairedVerifiedConstMap` to a file at the given path.
    pub fn save_to_file(&self, path: &str) -> io::Result<()> {
        let mut f = std::fs::File::create(path)?;
        self.write_to(&mut f)?;
        Ok(())
    }

    /// Deserialize a `PairedVerifiedConstMap` from a file at the given path.
    pub fn load_from_file(path: &str) -> io::Result<Self> {
        let mut f = std::fs::File::open(path)?;
        Self::read_from(&mut f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NOT_FOUND;

    fn build_data(n: usize) -> (Vec<String>, Vec<u64>) {
        let keys: Vec<String> = (0..n).map(|i| format!("key-{}", i)).collect();
        let values: Vec<u64> = (0..n).map(|i| (i * 7) as u64).collect();
        (keys, values)
    }

    fn refs(keys: &[String]) -> Vec<&str> {
        keys.iter().map(|s| s.as_str()).collect()
    }

    // ---------- ConstMap ----------

    #[test]
    fn test_serialize_deserialize() {
        let keys = vec!["apple", "banana", "cherry", "date", "elderberry"];
        let values = vec![100u64, 200, 300, 400, 500];

        let cm = ConstMap::new(&keys, &values).unwrap();

        let mut buf = Vec::new();
        cm.write_to(&mut buf).unwrap();

        let cm2 = ConstMap::read_from(&mut &buf[..]).unwrap();

        for (i, &k) in keys.iter().enumerate() {
            assert_eq!(
                cm2.map(k),
                values[i],
                "after deserialize: map({}) mismatch",
                k
            );
        }
    }

    #[test]
    fn test_serialize_large() {
        let (keys, values) = build_data(100_000);
        let cm = ConstMap::new(&refs(&keys), &values).unwrap();

        let mut buf = Vec::new();
        let written = cm.write_to(&mut buf).unwrap();
        assert_eq!(written as usize, buf.len());

        let cm2 = ConstMap::read_from(&mut &buf[..]).unwrap();

        for (i, k) in keys.iter().enumerate() {
            assert_eq!(
                cm2.map(k),
                values[i],
                "after deserialize: map({}) mismatch",
                k
            );
        }
    }

    /// The data array is moved in chunks of `IO_CHUNK_WORDS`; sizes just below,
    /// at, and just above a chunk boundary all have to round-trip.
    #[test]
    fn test_serialize_chunk_boundaries() {
        for n in [
            1usize,
            2,
            IO_CHUNK_WORDS - 1,
            IO_CHUNK_WORDS,
            IO_CHUNK_WORDS + 1,
        ] {
            let (keys, values) = build_data(n);
            let cm = ConstMap::new(&refs(&keys), &values).unwrap();

            let mut buf = Vec::new();
            let written = cm.write_to(&mut buf).unwrap();
            assert_eq!(written as usize, buf.len(), "n={}", n);

            let cm2 = ConstMap::read_from(&mut &buf[..]).unwrap();
            assert_eq!(cm2.data, cm.data, "n={}", n);
            for (i, k) in keys.iter().enumerate() {
                assert_eq!(cm2.map(k), values[i], "n={}: map({}) mismatch", n, k);
            }
        }
    }

    /// A reader that hands back one byte at a time still has to be read to
    /// completion: `read_words` must not assume a chunk arrives in one read.
    #[test]
    fn test_read_from_short_reads() {
        let (keys, values) = build_data(20_000);
        let cm = ConstMap::new(&refs(&keys), &values).unwrap();

        let mut buf = Vec::new();
        cm.write_to(&mut buf).unwrap();

        let mut r = OneByteAtATime(&buf[..]);
        let cm2 = ConstMap::read_from(&mut r).unwrap();
        assert_eq!(cm2.data, cm.data);
    }

    #[test]
    fn test_read_from_truncated() {
        let (keys, values) = build_data(10_000);
        let cm = ConstMap::new(&refs(&keys), &values).unwrap();

        let mut buf = Vec::new();
        cm.write_to(&mut buf).unwrap();

        for cut in [0usize, 4, 8, 20, 24, buf.len() / 2, buf.len() - 1] {
            let err = ConstMap::read_from(&mut &buf[..cut]);
            assert!(err.is_err(), "truncating to {} bytes should fail", cut);
        }
    }

    #[test]
    fn test_write_to_short_writer() {
        let (keys, values) = build_data(20_000);
        let cm = ConstMap::new(&refs(&keys), &values).unwrap();

        // Fails partway through the data array, not in the header.
        let mut w = ShortWriter { left: 1024 };
        assert!(cm.write_to(&mut w).is_err());
    }

    #[test]
    fn test_serialize_corrupted() {
        let keys = vec!["apple", "banana", "cherry"];
        let values = vec![10u64, 20, 30];

        let cm = ConstMap::new(&keys, &values).unwrap();

        let mut buf = Vec::new();
        cm.write_to(&mut buf).unwrap();

        // Flip a byte in the middle.
        let mid = buf.len() / 2;
        buf[mid] ^= 0xff;

        assert!(ConstMap::read_from(&mut &buf[..]).is_err());
    }

    #[test]
    fn test_save_load_file() {
        let keys = vec!["one", "two", "three"];
        let values = vec![1u64, 2, 3];

        let cm = ConstMap::new(&keys, &values).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.cmap");
        let path_str = path.to_str().unwrap();

        cm.save_to_file(path_str).unwrap();
        let cm2 = ConstMap::load_from_file(path_str).unwrap();

        for (i, &k) in keys.iter().enumerate() {
            assert_eq!(cm2.map(k), values[i], "after load: map({}) mismatch", k);
        }

        assert!(path.exists());
        assert!(std::fs::metadata(&path).unwrap().len() > 0);
    }

    #[test]
    fn test_serialize_empty() {
        let cm = ConstMap::new(&[], &[]).unwrap();

        let mut buf = Vec::new();
        cm.write_to(&mut buf).unwrap();

        let cm2 = ConstMap::read_from(&mut &buf[..]).unwrap();
        assert!(cm2.data.is_empty());
    }

    // ---------- VerifiedConstMap ----------

    #[test]
    fn test_verified_serialize_deserialize() {
        let keys = vec!["apple", "banana", "cherry", "date", "elderberry"];
        let values = vec![100u64, 200, 300, 400, 500];

        let vm = VerifiedConstMap::new(&keys, &values).unwrap();

        let mut buf = Vec::new();
        let written = vm.write_to(&mut buf).unwrap();
        assert_eq!(written as usize, buf.len());

        let vm2 = VerifiedConstMap::read_from(&mut &buf[..]).unwrap();

        for (i, &k) in keys.iter().enumerate() {
            assert_eq!(
                vm2.map(k),
                values[i],
                "after deserialize: map({}) mismatch",
                k
            );
        }
        for k in ["grape", "kiwi", "mango"] {
            assert_eq!(vm2.map(k), NOT_FOUND, "after deserialize: map({})", k);
        }
    }

    /// Both `u64` arrays must begin on a 64-bit boundary within the file.
    #[test]
    fn test_verified_serialize_alignment() {
        let (keys, values) = build_data(1000);
        let vm = VerifiedConstMap::new(&refs(&keys), &values).unwrap();

        let mut buf = Vec::new();
        vm.write_to(&mut buf).unwrap();

        assert_eq!(VERIFIED_HEADER_SIZE % 8, 0, "data must start aligned");
        let checks_offset = VERIFIED_HEADER_SIZE + 8 * vm.data.len() as u64;
        assert_eq!(checks_offset % 8, 0, "checks must start aligned");
        assert_eq!(
            buf.len() as u64,
            checks_offset + 8 * vm.data.len() as u64 + 8
        );

        // The padding word itself is zero.
        assert_eq!(&buf[28..32], &[0, 0, 0, 0]);
    }

    #[test]
    fn test_verified_serialize_chunk_boundaries() {
        for n in [
            1usize,
            2,
            IO_CHUNK_WORDS - 1,
            IO_CHUNK_WORDS,
            IO_CHUNK_WORDS + 1,
        ] {
            let (keys, values) = build_data(n);
            let vm = VerifiedConstMap::new(&refs(&keys), &values).unwrap();

            let mut buf = Vec::new();
            let written = vm.write_to(&mut buf).unwrap();
            assert_eq!(written as usize, buf.len(), "n={}", n);

            let vm2 = VerifiedConstMap::read_from(&mut &buf[..]).unwrap();
            assert_eq!(vm2.data, vm.data, "n={}", n);
            assert_eq!(vm2.checks, vm.checks, "n={}", n);
        }
    }

    #[test]
    fn test_verified_round_trip_lookups() {
        let (keys, values) = build_data(50_000);
        let vm = VerifiedConstMap::new(&refs(&keys), &values).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.vmap");
        let path_str = path.to_str().unwrap();

        vm.save_to_file(path_str).unwrap();
        let vm2 = VerifiedConstMap::load_from_file(path_str).unwrap();

        for (i, k) in keys.iter().enumerate() {
            assert_eq!(vm2.map(k), values[i], "after load: map({}) mismatch", k);
        }
        for i in 0..1000 {
            let k = format!("missing-{}", i);
            assert_eq!(vm2.map(&k), NOT_FOUND, "after load: map({})", k);
        }
    }

    #[test]
    fn test_verified_serialize_empty() {
        let vm = VerifiedConstMap::new(&[], &[]).unwrap();

        let mut buf = Vec::new();
        vm.write_to(&mut buf).unwrap();

        let vm2 = VerifiedConstMap::read_from(&mut &buf[..]).unwrap();
        assert!(vm2.data.is_empty());
        assert_eq!(vm2.map("anything"), NOT_FOUND);
    }

    #[test]
    fn test_verified_read_from_short_reads() {
        let (keys, values) = build_data(20_000);
        let vm = VerifiedConstMap::new(&refs(&keys), &values).unwrap();

        let mut buf = Vec::new();
        vm.write_to(&mut buf).unwrap();

        let mut r = OneByteAtATime(&buf[..]);
        let vm2 = VerifiedConstMap::read_from(&mut r).unwrap();
        assert_eq!(vm2.data, vm.data);
        assert_eq!(vm2.checks, vm.checks);
    }

    #[test]
    fn test_verified_read_from_truncated() {
        let (keys, values) = build_data(10_000);
        let vm = VerifiedConstMap::new(&refs(&keys), &values).unwrap();

        let mut buf = Vec::new();
        vm.write_to(&mut buf).unwrap();

        for cut in [0usize, 4, 8, 28, 32, buf.len() / 2, buf.len() - 1] {
            assert!(
                VerifiedConstMap::read_from(&mut &buf[..cut]).is_err(),
                "truncating to {} bytes should fail",
                cut
            );
        }
    }

    #[test]
    fn test_verified_serialize_corrupted() {
        let (keys, values) = build_data(1000);
        let vm = VerifiedConstMap::new(&refs(&keys), &values).unwrap();

        let mut buf = Vec::new();
        vm.write_to(&mut buf).unwrap();

        // A flip in the checks array must be caught too, not just in data.
        let last = buf.len() - 16;
        buf[last] ^= 0xff;

        assert!(VerifiedConstMap::read_from(&mut &buf[..]).is_err());
    }

    #[test]
    fn test_verified_write_to_short_writer() {
        let (keys, values) = build_data(20_000);
        let vm = VerifiedConstMap::new(&refs(&keys), &values).unwrap();

        let mut w = ShortWriter { left: 1024 };
        assert!(vm.write_to(&mut w).is_err());
    }

    #[test]
    fn test_verified_mismatched_arrays() {
        let (keys, values) = build_data(100);
        let mut vm = VerifiedConstMap::new(&refs(&keys), &values).unwrap();
        vm.checks.pop();

        let mut buf = Vec::new();
        assert!(vm.write_to(&mut buf).is_err());
    }

    /// The two formats must not be interchangeable: each reader reports the
    /// other's file rather than silently misinterpreting it.
    #[test]
    fn test_serialize_formats_are_distinct() {
        let (keys, values) = build_data(100);
        let cm = ConstMap::new(&refs(&keys), &values).unwrap();
        let vm = VerifiedConstMap::new(&refs(&keys), &values).unwrap();

        let mut cbuf = Vec::new();
        cm.write_to(&mut cbuf).unwrap();
        let mut vbuf = Vec::new();
        vm.write_to(&mut vbuf).unwrap();

        assert_eq!(&cbuf[..8], MAGIC);
        assert_eq!(&vbuf[..8], VERIFIED_MAGIC);

        let err = VerifiedConstMap::read_from(&mut &cbuf[..]).unwrap_err();
        assert!(
            err.to_string().contains("ConstMap file"),
            "unexpected error: {}",
            err
        );

        let err = ConstMap::read_from(&mut &vbuf[..]).unwrap_err();
        assert!(
            err.to_string().contains("VerifiedConstMap file"),
            "unexpected error: {}",
            err
        );

        let err = ConstMap::read_from(&mut &b"NOTAMAP!12345678"[..]).unwrap_err();
        assert!(
            err.to_string().contains("invalid magic"),
            "unexpected error: {}",
            err
        );
    }

    // ---------- PairedVerifiedConstMap ----------

    #[test]
    fn test_paired_round_trip_lookups() {
        let (keys, values) = build_data(50_000);
        let pm = PairedVerifiedConstMap::new(&refs(&keys), &values).unwrap();

        let mut buf = Vec::new();
        let written = pm.write_to(&mut buf).unwrap();
        assert_eq!(written as usize, buf.len());

        let pm2 = PairedVerifiedConstMap::read_from(&mut &buf[..]).unwrap();
        for (i, k) in keys.iter().enumerate() {
            assert_eq!(
                pm2.map(k),
                values[i],
                "after deserialize: map({}) mismatch",
                k
            );
        }
        for i in 0..10_000 {
            let k = format!("absent-{}", i);
            assert_eq!(
                pm2.map(&k),
                NOT_FOUND,
                "after deserialize: map({}) should be NOT_FOUND",
                k
            );
        }
    }

    #[test]
    fn test_paired_round_trip_file() {
        let (keys, values) = build_data(1000);
        let pm = PairedVerifiedConstMap::new(&refs(&keys), &values).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("paired.cmap");
        let path = path.to_str().unwrap();
        pm.save_to_file(path).unwrap();
        let pm2 = PairedVerifiedConstMap::load_from_file(path).unwrap();
        for (i, k) in keys.iter().enumerate() {
            assert_eq!(pm2.map(k), values[i]);
        }
        assert_eq!(pm2.map("not-there"), NOT_FOUND);
    }

    #[test]
    fn test_paired_serialize_empty() {
        let pm = PairedVerifiedConstMap::new(&[], &[]).unwrap();
        let mut buf = Vec::new();
        pm.write_to(&mut buf).unwrap();
        let pm2 = PairedVerifiedConstMap::read_from(&mut &buf[..]).unwrap();
        assert!(pm2.slots.is_empty());
        assert_eq!(pm2.map("anything"), NOT_FOUND);
    }

    /// The paired file is the verified file's size: same header, same number
    /// of words, same trailer.
    #[test]
    fn test_paired_serialize_same_size_as_verified() {
        let (keys, values) = build_data(1000);
        let vm = VerifiedConstMap::new(&refs(&keys), &values).unwrap();
        let mut vbuf = Vec::new();
        let vn = vm.write_to(&mut vbuf).unwrap();
        let mut pbuf = Vec::new();
        let pn = vm.paired().write_to(&mut pbuf).unwrap();
        assert_eq!(vn, pn);
        assert_eq!(vbuf.len(), pbuf.len());
    }

    #[test]
    fn test_paired_serialize_corrupted() {
        let (keys, values) = build_data(1000);
        let pm = PairedVerifiedConstMap::new(&refs(&keys), &values).unwrap();
        let mut buf = Vec::new();
        pm.write_to(&mut buf).unwrap();

        // Flip a value byte, a fingerprint byte, and a byte of the checksum
        // itself; all three must be reported as checksum mismatches.
        let n = buf.len();
        for at in [
            VERIFIED_HEADER_SIZE as usize + 40,
            VERIFIED_HEADER_SIZE as usize + 48,
            n - 3,
        ] {
            let mut corrupt = buf.clone();
            corrupt[at] ^= 0xff;
            let err = PairedVerifiedConstMap::read_from(&mut &corrupt[..]).unwrap_err();
            assert!(
                err.to_string().contains("checksum"),
                "flipping byte {}: unexpected error: {}",
                at,
                err
            );
        }
    }

    #[test]
    fn test_paired_serialize_truncated() {
        let (keys, values) = build_data(1000);
        let pm = PairedVerifiedConstMap::new(&refs(&keys), &values).unwrap();
        let mut buf = Vec::new();
        pm.write_to(&mut buf).unwrap();
        for len in [0, 8, 31, 32, 100, buf.len() - 1] {
            assert!(
                PairedVerifiedConstMap::read_from(&mut &buf[..len]).is_err(),
                "truncating to {} bytes was not caught",
                len
            );
        }
    }

    /// A file whose segment parameters do not describe its slot count is
    /// refused even when its checksum is valid.
    #[test]
    fn test_paired_read_rejects_inconsistent_parameters() {
        let (keys, values) = build_data(1000);
        let pm = PairedVerifiedConstMap::new(&refs(&keys), &values).unwrap();
        let mut buf = Vec::new();
        pm.write_to(&mut buf).unwrap();

        fn rewrite_checksum(raw: &mut [u8]) {
            let n = raw.len() - 8;
            let mut h = Fnv64a::new();
            h.write(&raw[..n]);
            raw[n..].copy_from_slice(&h.finish().to_le_bytes());
        }

        // Double the segment count.
        let mut bad = buf.clone();
        let segment_count = u32::from_le_bytes(bad[20..24].try_into().unwrap());
        bad[20..24].copy_from_slice(&(2 * segment_count).to_le_bytes());
        rewrite_checksum(&mut bad);
        let err = PairedVerifiedConstMap::read_from(&mut &bad[..]).unwrap_err();
        assert!(
            err.to_string().contains("segment parameters"),
            "unexpected error: {}",
            err
        );

        // A segment length that is not a power of two.
        let mut bad = buf.clone();
        bad[16..20].copy_from_slice(&3u32.to_le_bytes());
        rewrite_checksum(&mut bad);
        let err = PairedVerifiedConstMap::read_from(&mut &bad[..]).unwrap_err();
        assert!(
            err.to_string().contains("power of two"),
            "unexpected error: {}",
            err
        );

        // A crafted file that passes the other two checks with a zero
        // segment count: segment_length 4, segment_count 0, 8 slots, so
        // (0 + 2) * 4 == 8. With it, h0 is always 0 and h2 lands in [8, 12),
        // past the slots.
        let mut crafted = Vec::new();
        crafted.extend_from_slice(PAIRED_MAGIC);
        crafted.extend_from_slice(&0u64.to_le_bytes()); // seed
        crafted.extend_from_slice(&4u32.to_le_bytes()); // segment_length
        crafted.extend_from_slice(&0u32.to_le_bytes()); // segment_count
        crafted.extend_from_slice(&8u32.to_le_bytes()); // slots
        crafted.extend_from_slice(&0u32.to_le_bytes()); // padding
        crafted.extend_from_slice(&[0u8; 8 * 16]);
        crafted.extend_from_slice(&[0u8; 8]); // checksum, rewritten below
        rewrite_checksum(&mut crafted);
        let err = PairedVerifiedConstMap::read_from(&mut &crafted[..]).unwrap_err();
        assert!(
            err.to_string().contains("segment count"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn test_paired_write_to_short_writer() {
        let (keys, values) = build_data(1000);
        let pm = PairedVerifiedConstMap::new(&refs(&keys), &values).unwrap();
        for left in [0, 8, 32, 100] {
            assert!(pm.write_to(&mut ShortWriter { left }).is_err());
        }
    }

    /// The paired reader refuses the other two formats, and the other two
    /// readers refuse a paired file.
    #[test]
    fn test_paired_format_is_distinct() {
        let (keys, values) = build_data(100);
        let cm = ConstMap::new(&refs(&keys), &values).unwrap();
        let vm = VerifiedConstMap::new(&refs(&keys), &values).unwrap();
        let pm = vm.paired();

        let mut cbuf = Vec::new();
        cm.write_to(&mut cbuf).unwrap();
        let mut vbuf = Vec::new();
        vm.write_to(&mut vbuf).unwrap();
        let mut pbuf = Vec::new();
        pm.write_to(&mut pbuf).unwrap();
        assert_eq!(&pbuf[..8], PAIRED_MAGIC);

        let err = PairedVerifiedConstMap::read_from(&mut &cbuf[..]).unwrap_err();
        assert!(
            err.to_string().contains("ConstMap file"),
            "unexpected error: {}",
            err
        );
        let err = PairedVerifiedConstMap::read_from(&mut &vbuf[..]).unwrap_err();
        assert!(
            err.to_string().contains("VerifiedConstMap file"),
            "unexpected error: {}",
            err
        );
        let err = ConstMap::read_from(&mut &pbuf[..]).unwrap_err();
        assert!(
            err.to_string().contains("PairedVerifiedConstMap file"),
            "unexpected error: {}",
            err
        );
        let err = VerifiedConstMap::read_from(&mut &pbuf[..]).unwrap_err();
        assert!(
            err.to_string().contains("PairedVerifiedConstMap file"),
            "unexpected error: {}",
            err
        );
    }

    // ---------- helpers ----------

    /// A reader that returns at most one byte per call.
    struct OneByteAtATime<'a>(&'a [u8]);

    impl Read for OneByteAtATime<'_> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if buf.is_empty() || self.0.is_empty() {
                return Ok(0);
            }
            buf[0] = self.0[0];
            self.0 = &self.0[1..];
            Ok(1)
        }
    }

    /// A writer that accepts a fixed number of bytes and then fails.
    struct ShortWriter {
        left: usize,
    }

    impl Write for ShortWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            if self.left == 0 {
                return Err(io::Error::new(io::ErrorKind::WriteZero, "no space left"));
            }
            let n = buf.len().min(self.left);
            self.left -= n;
            Ok(n)
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
}
