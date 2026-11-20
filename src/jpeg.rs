//! A baseline JPEG header parser: just what the hardware decoder needs (size, sampling, quantization and Huffman
//! tables, restart interval, where the entropy-coded data starts). The pixels are decoded by the hardware.
//!
//! Mirrors MPP `mpp/codec/dec/jpeg/jpegd_parser.c` (baseline SOF0 only, the standard tables when there is no DHT).

use anyhow::{bail, ensure, Context, Result};

/// One Huffman table as stored in DHT: code counts per length, then the symbols.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Huffman {
    pub bits: [u8; 16],
    pub vals: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Component {
    pub id: u8,
    pub h: u8,
    pub v: u8,
    pub quant: u8,
    pub dc: u8,
    pub ac: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sampling {
    Yuv420,
    Yuv422,
}

#[derive(Clone, Debug)]
pub struct Header {
    pub width: u32,
    pub height: u32,
    pub components: Vec<Component>,
    /// Quantization tables by id, in the file's zig-zag order.
    pub quant: [[u16; 64]; 4],
    pub dc: [Option<Huffman>; 4],
    pub ac: [Option<Huffman>; 4],
    pub restart_interval: u16,
    /// Byte offset of the first entropy-coded byte (just after the SOS header).
    pub scan_offset: usize,
}

impl Header {
    pub fn sampling(&self) -> Result<Sampling> {
        let s: Vec<(u8, u8)> = self.components.iter().map(|c| (c.h, c.v)).collect();
        match s.as_slice() {
            [(2, 2), (1, 1), (1, 1)] => Ok(Sampling::Yuv420),
            [(2, 1), (1, 1), (1, 1)] => Ok(Sampling::Yuv422),
            other => bail!("unsupported JPEG sampling {other:?} (only YUV 4:2:0 and 4:2:2)"),
        }
    }
}

const SOI: u8 = 0xd8;
const EOI: u8 = 0xd9;
const SOF0: u8 = 0xc0;
const DHT: u8 = 0xc4;
const DQT: u8 = 0xdb;
const DRI: u8 = 0xdd;
const SOS: u8 = 0xda;

fn be16(data: &[u8], at: usize) -> Result<u16> {
    let b = data.get(at..at + 2).context("JPEG truncated")?;
    Ok(u16::from_be_bytes([b[0], b[1]]))
}

/// Parses everything up to and including SOS.
pub fn parse(data: &[u8]) -> Result<Header> {
    ensure!(
        data.len() > 4 && data[0] == 0xff && data[1] == SOI,
        "not a JPEG (no SOI)"
    );
    let mut header = Header {
        width: 0,
        height: 0,
        components: Vec::new(),
        quant: [[0; 64]; 4],
        dc: Default::default(),
        ac: Default::default(),
        restart_interval: 0,
        scan_offset: 0,
    };
    let mut found_dht = false;
    let mut at = 2;
    loop {
        // markers may be padded with any number of 0xff fill bytes
        ensure!(
            data.get(at) == Some(&0xff),
            "JPEG marker expected at byte {at}"
        );
        while data.get(at) == Some(&0xff) {
            at += 1;
        }
        let marker = *data.get(at).context("JPEG truncated")?;
        at += 1;
        if marker == EOI {
            bail!("JPEG ends before its scan");
        }
        if (0xd0..=0xd7).contains(&marker) || marker == 0x01 {
            continue;
        }
        let len = be16(data, at)? as usize;
        ensure!(len >= 2, "bad JPEG segment length");
        let seg = data
            .get(at + 2..at + len)
            .context("JPEG segment runs past the end")?;
        match marker {
            SOF0 => {
                ensure!(seg.len() >= 6, "short SOF0");
                ensure!(seg[0] == 8, "{}-bit JPEG (only 8-bit baseline)", seg[0]);
                header.height = u16::from_be_bytes([seg[1], seg[2]]) as u32;
                header.width = u16::from_be_bytes([seg[3], seg[4]]) as u32;
                let n = seg[5] as usize;
                ensure!(n == 3, "{n}-component JPEG (only YCbCr)");
                ensure!(seg.len() >= 6 + 3 * n, "short SOF0");
                for i in 0..n {
                    let c = &seg[6 + 3 * i..9 + 3 * i];
                    ensure!(c[2] < 4, "quantization table id {} out of range", c[2]);
                    header.components.push(Component {
                        id: c[0],
                        h: c[1] >> 4,
                        v: c[1] & 15,
                        quant: c[2],
                        dc: 0,
                        ac: 0,
                    });
                }
            }
            0xc1..=0xcf if marker != DHT && marker != 0xc8 && marker != 0xcc => {
                bail!("JPEG SOF{} (only baseline SOF0)", marker - 0xc0)
            }
            DHT => {
                let mut p = 0;
                while p < seg.len() {
                    let class = seg[p] >> 4;
                    let id = (seg[p] & 15) as usize;
                    ensure!(class <= 1 && id < 4, "bad DHT table class/id");
                    let mut bits = [0u8; 16];
                    bits.copy_from_slice(seg.get(p + 1..p + 17).context("short DHT")?);
                    let count: usize = bits.iter().map(|&b| b as usize).sum();
                    let max = if class == 0 { 12 } else { 162 };
                    ensure!(
                        count <= max,
                        "DHT with {count} codes (baseline allows {max})"
                    );
                    let vals = seg
                        .get(p + 17..p + 17 + count)
                        .context("short DHT")?
                        .to_vec();
                    let table = Some(Huffman { bits, vals });
                    if class == 0 {
                        header.dc[id] = table;
                    } else {
                        header.ac[id] = table;
                    }
                    p += 17 + count;
                }
                found_dht = true;
            }
            DQT => {
                let mut p = 0;
                while p < seg.len() {
                    let precision = seg[p] >> 4;
                    let id = (seg[p] & 15) as usize;
                    ensure!(precision <= 1 && id < 4, "bad DQT precision/id");
                    let size = if precision == 0 { 64 } else { 128 };
                    let body = seg.get(p + 1..p + 1 + size).context("short DQT")?;
                    for i in 0..64 {
                        header.quant[id][i] = if precision == 0 {
                            body[i] as u16
                        } else {
                            u16::from_be_bytes([body[2 * i], body[2 * i + 1]])
                        };
                    }
                    p += 1 + size;
                }
            }
            DRI => {
                ensure!(seg.len() >= 2, "short DRI");
                header.restart_interval = u16::from_be_bytes([seg[0], seg[1]]);
            }
            SOS => {
                ensure!(
                    !header.components.is_empty(),
                    "JPEG scan before its frame header"
                );
                let n = *seg.first().context("short SOS")? as usize;
                ensure!(
                    n == header.components.len(),
                    "non-interleaved JPEG scan ({n} components)"
                );
                for i in 0..n {
                    let s = seg.get(1 + 2 * i..3 + 2 * i).context("short SOS")?;
                    let component = header
                        .components
                        .iter_mut()
                        .find(|c| c.id == s[0])
                        .context("SOS names a component the frame lacks")?;
                    component.dc = s[1] >> 4;
                    component.ac = s[1] & 15;
                    ensure!(
                        component.dc < 4 && component.ac < 4,
                        "Huffman table id out of range"
                    );
                }
                header.scan_offset = at + len;
                break;
            }
            _ => {}
        }
        at += len;
    }
    if !found_dht {
        let (dc, ac) = standard_tables();
        header.dc = dc;
        header.ac = ac;
    }
    for c in &header.components {
        ensure!(
            header.dc[c.dc as usize].is_some() && header.ac[c.ac as usize].is_some(),
            "component {} uses a Huffman table that is not defined",
            c.id
        );
    }
    ensure!(header.width > 0 && header.height > 0, "JPEG without a size");
    Ok(header)
}

/// The length of the JPEG through its EOI marker (a capture buffer can carry padding after it).
pub fn trimmed_len(data: &[u8]) -> usize {
    let start = data.len().saturating_sub(4096);
    data[start..]
        .windows(2)
        .rposition(|w| w == [0xff, EOI])
        .map(|p| start + p + 2)
        .unwrap_or(data.len())
}

/// The JPEG standard's Annex K tables, for streams (most UVC MJPEG) that leave out DHT.
fn standard_tables() -> ([Option<Huffman>; 4], [Option<Huffman>; 4]) {
    const DC_LUMA_BITS: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
    const DC_CHROMA_BITS: [u8; 16] = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
    const AC_LUMA_BITS: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d];
    const AC_CHROMA_BITS: [u8; 16] = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77];
    #[rustfmt::skip]
    const AC_LUMA_VALS: [u8; 162] = [
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
        0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
        0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
        0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
        0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
        0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
        0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
        0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
        0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
        0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
        0xf9, 0xfa,
    ];
    #[rustfmt::skip]
    const AC_CHROMA_VALS: [u8; 162] = [
        0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
        0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
        0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
        0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
        0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
        0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
        0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
        0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
        0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
        0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
        0xf9, 0xfa,
    ];
    let dc_vals: Vec<u8> = (0..12).collect();
    let table = |bits: [u8; 16], vals: &[u8]| {
        Some(Huffman {
            bits,
            vals: vals.to_vec(),
        })
    };
    (
        [
            table(DC_LUMA_BITS, &dc_vals),
            table(DC_CHROMA_BITS, &dc_vals),
            None,
            None,
        ],
        [
            table(AC_LUMA_BITS, &AC_LUMA_VALS),
            table(AC_CHROMA_BITS, &AC_CHROMA_VALS),
            None,
            None,
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal 16x8 4:2:2 JPEG header: DQT, SOF0, SOS (no DHT, like UVC MJPEG).
    fn header_422_without_dht() -> Vec<u8> {
        let mut j = vec![0xff, 0xd8];
        j.extend([0xff, 0xdb, 0x00, 0x43, 0x00]);
        j.extend((1..=64).map(|q| q as u8));
        j.extend([0xff, 0xc0, 0x00, 0x11, 0x08, 0x00, 0x08, 0x00, 0x10, 0x03]);
        j.extend([0x01, 0x21, 0x00, 0x02, 0x11, 0x00, 0x03, 0x11, 0x00]);
        j.extend([
            0xff, 0xda, 0x00, 0x0c, 0x03, 0x01, 0x00, 0x02, 0x11, 0x03, 0x11, 0x00, 0x3f, 0x00,
        ]);
        j.extend([0x12, 0x34, 0xff, 0xd9, 0, 0, 0]);
        j
    }

    #[test]
    fn parses_a_dht_less_422_header() {
        let data = header_422_without_dht();
        let h = parse(&data).unwrap();
        assert_eq!((h.width, h.height), (16, 8));
        assert_eq!(h.sampling().unwrap(), Sampling::Yuv422);
        assert_eq!(h.quant[0][63], 64);
        assert_eq!(h.components[1].dc, 1);
        assert_eq!(h.ac[1].as_ref().unwrap().vals.len(), 162);
        assert_eq!(data[h.scan_offset], 0x12);
        assert_eq!(trimmed_len(&data), data.len() - 3);
    }

    #[test]
    fn refuses_progressive() {
        let mut data = header_422_without_dht();
        let sof = data.windows(2).position(|w| w == [0xff, 0xc0]).unwrap();
        data[sof + 1] = 0xc2;
        assert!(format!("{:#}", parse(&data).unwrap_err()).contains("SOF2"));
    }
}
