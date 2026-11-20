//! The RK3588 hardware JPEG decoder (RKJPEGD, MPP's "jpegd_rkv"): one 42-register task per frame, the Huffman and
//! quantization tables in a small dmabuf, output YUV semi-planar (NV12 for 4:2:0, NV16 for 4:2:2) with the luma plane
//! `width * height` bytes and the chroma plane right after it.
//!
//! Mirrors MPP `mpp/hal/vpu/jpegd/hal_jpegd_rkv.c`, `hal_jpegd_vpu7xx_com.c`, `hal_jpegd_rkv_reg.h` and the kernel's
//! `mpp_jpgdec.c` (registers 9..=13 hold dmabuf fds the kernel translates).

use anyhow::{bail, ensure, Context, Result};

use crate::dma::{DmaBuf, Heap};
use crate::jpeg::{self, Header, Huffman, Sampling};
use crate::service::{client, AddrOffset, Read, Session, Write};

const REG_COUNT: usize = 42;
const QUANT_OFFSET: usize = 0;
const MINCODE_OFFSET: usize = 0x180;
const VALUE_OFFSET: usize = 0x2c0;
const TABLE_SIZE: usize = 4096;

/// The zig-zag scan position -> raster position (MPP `zzOrder`).
#[rustfmt::skip]
const ZIGZAG_TO_RASTER: [u8; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// `jpeg_mode` values (hal_jpegd_vpu7xx_com.h `YUV_MODE_*`).
fn jpeg_mode(sampling: Sampling) -> u32 {
    match sampling {
        Sampling::Yuv420 => 2,
        Sampling::Yuv422 => 3,
    }
}

pub struct JpegDecoder {
    session: Session,
    tables: DmaBuf,
    stream: DmaBuf,
    width: u32,
    height: u32,
}

/// The decoder's register words for one frame; `table_fd`/`stream_fd`/`out_fd` go in the address registers.
pub fn registers(
    header: &Header,
    stream_len: usize,
    table_fd: u32,
    stream_fd: u32,
    out_fd: u32,
) -> Result<([u32; REG_COUNT], [AddrOffset; 3])> {
    let sampling = header.sampling()?;
    let (cb, cr) = (header.components[1], header.components[2]);
    ensure!(
        cb.dc == cr.dc && cb.ac == cr.ac,
        "Cb and Cr use different Huffman tables (the decoder shares one set)"
    );
    let mut r = [0u32; REG_COUNT];
    // reg1: dec_e, dec_timeout_e, buf_empty_e
    r[1] = 1 | 1 << 2 | 1 << 3;
    // reg2: fill_down_e (always on for 4:2:0; MPP's fill_bottom for 4:2:2), raster output, no format conversion
    let fill_down = match sampling {
        Sampling::Yuv420 => true,
        Sampling::Yuv422 => header.height & 15 != 0 && header.height & 15 <= 8,
    };
    r[2] = (fill_down as u32) << 24;
    r[3] = (header.height - 1) << 16 | (header.width - 1);
    let qtables_sel = if cr.quant != cb.quant { 3 } else { 2 };
    let htables_sel = 2;
    r[4] = jpeg_mode(sampling) | qtables_sel << 8 | htables_sel << 12;
    if header.restart_interval != 0 {
        r[4] |= 1 << 15 | (header.restart_interval as u32 - 1) << 16;
    }
    let y_hor_stride = header.width.div_ceil(16);
    let out_height = if fill_down {
        header.height.next_multiple_of(16)
    } else {
        header.height.next_multiple_of(8)
    };
    r[5] = y_hor_stride | y_hor_stride << 16;
    r[6] = (y_hor_stride * out_height) << 4;
    let qtbl_len = qtables_sel * 8 - 1;
    let htbl_value_len = htables_sel * 12 - 1;
    let htbl_mincode_len = (3 - 1) * 6 - 1;
    r[7] =
        qtbl_len | htbl_mincode_len << 8 | htbl_value_len << 16 | ((y_hor_stride >> 16) & 1) << 24;
    let hw_offset = header.scan_offset & !15;
    let start_byte = (header.scan_offset % 16) as u32;
    let stream_words = ((stream_len - hw_offset).next_multiple_of(16) - 1) >> 4;
    r[8] = start_byte | (stream_words as u32) << 4;
    r[9] = table_fd;
    r[10] = table_fd;
    r[11] = table_fd;
    r[12] = stream_fd;
    r[13] = out_fd;
    // reg14: error_prc_mode=1, strm_ffff_err_mode=2, strm_other_mark_mode=2
    r[14] = 1 | 2 << 5 | 2 << 7;
    r[16] = 0xff;
    // reg30: axi_perf_work_e, axi_perf_clr_e, axi_cnt_type, rd_latency_id=0xa
    r[30] = 1 | 1 << 1 | 1 << 3 | 0xa << 4;
    let offsets = [
        AddrOffset {
            index: 12,
            offset: hw_offset as u32,
        },
        AddrOffset {
            index: 10,
            offset: MINCODE_OFFSET as u32,
        },
        AddrOffset {
            index: 11,
            offset: VALUE_OFFSET as u32,
        },
    ];
    Ok((r, offsets))
}

/// Writes the tables the hardware reads: per component a raster-order quantization table, then per component the
/// Huffman min-code / accumulated-address words and the symbol values (components 1 and 2 share one set).
pub fn write_tables(header: &Header, table: &mut [u8]) -> Result<()> {
    table[..VALUE_OFFSET + 3 * 192].fill(0);
    for (j, c) in header.components.iter().enumerate() {
        let q = &header.quant[c.quant as usize];
        for (i, &raster) in ZIGZAG_TO_RASTER.iter().enumerate() {
            let at = QUANT_OFFSET + (j * 64 + raster as usize) * 2;
            table[at..at + 2].copy_from_slice(&q[i].to_le_bytes());
        }
    }
    let sets = [
        (header.components[0].dc, header.components[0].ac),
        (header.components[1].dc, header.components[1].ac),
        (header.components[1].dc, header.components[1].ac),
    ];
    for (k, (dc, ac)) in sets.into_iter().enumerate() {
        let dc = header.dc[dc as usize]
            .as_ref()
            .context("missing DC table")?;
        let ac = header.ac[ac as usize]
            .as_ref()
            .context("missing AC table")?;
        let mut words = Vec::with_capacity(48);
        for table in [dc, ac] {
            let (min_code, acc_addr) = min_codes(table);
            words.extend_from_slice(&min_code);
            words.extend((0..8).map(|i| acc_addr[2 * i] | acc_addr[2 * i + 1] << 8));
        }
        for (i, w) in words.iter().enumerate() {
            let at = MINCODE_OFFSET + (k * 48 + i) * 2;
            table[at..at + 2].copy_from_slice(&w.to_le_bytes());
        }
        let values = &mut table[VALUE_OFFSET + k * 192..VALUE_OFFSET + (k + 1) * 192];
        values[..dc.vals.len()].copy_from_slice(&dc.vals);
        values[16..16 + ac.vals.len()].copy_from_slice(&ac.vals);
    }
    Ok(())
}

/// MPP `jpegd_vpu7xx_write_htbl`'s per-table min code per length and running code count.
fn min_codes(table: &Huffman) -> ([u16; 16], [u16; 16]) {
    let mut min_code = [0u16; 16];
    let mut acc_addr = [0u16; 16];
    let (mut code, mut addr) = (0u16, 0u16);
    for j in 0..16 {
        let len = table.bits[j] as u16;
        min_code[j] = if len == 0 && j > 0 {
            code.max(min_code[j - 1] << 1)
        } else {
            code
        };
        code += len;
        addr += len;
        acc_addr[j] = addr;
        code <<= 1;
    }
    min_code[0] = if table.bits[15] != 0 {
        min_code[15] + table.bits[15] as u16 - 1
    } else {
        min_code[15]
    };
    (min_code, acc_addr)
}

impl JpegDecoder {
    /// A decoder session for `width`x`height` JPEGs of up to `max_jpeg` bytes.
    pub fn new(heap: &Heap, width: u32, height: u32, max_jpeg: usize) -> Result<Self> {
        Ok(Self {
            session: Session::new(client::RKJPEGD).context("opening the JPEG decoder")?,
            tables: heap.alloc(TABLE_SIZE)?,
            stream: heap.alloc(max_jpeg.next_multiple_of(4096) + 4096)?,
            width,
            height,
        })
    }

    /// Decodes `jpeg` into `out` (at least `width * height * 2` bytes); returns its sampling.
    pub fn decode(&mut self, jpeg: &[u8], out: &DmaBuf) -> Result<Sampling> {
        let len = jpeg::trimmed_len(jpeg);
        let jpeg = &jpeg[..len];
        let header = jpeg::parse(jpeg)?;
        if (header.width, header.height) != (self.width, self.height) {
            bail!(
                "JPEG is {}x{}, decoder set up for {}x{}",
                header.width,
                header.height,
                self.width,
                self.height
            );
        }
        ensure!(
            len + 64 <= self.stream.len(),
            "{len}-byte JPEG is larger than the {}-byte input buffer",
            self.stream.len()
        );
        ensure!(
            out.len() >= (self.width * self.height * 2) as usize,
            "output buffer too small"
        );
        let sampling = header.sampling()?;
        self.stream.write(|b| {
            b[..len].copy_from_slice(jpeg);
            // the hardware reads whole 16-byte words past the end of the stream
            b[len..len + 64].fill(0);
        })?;
        self.tables.write(|t| write_tables(&header, t))??;
        let (regs, offsets) =
            registers(&header, len, self.tables.fd(), self.stream.fd(), out.fd())?;
        let mut status = [0u32; REG_COUNT];
        self.session.run(
            &[Write {
                offset: 0,
                words: &regs,
            }],
            &mut [Read {
                offset: 0,
                words: &mut status,
            }],
            &offsets,
        )?;
        let irq = status[1];
        let ok = irq & (1 << 8) != 0 && irq & (1 << 9) != 0 && irq & (0xf << 10) == 0;
        ensure!(ok, "JPEG decode failed: interrupt status {irq:#010x}");
        Ok(sampling)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The decoder registers MPP (the vendor gst plugin's build) programmed for /tmp/jt/f000.jpg on AOS: 1280x720
    /// 4:2:0, entropy data at byte 0x26f of a 15495-byte file (fds replaced by 1, 2, 3).
    #[test]
    fn matches_the_vendor_registers_for_a_720p_420_frame() {
        let mut header = jpeg::parse(&synthetic_720p_header()).unwrap();
        header.scan_offset = 0x26f;
        let (r, offsets) = registers(&header, 0x3c87, 1, 2, 3).unwrap();
        let want: [u32; 17] = [
            0, 0x0d, 0x01000000, 0x02cf04ff, 0x00002202, 0x00500050, 0x000e1000, 0x00170b0f,
            0x00003a2f, 1, 1, 1, 2, 3, 0x141, 0, 0xff,
        ];
        assert_eq!(&r[..17], &want);
        assert_eq!(r[30], 0xab);
        assert_eq!(
            offsets[0],
            AddrOffset {
                index: 12,
                offset: 0x260
            }
        );
        assert_eq!(
            offsets[1],
            AddrOffset {
                index: 10,
                offset: 0x180
            }
        );
        assert_eq!(
            offsets[2],
            AddrOffset {
                index: 11,
                offset: 0x2c0
            }
        );
    }

    fn synthetic_720p_header() -> Vec<u8> {
        let mut j = vec![0xff, 0xd8, 0xff, 0xdb, 0x00, 0x84, 0x00];
        j.extend([1u8; 64]);
        j.push(0x01);
        j.extend([2u8; 64]);
        j.extend([0xff, 0xc0, 0x00, 0x11, 0x08, 0x02, 0xd0, 0x05, 0x00, 0x03]);
        j.extend([0x01, 0x22, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01]);
        j.extend([
            0xff, 0xda, 0x00, 0x0c, 0x03, 0x01, 0x00, 0x02, 0x11, 0x03, 0x11, 0x00, 0x3f, 0x00,
        ]);
        j.extend([0u8; 16]);
        j
    }

    #[test]
    fn min_codes_of_the_standard_luma_dc_table() {
        let h = jpeg::parse(&synthetic_720p_header()).unwrap();
        let (min, acc) = min_codes(h.dc[0].as_ref().unwrap());
        // one 2-bit code, five 3-bit codes, then one per length: canonical first codes 0b00, 0b010, 0b1110, 0b11110
        assert_eq!(&min[1..5], &[0, 2, 14, 30]);
        assert_eq!(acc[15], 12);
    }
}
