//! Bit level IO operations


use std::io::{self, Read, Write};
use std::cmp;
use std::sync::Arc;
use byteorder::{ReadBytesExt, WriteBytesExt, LittleEndian, ByteOrder};

#[cfg(feature="cgmath")]
mod cgmath;

pub trait DeltaEncodable: Sized {

    fn encode<W>(&self, base: Option<&Self>, w: &mut Writer<W>) -> io::Result<()>
        where W: Write;

    fn decode<R>(base: Option<&Self>, r: &mut Reader<R>) -> io::Result<Self>
        where R: Read;
}

impl <T> DeltaEncodable for Arc<T>
    where T: DeltaEncodable
{
    #[inline]
    fn encode<W>(&self, base: Option<&Self>, w: &mut Writer<W>) -> io::Result<()>
        where W: Write
    {
        T::encode(self, base.map(|v| &**v), w)
    }

    #[inline]
    fn decode<R>(base: Option<&Self>, r: &mut Reader<R>) -> io::Result<Self>
        where R: Read
    {
        Ok(Arc::new(T::decode(base.map(|v| &**v), r)?))
    }
}



/// Encodes a `&str` without allocating.
///
/// Only encodes if the passed string is different from the base
/// otherwise it just uses a single bit
pub fn encode_str<W>(v: &str, base: Option<&str>, w: &mut Writer<W>) -> io::Result<()>
    where W: Write
{
    if let Some(base) = base {
        if base == v {
            w.write_bool(false)?;
            return Ok(())
        }
    }
    w.write_bool(true)?;
    write_str(v, w)
}


/// Writes a `&str` without allocating
pub fn write_str<W>(v: &str, w: &mut Writer<W>) -> io::Result<()>
    where W: Write
{
    let bytes = v.as_bytes();
    let len = bytes.len();

    write_len_bits(w, len)?;
    for val in bytes {
        w.write_unsigned(u64::from(*val), 8)?;
    }
    Ok(())
}

/// Decodes a `String` without having to allocate for the base
/// parameter.
///
/// Clones the base if the string is the same as before.
pub fn decode_string<R>(base: Option<&str>, r: &mut Reader<R>) -> io::Result<String>
    where R: Read
{
    if r.read_bool()? {
        read_string(r)
    } else if let Some(base) = base {
            Ok(base.to_owned())
    } else {
        Err(io::Error::new(io::ErrorKind::InvalidInput, "Missing previous string state"))
    }
}

/// Reads a string from the reader
pub fn read_string<R>(r: &mut Reader<R>) -> io::Result<String>
    where R: Read
{
    let len = read_len_bits(r)?;
    let mut buf = Vec::with_capacity(len);
    for _ in 0 .. len {
        buf.push(r.read_unsigned(8)? as u8);
    }
    match String::from_utf8(buf) {
        Ok(val) => Ok(val),
        Err(err) => Err(io::Error::new(io::ErrorKind::InvalidData, err))
    }
}

impl DeltaEncodable for String {
    #[inline]
    fn encode<W>(&self, base: Option<&Self>, w: &mut Writer<W>) -> io::Result<()>
        where W: Write
    {
        encode_str(self, base.map(|v| v.as_str()), w)
    }

    #[inline]
    fn decode<R>(base: Option<&Self>, r: &mut Reader<R>) -> io::Result<Self>
        where R: Read
    {
        decode_string(base.map(|v| v.as_str()), r)
    }
}

impl DeltaEncodable for Arc<str> {
    #[inline]
    fn encode<W>(&self, base: Option<&Self>, w: &mut Writer<W>) -> io::Result<()>
        where W: Write
    {
        encode_str(self, base.map(|v| &**v), w)
    }

    #[inline]
    fn decode<R>(base: Option<&Self>, r: &mut Reader<R>) -> io::Result<Self>
        where R: Read
    {
        decode_string(base.map(|v| &**v), r).map(|v| v.into())
    }
}

#[derive(Debug, PartialOrd, PartialEq, Clone)]
pub struct AlwaysVec<T>(pub Vec<T>);

impl <T> DeltaEncodable for AlwaysVec<T>
    where T: DeltaEncodable
{
    #[inline]
    fn encode<W>(&self, base: Option<&Self>, w: &mut Writer<W>) -> io::Result<()>
        where W: Write
    {
        write_len_bits(w, self.0.len())?;
        for (idx, val) in self.0.iter().enumerate() {
            T::encode(val, base.and_then(|v | v.0.get(idx)), w)?;
        }
        Ok(())
    }

    #[inline]
    fn decode<R>(base: Option<&Self>, r: &mut Reader<R>) -> io::Result<Self>
        where R: Read
    {
        let len = read_len_bits(r)?;
        let mut buf = Vec::with_capacity(len);
        for idx in 0 .. len {
            buf.push(T::decode(base.and_then(|v| v.0.get(idx)), r)?);
        }
        Ok(AlwaysVec(buf))
    }
}

impl <T> DeltaEncodable for Vec<T>
    where T: DeltaEncodable,
          Vec<T>: PartialEq + Clone
{
    #[inline]
    fn encode<W>(&self, base: Option<&Self>, w: &mut Writer<W>) -> io::Result<()>
        where W: Write
    {
        if let Some(base) = base {
            if base == self {
                w.write_bool(false)?;
                return Ok(())
            }
        }
        w.write_bool(true)?;

        write_len_bits(w, self.len())?;
        for (idx, val) in self.iter().enumerate() {
            T::encode(val, base.and_then(|v | v.get(idx)), w)?;
        }
        Ok(())
    }

    #[inline]
    fn decode<R>(base: Option<&Self>, r: &mut Reader<R>) -> io::Result<Self>
        where R: Read
    {
        if r.read_bool()? {
            let len = read_len_bits(r)?;
            let mut buf = Vec::with_capacity(len);
            for idx in 0 .. len {
                buf.push(T::decode(base.and_then(|v| v.get(idx)), r)?);
            }
            Ok(buf)
        } else if let Some(base) = base {
            Ok(base.to_owned())
        } else {
            Err(io::Error::new(io::ErrorKind::InvalidInput, "Missing previous vec state"))
        }
    }
}

impl <T> DeltaEncodable for Option<T>
    where T: DeltaEncodable
{
    #[inline]
    fn encode<W>(&self, base: Option<&Self>, w: &mut Writer<W>) -> io::Result<()>
        where W: Write
    {
        if let Some(ref s) = *self {
            w.write_bool(true)?;
            T::encode(s, base.and_then(|v| v.as_ref()), w)?;
        } else {
            w.write_bool(false)?;
        }
        Ok(())
    }

    #[inline]
    fn decode<R>(base: Option<&Self>, r: &mut Reader<R>) -> io::Result<Self>
        where R: Read
    {
        if r.read_bool()? {
            Ok(Some(
                T::decode(base.and_then(|v| v.as_ref()), r)?
            ))
        } else {
            Ok(None)
        }
    }
}


impl DeltaEncodable for f32 {
    #[inline]
    fn encode<W>(&self, _base: Option<&Self>, w: &mut Writer<W>) -> io::Result<()>
        where W: Write
    {
        w.write_f32(*self)
    }

    #[inline]
    fn decode<R>(_base: Option<&Self>, r: &mut Reader<R>) -> io::Result<Self>
        where R: Read
    {
        r.read_f32()
    }
}

/// Helper to read a dynamic length prefix
pub fn read_len_bits<R: Read>(r: &mut Reader<R>) -> io::Result<usize> {
    let len_bits = match r.read_unsigned(2)? {
        0 => 5,
        1 => 8,
        2 => 10,
        3 => 16,
        _ => unreachable!(),
    };
    Ok(r.read_unsigned(len_bits)? as usize)
}

/// Helper to write a dynamic length prefix
pub fn write_len_bits<W: Write>(w: &mut Writer<W>, len: usize) -> io::Result<()> {
    // In debug mode crash on large lengths
    debug_assert!(len <= 0xFFFF);

    let (ty, size) = match len {
        0 ... 31 => (0, 5),
        32 ... 255 => (1, 8),
        256 ... 1023 => (2, 10),
        1024 ... 65_536 => (3, 16),
        _ => return Err(io::Error::new(io::ErrorKind::InvalidInput, "String too long")),
    };

    w.write_unsigned(ty, 2)?;
    w.write_unsigned(len as u64, size)?;
    Ok(())
}
/// Reads values at the bit boundary instead of
/// byte. Padded to 32 bits
pub struct Reader<R> {
    r: R,
    current: u32,
    offset: u8,
}

impl <R: Read> Reader<R> {
    /// Creates a bit reader that reads from the
    /// underlying reader.
    #[inline]
    pub fn new(r: R) -> Reader<R> {
        Reader {
            r,
            current: 0,
            offset: 32,
        }
    }

    /// Reads an unsigned value from the reader
    #[inline]
    pub fn read_unsigned(&mut self, mut bit_size: u8) -> io::Result<u64> {
        let mut val = 0;
        let mut bits_done = 0;
        while bit_size > 0 {
            // Ran out of data, get the next
            // batch
            if self.offset == 32 {
                self.offset = 0;
                self.current = self.r.read_u32::<LittleEndian>()?;
            }
            let bits = cmp::min(bit_size, 32 - self.offset);
            let mask = ((1u64 << bits) - 1) as u32;
            let part = self.current & mask;
            self.offset += bits;
            self.current = self.current.checked_shr(u32::from(bits)).unwrap_or(0);
            val |= u64::from(part) << bits_done;
            bits_done += bits;
            bit_size -= bits;
        }
        Ok(val)
    }

    /// Reads a signed value from the reader
    #[inline]
    pub fn read_signed(&mut self, bit_size: u8) -> io::Result<i64> {
        let val = self.read_unsigned(bit_size)? << (64 - bit_size);
        Ok((val as i64) >> (64 - bit_size))
    }

    /// Reads a boolean as a single bit from the reader
    #[inline]
    pub fn read_bool(&mut self) -> io::Result<bool> {
        Ok(self.read_unsigned(1)? != 0)
    }

    /// Reads a 32 bit float from the reader
    #[inline]
    pub fn read_f32(&mut self) -> io::Result<f32> {
        let mut buf = [0u8; 4];
        for b in &mut buf {
            *b = self.read_unsigned(8)? as u8;
        }
        Ok(LittleEndian::read_f32(&buf))
    }

    /// Reads a 64 bit float from the reader
    #[inline]
    pub fn read_f64(&mut self) -> io::Result<f64> {
        let mut buf = [0u8; 8];
        for b in &mut buf {
            *b = self.read_unsigned(8)? as u8;
        }
        Ok(LittleEndian::read_f64(&buf))
    }
}

/// Writes values at the bit boundary instead of
/// byte. Padded to 32 bits
#[derive(Clone, Debug, PartialEq)]
pub struct Writer<W> {
    w: W,
    current: u32,
    offset: u8,
}

impl Writer<Vec<u8>> {
    /// Copies the contents of this writer into
    /// the other writer.
    #[inline]
    pub fn copy_into<W: Write>(&self, other: &mut Writer<W>)  -> io::Result<()>{
        for b in &self.w {
            other.write_unsigned(u64::from(*b), 8)?;
        }
        if self.offset > 0 {
            other.write_unsigned(u64::from(self.current), self.offset)?;
        }
        Ok(())
    }

    /// Returns the size of this write in bits
    #[inline]
    pub fn bit_len(&self) -> usize {
        self.w.len() * 8 + self.offset as usize
    }

    /// Returns a readable view on the data within
    #[inline]
    pub fn read_view(&self) -> io::Chain<io::Cursor<&[u8]>, io::Cursor<[u8; 4]>> {
        let mut cur_data = [0; 4];
        LittleEndian::write_u32(&mut cur_data, self.current);
        io::Cursor::new(&self.w[..])
            .chain(io::Cursor::new(cur_data))
    }

    /// Clears this writer for reuse
    #[inline]
    pub fn clear(&mut self) {
        self.w.clear();
        self.current = 0;
        self.offset = 0;
    }
}

impl <W: Write> Writer<W> {
    /// Creates a bit reader that writes to the
    /// underlying writer;
    #[inline]
    pub fn new(w: W) -> Writer<W> {
        Writer {
            w,
            current: 0,
            offset: 0,
        }
    }

    /// Writes an unsigned value to the writer
    #[inline]
    pub fn write_unsigned(&mut self, mut val: u64, mut bit_size: u8) -> io::Result<()> {
        while bit_size > 0 {
            let bits = cmp::min(bit_size, 32 - self.offset);
            let mask = (1 << bits) - 1;
            let masked_val = val & mask;
            self.current |= (masked_val as u32) << self.offset;
            self.offset += bits;
            val >>= bits;
            bit_size -= bits;
            if self.offset == 32 {
                self.w.write_u32::<LittleEndian>(self.current)?;
                self.current = 0;
                self.offset = 0;
            }
        }
        Ok(())
    }

    /// Writes a signed value to the writer
    #[inline]
    pub fn write_signed(&mut self, mut val: i64, bit_size: u8) -> io::Result<()> {
        val <<= 64 - bit_size;
        let val = (val as u64) >> (64 - bit_size);
        self.write_unsigned(val, bit_size)
    }

    /// Writes a boolean as a single bit to the writer
    #[inline]
    pub fn write_bool(&mut self, val: bool) -> io::Result<()> {
        self.write_unsigned(val as u64, 1)
    }

    /// Writes a 32 bit float to the writer
    #[inline]
    pub fn write_f32(&mut self, val: f32) -> io::Result<()> {
        let mut buf = [0u8; 4];
        LittleEndian::write_f32(&mut buf[..], val);
        for v in &buf {
            self.write_unsigned(u64::from(*v), 8)?;
        }
        Ok(())
    }

    /// Writes a 64 bit float to the writer
    #[inline]
    pub fn write_f64(&mut self, val: f64) -> io::Result<()> {
        let mut buf = [0u8; 8];
        LittleEndian::write_f64(&mut buf[..], val);
        for v in &buf {
            self.write_unsigned(u64::from(*v), 8)?;
        }
        Ok(())
    }

    /// Flushes the writer and returns the underlying
    /// writer back.
    #[inline]
    pub fn finish(mut self) -> io::Result<W> {
        if self.offset > 0 {
            self.w.write_u32::<LittleEndian>(self.current)?;
        }
        Ok(self.w)
    }
}

#[test]
#[cfg(test)]
fn test_copy() {
    let buf = vec![];
    let mut writer = Writer::new(buf);
    writer.write_unsigned(0xA8FD_23CB, 32).unwrap();

    let mut writer2 = Writer::new(vec![]);
    writer2.write_unsigned(0x1F, 5).unwrap();
    writer.copy_into(&mut writer2).unwrap();
    let buf = writer2.finish().unwrap();

    let mut reader = Reader::new(io::Cursor::new(buf));
    assert_eq!(reader.read_unsigned(5).unwrap(), 0x1F);
    assert_eq!(reader.read_unsigned(32).unwrap(), 0xA8FD_23CB);
}


#[test]
#[cfg(test)]
fn test_bits() {
    let unsigned_test_vals = [
        (55, 6),
        (32, 6),
        (4, 4),
        (1, 1),
        (127, 7),
        (126, 7),
        // Overload test
        (254, 8),
        (254, 8),
        (254, 8),
        (254, 8),
        (254, 8),
        (254, 8),
        (254, 8),
        (254, 8),
        (254, 8),
        (254, 8),
        (254, 8),
        (254, 8),
        // Large test
        (0xFFFF_FFFF_FFFF_FFFF, 64)
    ];
    let signed_test_vals = [
        (-55, 7),
        (-32, 7),
        (4, 5),
        (-1, 2),
        (127, 8),
        (-126, 8),
        // Overload test
        (254, 9),
        (-254, 9),
        (254, 9),
        (-254, 9),
        (-254, 9),
        (254, 9),
        (254, 9),
        (-254, 9),
        (-254, 9),
        (254, 9),
        (-254, 9),
        (254, 9),
        // Large test
        (-0x7FFF_FFFF_FFFF_FFFF, 64)
    ];
    let float_test_vals = [
        0.799_053_067_0,
        0.330_235_920_9,
        0.928_176_656_8,
        0.081_341_180_8,
        0.468_306_402_9,
        0.562_892_797_4,
        0.219_208_626_3,
        0.049_358_503_3,
        0.142_168_047_5,
        0.859_083_713_8,
    ];

    let buf = vec![];
    let mut writer = Writer::new(buf);

    for &(val, bits) in &unsigned_test_vals {
        writer.write_unsigned(val, bits).unwrap();
    }
    for &(val, bits) in &signed_test_vals {
        writer.write_signed(val, bits).unwrap();
    }
    for val in &float_test_vals {
        writer.write_f32(*val as f32).unwrap();
        writer.write_f64(*val).unwrap();
    }
    writer.write_bool(true).unwrap();
    writer.write_bool(true).unwrap();
    writer.write_bool(false).unwrap();
    writer.write_bool(true).unwrap();
    writer.write_bool(false).unwrap();

    let buf = writer.finish().unwrap();
    println!("Buffer: {:?}", buf);

    let mut reader = Reader::new(io::Cursor::new(buf));

    for &(val, bits) in &unsigned_test_vals {
        assert_eq!(val, reader.read_unsigned(bits).unwrap());
    }

    for &(val, bits) in &signed_test_vals {
        assert_eq!(val, reader.read_signed(bits).unwrap());
    }

    for &val in &float_test_vals {
        assert_eq!(val as f32, reader.read_f32().unwrap());
        assert_eq!(val, reader.read_f64().unwrap());
    }

    assert_eq!(true, reader.read_bool().unwrap());
    assert_eq!(true, reader.read_bool().unwrap());
    assert_eq!(false, reader.read_bool().unwrap());
    assert_eq!(true, reader.read_bool().unwrap());
    assert_eq!(false, reader.read_bool().unwrap());
}