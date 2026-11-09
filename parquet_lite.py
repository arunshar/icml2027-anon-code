"""This module is a small pure-Python reader for flat Parquet files of string columns.

The WO-I dataset builder reads the two GSM8K parquet files with this module. The repository venv
has no pyarrow or pandas, and installing a package needs its own approval, so this reader covers
only what those files use. It reads the Thrift compact metadata, Snappy or uncompressed pages,
PLAIN and dictionary encodings, and version 1 and version 2 data pages. It supports flat schemas
with REQUIRED or OPTIONAL BYTE_ARRAY columns. Any other feature raises ParquetError, so the reader
never guesses at a layout it does not understand.

The format facts come from the Apache Parquet specification (parquet.thrift) and the Snappy
format description. The tests build small files with a separate writer and check the reader
against them.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, List, Optional, Sequence

MAGIC = b"PAR1"

# These are the Thrift compact protocol type codes.
_STOP, _TRUE, _FALSE, _BYTE, _I16, _I32, _I64, _DOUBLE, _BINARY, _LIST, _SET, _MAP, _STRUCT = range(13)

# These are the Parquet enum values that the reader uses.
TYPE_BYTE_ARRAY = 6
REP_REQUIRED, REP_OPTIONAL, REP_REPEATED = 0, 1, 2
CODEC_UNCOMPRESSED, CODEC_SNAPPY = 0, 1
PAGE_DATA, PAGE_INDEX, PAGE_DICTIONARY, PAGE_DATA_V2 = 0, 1, 2, 3
ENC_PLAIN, ENC_PLAIN_DICTIONARY, ENC_RLE, ENC_RLE_DICTIONARY = 0, 2, 3, 8


class ParquetError(ValueError):
    """The file uses a feature this reader does not support, or it is malformed."""


# ------------------------------------------------------------------ thrift compact protocol


class _Reader:
    def __init__(self, data: bytes, pos: int = 0):
        self.data = data
        self.pos = pos

    def byte(self) -> int:
        if self.pos >= len(self.data):
            raise ParquetError("unexpected end of data")
        b = self.data[self.pos]
        self.pos += 1
        return b

    def take(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.data):
            raise ParquetError("unexpected end of data")
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out

    def varint(self) -> int:
        shift, out = 0, 0
        while True:
            b = self.byte()
            out |= (b & 0x7F) << shift
            if not b & 0x80:
                return out
            shift += 7
            if shift > 70:
                raise ParquetError("varint is too long")

    def zigzag(self) -> int:
        n = self.varint()
        return (n >> 1) ^ -(n & 1)


def _read_value(r: _Reader, ttype: int):
    if ttype == _TRUE:
        return True
    if ttype == _FALSE:
        return False
    if ttype == _BYTE:
        return struct.unpack("<b", r.take(1))[0]
    if ttype in (_I16, _I32, _I64):
        return r.zigzag()
    if ttype == _DOUBLE:
        return struct.unpack("<d", r.take(8))[0]
    if ttype == _BINARY:
        return r.take(r.varint())
    if ttype in (_LIST, _SET):
        head = r.byte()
        size, etype = head >> 4, head & 0x0F
        if size == 15:
            size = r.varint()
        if etype in (_TRUE, _FALSE):
            # A boolean inside a container is one byte, where 1 means true.
            return [r.byte() == 1 for _ in range(size)]
        return [_read_value(r, etype) for _ in range(size)]
    if ttype == _MAP:
        size = r.varint()
        if size == 0:
            return {}
        kv = r.byte()
        ktype, vtype = kv >> 4, kv & 0x0F
        return {_read_value(r, ktype): _read_value(r, vtype) for _ in range(size)}
    if ttype == _STRUCT:
        return read_struct(r)
    raise ParquetError(f"unknown thrift type {ttype}")


def read_struct(r: _Reader) -> Dict[int, object]:
    """Read one compact-protocol struct into a dict keyed by field id."""
    out: Dict[int, object] = {}
    last = 0
    while True:
        head = r.byte()
        ttype = head & 0x0F
        if ttype == _STOP:
            return out
        delta = head >> 4
        fid = last + delta if delta else r.zigzag()
        out[fid] = _read_value(r, ttype)
        last = fid


# ------------------------------------------------------------------ snappy


def snappy_decompress(data: bytes) -> bytes:
    """Decompress one raw Snappy block, as Parquet stores it (no framing format)."""
    r = _Reader(data)
    expected = r.varint()
    out = bytearray()
    while r.pos < len(data):
        tag = r.byte()
        kind = tag & 3
        if kind == 0:
            length = tag >> 2
            if length >= 60:
                nbytes = length - 59
                length = int.from_bytes(r.take(nbytes), "little")
            out += r.take(length + 1)
            continue
        if kind == 1:
            length = 4 + ((tag >> 2) & 7)
            offset = ((tag >> 5) << 8) | r.byte()
        elif kind == 2:
            length = (tag >> 2) + 1
            offset = int.from_bytes(r.take(2), "little")
        else:
            length = (tag >> 2) + 1
            offset = int.from_bytes(r.take(4), "little")
        if offset == 0 or offset > len(out):
            raise ParquetError("snappy copy offset is out of range")
        start = len(out) - offset
        for i in range(length):  # a copy may overlap its own output
            out.append(out[start + i])
    if len(out) != expected:
        raise ParquetError(f"snappy length mismatch: {len(out)} != {expected}")
    return bytes(out)


def _decompress(codec: int, data: bytes, uncompressed_size: int) -> bytes:
    if codec == CODEC_UNCOMPRESSED:
        out = data
    elif codec == CODEC_SNAPPY:
        out = snappy_decompress(data)
    else:
        raise ParquetError(f"compression codec {codec} is not supported")
    if len(out) != uncompressed_size:
        raise ParquetError("a page's uncompressed size differs from its header")
    return out


# ------------------------------------------------------------------ encodings


def rle_bitpacked_hybrid(data: bytes, bit_width: int, count: int) -> List[int]:
    """Decode `count` values of the RLE and bit-packed hybrid encoding, with no length prefix."""
    if not 0 <= bit_width <= 32:
        raise ParquetError(f"bit width {bit_width} is out of range")
    r = _Reader(data)
    out: List[int] = []
    width_bytes = (bit_width + 7) // 8
    while len(out) < count:
        header = r.varint()
        if header & 1:
            groups = header >> 1
            nbytes = groups * bit_width
            raw = int.from_bytes(r.take(nbytes), "little")
            mask = (1 << bit_width) - 1
            for i in range(groups * 8):
                out.append((raw >> (i * bit_width)) & mask if bit_width else 0)
        else:
            run = header >> 1
            value = int.from_bytes(r.take(width_bytes), "little") if width_bytes else 0
            out.extend([value] * run)
    return out[:count]


def plain_byte_arrays(data: bytes, count: int) -> List[bytes]:
    r = _Reader(data)
    out = []
    for _ in range(count):
        n = struct.unpack("<I", r.take(4))[0]
        out.append(r.take(n))
    return out


def _bit_width(max_value: int) -> int:
    return max_value.bit_length()


# ------------------------------------------------------------------ file structure


def _footer(data: bytes) -> Dict[int, object]:
    if len(data) < 12 or data[:4] != MAGIC or data[-4:] != MAGIC:
        raise ParquetError("the file does not start and end with PAR1")
    meta_len = struct.unpack("<I", data[-8:-4])[0]
    start = len(data) - 8 - meta_len
    if start < 4:
        raise ParquetError("the footer length is out of range")
    return read_struct(_Reader(data[start:len(data) - 8]))


def _flat_columns(schema: Sequence[Dict[int, object]]) -> List[Dict[str, object]]:
    if not schema:
        raise ParquetError("the schema is empty")
    root, leaves = schema[0], schema[1:]
    if root.get(5) != len(leaves):
        raise ParquetError("only a flat schema of leaf columns is supported")
    cols = []
    for el in leaves:
        if el.get(5):
            raise ParquetError("nested columns are not supported")
        rep = el.get(3, REP_REQUIRED)
        if rep == REP_REPEATED:
            raise ParquetError("repeated columns are not supported")
        cols.append({"name": el[4].decode("utf-8"), "type": el.get(1), "optional": rep == REP_OPTIONAL})
    return cols


def _read_column_chunk(data: bytes, meta: Dict[int, object], optional: bool) -> List[Optional[bytes]]:
    if meta.get(1) != TYPE_BYTE_ARRAY:
        raise ParquetError("only BYTE_ARRAY columns are supported")
    codec = meta.get(4, CODEC_UNCOMPRESSED)
    num_values = meta[5]
    offset = meta.get(11) if meta.get(11) is not None else meta[9]
    if meta.get(11) is not None and meta[9] < meta[11]:
        offset = meta[9]
    end = offset + meta[7]
    r = _Reader(data, offset)
    dictionary: Optional[List[bytes]] = None
    values: List[Optional[bytes]] = []
    while len(values) < num_values:
        if r.pos >= end:
            raise ParquetError("a column chunk ended before all its values were read")
        header = read_struct(r)
        ptype, usize, csize = header[1], header[2], header[3]
        raw = r.take(csize)
        if ptype == PAGE_DICTIONARY:
            dh = header[7]
            if dh.get(2) not in (ENC_PLAIN, ENC_PLAIN_DICTIONARY):
                raise ParquetError("a dictionary page must be PLAIN encoded")
            dictionary = plain_byte_arrays(_decompress(codec, raw, usize), dh[1])
            continue
        if ptype == PAGE_INDEX:
            continue
        if ptype == PAGE_DATA:
            dh = header[5]
            n = dh[1]
            encoding = dh[2]
            page = _decompress(codec, raw, usize)
            pr = _Reader(page)
            if optional:
                if dh.get(3, ENC_RLE) != ENC_RLE:
                    raise ParquetError("definition levels must be RLE encoded")
                dlen = struct.unpack("<I", pr.take(4))[0]
                defs = rle_bitpacked_hybrid(pr.take(dlen), 1, n)
            else:
                defs = [1] * n
            body = page[pr.pos:]
        elif ptype == PAGE_DATA_V2:
            dh = header[8]
            n = dh[1]
            encoding = dh[4]
            dlen, rlen = dh[5], dh[6]
            if rlen:
                raise ParquetError("repetition levels are not supported")
            levels = raw[:dlen]
            rest = raw[dlen:]
            compressed = dh.get(7, True)
            body = _decompress(codec, rest, usize - dlen) if compressed else rest
            defs = rle_bitpacked_hybrid(levels, 1, n) if optional else [1] * n
        else:
            raise ParquetError(f"page type {ptype} is not supported")
        present = sum(defs)
        if encoding == ENC_PLAIN:
            items = plain_byte_arrays(body, present)
        elif encoding in (ENC_PLAIN_DICTIONARY, ENC_RLE_DICTIONARY):
            if dictionary is None:
                raise ParquetError("a dictionary-encoded page has no dictionary")
            width = body[0] if body else 0
            idx = rle_bitpacked_hybrid(body[1:], width, present)
            if any(i >= len(dictionary) for i in idx):
                raise ParquetError("a dictionary index is out of range")
            items = [dictionary[i] for i in idx]
        else:
            raise ParquetError(f"value encoding {encoding} is not supported")
        it = iter(items)
        values.extend(next(it) if d else None for d in defs)
    if len(values) != num_values:
        raise ParquetError("a column chunk holds more values than its metadata says")
    return values


def read_string_table(path) -> Dict[str, List[Optional[str]]]:
    """Read every column of a flat parquet file of strings, and return a dict from column name to values."""
    data = Path(path).read_bytes()
    meta = _footer(data)
    cols = _flat_columns(meta[2])
    out: Dict[str, List[Optional[str]]] = {c["name"]: [] for c in cols}
    total = 0
    for rg in meta.get(4, []):
        chunks = rg[1]
        if len(chunks) != len(cols):
            raise ParquetError("a row group has the wrong number of columns")
        for col, chunk in zip(cols, chunks):
            if chunk.get(1):
                raise ParquetError("external column files are not supported")
            cmeta = chunk[3]
            path_in_schema = [p.decode("utf-8") for p in cmeta[3]]
            if path_in_schema != [col["name"]]:
                raise ParquetError("a column chunk does not match the schema order")
            vals = _read_column_chunk(data, cmeta, col["optional"])
            if len(vals) != rg[3]:
                raise ParquetError("a column chunk and its row group disagree on the row count")
            out[col["name"]].extend(None if v is None else v.decode("utf-8") for v in vals)
        total += rg[3]
    if total != meta[3]:
        raise ParquetError("the row groups do not add up to the file's row count")
    return out
