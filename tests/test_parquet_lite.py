"""These tests check the pure-Python parquet reader against hand-built files and published vectors.

The writer below is separate from the reader and follows the Parquet and Thrift compact protocol
specifications. The last test reads the real GSM8K files when they are present on this laptop,
and it skips otherwise, so the suite never downloads anything.
"""
import hashlib
import json
import struct
from pathlib import Path

import pytest

import parquet_lite as P

HERE = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------------ a minimal writer


def varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def zigzag(n):
    return varint((n << 1) ^ (n >> 63))


I32, I64, BIN, LIST, STRUCT, TRUE, FALSE = 5, 6, 8, 9, 12, 1, 2


def tstruct(fields):
    """Encode a struct from (field id, type, value) triples given in increasing id order."""
    out, last = bytearray(), 0
    for fid, ttype, value in fields:
        if ttype == "bool":
            ttype = TRUE if value else FALSE
        delta = fid - last
        if 0 < delta <= 15:
            out.append((delta << 4) | ttype)
        else:
            out.append(ttype)
            out += zigzag(fid)
        out += tvalue(ttype, value)
        last = fid
    out.append(0)
    return bytes(out)


def tvalue(ttype, value):
    if ttype in (TRUE, FALSE):
        return b""
    if ttype in (I32, I64):
        return zigzag(value)
    if ttype == BIN:
        raw = value.encode() if isinstance(value, str) else value
        return varint(len(raw)) + raw
    if ttype == STRUCT:
        return value
    if ttype == LIST:
        etype, items = value
        head = (len(items) << 4 | etype) if len(items) < 15 else (0xF0 | etype)
        out = bytes([head]) + (b"" if len(items) < 15 else varint(len(items)))
        return out + b"".join(tvalue(etype, v) for v in items)
    raise AssertionError(ttype)


def plain(values):
    return b"".join(struct.pack("<I", len(v)) + v for v in values)


def bitpacked(values, width):
    groups = (len(values) + 7) // 8
    padded = list(values) + [0] * (groups * 8 - len(values))
    raw = 0
    for i, v in enumerate(padded):
        raw |= v << (i * width)
    return varint(groups << 1 | 1) + raw.to_bytes(groups * width, "little")


def snappy_literal(data):
    """Return a valid Snappy block made of literal runs only."""
    out = bytearray(varint(len(data)))
    for i in range(0, len(data), 64):
        chunk = data[i:i + 64]
        n = len(chunk) - 1
        out += bytes([n << 2]) if n < 60 else bytes([60 << 2, n])
        out += chunk
    return bytes(out)


def page(ptype, body, uncompressed_size, header_field):
    return tstruct([(1, I32, ptype), (2, I32, uncompressed_size), (3, I32, len(body)), header_field]) + body


def write_file(path, columns, nrows, v2=False):
    """Write a file from (name, values, encoding, codec) columns, where encoding is plain or dict."""
    blob = bytearray(P.MAGIC)
    chunks = []
    for name, values, encoding, codec in columns:
        start = len(blob)
        present = [v.encode() for v in values if v is not None]
        defs = [0 if v is None else 1 for v in values]
        dict_offset = None
        comp = (lambda b: snappy_literal(b)) if codec == P.CODEC_SNAPPY else (lambda b: b)
        if encoding == "dict":
            uniq = sorted(set(present))
            dict_body = plain(uniq)
            c = comp(dict_body)
            dict_offset = len(blob)
            blob += page(P.PAGE_DICTIONARY, c, len(dict_body),
                         (7, STRUCT, tstruct([(1, I32, len(uniq)), (2, I32, P.ENC_PLAIN)])))
            idx = [uniq.index(v) for v in present]
            width = max(1, (len(uniq) - 1).bit_length())
            values_body, value_enc = bytes([width]) + bitpacked(idx, width), P.ENC_RLE_DICTIONARY
        else:
            values_body, value_enc = plain(present), P.ENC_PLAIN
        data_offset = len(blob)
        levels = bitpacked(defs, 1)
        if v2:
            c = comp(values_body)
            hdr = tstruct([(1, I32, len(values)), (2, I32, defs.count(0)), (3, I32, len(values)),
                           (4, I32, value_enc), (5, I32, len(levels)), (6, I32, 0), (7, "bool", True)])
            body = levels + c
            blob += tstruct([(1, I32, P.PAGE_DATA_V2), (2, I32, len(levels) + len(values_body)),
                             (3, I32, len(body)), (8, STRUCT, hdr)]) + body
        else:
            raw = struct.pack("<I", len(levels)) + levels + values_body
            c = comp(raw)
            hdr = tstruct([(1, I32, len(values)), (2, I32, value_enc), (3, I32, P.ENC_RLE), (4, I32, P.ENC_RLE)])
            blob += page(P.PAGE_DATA, c, len(raw), (5, STRUCT, hdr))
        meta = [(1, I32, P.TYPE_BYTE_ARRAY), (2, LIST, (I32, [value_enc, P.ENC_RLE])),
                (3, LIST, (BIN, [name])), (4, I32, codec), (5, I64, len(values)),
                (6, I64, len(blob) - start), (7, I64, len(blob) - start), (9, I64, data_offset)]
        if dict_offset is not None:
            meta.append((11, I64, dict_offset))
        chunks.append(tstruct([(2, I64, start), (3, STRUCT, tstruct(meta))]))
    schema = [tstruct([(4, BIN, "schema"), (5, I32, len(columns))])]
    schema += [tstruct([(1, I32, P.TYPE_BYTE_ARRAY), (3, I32, P.REP_OPTIONAL), (4, BIN, name)])
               for name, *_ in columns]
    rowgroup = tstruct([(1, LIST, (STRUCT, chunks)), (2, I64, 0), (3, I64, nrows)])
    footer = tstruct([(1, I32, 1), (2, LIST, (STRUCT, schema)), (3, I64, nrows),
                      (4, LIST, (STRUCT, [rowgroup]))])
    blob += footer + struct.pack("<I", len(footer)) + P.MAGIC
    Path(path).write_bytes(bytes(blob))


# ------------------------------------------------------------------ vectors


def test_snappy_literal_and_both_copy_kinds_decode():
    assert P.snappy_decompress(bytes([12, 0x08]) + b"abc" + bytes([0x15, 0x03])) == b"abcabcabcabc"
    literal = b"hello world, "
    stream = bytes([24, (len(literal) - 1) << 2]) + literal + bytes([2 | (10 << 2), 13, 0])
    assert P.snappy_decompress(stream) == b"hello world, hello world"
    long = bytes(range(100))
    assert P.snappy_decompress(bytes([100, 60 << 2, 99]) + long) == long


def test_snappy_refuses_a_bad_offset_and_a_wrong_length():
    with pytest.raises(P.ParquetError):
        P.snappy_decompress(bytes([8, 0x15, 0x09]))
    with pytest.raises(P.ParquetError):
        P.snappy_decompress(bytes([5, 0x04]) + b"ab")


def test_rle_hybrid_matches_the_specification_example_and_runs():
    # The Parquet specification packs 0 to 7 at bit width 3 into 0x88 0xC6 0xFA.
    assert P.rle_bitpacked_hybrid(bytes([0x03, 0x88, 0xC6, 0xFA]), 3, 8) == list(range(8))
    assert P.rle_bitpacked_hybrid(bytes([10, 7]), 3, 5) == [7] * 5
    assert P.rle_bitpacked_hybrid(bytes([4, 1]) + bitpacked([0, 1, 1], 1), 1, 5) == [1, 1, 0, 1, 1]


def test_thrift_reader_handles_long_field_deltas_and_long_lists():
    raw = tstruct([(1, I32, -7), (40, BIN, "x"), (41, LIST, (I32, list(range(20))))])
    parsed = P.read_struct(P._Reader(raw))
    assert parsed == {1: -7, 40: b"x", 41: list(range(20))}


# ------------------------------------------------------------------ files


@pytest.mark.parametrize("v2", [False, True])
def test_reader_decodes_plain_dictionary_snappy_and_nulls(tmp_path, v2):
    q = ["what is 1+1?", None, "café costs $3", "a", "b"]
    a = ["#### 2", "#### 5", "#### 3", "#### 2", "#### 2"]
    path = tmp_path / "t.parquet"
    write_file(path, [("question", q, "plain", P.CODEC_UNCOMPRESSED),
                      ("answer", a, "dict", P.CODEC_SNAPPY)], 5, v2=v2)
    assert P.read_string_table(path) == {"question": q, "answer": a}


def test_reader_refuses_a_truncated_file_and_an_unknown_codec(tmp_path):
    path = tmp_path / "t.parquet"
    write_file(path, [("question", ["x"], "plain", P.CODEC_UNCOMPRESSED)], 1)
    raw = path.read_bytes()
    (tmp_path / "cut.parquet").write_bytes(raw[:-3])
    with pytest.raises(P.ParquetError):
        P.read_string_table(tmp_path / "cut.parquet")
    write_file(path, [("question", ["x"], "plain", 6)], 1)
    with pytest.raises(P.ParquetError):
        P.read_string_table(path)


REAL = HERE / "data" / "gsm8k_main"


@pytest.mark.skipif(not (REAL / "train-00000-of-00001.parquet").exists(),
                    reason="the GSM8K parquet files are not on this machine")
def test_real_gsm8k_files_match_their_recorded_digests_and_sizes():
    import make_reasoning_dataset as M
    for split, spec in M.SOURCE["files"].items():
        path = REAL / f"{split}-00000-of-00001.parquet"
        assert path.stat().st_size == spec["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == spec["sha256"]
        table = P.read_string_table(path)
        assert len(table["question"]) == spec["rows"]
        # The Hub's dataset_info gives num_bytes as the UTF-8 bytes of both columns plus four
        # offset bytes per value, 3,963,202 for train and 713,732 for test. Matching it checks
        # the length of every decoded string.
        total = sum(len(x.encode()) for col in table.values() for x in col) + 8 * spec["rows"]
        assert total == {"train": 3963202, "test": 713732}[split]
