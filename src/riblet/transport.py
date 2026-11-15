import struct

MAX_PACKET = 512 * 2**20


def require_owner(rank, kind):
    owners = {"model": 0, "parameters": 0, "input": 1, "output": 1}
    if kind not in owners or type(rank) is not int or rank != owners[kind]:
        raise PermissionError("asset access does not match fixed model0/data1 ownership")


def encode_share(share):
    meta = bytes(share.meta)
    chunks = [bytes(v) for v in share.share_chunks]
    if len(meta) > 2**20 or len(chunks) > 65536:
        raise ValueError("share metadata/chunk bounds")
    length = 12 + len(meta) + sum(8 + len(c) for c in chunks)
    if length > MAX_PACKET:
        raise ValueError("share exceeds supported model tensor size")
    return (
        b"RBS1"
        + struct.pack("<II", len(meta), len(chunks))
        + meta
        + b"".join(struct.pack("<Q", len(c)) + c for c in chunks)
    )


def decode_share(packet):
    packet = bytes(packet)
    if len(packet) < 12 or len(packet) > MAX_PACKET or packet[:4] != b"RBS1":
        raise ValueError("invalid share frame")
    meta_length, count = struct.unpack_from("<II", packet, 4)
    if meta_length > 2**20 or count > 65536 or 12 + meta_length > len(packet):
        raise ValueError("invalid share metadata")
    at = 12 + meta_length
    meta = packet[12:at]
    chunks = []
    for _ in range(count):
        if at + 8 > len(packet):
            raise ValueError("truncated share chunk header")
        size = struct.unpack_from("<Q", packet, at)[0]
        at += 8
        if at + size > len(packet):
            raise ValueError("truncated share chunk")
        chunks.append(packet[at : at + size])
        at += size
    if at != len(packet):
        raise ValueError("trailing share bytes")
    from spu import libspu

    value = libspu.Share()
    value.meta = meta
    value.share_chunks = chunks
    return value
