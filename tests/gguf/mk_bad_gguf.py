"""Write crafted GGUF files that must be REJECTED (plus one good control) to test src/gguf.cpp checks.
usage: python mk_bad_gguf.py <outdir>"""
import os, struct, sys

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
U32, U64, STR = 4, 10, 8  # gguf value types
F32_T = 0                 # ggml type F32 (block 1, 4 bytes)


def s(x): b = x.encode(); return struct.pack('<Q', len(b)) + b


def kv_u32(k, v): return s(k) + struct.pack('<II', U32, v)


def kv_str(k, v): return s(k) + struct.pack('<I', STR) + s(v)


def build(kvs, tensors, n_tensors=None, n_kv=None, align=32, data=b'\0' * 64):
    h = b'GGUF' + struct.pack('<I', 3)
    h += struct.pack('<QQ', len(tensors) if n_tensors is None else n_tensors, len(kvs) if n_kv is None else n_kv)
    for k in kvs: h += k
    for name, dims, typ, off in tensors:
        h += s(name) + struct.pack('<I', len(dims)) + b''.join(struct.pack('<Q', d) for d in dims) + struct.pack('<IQ', typ, off)
    pad = (-len(h)) % align if align else 0
    return h + b'\0' * pad + data


cases = {
    'good_control': build([kv_str('general.architecture', 'x')], [('t', [16], F32_T, 0)]),
    'align_zero': build([kv_u32('general.alignment', 0)], [('t', [16], F32_T, 0)], align=32),
    'align_not_pow2': build([kv_u32('general.alignment', 24)], [('t', [16], F32_T, 0)], align=24),
    'align_string': build([kv_str('general.alignment', '32')], [('t', [16], F32_T, 0)]),
    # 2^32 * 2^32 rows of 4 bytes wraps the old int64 product to 0 -> passed the old EOF check
    'dims_wrap': build([], [('t', [1, 1 << 32, 1 << 32], F32_T, 0)]),
    'dim_zero': build([], [('t', [16, 0], F32_T, 0)]),
    'dim_huge': build([], [('t', [1 << 63], F32_T, 0)]),
    # offset near 2^64: old check data_start + offset + nbytes wrapped below file_size
    'offset_wrap': build([], [('t', [16], F32_T, (1 << 64) - 32)]),
    'offset_misaligned': build([], [('t', [4], F32_T, 4)]),
    'past_eof': build([], [('t', [1024], F32_T, 0)]),
    'n_tensors_huge': build([], [], n_tensors=1 << 40),
    'n_kv_huge': build([], [], n_kv=1 << 40),
    'array_huge': build([s('a') + struct.pack('<IIQ', 9, U32, 1 << 40)], []),
}
for name, b in cases.items():
    with open(os.path.join(out, name + '.gguf'), 'wb') as f: f.write(b)
print(' '.join(cases))
