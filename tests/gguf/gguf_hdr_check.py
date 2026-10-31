"""Read-only: apply the new src/gguf.cpp header checks to GGUF files and print OK or the first rule they break.
usage: python3 gguf_hdr_check.py <file.gguf> ..."""
import struct, sys, os

SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}  # gguf value type -> bytes


def check(path):
    fs = os.path.getsize(path)
    with open(path, 'rb') as f:
        def rd(n):
            b = f.read(n)
            if len(b) != n: raise ValueError('unexpected end')
            return b
        u32 = lambda: struct.unpack('<I', rd(4))[0]
        u64 = lambda: struct.unpack('<Q', rd(8))[0]
        def s(): return rd(u64())
        if rd(4) != b'GGUF': return 'bad magic'
        u32()
        nt, nkv = u64(), u64()
        if nkv > fs // 13 or nt > fs // 32: return 'bad kv / tensor count'
        align = 32
        for _ in range(nkv):
            k = s(); t = u32()
            if t == 9:
                at = u32(); n = u64()
                if at == 8:
                    for _ in range(n): s()
                else: f.seek(SIZES[at] * n, 1)
            elif t == 8: v = s()
            else:
                v = rd(SIZES[t])
                if k == b'general.alignment':
                    if t in (6, 7, 12): return 'alignment not integer'
                    align = int.from_bytes(v, 'little')
        if (align == 0 or align & (align - 1) or align > 1 << 30): return f'bad alignment {align}'
        bad = 0
        for _ in range(nt):
            name = s(); nd = u32()
            ne = [u64() for _ in range(nd)]
            u32(); off = u64()
            if any(d < 1 or d >= 1 << 62 for d in ne): return f'bad dimension {name!r} {ne}'
            if off % align: bad += 1
        if bad: return f'{bad} misaligned tensor offsets (alignment {align})'
    return 'OK'


for p in sys.argv[1:]:
    try: r = check(p)
    except Exception as e: r = f'parse error {e!r}'
    print(f'{r:40s} {p}')
