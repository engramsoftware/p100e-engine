#!/usr/bin/env python3
"""Build the register-bank-fixed sm_60 SASS of p100e's prefill fold GEMM kernel (fold_seg_kernel<8, false>,
src/fast/gemm.cu) into src/fast/gemm_sass_fold8.h (the fork's GGML_CUDA_GEMM_FOLD_SASS).

Pascal reads registers from 4 banks (index mod 4); an FMA-type instruction whose source operands sit in one bank
(and do not come from the operand reuse cache) stalls a cycle. ptxas leaves ~37 % of the fold kernel's HFMA2s with
such a conflict. Renaming registers (a permutation of register names) changes no instruction, so the results are
bit-identical by construction. The renaming choice is Kmic-68's bankfix.py (MIT):
greedy placement of the HFMA2 accumulators into the bank that adds the fewest conflicts, over the accumulators and the
free registers (here capped at the kernel's register count, so occupancy and the REGCOUNT attribute do not change);
registers that are part of a 64/128-bit operand anywhere are never moved.
Kmic re-assembles with CuAssembler; this tool needs no assembler: it patches the register fields of the instruction
words in the cubin (Maxwell/Pascal register fields at bits 0, 8, 20, 39) and uses nvdisasm as an exact oracle - the
patched cubin's disassembly must equal the original disassembly with the registers renamed, line for line (any
instruction whose fields do not match that is searched over field subsets; failure aborts).

The header also pins what the launcher (fast::fold_gemm) hands the binary: static_asserts on sizeof / offsetof of
seg_tab and on the kernel constants, and the parameter sizes read from the cubin's EIATTR_KPARAM_INFO
(gemm_sass_fold8_param_size, checked against the launch arguments in gemm.cu). gemm.cu includes it after the kernel,
where seg_tab and the constants are visible.

usage: fold_bankfix.py [--check]     (needs nvcc / nvdisasm on PATH and the MSVC environment for nvcc)
  --check: exit 1 if the header was built from a different kernel text or by an older version of this tool (rerun
           after any change to the kernel). Pure Python (no nvcc); the CMake build runs it before compiling gemm.cu.
"""
import hashlib, os, re, struct, subprocess, sys, tempfile
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC = os.path.join(ROOT, 'src', 'fast', 'gemm.cu')
OUT = os.path.join(ROOT, 'src', 'fast', 'gemm_sass_fold8.h')
FORMAT = 2   # header format: 2 = with the layout / parameter static_asserts (bump when the header contents change)
NVCC_FLAGS = ['-cubin', '-O3', '-DNDEBUG', '-std=c++17', '--expt-relaxed-constexpr', '-arch=sm_60',
              '-allow-unsupported-compiler', '-Wno-deprecated-gpu-targets', '-Xcudafe', '--diag_suppress=177',
              '-Xcudafe', '--diag_suppress=550']


def kernel_source():
    s = open(SRC, encoding='utf-8').read()
    consts = re.search(r'constexpr int BM = 128;.*?constexpr int XEXP = 3;[^\n]*\n', s, re.S).group(0)
    foldh = re.search(r'__device__ __forceinline__ float fold_h\(const half2 p\) \{.*?\n\}\n', s, re.S).group(0)
    a = s.index('struct seg_tab {')
    b = s.index('} // namespace', a)
    ktxt = consts + foldh + s[a:b]
    return ktxt, ('#include <cuda_fp16.h>\n#include <cstdint>\n' + ktxt +
                  '\nvoid * fold8_inst() { return (void *) &fold_seg_kernel<8, false>; }\n')


# an instruction line of the kernel's .text listing: "/*addr*/ [{] text ;" (encoding on the same line) or, for the second
# instruction of a dual-issue pair, "/*addr*/ text }" (its encoding follows on the next line). The words are read from
# the cubin itself (section offset + addr), so both forms are handled alike.
INS_RE = re.compile(r'^\s*/\*([0-9a-f]{4,})\*/\s+\{?\s*([^;{}]*?)\s*[;}]')
REG_RE = re.compile(r'(?<![\w.])R(\d+)((?:\.reuse)?)')
FMA_OPS = ('HFMA2', 'HMUL2', 'HADD2', 'FFMA', 'FADD', 'FMUL')
FIELDS = (0, 8, 20, 39)   # Maxwell / Pascal register fields: Rd, Ra, Rb, Rc
WIDE = False              # --wide: see bankfix()


def disasm(nvdisasm, cubin):
    """(addr, text, 64-bit word) of every instruction of the cubin's (single) kernel"""
    out = subprocess.run([nvdisasm, '-hex', cubin], capture_output=True, text=True, check=True).stdout
    blob = open(cubin, 'rb').read()
    _, off, size = elf_text_section(blob, '.text.')
    ins = []
    in_text = False
    for l in out.split('\n'):
        if re.match(r'^\s*\.section\s', l):   # not .sectionflags / .sectioninfo
            in_text = '.text.' in l
            continue
        m = INS_RE.match(l) if in_text else None
        if m and m.group(2) and not m.group(2).startswith('.'):
            a = int(m.group(1), 16)
            if a % 32 == 0 or a >= size:
                raise SystemExit(f'unexpected instruction address 0x{a:x} (control-word slot or past the section)')
            ins.append((a, m.group(2).strip(), struct.unpack_from('<Q', blob, off + a)[0]))
    n_slots = sum(1 for a in range(0, size, 8) if a % 32 != 0)
    if len(ins) != n_slots:
        raise SystemExit(f'parsed {len(ins)} instructions, the section holds {n_slots} instruction slots')
    return ins


def opcode(t):
    t = re.sub(r'^@!?P\w+\s+', '', t.strip())
    return t.split()[0] if t else ''


def vector_regs(ins):   # bankfix.py: registers of any 64/128-bit operand (never moved)
    vec = set()
    for _, t, _ in ins:
        op = opcode(t)
        width = 4 if '.128' in op else 2 if ('.64' in op or op.startswith('D')) else 1
        regs = [int(r) for r, _ in REG_RE.findall(t)]
        if width > 1 and regs:
            for r in (regs[:1] if not op.startswith('ST') else regs[-1:]):
                vec.update(range(r, r + width))
        if re.match(r'(LDG|STG|LD|ST|ATOM|RED)\b', op) and '.E' in op:
            for a in re.findall(r'\[R(\d+)', t):
                vec.update((int(a), int(a) + 1))
        if op.startswith('ST') and width > 1:
            vec.update(range(regs[-1], regs[-1] + width))
        if op.startswith('SHF') and '.64' in op or op.startswith('LEA.HI.X'):
            pass
    return vec


def sources(t):
    body = re.sub(r'^@!?P\w+\s+', '', t.strip()).rstrip(' ;}')
    parts = body.split(None, 1)
    if len(parts) < 2:
        return None, []
    ops = [o.strip() for o in parts[1].split(',')]
    dest = None
    m = re.match(r'-?\|?R(\d+)', ops[0])
    if m:
        dest = int(m.group(1))
    src = []
    for slot, o in enumerate(ops[1:]):
        m = re.match(r'-?\|?R(\d+)(\.reuse)?', o)
        if m:
            src.append((int(m.group(1)), bool(m.group(2)), slot))
    return dest, src


def conflicts(hot, bank):   # bankfix.py's count: same-bank source pairs not served by the reuse cache
    n = 0
    prev = {}
    for t in hot:
        d, src = sources(t)
        banks = defaultdict(int)
        seen = set()
        for r, reuse, slot in src:
            if prev.get(slot) != r and r != 255 and r not in seen:
                banks[bank(r)] += 1
                seen.add(r)
        n += sum(v - 1 for v in banks.values() if v > 1)
        prev = {slot: r for r, reuse, slot in src if reuse}
    return n


def bankfix(ins, cap, exclude=frozenset()):
    hf = [a for a, t, _ in ins if opcode(t).startswith(('HFMA2', 'HMUL2'))]
    lo, hi = min(hf), max(hf)
    hot = [t for a, t, _ in ins if lo <= a <= hi and opcode(t).split('.')[0] in FMA_OPS]
    used = set()
    for _, t, _ in ins:
        used.update(int(r) for r, _ in REG_RE.findall(t))
    used.discard(255)
    vec = vector_regs(ins) | set(exclude)
    before = conflicts(hot, lambda r: r % 4)
    acc = set()
    for t in hot:
        d, s = sources(t)
        if opcode(t).startswith(('HFMA2', 'HMUL2')) and d is not None:
            acc.add(d)
            for r, _, slot in s:
                if slot == 2:
                    acc.add(r)
    acc -= vec
    free = [r for r in range(0, cap) if r not in used]
    pool = set(acc) | set(free)
    if WIDE:   # bankfix.py BANKFIX_WIDE: accumulators may also take any other scalar register's place
        pool |= {r for r in used if r < cap and r not in vec}
    pool = sorted(pool)
    mapping = {}
    avail = defaultdict(list)
    for r in pool:
        avail[r % 4].append(r)
    cur = lambda r: mapping.get(r, r)
    order = sorted(acc)
    for it in range(3):
        for r in order:
            if r in mapping:
                avail[mapping[r] % 4].append(mapping.pop(r))
            best_b, best_c = None, None
            for b in range(4):
                if not avail[b]:
                    continue
                mapping[r] = avail[b][0]
                c = conflicts(hot, lambda x: cur(x) % 4)
                del mapping[r]
                if best_c is None or c < best_c:
                    best_b, best_c = b, c
            mapping[r] = avail[best_b].pop(0)
    perm = dict(mapping)
    targets = set(perm.values())
    vacated = [r for r in sorted(acc) if r not in targets]
    displaced = [r for r in sorted(targets) if r not in acc and r in used]
    assert len(displaced) <= len(vacated)
    for a, b in zip(displaced, vacated):
        perm[a] = b
    assert len(set(perm.values())) == len(perm), 'not a permutation'
    perm = {a: b for a, b in perm.items() if a != b}
    after = conflicts([ren(t, perm) for t in hot], lambda r: r % 4)
    return perm, before, after, len([t for t in hot if opcode(t).startswith(('HFMA2', 'HMUL2'))]), (lo, hi)


def ren(t, perm):
    return REG_RE.sub(lambda m: f"R{perm.get(int(m.group(1)), int(m.group(1)))}{m.group(2)}", t)


def elf_text_section(blob, name_prefix):
    shoff, = struct.unpack_from('<Q', blob, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from('<HHH', blob, 0x3A)
    def sh(i):
        return struct.unpack_from('<IIQQQQIIQQ', blob, shoff + i * shentsize)
    stro = sh(shstrndx)[4]
    for i in range(shnum):
        h = sh(i)
        nm = blob[stro + h[0]: blob.index(b'\0', stro + h[0])].decode()
        if nm.startswith(name_prefix):
            return nm, h[4], h[5]
    raise SystemExit('no ' + name_prefix + ' section')


def patch_word(w, old_new, fields):
    for f in fields:
        v = (w >> f) & 0xff
        if v in old_new:
            w = (w & ~(0xff << f)) | (old_new[v] << f)
    return w


def build(ins, orig, off, regcount, nvdis, out_cub, kname):
    """one variant (WIDE as set): the verified patched cubin, (before, after, nfma)"""
    exclude = set()
    for attempt in range(8):
        perm, before, after, nfma, (lo, hi) = bankfix(ins, regcount, exclude)
        print(f'{"wide" if WIDE else "narrow"}: {kname}: {len(ins)} instructions, regcount {regcount}, hot loop '
              f'0x{lo:x}-0x{hi:x}, {nfma} HFMA2/HMUL2')
        print(f'  bank conflicts before {before}, after {after} (renamed {len(perm)} registers, {len(exclude)} excluded)')
        # patch every instruction that names a moved register (register fields only); nvdisasm must then show
        # exactly the renamed text for each of them
        blob = bytearray(orig)
        expect = {a: ren(t, perm) for a, t, _ in ins}
        touched = [(a, t, w) for a, t, w in ins if expect[a] != t]
        miscount = set()
        for a, t, w in touched:
            names = {int(r) for r, _ in REG_RE.findall(t)}
            old_new = {r: perm[r] for r in names if r in perm}
            # exactly as many fields patched as the text shows moved-register operands: no hidden (unprinted) bits
            # may change
            occ = sum(1 for r, _ in REG_RE.findall(t) if int(r) in perm)
            if sum(1 for f in FIELDS if ((w >> f) & 0xff) in old_new) != occ:
                miscount.add(a)
            struct.pack_into('<Q', blob, off + a, patch_word(w, old_new, FIELDS))
        open(out_cub, 'wb').write(blob)
        got = {a: t for a, t, _ in disasm(nvdis, out_cub)}
        bad = [(a, t, w) for a, t, w in touched if got.get(a) != expect[a] or a in miscount]
        if not bad:
            print(f'  verified: {len(touched)} patched instructions disassemble to the renamed text; all others unchanged')
            return bytes(blob), (before, after, nfma)
        # an instruction whose registers do not sit in the standard fields: keep its registers out of the renaming
        for a, t, w in bad:
            print(f'  0x{a:04x}: register fields not standard ("{t}"): its registers stay put')
            exclude.update(int(r) for r, _ in REG_RE.findall(t))
    raise SystemExit('patch failed')


# sizes of the field types seg_tab may use (64-bit device ABI: pointers and long long are 8 bytes)
FIELD_SIZE = {'int': 4, 'unsigned': 4, 'float': 4, 'long long': 8, 'int64_t': 8, 'size_t': 8, 'double': 8}


def seg_tab_layout(ktxt):
    """[(field, offset)], size, align of struct seg_tab, from its text (C layout rules; every field type must be in
    FIELD_SIZE or a pointer). Cross-checked against the cubin's parameter 0 size before anything is written."""
    body = re.search(r'struct seg_tab \{(.*?)\};', ktxt, re.S).group(1)
    fields, off, align = [], 0, 1
    for line in body.split(';'):
        line = re.sub(r'//[^\n]*', '', line).strip()
        if not line:
            continue
        m = re.match(r'^(.*?)(\w+)\s*(?:\[(\d+)\])?$', line, re.S)
        if not m:
            raise SystemExit(f'seg_tab: cannot parse field "{line}"')
        typ, name, n = ' '.join(m.group(1).split()), m.group(2), int(m.group(3) or 1)
        if '*' in typ:
            sz = 8
        else:
            t = typ.replace('const ', '').strip()
            if t not in FIELD_SIZE:
                raise SystemExit(f'seg_tab: unknown field type "{typ}" (extend FIELD_SIZE)')
            sz = FIELD_SIZE[t]
        off = (off + sz - 1) // sz * sz
        fields.append((name, off))
        off += sz * n
        align = max(align, sz)
    return fields, (off + align - 1) // align * align, align


def kconsts(ktxt):
    """the integer constants of the kernel text (BM, BK2, FOLD_K2_DEFAULT, XEXP)"""
    return re.findall(r'constexpr int (\w+) = (-?\d+);', ktxt)


def kparams(blob, kname):
    """[(offset, size)] of the kernel's parameters by ordinal, from its EIATTR_KPARAM_INFO records (.nv.info.<kname>)"""
    shoff, = struct.unpack_from('<Q', blob, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from('<HHH', blob, 0x3A)
    def sh(i):
        return struct.unpack_from('<IIQQQQIIQQ', blob, shoff + i * shentsize)
    stro = sh(shstrndx)[4]
    for i in range(shnum):
        h = sh(i)
        nm = blob[stro + h[0]: blob.index(b'\0', stro + h[0])].decode()
        if nm != '.nv.info.' + kname:
            continue
        p, end, params = h[4], h[4] + h[5], {}
        while p < end:
            fmt, attr, v = struct.unpack_from('<BBH', blob, p)
            p += 4
            if fmt == 4:   # EIFMT_SVAL: v = payload bytes
                if attr == 0x17:   # EIATTR_KPARAM_INFO: index u32, ordinal u16, offset u16, flags u32 (size: bits 18..31)
                    _, ordn, poff, flags = struct.unpack_from('<IHHI', blob, p)
                    params[ordn] = (poff, (flags >> 18) & 0x3fff)
                p += v
        if not params or sorted(params) != list(range(len(params))):
            raise SystemExit(f'no / incomplete EIATTR_KPARAM_INFO for {kname}')
        return [params[o] for o in range(len(params))]
    raise SystemExit('no .nv.info section for ' + kname)


def main():
    ktxt, cu = kernel_source()
    h = hashlib.sha1(ktxt.encode()).hexdigest()[:16]
    global WIDE
    if '--check' in sys.argv:
        m = re.search(r'gemm_sass_fold8_src_hash\[\] = "(\w+)"', open(OUT).read()) if os.path.exists(OUT) else None
        f = re.search(r'gemm_sass_fold8_format = (\d+);', open(OUT).read()) if m else None
        if not m or m.group(1) != h:
            print(f'{OUT} is stale (kernel hash {h}): rerun {__file__}')
            sys.exit(1)
        if not f or int(f.group(1)) != FORMAT:
            print(f'{OUT} was written by an older fold_bankfix.py (format {f.group(1) if f else 1}, now {FORMAT}): '
                  f'rerun {__file__}')
            sys.exit(1)
        print('gemm_sass_fold8.h matches the fold kernel')
        return
    nvcc, nvdis = 'nvcc', 'nvdisasm'
    tmp = tempfile.mkdtemp(prefix='foldsass-')
    src = os.path.join(tmp, 'fold8.cu')
    cub = os.path.join(tmp, 'fold8.cubin')
    open(src, 'w').write(cu)
    subprocess.check_call([nvcc] + NVCC_FLAGS + ['-o', cub, src])
    blob = bytearray(open(cub, 'rb').read())
    secname, off, size = elf_text_section(blob, '.text.')
    kname = secname[len('.text.'):]
    ins = disasm(nvdis, cub)
    regcount = struct.unpack_from('<I', blob, blob.index(b'\x04\x2f\x08\x00') + 8)[0]
    orig = bytes(blob)
    params = kparams(orig, kname)
    fields, tsize, talign = seg_tab_layout(ktxt)
    if params[0][1] != tsize:
        raise SystemExit(f'seg_tab layout: computed size {tsize}, the cubin\'s parameter 0 has {params[0][1]} bytes')
    out_cub = os.path.join(tmp, 'fold8_bf.cubin')
    variants = []
    for wide in (False, True):   # [0] the accumulators + free registers (bankfix.py), [1] BANKFIX_WIDE
        WIDE = wide
        variants.append(build(ins, orig, off, regcount, nvdis, out_cub, kname))
    if '--dry' in sys.argv:
        print('--dry: header not written')
        return
    with open(OUT, 'w') as f:
        f.write('// Generated by tools/sass/fold_bankfix.py -- do not edit. fold_seg_kernel<8, false> (src/fast/gemm.cu)\n')
        f.write('// compiled alone for sm_60 and register-renamed against operand bank conflicts (a permutation of register\n')
        f.write('// names, bit-identical results). Variant 0: accumulators + free registers; variant 1: BANKFIX_WIDE;\n')
        f.write('// variant 2: the unpatched compile (debugging: the loader / launch path without the renaming).\n')
        for i, (_, (before, after, nfma)) in enumerate(variants):
            f.write(f'//   variant {i}: {before} -> {after} conflicting HFMA2 source pairs over {nfma} HFMA2/HMUL2\n')
        f.write('// Included by src/fast/gemm.cu after the kernel (inside namespace p100e::fast, where seg_tab and the kernel\n')
        f.write('// constants are visible): the asserts below pin what the launcher hands this binary.\n')
        f.write('#pragma once\n')
        f.write(f'static const char gemm_sass_fold8_src_hash[] = "{h}";\n')
        f.write(f'static constexpr int gemm_sass_fold8_format = {FORMAT};\n')
        f.write(f'static const char gemm_sass_fold8_name[] = "{kname}";\n')
        f.write('// the cubin\'s kernel parameters (EIATTR_KPARAM_INFO) by ordinal; fold_gemm static_asserts its argument list\n')
        f.write('// against these sizes\n')
        f.write(f'static constexpr int gemm_sass_fold8_nparam = {len(params)};\n')
        f.write(f'static constexpr unsigned gemm_sass_fold8_param_size[{len(params)}] = '
                f'{{{", ".join(str(sz) for _, sz in params)}}};\n')
        f.write(f'static_assert(sizeof(seg_tab) == {tsize} && alignof(seg_tab) == {talign}, '
                f'"seg_tab changed: rerun tools/sass/fold_bankfix.py");\n')
        f.write('static_assert(' + ' &&\n              '.join(f'offsetof(seg_tab, {n}) == {o}' for n, o in fields) +
                ',\n              "seg_tab changed: rerun tools/sass/fold_bankfix.py");\n')
        f.write('static_assert(' + ' && '.join(f'{n} == {v}' for n, v in kconsts(ktxt)) +
                ', "fold kernel constants changed: rerun tools/sass/fold_bankfix.py");\n')
        for i, (data, _) in enumerate(variants):
            f.write(f'static const unsigned char gemm_sass_fold8_v{i}[{len(data)}] = {{\n')
            for j in range(0, len(data), 24):
                f.write('    ' + ', '.join(f'0x{b:02x}' for b in data[j:j + 24]) + ',\n')
            f.write('};\n')
        f.write(f'static const unsigned char gemm_sass_fold8_v2[{len(orig)}] = {{\n')
        for j in range(0, len(orig), 24):
            f.write('    ' + ', '.join(f'0x{b:02x}' for b in orig[j:j + 24]) + ',\n')
        f.write('};\n')
        f.write('static const unsigned char * const gemm_sass_fold8[3] = {gemm_sass_fold8_v0, gemm_sass_fold8_v1, gemm_sass_fold8_v2};\n')
    print(f'wrote {OUT} (2 variants, source hash {h})')


if __name__ == '__main__':
    main()
