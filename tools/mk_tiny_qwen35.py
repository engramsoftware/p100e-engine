"""Write a tiny random-weight qwen35 (Qwen3.8 hybrid) GGUF for local engine tests on a small GPU.

The real model does not fit a small GPU; this one exercises the same code paths (gated delta net + full
attention layers with D = 256 and GQA 6, an MTP / nextn block, q4_K / q8_0 / f16 weights with fast kernels, tensor
parallel shapes for 1 or 2 ranks) in a few MB. Tokens are garbage; bit identity checks (sessions alone vs batched,
offload / restore, MTP vs plain) are meaningful.

usage: python mk_tiny_qwen35.py out.gguf [--layers 8] [--seed 1]
       [--embd 1024 --ff 2048 --heads 12,2 --ssm 4,8 --vocab 4096 --fast]
The shape options default to the shapes above. --heads 24,4 --ssm 16,48 are Qwen3.8-27B's head counts (tensor parallel
over 1, 2 or 4 ranks; over 3 ranks they also exercise every uneven path with the
default --ff 2048 / --vocab 4096 / --embd 1024: KV heads 2/1/1, k heads in fours, 8 FFN and 16 vocab blocks of 256,
eh_proj 8 column blocks with rank 1 straddling the halves); --fast draws the random bytes with random.randbytes (much faster, a different
byte stream than the default, so the default file stays as it was).
No dependencies beyond the standard library (writes GGUF v3 directly).
"""
import argparse
import random
import struct

GGML_F32, GGML_F16, GGML_Q8_0, GGML_Q2_K, GGML_Q4_K, GGML_Q6_K = 0, 1, 8, 10, 12, 14
BLK = {GGML_Q8_0: (32, 34), GGML_Q2_K: (256, 84), GGML_Q4_K: (256, 144), GGML_Q6_K: (256, 210)}


def f16(x):
    return struct.pack('<e', x)


def gstr(s):
    b = s.encode()
    return struct.pack('<Q', len(b)) + b


class Writer:
    def __init__(self):
        self.kv = []
        self.tensors = []   # (name, type, ne, bytes)

    def u32(self, k, v): self.kv.append(gstr(k) + struct.pack('<II', 4, v))
    def f32(self, k, v): self.kv.append(gstr(k) + struct.pack('<If', 6, v))
    def str(self, k, v): self.kv.append(gstr(k) + struct.pack('<I', 8) + gstr(v))

    def tensor(self, name, typ, ne, data):
        self.tensors.append((name, typ, ne, data))

    def write(self, path):
        align = 32
        out = bytearray(b'GGUF' + struct.pack('<IQQ', 3, len(self.tensors), len(self.kv)))
        for k in self.kv:
            out += k
        off = 0
        offs = []
        for name, typ, ne, data in self.tensors:
            offs.append(off)
            out += gstr(name) + struct.pack('<I', len(ne)) + b''.join(struct.pack('<Q', n) for n in ne)
            out += struct.pack('<IQ', typ, off)
            off += (len(data) + align - 1) // align * align
        out += b'\0' * ((-len(out)) % align)
        for (name, typ, ne, data), o in zip(self.tensors, offs):
            out += data
            out += b'\0' * ((-len(data)) % align)
        open(path, 'wb').write(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--layers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--q2k', action='store_true', help='ffn_gate / ffn_up / ffn_down and attn_q as Q2_K (fast q2kr layout test)')
    ap.add_argument('--embd', type=int, default=1024)
    ap.add_argument('--ff', type=int, default=2048)
    ap.add_argument('--heads', default='12,2', help='attention heads,KV heads (head dim 256)')
    ap.add_argument('--ssm', default='4,8', help='delta-net k heads,v heads (state 128)')
    ap.add_argument('--vocab', type=int, default=4096)
    ap.add_argument('--fast', action='store_true', help='random bytes via random.randbytes (other stream than the default)')
    ap.add_argument('--ssm-out-q8', action='store_true', help='ssm_out as Q8_0 (as a Q6_K model; q8r layout test)')
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    E, FF, HD, NROT = a.embd, a.ff, 256, 64
    H, HKV = (int(v) for v in a.heads.split(','))
    HK, HV = (int(v) for v in a.ssm.split(','))
    DK = 128
    DV = DK
    VOCAB = a.vocab
    L = a.layers
    C = 2 * HK * DK + HV * DV
    w = Writer()
    arch = 'qwen35'
    w.str('general.architecture', arch)
    w.str('general.name', 'tiny-qwen35-random')
    for k, v in [('embedding_length', E), ('block_count', L + 1), ('nextn_predict_layers', 1),
                 ('feed_forward_length', FF), ('attention.head_count', H), ('attention.head_count_kv', HKV),
                 ('attention.key_length', HD), ('attention.value_length', HD), ('rope.dimension_count', NROT),
                 ('ssm.conv_kernel', 4), ('ssm.group_count', HK), ('ssm.time_step_rank', HV), ('ssm.state_size', DK),
                 ('ssm.inner_size', HV * DV), ('full_attention_interval', 4), ('context_length', 262144)]:
        w.u32(arch + '.' + k, v)
    w.f32(arch + '.attention.layer_norm_rms_epsilon', 1e-6)
    w.f32(arch + '.rope.freq_base', 1e7)

    def f32v(n, base=0.0, scale=0.1):
        return b''.join(struct.pack('<f', base + scale * (rnd.random() * 2 - 1)) for _ in range(n))

    def quant(typ, n_in, n_out):
        bl, bs = BLK[typ]
        nb = n_in // bl * n_out
        data = bytearray(rnd.randbytes(nb * bs)) if a.fast else bytearray(rnd.getrandbits(8) for _ in range(nb * bs))
        offs = {GGML_Q8_0: [0], GGML_Q2_K: [80, 82], GGML_Q4_K: [0, 2], GGML_Q6_K: [208]}[typ]
        for b in range(nb):
            for o in offs:
                # small scales: |w| <~ 0.06 (q4_K: d * 63 * 15, q6_K: d * 127 * 32, q8_0: d * 127), so the random
                # residual stream stays finite through the layers
                data[b * bs + o:b * bs + o + 2] = f16((0.5 + rnd.random()) / {GGML_Q8_0: 4096.0, GGML_Q2_K: 2048.0, GGML_Q4_K: 16384.0,
                                                                             GGML_Q6_K: 65536.0}[typ])
        return bytes(data)

    def qt(name, typ, n_in, n_out):
        w.tensor(name, typ, [n_in, n_out], quant(typ, n_in, n_out))

    def vec(name, n, base=1.0, scale=0.1):
        w.tensor(name, GGML_F32, [n], f32v(n, base, scale))

    w.tensor('token_embd.weight', GGML_F16, [E, VOCAB],
             b''.join(f16(rnd.random() * 2 - 1) for _ in range(E * VOCAB)))
    vec('output_norm.weight', E)
    qt('output.weight', GGML_Q6_K, E, VOCAB)

    def attn_block(i):
        qt(f'blk.{i}.attn_q.weight', GGML_Q2_K if a.q2k else GGML_Q4_K, E, H * 2 * HD)
        qt(f'blk.{i}.attn_k.weight', GGML_Q8_0, E, HKV * HD)
        qt(f'blk.{i}.attn_v.weight', GGML_Q8_0, E, HKV * HD)
        qt(f'blk.{i}.attn_output.weight', GGML_Q4_K, H * HD, E)
        vec(f'blk.{i}.attn_q_norm.weight', HD)
        vec(f'blk.{i}.attn_k_norm.weight', HD)

    def ffn(i):
        vec(f'blk.{i}.attn_norm.weight', E)
        vec(f'blk.{i}.post_attention_norm.weight', E)
        qt(f'blk.{i}.ffn_gate.weight', GGML_Q2_K if a.q2k else GGML_Q4_K, E, FF)
        qt(f'blk.{i}.ffn_up.weight', GGML_Q2_K if a.q2k else GGML_Q4_K, E, FF)
        qt(f'blk.{i}.ffn_down.weight', GGML_Q2_K if a.q2k else GGML_Q6_K, FF, E)

    for i in range(L):
        ffn(i)
        if (i + 1) % 4 == 0:
            attn_block(i)
        else:
            qt(f'blk.{i}.attn_qkv.weight', GGML_Q4_K, E, C)
            qt(f'blk.{i}.attn_gate.weight', GGML_Q4_K, E, HV * DV)
            qt(f'blk.{i}.ssm_beta.weight', GGML_Q8_0, E, HV)
            qt(f'blk.{i}.ssm_alpha.weight', GGML_Q8_0, E, HV)
            w.tensor(f'blk.{i}.ssm_conv1d.weight', GGML_F32, [4, C], f32v(4 * C, 0.0, 0.5))
            vec(f'blk.{i}.ssm_dt.bias', HV, 0.0, 0.5)
            vec(f'blk.{i}.ssm_a', HV, -1.0, 0.5)
            vec(f'blk.{i}.ssm_norm.weight', DV)
            qt(f'blk.{i}.ssm_out.weight', GGML_Q8_0 if a.ssm_out_q8 else GGML_Q4_K, HV * DV, E)
    # MTP / nextn block
    i = L
    ffn(i)
    attn_block(i)
    qt(f'blk.{i}.nextn.eh_proj.weight', GGML_Q8_0, 2 * E, E)
    vec(f'blk.{i}.nextn.enorm.weight', E)
    vec(f'blk.{i}.nextn.hnorm.weight', E)
    vec(f'blk.{i}.nextn.shared_head_norm.weight', E)
    w.write(a.out)
    print('wrote', a.out, len(w.tensors), 'tensors')


if __name__ == '__main__':
    main()
