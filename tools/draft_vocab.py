"""Reduced draft vocabulary for the MTP draft head (p100e-run --draft-vocab).

The draft head's LM head reads all 248320 rows of output.weight (q6_K, 0.92 ms per GPU per draft). The draft only
has to guess the main model's next token, so it can score a frequency-ranked subset of the vocab instead; verify
still runs the full head, so the generated text is unchanged (only the acceptance rate can drop).

Steps (python -I tools/draft_vocab.py <cmd> ...):
  vocab  <p100e-info --vocab dump> <template vocab gguf> <gguf-py dir> <out.gguf>
         rebuild a vocab-only GGUF with the model's own tokenizer (tokens, types, merges, pre) so llama-tokenize can
         tokenize text exactly like the model. The template supplies the non-tokenizer keys llama.cpp needs.
  count  <llama-tokenize.exe> <vocab.gguf> <out_counts.tsv> <corpus files...>
         tokenize each corpus file (chunked) and count token ids
  rank   <p100e-info --vocab dump> <counts.tsv> <out_ranked.txt>
         ranked token-id list: every non-normal token (control / user-defined: chat and tool markup, <think>) and
         the 256 byte tokens first, then by corpus count (desc), then by token id (BPE merge order) for the
         tokens the corpus never produced. p100e-run takes the first N ids of this file.
"""
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path


def read_dump(path):
    kv, arrays, cur = {}, {}, None
    with open(path, "r", encoding="ascii", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line.startswith("KVS "):
                _, k, h = line.split(" ", 2)
                kv[k] = bytes.fromhex(h).decode("utf-8")
                cur = None
            elif line.startswith("KVN "):
                _, k, t, i, fl = line.split(" ")
                kv[k] = int(i)
                cur = None
            elif line.startswith("KVA "):
                _, k, t, n = line.split(" ")
                cur = k
                arrays[k] = []
            elif line.startswith("S ") and cur:
                arrays[cur].append(bytes.fromhex(line[2:]).decode("utf-8"))
            elif line.startswith("N ") and cur:
                arrays[cur].append(int(float(line[2:])))
    return kv, arrays


def cmd_vocab(dump, template, gguf_py, out):
    sys.path.insert(0, gguf_py)
    import gguf
    kv, arrays = read_dump(dump)
    rd = gguf.GGUFReader(template)
    arch = rd.fields["general.architecture"].contents()
    w = gguf.GGUFWriter(out, arch)
    for k, f in rd.fields.items():
        if k.startswith("tokenizer.") or k.startswith("GGUF.") or k == "general.architecture":
            continue
        v = f.contents()
        t = f.types[0]
        if t == gguf.GGUFValueType.ARRAY:
            continue
        w.add_key_value(k, v, t)
    w.add_tokenizer_model(kv["tokenizer.ggml.model"])
    w.add_tokenizer_pre(kv["tokenizer.ggml.pre"])
    w.add_token_list(arrays["tokenizer.ggml.tokens"])
    w.add_token_types(arrays["tokenizer.ggml.token_type"])
    w.add_token_merges(arrays["tokenizer.ggml.merges"])
    w.add_bos_token_id(kv["tokenizer.ggml.bos_token_id"])
    w.add_eos_token_id(kv["tokenizer.ggml.eos_token_id"])
    w.add_pad_token_id(kv["tokenizer.ggml.padding_token_id"])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out}: {len(arrays['tokenizer.ggml.tokens'])} tokens, {len(arrays['tokenizer.ggml.merges'])} merges")


def tokenize(exe, vocab, text):
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as tf:
        tf.write(text)
        name = tf.name
    try:
        r = subprocess.run([exe, "-m", vocab, "-f", name, "--ids", "--log-disable", "--no-parse-special"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
    finally:
        Path(name).unlink()
    line = [l for l in r.stdout.splitlines() if l.startswith("[")]
    if r.returncode != 0 or not line:
        raise RuntimeError(f"llama-tokenize failed ({r.returncode}): {r.stderr[-500:]}")
    s = line[-1].strip("[] ")
    return [int(x) for x in s.split(",")] if s else []


def cmd_count(exe, vocab, out, files):
    cnt = Counter()
    for fn in files:
        text = Path(fn).read_text(encoding="utf-8", errors="replace")
        n0 = sum(cnt.values())
        chunk = 1 << 20
        i = 0
        while i < len(text):
            j = min(len(text), i + chunk)
            while j < len(text) and text[j - 1] != "\n" and j - i < chunk + 4096:
                j += 1   # cut at a line end so no word is split
            cnt.update(tokenize(exe, vocab, text[i:j]))
            i = j
        print(f"{fn}: {sum(cnt.values()) - n0} tokens")
    with open(out, "w") as f:
        for t, c in cnt.most_common():
            f.write(f"{t}\t{c}\n")
    print(f"total {sum(cnt.values())} tokens, {len(cnt)} distinct -> {out}")


def cmd_rank(dump, counts, out):
    kv, arrays = read_dump(dump)
    types = arrays["tokenizer.ggml.token_type"]
    n = len(types)
    cnt = {}
    for line in open(counts):
        t, c = line.split()
        cnt[int(t)] = int(c)
    first = [i for i in range(n) if types[i] != 1]        # control / user-defined / unused: chat + tool markup
    first += [i for i in range(min(256, n)) if types[i] == 1]
    seen = set(first)
    rest = sorted((i for i in range(n) if i not in seen), key=lambda i: (-cnt.get(i, 0), i))
    order = first + rest
    with open(out, "w") as f:
        for i in order:
            f.write(f"{i}\n")
    tot = sum(cnt.values())
    for N in (8192, 16384, 24576, 32768, 49152, 65536):
        cov = sum(cnt.get(i, 0) for i in order[:N]) / max(1, tot)
        print(f"top {N:6d}: corpus coverage {100 * cov:.2f}%")
    print(f"{len(first)} special/byte tokens first; wrote {len(order)} ids -> {out}")


def vocab_hash(tokens):
    # FNV-1a 64 over every token's UTF-8 bytes + a 0 byte, in id order (src/tp_mtp.cu computes the same from the GGUF)
    h = 1469598103934665603
    for t in tokens:
        for b in t.encode("utf-8") + b"\0":
            h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def cmd_emit(dump, ranked, out, n):
    kv, arrays = read_dump(dump)
    toks = arrays["tokenizer.ggml.tokens"]
    ids = [int(l) for l in open(ranked)][: int(n)]
    with open(out, "w", newline="\n") as f:
        f.write("// generated by tools/draft_vocab.py emit: draft-head vocabulary ranking (most frequent first) for the\n")
        f.write(f"// Qwen3.8 tokenizer ({len(toks)} tokens, pre={kv['tokenizer.ggml.pre']}): control/user-defined and byte tokens,\n")
        f.write("// then token counts over wikitext-2 train + llama.cpp sources + docs (5.4M tokens), then token id order\n")
        f.write(f"static const uint64_t kDraftVocabHash = 0x{vocab_hash(toks):016x}ull;\n")
        f.write(f"static const int kDraftVocabTokens = {len(toks)};\n")
        f.write(f"static const int kDraftRankN = {len(ids)};\n")
        f.write("static const int32_t kDraftRank[] = {\n")
        for i in range(0, len(ids), 16):
            f.write(",".join(str(x) for x in ids[i:i + 16]) + ",\n")
        f.write("};\n")
    print(f"wrote {out}: {len(ids)} ids, vocab hash {vocab_hash(toks):016x}")


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "emit":
        cmd_emit(*sys.argv[2:6])
        sys.exit(0)
    if c == "vocab":
        cmd_vocab(*sys.argv[2:6])
    elif c == "count":
        cmd_count(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
    elif c == "rank":
        cmd_rank(*sys.argv[2:5])
    else:
        print(__doc__)
