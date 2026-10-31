"""Check p100e-dequant GPU output against gguf-py's numpy dequantizer (independent CPU reference).

usage: python -I tests/verify_dequant.py <model.gguf> <dump_dir> <gguf-py dir>
Reads <dump_dir>/manifest.tsv written by p100e-dequant; for every tensor compares the first <rows>
rows (f32 output: max abs diff, must be exact or within 1 ulp-ish; f16 output: max rel error vs
reference rounded to f16). Writes <dump_dir>/verify.tsv and exits 1 on any failure.
"""
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    model, dump, gguf_py = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    sys.path.insert(0, gguf_py)
    from gguf import GGUFReader
    from gguf.quants import dequantize

    reader = GGUFReader(model)
    tensors = {t.name: t for t in reader.tensors}
    lines = (dump / "manifest.tsv").read_text().splitlines()[1:]
    out = ["tensor\ttype\tn\tf32_max_abs\tf32_exact\tf16_max_abs_vs_ref16\tresult"]
    failed = 0
    for line in lines:
        name, tname, ne0, rows, fname, _ms = line.split("\t")
        if fname == "-":
            out.append(f"{name}\t{tname}\t0\t-\t-\t-\tNO_GPU_DEQUANT")
            failed += 1
            continue
        ne0, rows = int(ne0), int(rows)
        t = tensors[name]
        # gguf-py data is shaped [rows, row_bytes] for quantized types; take the first `rows` rows
        raw = np.asarray(t.data).reshape(-1, np.asarray(t.data).shape[-1])[:rows]
        ref = dequantize(raw, t.tensor_type).astype(np.float32).reshape(-1)[: ne0 * rows]
        g32 = np.fromfile(dump / f"{fname}.f32", dtype=np.float32)
        g16 = np.fromfile(dump / f"{fname}.f16", dtype=np.float16).astype(np.float32)
        d32 = np.abs(g32 - ref)
        exact = int((d32 == 0).sum())
        ref16 = ref.astype(np.float16).astype(np.float32)
        d16 = np.abs(g16 - ref16)
        tol32 = 1e-6 * max(1.0, float(np.abs(ref).max()))
        ok = d32.max() <= tol32 and d16.max() <= 2 * np.spacing(np.abs(ref16).max().astype(np.float16)).astype(np.float32)
        res = "PASS" if ok else "FAIL"
        failed += not ok
        out.append(f"{name}\t{tname}\t{ref.size}\t{d32.max():.3e}\t{exact}/{ref.size}\t{d16.max():.3e}\t{res}")
        print(out[-1])
    (dump / "verify.tsv").write_text("\n".join(out) + "\n")
    print(f"{len(lines) - failed}/{len(lines)} types pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
