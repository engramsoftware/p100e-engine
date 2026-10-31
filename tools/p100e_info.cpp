// p100e-info <model.gguf> [--tensors]
// Prints model metadata (non-array keys), a quant-type histogram and, with --tensors, every tensor.
#include "../src/gguf.h"

#include <cstdio>
#include <cstring>
#include <map>

using namespace p100e;

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: p100e-info <model.gguf> [--tensors]\n");
        return 1;
    }
    const bool list = argc > 2 && strcmp(argv[2], "--tensors") == 0;
    const bool vocab = argc > 2 && strcmp(argv[2], "--vocab") == 0;
    try {
        gguf_file g(argv[1]);
        if (vocab) {
            // tokenizer dump for offline tools (scripts that rebuild a vocab-only GGUF): every tokenizer.* key,
            // strings and string arrays hex-encoded (byte-safe), one item per line
            auto hex = [](const std::string & s) {
                static const char * d = "0123456789abcdef";
                std::string o;
                for (unsigned char c : s) { o += d[c >> 4]; o += d[c & 15]; }
                return o;
            };
            for (const auto & [k, v] : g.kv) {
                if (k.rfind("tokenizer.", 0) != 0 && k != "general.architecture") continue;
                if (v.type == gguf_vtype::STRING) { printf("KVS %s %s\n", k.c_str(), hex(v.s).c_str()); continue; }
                if (v.type != gguf_vtype::ARRAY) { printf("KVN %s %u %lld %.17g\n", k.c_str(), (unsigned) v.type, (long long) v.i, v.f); continue; }
                printf("KVA %s %u %llu\n", k.c_str(), (unsigned) v.arr_type, (unsigned long long) v.u);
                for (size_t i = 0; i < v.arr_str.size(); ++i) printf("S %s\n", hex(v.arr_str[i]).c_str());
                for (size_t i = 0; i < v.arr_num.size(); ++i) printf("N %.17g\n", v.arr_num[i]);
            }
            printf("VOCAB_END\n");
            return 0;
        }
        printf("gguf v%u  tensors %zu  kv %zu  alignment %llu  size %.2f GiB\n", g.version, g.tensors.size(),
               g.kv.size(), (unsigned long long) g.alignment, g.file_size / 1073741824.0);
        for (const auto & [k, v] : g.kv) {
            switch (v.type) {
                case gguf_vtype::STRING: printf("  %s = %.80s\n", k.c_str(), v.s.c_str()); break;
                case gguf_vtype::ARRAY:  printf("  %s = [%llu items]\n", k.c_str(), (unsigned long long) v.u); break;
                case gguf_vtype::F32:
                case gguf_vtype::F64:    printf("  %s = %g\n", k.c_str(), v.f); break;
                default:                 printf("  %s = %lld\n", k.c_str(), (long long) v.i); break;
            }
        }
        std::map<std::string, std::pair<int, double>> hist;
        bool missing = false;
        for (const auto & t : g.tensors) {
            const qtype_info & ti = get_type_info(t.type);
            auto & h = hist[ti.name];
            h.first++;
            h.second += t.nbytes / 1048576.0;
            if (!ti.has_gpu_dequant) missing = true;
            if (list) {
                printf("%-48s %-8s [%lld, %lld, %lld, %lld] %.2f MiB\n", t.name.c_str(), ti.name,
                       (long long) t.ne[0], (long long) t.ne[1], (long long) t.ne[2], (long long) t.ne[3],
                       t.nbytes / 1048576.0);
            }
        }
        printf("type histogram:\n");
        for (const auto & [name, h] : hist) printf("  %-8s %5d tensors %10.1f MiB\n", name.c_str(), h.first, h.second);
        printf(missing ? "WARNING: some tensor types have no GPU dequant yet\n" : "all tensor types have GPU dequant\n");
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
