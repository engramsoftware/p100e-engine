"""p100e-server slot-loop tests on the CPU fake engine (no GPU, no weights): build/p100e-server-fake.exe.

  python -I tests/server/fake_engine_test.py <p100e-server-fake.exe> <model.gguf> [--port 18190] [--keep-logs DIR]

The GGUF is only read for its tokenizer (vocab-only load, as p100e-server --p100e-no-engine). The fake engine
(tests/server/fake_engine.cpp) emits a fixed token cycle, never end-of-generation, drafts the true continuation (or a
wrong token every P100E_FAKE_DRAFT_MISS drafts) and throws on any call the real engine could not serve.

  queue_after_launch_failure  -np 3 on the engine's one sequence: A runs (slow steps), B (a token id outside the
                              vocabulary: launch_slot_with_task fails) and C wait in the queue. When A ends, B fails
                              with 400 and C must still run - the launch failure has to release the next queued task.
  ctx_limit_mtp               -c 64, greedy, no n_predict: the request runs into the context limit. Without MTP, with
                              MTP k=3 (all drafts right) and MTP k=3 with every 3rd draft wrong: identical text, the
                              same count (n_ctx - n_prompt), truncated - the cut must not depend on accepted drafts.
  metrics                     /metrics after a request: n_decode counts the prompt and decode forwards,
                              n_busy_slots_per_decode > 0, n_tokens_max == the slot's tokens at the end.
  engine_fatal                P100E_FAKE_FATAL_AT=3: the 3rd decode throws engine_fatal_error (engine_tp's P2P
                              allreduce timeout): the server must log it and exit with code 3 (supervisor restart)
                              instead of answering an error and carrying on.
  live_status                 slow fake steps (400 ms): /slots /metrics /props /health answer from the loop's
                              snapshot in < 100 ms mid-step, prompt progress per chunk, decoded tokens rising, idle
                              slots keep the last task's numbers, /props build_info names the p100e commit.
Prints one PASS / FAIL line per check and FAKE_TEST_SUMMARY; exit code 0 when everything passed.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request


def http(port, method, path, body=None, timeout=30.0):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data is not None else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8")
            return r.status, raw
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:  # timeout, connection refused
        return None, f"{type(e).__name__}: {e}"


class Server:
    def __init__(self, exe, gguf, port, args, env_extra, log_dir, name):
        self.port = port
        self.log_path = os.path.join(log_dir, f"{name}.log")
        env = dict(os.environ)
        env.update(env_extra)
        self.log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen([os.path.abspath(exe), "-m", gguf, "--host", "127.0.0.1", "--port", str(port)] + args,
                                     stdout=self.log, stderr=subprocess.STDOUT, env=env)

    def wait_ready(self, timeout=60.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                return False
            st, _ = http(self.port, "GET", "/health", timeout=2.0)
            if st == 200:
                return True
            time.sleep(0.2)
        return False

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        self.log.close()

    def log_text(self):
        with open(self.log_path, encoding="utf-8", errors="replace") as f:
            return f.read()


results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)


def test_queue_after_launch_failure(exe, gguf, port, log_dir):
    srv = Server(exe, gguf, port, ["-c", "3072", "-np", "3", "--p100e-mtp", "0"], {"P100E_FAKE_DELAY_MS": "40"},
                 log_dir, "queue")
    try:
        if not srv.wait_ready():
            check("queue: server start", False, srv.log_text()[-2000:])
            return
        out = {}

        def run(key, body, timeout):
            t0 = time.time()
            out[key] = http(port, "POST", "/completion", body, timeout=timeout) + (time.time() - t0,)

        # A: ~40 steps x 40 ms; B and C queue behind it (the fake engine has one sequence)
        ta = threading.Thread(target=run, args=("A", {"prompt": "alpha", "n_predict": 40, "temperature": 0}, 60))
        tb = threading.Thread(target=run, args=("B", {"prompt": [999999999], "n_predict": 4, "temperature": 0}, 60))
        # C's own deadline: A needs ~2 s; without the fix C waits until some other request ends (here: never)
        tc = threading.Thread(target=run, args=("C", {"prompt": "beta", "n_predict": 4, "temperature": 0}, 15))
        ta.start()
        time.sleep(0.5)
        tb.start()
        time.sleep(0.3)
        tc.start()
        for t in (ta, tb, tc):
            t.join()
        st_a, body_a, dt_a = out["A"]
        st_b, body_b, dt_b = out["B"]
        st_c, body_c, dt_c = out["C"]
        check("queue: A completes", st_a == 200, f"status {st_a}, {dt_a:.2f} s")
        check("queue: B (invalid token) is rejected after A", st_b == 400 and "invalid tokens" in body_b,
              f"status {st_b}, {dt_b:.2f} s, {body_b[:160]}")
        ok_c = st_c == 200 and json.loads(body_c).get("tokens_predicted") == 4
        check("queue: C runs after B's launch failure", ok_c, f"status {st_c}, {dt_c:.2f} s, {body_c[:160]}")
        log = srv.log_text()
        check("queue: B and C were deferred", log.count("defer task") >= 2 or dt_b > 1.0,
              f"B answered after {dt_b:.2f} s (A took {dt_a:.2f} s)")
    finally:
        srv.stop()


def test_ctx_limit_mtp(exe, gguf, port, log_dir):
    n_ctx = 64
    prompt = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda omicron alpha beta gamma delta"
    runs = {}
    for name, mtp, miss in (("mtp0", "0", "0"), ("mtp3", "3", "0"), ("mtp3_miss3", "3", "3")):
        srv = Server(exe, gguf, port, ["-c", str(n_ctx), "-np", "1", "--p100e-mtp", mtp, "--p100e-mtp-tau", "0"],
                     {"P100E_FAKE_DRAFT_MISS": miss}, log_dir, "ctx_" + name)
        try:
            if not srv.wait_ready():
                check(f"ctx: server start ({name})", False, srv.log_text()[-2000:])
                return
            st, body = http(port, "POST", "/completion", {"prompt": prompt, "temperature": 0, "n_predict": -1}, 60)
            if st != 200:
                check(f"ctx: request ({name})", False, f"status {st}: {body[:300]}")
                return
            j = json.loads(body)
            runs[name] = j
            print(f"   {name}: prompt {j.get('tokens_evaluated')}, predicted {j.get('tokens_predicted')}, "
                  f"truncated {j.get('truncated')}, stop_type {j.get('stop_type')}, draft accepted "
                  f"{j.get('timings', {}).get('draft_n_accepted')} / {j.get('timings', {}).get('draft_n')}", flush=True)
        finally:
            srv.stop()
    base = runs["mtp0"]
    n_prompt = base["tokens_evaluated"]
    check("ctx: no MTP stops at the context limit",
          base["truncated"] and base["tokens_predicted"] == n_ctx - n_prompt,
          f"predicted {base['tokens_predicted']}, expected n_ctx - n_prompt = {n_ctx - n_prompt}")
    for name in ("mtp3", "mtp3_miss3"):
        j = runs[name]
        check(f"ctx: {name} predicted == no MTP", j["tokens_predicted"] == base["tokens_predicted"],
              f"{j['tokens_predicted']} vs {base['tokens_predicted']}")
        check(f"ctx: {name} text == no MTP", j["content"] == base["content"])
        check(f"ctx: {name} truncated", bool(j["truncated"]))
        check(f"ctx: {name} drafted", j.get("timings", {}).get("draft_n", 0) > 0,
              f"draft_n {j.get('timings', {}).get('draft_n')}")


def parse_metrics(text):
    m = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        k, _, v = line.partition(" ")
        try:
            m[k] = float(v)
        except ValueError:
            pass
    return m


def test_metrics(exe, gguf, port, log_dir):
    srv = Server(exe, gguf, port, ["-c", "4096", "-np", "1", "-ub", "16", "-b", "16", "--metrics", "--p100e-mtp", "0"],
                 {}, log_dir, "metrics")
    try:
        if not srv.wait_ready():
            check("metrics: server start", False, srv.log_text()[-2000:])
            return
        prompt = " ".join(["alpha beta gamma delta"] * 10)   # 40 tokens: 3 prompt chunks of 16
        st, body = http(port, "POST", "/completion", {"prompt": prompt, "temperature": 0, "n_predict": 8}, 60)
        if st != 200:
            check("metrics: request", False, f"status {st}: {body[:300]}")
            return
        j = json.loads(body)
        n_prompt, n_gen = j["tokens_evaluated"], j["tokens_predicted"]
        st, text = http(port, "GET", "/metrics", timeout=10)
        if st != 200:
            check("metrics: /metrics", False, f"status {st}: {text[:300]}")
            return
        m = parse_metrics(text)
        n_chunks = -(-n_prompt // 16)
        # MTP off: the first token comes from the last prompt chunk, every other token from one decode step
        exp_decode = n_chunks + (n_gen - 1)
        check("metrics: n_decode_total counts prompt + decode forwards", m.get("llamacpp:n_decode_total") == exp_decode,
              f"{m.get('llamacpp:n_decode_total')} vs {n_chunks} prompt chunks + {n_gen - 1} decode steps")
        check("metrics: n_busy_slots_per_decode == 1", m.get("llamacpp:n_busy_slots_per_decode") == 1.0,
              f"{m.get('llamacpp:n_busy_slots_per_decode')}")
        # the slot's tokens at the last forward: prompt + every emitted token but the last (never fed)
        check("metrics: n_tokens_max", m.get("llamacpp:n_tokens_max") == n_prompt + n_gen - 1,
              f"{m.get('llamacpp:n_tokens_max')} vs {n_prompt + n_gen - 1}")
    finally:
        srv.stop()


def test_engine_fatal(exe, gguf, port, log_dir):
    srv = Server(exe, gguf, port, ["-c", "4096", "-np", "1", "--p100e-mtp", "0"], {"P100E_FAKE_FATAL_AT": "3"},
                 log_dir, "fatal")
    try:
        if not srv.wait_ready():
            check("fatal: server start", False, srv.log_text()[-2000:])
            return
        st, body = http(port, "POST", "/completion", {"prompt": "alpha", "temperature": 0, "n_predict": 16}, 30)
        try:
            rc = srv.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            rc = None
        check("fatal: the request does not complete", st != 200, f"status {st}, {str(body)[:160]}")
        check("fatal: the server exits with code 3", rc == 3, f"exit code {rc}")
        log = srv.log_text()
        check("fatal: the log names the fatal engine error", "FATAL engine error" in log and "P100E_FAKE_FATAL_AT" in log,
              log[-400:].replace(chr(10), " | "))
    finally:
        srv.stop()


def test_concurrent_cache(exe, gguf, port, log_dir):
    """3 sequences at once (P100E_FAKE_SEQS=3, -np 3, MTP 3 so the 8-row step budget trims drafts), each output
    equal to the same request alone; a repeated prompt reuses the cache (usage cached_tokens) with the same output"""
    srv = Server(exe, gguf, port, ["-c", "12288", "-np", "3", "--p100e-mtp", "3", "--p100e-mtp-tau", "0", "--metrics"],
                 {"P100E_FAKE_DELAY_MS": "20", "P100E_FAKE_SEQS": "3"}, log_dir, "concurrent")
    try:
        if not srv.wait_ready():
            check("concurrent: server start", False, srv.log_text()[-2000:])
            return
        prompts = ["alpha beta", "gamma delta epsilon", "zeta eta theta iota"]
        body = lambda p: {"messages": [{"role": "user", "content": p}], "max_tokens": 40, "temperature": 0}
        alone = {}
        for p in prompts:
            st, b = http(port, "POST", "/v1/chat/completions", body(p), timeout=60)
            alone[p] = json.loads(b)["choices"][0]["message"]["content"] if st == 200 else None
        out = {}

        def run(p):
            st, b = http(port, "POST", "/v1/chat/completions", body(p), timeout=60)
            out[p] = (st, json.loads(b) if st == 200 else b)

        th = [threading.Thread(target=run, args=(p,)) for p in prompts]
        t0 = time.time()
        for t in th:
            t.start()
        for t in th:
            t.join()
        dt = time.time() - t0
        same = all(out[p][0] == 200 and out[p][1]["choices"][0]["message"]["content"] == alone[p] for p in prompts)
        check("concurrent: 3 requests at once == each alone", same, f"{dt:.2f} s")
        cached = [out[p][1]["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0) for p in prompts
                  if out[p][0] == 200]
        check("concurrent: repeated prompts reuse the cache", len(cached) == 3 and all(c > 0 for c in cached), str(cached))
        st, mt = http(port, "GET", "/metrics")
        hits = parse_metrics(mt).get("p100e:prompt_cache_hits_total", 0) if st == 200 else -1
        check("concurrent: /metrics prompt cache hits", hits >= 3, f"hits {hits}")
    finally:
        srv.stop()


def test_decode_share(exe, gguf, port, log_dir):
    """latency-fixes: while slot B prefills a long prompt (many ubatch chunks), slot A keeps decoding. A fake prompt
    chunk costs 100 ms and a decode step 5 ms (P100 ratio ~4.5 s / 25 ms, scaled down; Windows sleeps round 5 ms up
    to ~15 ms), so decode share 0 (the old loop: one step per chunk) gives A one token per chunk; share 0.5 about 7.
    A's overall rate also counts its steps after B's prefill (fast in both runs), so the check asks for > 1.5x.
    Checks: A's rate rises, B's prompt takes <= ~2.2x longer, both outputs equal the same requests alone."""
    body_a = {"prompt": "alpha", "n_predict": 60, "temperature": 0, "cache_prompt": False, "id_slot": 1}
    body_b = {"prompt": "beta " * 1500, "n_predict": 4, "temperature": 0, "cache_prompt": False, "id_slot": 0}
    res = {}
    for share in ("0", "0.5"):
        srv = Server(exe, gguf, port, ["-c", "8192", "-np", "2", "-ub", "64", "-b", "64", "--p100e-mtp", "0"],
                     {"P100E_FAKE_DELAY_MS": "5", "P100E_FAKE_PREFILL_DELAY_MS": "100", "P100E_FAKE_SEQS": "2",
                      "P100E_DECODE_SHARE": share}, log_dir,
                     f"decode_share_{share}")
        try:
            if not srv.wait_ready():
                check(f"decode share {share}: server start", False, srv.log_text()[-2000:])
                return
            alone = {}
            for k, b in (("A", body_a), ("B", body_b)):
                st, raw = http(port, "POST", "/completion", b, timeout=120)
                alone[k] = json.loads(raw)["content"] if st == 200 else None
            out = {}

            def run(k, b):
                st, raw = http(port, "POST", "/completion", b, timeout=120)
                out[k] = (st, json.loads(raw) if st == 200 else raw, time.time())

            tb = threading.Thread(target=run, args=("B", body_b))
            tb.start()
            time.sleep(0.25)   # B is prefilling (~24 chunks of 64 rows)
            ta = threading.Thread(target=run, args=("A", body_a))
            ta.start()
            tb.join()
            ta.join()
            ok = all(out[k][0] == 200 and out[k][1]["content"] == alone[k] for k in ("A", "B"))
            check(f"decode share {share}: A and B == each alone", ok)
            # A's decode rate while B prefilled: tokens of A over B's prompt window (B's prompt_ms from its start)
            pb = out["B"][1]["timings"]["prompt_ms"] / 1000.0 if ok else 0
            na = out["A"][1]["timings"]["predicted_n"] if ok else 0
            ta_s = out["A"][1]["timings"]["predicted_ms"] / 1000.0 if ok else 1
            res[share] = (pb, na / ta_s if ta_s else 0)
            pa = out["A"][1]["timings"]["prompt_ms"] / 1000.0 if ok else -1
            print(f"   share {share}: B prompt {pb:.2f} s, A prompt {pa:.2f} s, A overall {res[share][1]:.1f} t/s",
                  flush=True)
            # shortest prompt first: A's one-chunk prompt is not queued behind B's whole prefill
            check(f"decode share {share}: A's short prompt is served during B's prefill", 0 <= pa < 0.6 * pb,
                  f"A prompt {pa:.2f} s, B prompt {pb:.2f} s")
        finally:
            srv.stop()
    if len(res) == 2:
        check("decode share: A decodes > 1.5x faster during a long prefill with share 0.5",
              res["0.5"][1] > 1.5 * res["0"][1], f"{res['0'][1]:.1f} -> {res['0.5'][1]:.1f} t/s")
        check("decode share: B's prefill takes <= 2.2x longer (share 0.5)", res["0.5"][0] < 2.2 * max(res["0"][0], 1e-3),
              f"{res['0'][0]:.2f} -> {res['0.5'][0]:.2f} s")


def test_live_status(exe, gguf, port, log_dir):
    """live status: /slots, /metrics, /props, /health answer from the task loop's snapshot while the engine is inside a
    slow step (P100E_FAKE_DELAY_MS=400 per prefill chunk / decode step): every reply well under one step, prompt rows
    rising chunk by chunk with the whole prompt size known from the start, decoded tokens rising, and an idle slot
    keeps its last task's numbers"""
    delay = 400
    srv = Server(exe, gguf, port, ["-c", "4096", "-np", "2", "-ub", "16", "-b", "16", "--metrics", "--p100e-mtp", "0"],
                 {"P100E_FAKE_DELAY_MS": str(delay)}, log_dir, "live")
    try:
        if not srv.wait_ready():
            check("live: server start", False, srv.log_text()[-2000:])
            return
        prompt = " ".join(["alpha beta gamma delta"] * 20)   # ~80 tokens: 5+ prompt chunks of 16
        n_predict = 8
        out = {}

        def run():
            out["r"] = http(port, "POST", "/completion", {"prompt": prompt, "temperature": 0, "n_predict": n_predict}, 120)

        th = threading.Thread(target=run)
        th.start()
        polls = []   # (dt slots, state, n_prompt_tokens, processed, decoded)
        t_other = []
        t_end = time.time() + 30
        while th.is_alive() and time.time() < t_end:
            t0 = time.time()
            st, b = http(port, "GET", "/slots", timeout=10)
            dt = time.time() - t0
            if st == 200:   # the busy slot (LRU picks any), else slot 0
                sl = json.loads(b)
                s0 = next((x for x in sl if x.get("is_processing")), sl[0])
                polls.append((dt, s0.get("state"), s0.get("n_prompt_tokens"), s0.get("n_prompt_tokens_processed"),
                              s0["next_token"][0]["n_decoded"] if s0.get("next_token") else None))
            for path in ("/metrics", "/props", "/health"):
                t1 = time.time()
                st2, b2 = http(port, "GET", path, timeout=10)
                t_other.append((path, time.time() - t1, st2, b2))
            time.sleep(0.1)
        th.join()
        st, body = out.get("r", (None, ""))
        check("live: request completes", st == 200, f"status {st}")
        if st != 200:
            return
        j = json.loads(body)
        n_prompt = j["tokens_evaluated"]
        busy = [p for p in polls if p[1] in ("processing_prompt", "generating", "started")]
        worst = max((p[0] for p in polls), default=99)
        worst_other = max((t[1] for t in t_other), default=99)
        print(f"   {len(polls)} /slots polls during the request ({len(busy)} busy), slowest {worst * 1e3:.1f} ms; "
              f"/metrics /props /health slowest {worst_other * 1e3:.1f} ms (engine step {delay} ms)", flush=True)
        check("live: /slots answers fast mid-step", len(busy) >= 10 and worst < 0.1, f"slowest {worst * 1e3:.1f} ms")
        check("live: /metrics /props /health answer fast mid-step",
              worst_other < 0.1 and all(t[2] == 200 for t in t_other), f"slowest {worst_other * 1e3:.1f} ms")
        pp = [p for p in busy if p[1] == "processing_prompt"]
        procs = sorted(set(p[3] for p in pp))
        check("live: whole prompt size known during prefill", pp and all(p[2] == n_prompt for p in pp),
              f"n_prompt_tokens {sorted(set(p[2] for p in pp))} vs {n_prompt}")
        check("live: n_prompt_tokens_processed rises per chunk", len(procs) >= 3 and all(v % 16 == 0 for v in procs[:-1]),
              f"seen {procs}")
        decs = sorted(set(p[4] for p in busy if p[1] == "generating"))
        check("live: n_decoded rises during decode", len(decs) >= 3, f"seen {decs}")
        m_busy = [t for t in t_other if t[0] == "/metrics" and parse_metrics(t[3]).get("p100e:slots_busy") == 1.0
                  and ('p100e:slot_busy{slot="0"} 1' in t[3] or 'p100e:slot_busy{slot="1"} 1' in t[3])]
        check("live: /metrics slot_busy gauge while busy", len(m_busy) > 0)
        m_pp = [parse_metrics(t[3]).get("p100e:prompt_tokens_processing", 0) for t in t_other if t[0] == "/metrics"]
        check("live: /metrics prompt_tokens_processing rises", len(set(v for v in m_pp if v > 0)) >= 2, str(sorted(set(m_pp))))
        props = [json.loads(t[3]) for t in t_other if t[0] == "/props"]
        bi = props[0].get("build_info", "") if props else ""
        check("live: /props build_info names the p100e commit", bi.startswith("p100e-") and "fork" in bi, bi)
        st, b = http(port, "GET", "/slots")
        sl = json.loads(b)
        used = [x for x in sl if x.get("id_task") is not None]
        s0 = used[0] if len(used) == 1 else sl[0]
        nt = s0["next_token"][0]
        idle_ok = (not s0["is_processing"] and s0["n_prompt_tokens"] == n_prompt and nt["n_decoded"] == n_predict
                   and not nt["has_next_token"] and s0["n_prompt_tokens_processed"] == n_prompt and s0["p100e"]["t_end_ms"] > 0)
        check("live: idle slot keeps its last task", idle_ok, json.dumps({k: s0.get(k) for k in
              ("is_processing", "n_prompt_tokens", "n_prompt_tokens_processed", "next_token", "p100e")}))
        s1 = [x for x in sl if x is not s0][0]
        check("live: never-used slot has next_token", len(used) == 1 and s1.get("next_token") and s1["next_token"][0]["n_decoded"] == 0)
    finally:
        srv.stop()


def main():
    ap = argparse.ArgumentParser(description="p100e-server slot-loop tests on the CPU fake engine")
    ap.add_argument("exe", help="build/p100e-server-fake.exe")
    ap.add_argument("gguf", help="a qwen35 GGUF (only its tokenizer is read)")
    ap.add_argument("--port", type=int, default=18190)
    ap.add_argument("--keep-logs", help="directory for the server logs (default: a temp dir, removed on success)")
    a = ap.parse_args()
    tmp = None
    log_dir = a.keep_logs
    if log_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="p100e_fake_")
        log_dir = tmp.name
    os.makedirs(log_dir, exist_ok=True)

    test_queue_after_launch_failure(a.exe, a.gguf, a.port, log_dir)
    test_ctx_limit_mtp(a.exe, a.gguf, a.port, log_dir)
    test_metrics(a.exe, a.gguf, a.port, log_dir)
    test_engine_fatal(a.exe, a.gguf, a.port, log_dir)
    test_concurrent_cache(a.exe, a.gguf, a.port, log_dir)
    test_decode_share(a.exe, a.gguf, a.port, log_dir)
    test_live_status(a.exe, a.gguf, a.port, log_dir)

    n_ok = sum(ok for _, ok in results)
    print(f"FAKE_TEST_SUMMARY {n_ok}/{len(results)} passed (logs: {log_dir})", flush=True)
    if tmp is not None and n_ok == len(results):
        tmp.cleanup()
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
