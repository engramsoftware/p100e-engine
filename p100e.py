#!/usr/bin/env python3
"""Easy front end for p100e. Works on Windows and Linux with no extra packages.

    python p100e.py check                 is everything installed?
    python p100e.py setup                 build the server
    python p100e.py setup --vision        build it with image input
    python p100e.py start model.gguf      start the server

Run any of them with --help for the options.
"""
import argparse
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WINDOWS = os.name == "nt"
EXE = ".exe" if WINDOWS else ""
BUILD = ROOT / "build"
FORK = ROOT / "third_party" / "llama.cpp-fork"
FORK_BUILD = FORK / "build"
KEY_FILE = ROOT / "keys.txt"
MIN_CMAKE = (3, 24)
EXIT_ENGINE_FATAL = 3   # p100e-server exits with this on a fatal GPU error (docs/FLAGS.md)
MAX_GPUS = 4
DEFAULT_CONTEXT = 131072
ONE_GPU_CONTEXT = 32768   # fits next to a 13 GB model on one 16 GB P100 with q4_0 KV and a small ubatch
ONE_GPU_UBATCH = 32


def run_text(cmd, env=None):
    """Output of a command, or None if it cannot run."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=env)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout + r.stderr if r.returncode == 0 else None


def find_vcvars():
    """Path of vcvars64.bat, from any Visual Studio edition."""
    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(pf86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    out = run_text([str(vswhere), "-latest", "-products", "*", "-property", "installationPath"])
    roots = [Path(line.strip()) for line in (out or "").splitlines() if line.strip()]
    roots.append(Path(pf86) / "Microsoft Visual Studio" / "2022" / "BuildTools")
    for root in roots:
        path = root / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        if path.exists():
            return path
    return None


def find_cuda():
    """Folder of CUDA Toolkit 12.9."""
    if WINDOWS:
        base = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        places = [base / "NVIDIA GPU Computing Toolkit" / "CUDA" / "v12.9"]
    else:
        places = [Path(p) for p in (os.environ.get("CUDA_HOME"), "/usr/local/cuda-12.9", "/usr/local/cuda") if p]
    for place in places:
        nvcc = place / "bin" / f"nvcc{EXE}"
        if nvcc.exists() and "12.9" in (run_text([str(nvcc), "--version"]) or ""):
            return place
    return None


def find_gpus():
    """GPUs as (index, name, memory in MiB). The index is in PCI order."""
    out = run_text(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"])
    gpus = []
    for line in (out or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[0].isdigit() and parts[2].isdigit():
            gpus.append((int(parts[0]), parts[1], int(parts[2])))
    return gpus


def pick_gpus(gpus):
    """Prefer the P100s; otherwise use every GPU. At most MAX_GPUS."""
    p100s = [g for g in gpus if "P100" in g[1]]
    return (p100s or gpus)[:MAX_GPUS]


def cmake_ok():
    out = run_text(["cmake", "--version"])
    if not out:
        return False
    numbers = out.split()[2].split(".")
    return (int(numbers[0]), int(numbers[1])) >= MIN_CMAKE


def ninja_ok():
    """On PATH, or the copy that comes with Visual Studio (build.cmd loads its environment)."""
    if shutil.which("ninja"):
        return True
    vcvars = find_vcvars() if WINDOWS else None
    if not vcvars:
        return False
    vs_root = vcvars.parents[3]
    return (vs_root / "Common7" / "IDE" / "CommonExtensions" / "Microsoft" / "CMake" / "Ninja" / "ninja.exe").exists()


def checks():
    """(name, ok, how to fix it)."""
    linux_apt = "sudo apt install build-essential cmake ninja-build git python3"
    items = [
        ("CMake 3.24 or newer", cmake_ok(), "Install from https://cmake.org/download/" if WINDOWS else linux_apt),
        ("Ninja", ninja_ok(), "Install from https://ninja-build.org/" if WINDOWS else linux_apt),
        ("git", bool(shutil.which("git")), "Install from https://git-scm.com/" if WINDOWS else linux_apt),
        ("CUDA Toolkit 12.9", find_cuda() is not None,
         "Install version 12.9 from NVIDIA's CUDA archive. Version 13 does not support the P100."),
        ("NVIDIA driver", bool(find_gpus()), "Install the NVIDIA driver, then check that nvidia-smi works."),
    ]
    if WINDOWS:
        items.insert(0, ("Visual Studio C++ tools", find_vcvars() is not None,
                         "Install Visual Studio 2022 Build Tools with the C++ workload."))
    else:
        items.insert(0, ("GCC", bool(shutil.which("g++")), linux_apt))
    return items


def cmd_check(_args):
    missing = 0
    for name, ok, fix in checks():
        print(f"  {'OK     ' if ok else 'MISSING'}  {name}")
        if not ok:
            print(f"           Fix: {fix}")
            missing += 1
    gpus = find_gpus()
    for index, name, mib in gpus:
        print(f"  GPU {index}    {name}, {mib // 1024} GB")
    if missing:
        print(f"\n{missing} item(s) missing. Fix them, then run this again.")
        return 1
    print("\nEverything is ready. Next: python p100e.py setup")
    return 0


def build_env():
    env = dict(os.environ)
    cuda = find_cuda()
    if WINDOWS:
        env["P100E_VCVARS"] = str(find_vcvars())
        env["P100E_CUDA_PATH"] = str(cuda)
    else:
        env["CUDA_HOME"] = str(cuda)
    return env


def build_fork(args, env):
    """The shared-library build of the llama.cpp fork that image input needs."""
    done = FORK_BUILD / "tools" / "mtmd" / "mtmd.lib" if WINDOWS else FORK_BUILD / "bin" / "libmtmd.so"
    if done.exists():
        print("Image input libraries are already built.")
        return 0
    print("Building image input (this takes a while)...")
    configure = [
        "cmake", "-S", str(FORK), "-B", str(FORK_BUILD), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=ON", "-DGGML_CUDA=ON", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_TOOLS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_BUILD_APP=OFF", "-DLLAMA_BUILD_MTMD=ON",
    ]
    build = ["cmake", "--build", str(FORK_BUILD), "-j", str(args.jobs)]
    if not WINDOWS:
        env = dict(env, PATH=f"{env['CUDA_HOME']}/bin:{env['PATH']}")
        rc = subprocess.call(configure, env=env)
        return rc or subprocess.call(build, env=env)
    configure.append("-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler")
    script = ["@echo off", f'call "{env["P100E_VCVARS"]}" >nul || exit /b 1',
              f'set "PATH={env["P100E_CUDA_PATH"]}\\bin;%PATH%"',
              subprocess.list2cmdline(configure) + " || exit /b 1", subprocess.list2cmdline(build)]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fork.cmd"
        path.write_text("\r\n".join(script) + "\r\n", encoding="ascii")
        return subprocess.call(["cmd", "/c", str(path)], env=env)


def cmd_setup(args):
    if any(not ok for _, ok, _ in checks()):
        print("Some tools are missing:\n")
        cmd_check(args)
        return 1
    env = build_env()
    env["JOBS"] = str(args.jobs)
    if args.vision:
        rc = build_fork(args, env)
        if rc:
            print("\nThe image input build failed. See the first error above.")
            return rc
    print(f"Building p100e for {args.arch} with {args.jobs} jobs (this takes a while)...")
    script = ["cmd", "/c", str(ROOT / "build.cmd"), args.arch] if WINDOWS else ["bash", str(ROOT / "build.sh"), args.arch]
    rc = subprocess.call(script, env=env, cwd=ROOT)
    if rc:
        print("\nThe build failed. See the first error above. If memory ran out, try --jobs 2.")
        return rc
    print("\nDone. Next: python p100e.py start path/to/model.gguf")
    return 0


def key_file():
    """keys.txt, made with a random key the first time."""
    if not KEY_FILE.exists():
        KEY_FILE.write_text(secrets.token_urlsafe(32) + "\n", encoding="ascii")
        if not WINDOWS:
            KEY_FILE.chmod(0o600)
        print(f"Made a new API key in {KEY_FILE}. Clients need the key from that file.")
    return KEY_FILE


def server_command(args, gpus):
    exe = Path(args.exe) if args.exe else BUILD / f"p100e-server{EXE}"
    one_gpu = len(gpus) == 1
    context = args.context or (ONE_GPU_CONTEXT if one_gpu else DEFAULT_CONTEXT)
    cmd = [str(exe), "-m", args.model, "--alias", args.name or Path(args.model).stem,
           "--host", "0.0.0.0" if args.share else "127.0.0.1", "--port", str(args.port),
           "-c", str(context), "-np", str(args.chats), "-ub", str(ONE_GPU_UBATCH if one_gpu else 2048), "--jinja"]
    cmd += ["--p100e-devices", ",".join(str(g) for g in gpus)]
    if args.mmproj:
        cmd += ["--mmproj", args.mmproj]
    if args.no_mtp or one_gpu:   # MTP needs 2 to 4 GPUs (src/tp_model.cu)
        cmd += ["--p100e-mtp", "0"]
    if args.small_cache or one_gpu:
        cmd += ["--p100e-kv", "q4_0"]
    if one_gpu:   # without a set pool size, the vision reserve leaves no room for the KV pool on one card
        cmd += ["--p100e-pool-tokens", str(context)]
    if args.share:
        cmd += ["--api-key-file", str(KEY_FILE if args.dry_run else key_file()), "--p100e-no-remote-images"]
    return cmd + args.extra


def cmd_start(args):
    if not Path(args.model).is_file():
        print(f"Model file not found: {args.model}")
        return 1
    if args.gpus:
        chosen = [int(g) for g in args.gpus.split(",")]
    else:
        found = find_gpus()
        if not found:
            print("No GPUs found. Check that nvidia-smi works, or choose them with --gpus 0,1")
            return 1
        picked = pick_gpus(found)
        chosen = [g[0] for g in picked]
        for index, name, mib in picked:
            print(f"Using GPU {index}: {name}, {mib // 1024} GB")
        if len({g[2] for g in picked}) > 1:
            print("These GPUs differ in size. If loading fails, pick with --gpus, or add --p100e-split after --.")
    cmd = server_command(args, chosen)
    # nvidia-smi counts in PCI order and CUDA by default does not, so make both agree
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID")
    if args.dry_run:
        print(subprocess.list2cmdline(cmd))
        return 0
    if not Path(cmd[0]).is_file():
        print(f"{cmd[0]} is not built yet. Run: python p100e.py setup")
        return 1
    where = "this machine's address" if args.share else "127.0.0.1"
    print(f"\nStarting. When the log says 'model loaded', connect to http://{where}:{args.port}/v1\n", flush=True)
    while True:
        rc = subprocess.call(cmd, env=env)
        if rc != EXIT_ENGINE_FATAL or not args.keep_running:
            return rc
        print("The server stopped on a GPU error. Starting it again in 5 seconds...", flush=True)
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description="Build and run p100e.")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("check", help="see if everything is installed").set_defaults(run=cmd_check)

    setup = commands.add_parser("setup", help="build the server")
    setup.add_argument("--vision", action="store_true", help="also build image input")
    setup.add_argument("--arch", default="60", help="GPU types: 60 = P100, 75 = GTX 1660, 86 = RTX 3070 (default 60)")
    setup.add_argument("--jobs", type=int, default=8, help="parallel compile jobs; use 2 if memory runs out")
    setup.set_defaults(run=cmd_setup)

    start = commands.add_parser("start", help="start the server",
                                epilog="Anything after -- goes straight to p100e-server.")
    start.add_argument("model", help="the model's .gguf file")
    start.add_argument("--mmproj", help="vision .gguf file, for image input")
    start.add_argument("--gpus", help="GPU numbers such as 0,1 (default: find them)")
    start.add_argument("--port", type=int, default=8080)
    start.add_argument("--context", type=int, help="total context in tokens (default 131072, or 32768 on one GPU)")
    start.add_argument("--chats", type=int, default=1, help="chats at once, sharing the context (default 1)")
    start.add_argument("--name", help="model name apps will see (default: the file's name)")
    start.add_argument("--small-cache", action="store_true", help="use less GPU memory per token")
    start.add_argument("--no-mtp", action="store_true", help="for models without the MTP block")
    start.add_argument("--share", action="store_true", help="allow other machines, protected by a key in keys.txt")
    start.add_argument("--keep-running", action="store_true", help="restart after a fatal GPU error")
    start.add_argument("--dry-run", action="store_true", help="print the command and stop")
    start.add_argument("--exe", help=argparse.SUPPRESS)
    start.set_defaults(run=cmd_start)

    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        split = argv.index("--")
        argv, extra = argv[:split], argv[split + 1:]
    args = parser.parse_args(argv)
    args.extra = extra
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
