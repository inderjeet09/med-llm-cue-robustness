"""Generate cloud/kaggle_batch.ipynb: runs a job queue (configs/queues/<name>.txt) on all Kaggle GPUs, one job per
GPU at a time, packing results after every finished job; skips jobs whose done-marker exists; retries a failure once.

Adapted from ~/Desktop/ecg-research/scripts/make_kaggle_batch_notebook.py (same author). Changes: queue lines are
generic commands with a done-marker; HF_TOKEN from Kaggle Secrets; pinned GPU libraries (configs/requirements-kaggle.txt); HF cache outside the
results zip; MedQA files placed from the flat dataset upload; provenance fallback env CODE_COMMIT.
Usage: python scripts/make_kaggle_batch_notebook.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
M = lambda s: nbf.v4.new_markdown_cell(s.strip())  # noqa: E731
C = lambda s: nbf.v4.new_code_cell(s.strip())  # noqa: E731

nb.cells = [
M("""
# medllm-robustness — Kaggle batch runner
Runs `configs/queues/<QUEUE>.txt` from the uploaded code bundle, **one job per GPU in parallel** (T4 ×2 → 2 jobs).
After every finished job it rewrites `/kaggle/working/<QUEUE>_results.zip`. Finished jobs are skipped on re-run.

**Settings:** Accelerator **GPU T4 ×2**, Internet **on**, Persistence off, environment "latest".
**Add-ons → Secrets:** `HF_TOKEN` (attached to this notebook). Input = your dataset (newest version).
Run with **Save Version → Save & Run All (Commit)**, then download the zip from the version's **Output** tab.
"""),
C("""
QUEUE = "quick_check"  # <- the only thing to change between batches (see configs/queues/)
MAX_HOURS = 11.0          # stop launching new jobs after this (sessions end at ~12 h)
"""),
C("""
import json, os, shutil, subprocess, sys, tarfile, time, zipfile
from pathlib import Path
import torch
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout)
N_GPU = torch.cuda.device_count(); assert N_GPU > 0, "Enable a GPU accelerator in Settings"
INP = Path("/kaggle/input")
WORK = Path("/kaggle/working/medllm")
commits = {c.read_text().strip() for c in INP.rglob("COMMIT.txt")}
assert len(commits) == 1, f"expected exactly one COMMIT.txt in the input, found {commits or 'none'} (is the dataset added?)"
COMMIT = commits.pop()
if not (WORK / "src").exists():
    WORK.mkdir(parents=True, exist_ok=True)
    # Use ONLY the code bundle that matches COMMIT.txt (a stale code_*.tar.gz left in the dataset must never run).
    tars = sorted(INP.rglob(f"code_{COMMIT}.tar.gz"))
    # Kaggle sometimes auto-extracts uploaded archives: then use the unpacked tree of the same commit.
    trees = sorted(p.parent.parent for p in INP.rglob("configs/queues")
                   if (p.parent.parent / "src").is_dir() and COMMIT in str(p.parent.parent))
    if tars:
        with tarfile.open(tars[0]) as t: t.extractall(WORK)
    elif trees:
        for d in ["src", "scripts", "configs", "tests", "prompts"]:
            shutil.copytree(trees[0] / d, WORK / d)
    else:
        print("Contents of /kaggle/input:", *sorted(str(p) for p in INP.rglob("*"))[:50], sep="\\n  ")
        raise FileNotFoundError(f"No code_{COMMIT}.tar.gz (or unpacked code_{COMMIT}) in the input: upload the code "
                                "bundle that matches COMMIT.txt")
med = WORK / "data/raw/medqa_us4"; med.mkdir(parents=True, exist_ok=True)
for f in ["train.jsonl", "dev.jsonl", "test.jsonl"]:
    src4 = next(INP.rglob(f), None)
    assert src4 or f == "test.jsonl", f"{f} missing from the input dataset"
    if src4: shutil.copy(src4, med / f)
if not (med / "test.jsonl").exists():
    print("NOTE: no test.jsonl uploaded; jobs that evaluate on the test split will stop immediately")
shutil.copy(next(INP.rglob("medqa_PREPARED.json")), med / "PREPARED.json")
hn = WORK / "data/raw/handler_nota"; hn.mkdir(parents=True, exist_ok=True)
shutil.copy(next(INP.rglob("handler_questions.csv")), hn / "questions.csv")
m5 = WORK / "data/raw/medqa_us5"; m5.mkdir(parents=True, exist_ok=True)   # 5-option (reproduction), if uploaded
for f in ["train", "dev"]:
    src5 = next(INP.rglob(f"medqa5_{f}.jsonl"), None)
    if src5: shutil.copy(src5, m5 / f"{f}.jsonl")
mtr = next(INP.rglob("mtr_items.jsonl"), None)                            # transfer test data, if uploaded
if mtr: (WORK / "data/raw/multiturn").mkdir(parents=True, exist_ok=True); shutil.copy(mtr, WORK / "data/raw/multiturn/items.jsonl")
os.environ["CODE_COMMIT"] = COMMIT
os.environ["HF_HOME"] = "/kaggle/tmp/hf"            # model downloads: NOT in /kaggle/working (kept out of the zip)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # less fragmentation on long prompts
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from Kaggle Secrets")
except Exception as e:
    print("WARNING: no HF_TOKEN secret (gated models such as Llama will fail):", type(e).__name__)
os.chdir(WORK)
# pinned GPU libraries (configs/requirements-kaggle.txt); each run also records them in summary.json -> provenance
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "configs/requirements-kaggle.txt"], check=True)
from importlib.metadata import version
print({p: version(p) for p in ["torch", "transformers", "peft", "accelerate", "bitsandbytes"]})
print("GPUs:", N_GPU, "| code commit:", COMMIT)
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:warnings", "tests"], capture_output=True, text=True)
print(r.stdout[-800:]); assert r.returncode == 0, "unit tests failed — do not run the batch"
"""),
C("""
jobs = []
for line in (WORK / f"configs/queues/{QUEUE}.txt").read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#"): continue
    marker, *cmd = line.split()
    jobs.append({"marker": marker, "cmd": cmd, "run": Path(marker).parent.as_posix().replace("/", "__")})
done = lambda j: (WORK / j["marker"]).exists()
todo = [j for j in jobs if not done(j)]
print(f"{len(jobs)} jobs in {QUEUE}, {len(jobs) - len(todo)} already done, {len(todo)} to run")
"""),
C("""
def pack():
    with zipfile.ZipFile(f"/kaggle/working/{QUEUE}_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for root in ["results", "experiments"]:
            for f in Path(root).rglob("*"):
                if f.is_file() and f.suffix not in (".pt", ".safetensors", ".bin"): z.write(f)
        z.write(f"configs/queues/{QUEUE}.txt")
        if Path("job_status.json").exists(): z.write("job_status.json")

t_start = time.time(); status = {}; running = {}; queue = list(todo); attempts = {}
def launch(job, gpu):
    attempts[job["run"]] = attempts.get(job["run"], 0) + 1
    log_dir = Path("experiments/_stdout"); log_dir.mkdir(parents=True, exist_ok=True)
    f = open(log_dir / f"{job['run']}_attempt{attempts[job['run']]}.txt", "w")
    cmd = [sys.executable if c == "python" else c for c in job["cmd"]]
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
    running[gpu] = (p, job, f, time.time()); print(time.strftime("%H:%M:%S"), f"GPU{gpu} start", job["run"], flush=True)

while queue or running:
    for gpu in range(N_GPU):
        if gpu not in running and queue and (time.time() - t_start) / 3600 < MAX_HOURS:
            launch(queue.pop(0), gpu)
    time.sleep(20)
    for gpu, (p, job, f, t0) in list(running.items()):
        if p.poll() is None: continue
        f.close(); del running[gpu]
        ok = p.returncode == 0 and done(job)
        status[job["run"]] = {"ok": ok, "returncode": p.returncode, "minutes": round((time.time() - t0) / 60, 1),
                              "attempts": attempts[job["run"]], "gpu": gpu}
        print(time.strftime("%H:%M:%S"), f"GPU{gpu} {'done' if ok else 'FAILED'}", job["run"], status[job["run"]], flush=True)
        if not ok and attempts[job["run"]] < 2: queue.append(job)          # retry once
        Path("job_status.json").write_text(json.dumps({"queue": QUEUE, "commit": COMMIT, "status": status,
                                                       "gpu_count": N_GPU}, indent=1))
        pack()
    if not running and queue and (time.time() - t_start) / 3600 >= MAX_HOURS:
        print("time limit reached; not launching", len(queue), "remaining jobs"); break
pack()
print("finished in", round((time.time() - t_start) / 3600, 2), "h ->", f"/kaggle/working/{QUEUE}_results.zip")
print(json.dumps(status, indent=1))
"""),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = ROOT / "cloud/kaggle_batch.ipynb"
nbf.write(nb, out)
print("written", out)
