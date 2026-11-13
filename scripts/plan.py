#!/usr/bin/env python3
"""Decide what this workflow run computes: job folder, shard count, release tag."""
import datetime, json, os, re, subprocess, sys

def out(**kw):
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        for k, v in kw.items():
            f.write(f"{k}={v}\n")

job = shards = tag = ""
if os.environ.get("EVENT") == "workflow_dispatch":
    job, shards, tag = os.environ.get("IN_JOB", ""), os.environ.get("IN_SHARDS", ""), os.environ.get("IN_TAG", "")
else:
    # the request file added or changed by this push
    try:
        changed = subprocess.run(["git", "diff", "--name-only", "--diff-filter=AM", "HEAD~1", "HEAD", "--", "requests/"],
                                 capture_output=True, text=True, check=True).stdout.split()
    except subprocess.CalledProcessError:
        changed = subprocess.run(["git", "ls-files", "requests/"], capture_output=True, text=True).stdout.split()
    changed = [c for c in changed if c.endswith(".env")]
    if not changed:
        print("no request file in this push; nothing to do")
        out(run="false", job="", shards="1", tag="", matrix="[0]")
        sys.exit(0)
    path = sorted(changed)[-1]
    kv = {}
    for line in open(path):
        m = re.match(r"\s*([A-Z_]+)\s*=\s*(.*?)\s*$", line)
        if m:
            kv[m.group(1)] = m.group(2).strip("\"'")
    job, shards = kv.get("JOB", ""), kv.get("SHARDS", "")
    tag = kv.get("TAG", "") or os.path.basename(path)[:-4]

if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", job) or not os.path.isfile(f"jobs/{job}/run.sh"):
    sys.exit(f"unknown job {job!r}: jobs/<job>/run.sh must exist")
n = int(shards or 20)
if not 1 <= n <= 20:
    sys.exit("SHARDS must be 1-20")
if not tag:
    tag = f"{job}-{datetime.datetime.utcnow():%Y%m%d-%H%M%S}"
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tag):
    sys.exit(f"bad tag {tag!r}")
print(f"job={job} shards={n} tag={tag}")
out(run="true", job=job, shards=n, tag=tag, matrix=json.dumps(list(range(n))))
