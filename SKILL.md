---
name: action-as-cpu
description: Run CPU-heavy work (rendering, simulation, batch conversion, parameter sweeps) on up to 20 parallel GitHub Actions runners through the htlin222/action-as-cpu repo, and collect the finished output from a GitHub release.
---

# action-as-cpu

Use this whenever a task needs far more CPU than the current machine has and can
be cut into independent pieces: frames of a render, chunks of a dataset, seeds
of a simulation, cells of a parameter grid. The repo `htlin222/action-as-cpu`
fans one job out over up to 20 GitHub-hosted runners (4 cores and 16 GB each on
this public repo, 6 hours per runner), merges the pieces, and publishes the
result as a release.

It does not help with work that cannot be split, needs a GPU, or needs secrets
or private data: the repo, its logs and its releases are public.

## How a job is described

A job is a folder `jobs/<job>/` with up to three scripts. Copy `jobs/_template/`.

| File | Runs | Contract |
| --- | --- | --- |
| `setup.sh` (optional) | on every runner | install dependencies (`pip`, `apt-get`) |
| `run.sh SHARD SHARDS OUTDIR` | once per runner, in parallel | do piece `SHARD` (0-based) of `SHARDS` and write results into `OUTDIR` |
| `merge.sh INDIR DISTDIR` (optional) | once, after all shards | combine everything in `INDIR` into deliverables in `DISTDIR` |

Rules that keep runs from failing:

- Split by index: item `i` belongs to the shard where `i % SHARDS == SHARD`.
  Interleaving spreads slow and fast items evenly across runners.
- Output file names must be unique across shards (`f0042.png`, `part-007.csv`);
  all shard outputs land in one folder for `merge.sh`.
- Make `run.sh` fail (non-zero exit) when its outputs are incomplete, and make
  `merge.sh` check that every expected piece is present.
- Every runner starts from a clean Ubuntu machine with Python 3.13; the job
  folder is the working directory. Put all code and small inputs in the folder.
- Each file in `DISTDIR` becomes a release asset (2 GB limit per file).

## Starting a run

1. Get the repo with push access and add or update `jobs/<job>/`.
2. Add a request file `requests/<tag>.env`:

   ```
   JOB=<job>
   SHARDS=20
   ```

   The file name (without `.env`) is the release tag, so use a new name for
   every run, for example `glass-lizard-v2`.
3. Commit and push to the default branch (`develope`). Pushing a new or changed
   file under `requests/` starts the `compute` workflow. It can also be started
   by hand from the Actions tab with the inputs `job`, `shards`, `tag`.

## Following the run and getting the result

```bash
gh api repos/htlin222/action-as-cpu/actions/runs --jq '.workflow_runs[0] | {id,status,conclusion,html_url}'
gh api repos/htlin222/action-as-cpu/actions/runs/<id>/jobs --paginate --jq '.jobs[] | {name,status,conclusion}'
gh api repos/htlin222/action-as-cpu/releases/tags/<tag> --jq '.assets[] | {name,size,browser_download_url}'
```

The finished output is the release `https://github.com/htlin222/action-as-cpu/releases/tag/<tag>`.
Download assets from `browser_download_url`.

If some shards fail, read their logs, fix the job, and push a request file with
a new tag. To keep finished shards, re-run only the failed jobs from the
Actions page; the release step then runs once all shards have succeeded.

## Before a big run

Try the job on the current machine with a tiny slice first, for example
`bash run.sh 0 200 /tmp/out` so that one shard is a handful of items, and time
it. Size the work so one shard stays well under the 6-hour runner limit.
