# action-as-cpu

Use GitHub Actions as a pool of CPUs: one job is cut into up to 20 shards, each
shard runs on its own runner in parallel, the pieces are merged, and the result
is published as a release.

- `SKILL.md`: the full instructions, written so an AI agent can follow them.
- `.github/workflows/compute.yml`: plan, fan out, merge, release.
- `jobs/_template/`: a minimal job to copy.
- `jobs/glass-lizard/`: a path-traced Blender animation rendered frame by frame.
- `requests/`: pushing `requests/<tag>.env` with `JOB=` and `SHARDS=` starts a run
  whose output appears under the release tag `<tag>`.
