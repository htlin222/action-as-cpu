# egfr-t790m

A 14 s, 1080x1920 molecular animation of the EGFR T790M gatekeeper mutation, ray-traced with open-source PyMOL.

- Protein: PDB 2JIT (EGFR kinase domain, T790M). Gefitinib from 2ITY and AMP-PNP from 2ITX, both superposed on 2JIT.
- `work.py` renders frames and writes each label anchor's screen position; `post.py` adds glow, floating labels and captions and encodes.
- Ligand paths are illustrative, not simulated.
