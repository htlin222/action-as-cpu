import argparse, os
ap = argparse.ArgumentParser(); ap.add_argument('--shard', type=int); ap.add_argument('--of', type=int); ap.add_argument('--out')
a = ap.parse_args()
items = range(1000)                      # the whole job
mine = [i for i in items if i % a.of == a.shard]   # this runner's slice
with open(os.path.join(a.out, f'part-{a.shard:03d}.txt'), 'w') as f:
    f.write(f'shard {a.shard}: sum of squares {sum(i * i for i in mine)}\n')
