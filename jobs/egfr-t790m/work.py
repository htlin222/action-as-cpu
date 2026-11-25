"""Ray-trace frames of the EGFR T790M animation with PyMOL.
Usage: work.py --shard S --of N --out DIR [--scale 1.0] [--quality hi|lo]"""
import argparse, json, os, time
import numpy as np
from pymol import cmd
from timeline import FPS, NF, state

ap = argparse.ArgumentParser()
ap.add_argument('--shard', type=int, default=0); ap.add_argument('--of', type=int, default=1)
ap.add_argument('--out', default='out'); ap.add_argument('--scale', type=float, default=1.0)
ap.add_argument('--quality', default='hi'); ap.add_argument('--frames', default='')
A = ap.parse_args()
W, H = int(1080 * A.scale), int(1920 * A.scale)
FOV, HALF = 22.0, 45.0
os.makedirs(A.out, exist_ok=True)

# --- scene -----------------------------------------------------------------
cmd.load('pdb/2JIT.pdb', 'mutraw'); cmd.load('pdb/2ITY.pdb', 'wtraw'); cmd.load('pdb/2ITX.pdb', 'atpraw')
cmd.create('wt', 'wtraw and chain A'); cmd.create('mut', 'mutraw and chain A and polymer'); cmd.create('ap', 'atpraw and chain A')
cmd.align('mut and name CA', 'wt and name CA'); cmd.align('ap and name CA', 'wt and name CA')
cmd.create('gef', 'wt and resn IRE'); cmd.create('atp', 'ap and resn ANP')
cmd.create('thr', 'wt and resi 790 and not name N+C+O'); cmd.create('met', 'mut and resi 790 and not name N+C+O')
cmd.delete('wtraw or mutraw or atpraw or wt or ap'); cmd.remove('hydro'); cmd.hide('everything'); cmd.bg_color('black')
for n, c in dict(sage=[.62, .76, .56], teal=[.20, .62, .56], indigo=[.62, .68, .88], deep=[.22, .30, .58], copper=[.90, .60, .44],
                 rust=[.72, .36, .24], pink=[1, .31, .60], amber=[1, .67, .24], cyan=[.39, .84, .95], sulf=[1, .85, .25]).items():
    cmd.set_color(n, c)
N = 'mut and resi 690-792'; C = 'mut and resi 793-1030'; AC = 'mut and (resi 752-768 or resi 855-876)'
cmd.show('cartoon', 'mut'); cmd.show('surface', 'mut')
cmd.color('sage', N); cmd.color('indigo', C); cmd.color('copper', AC)
cmd.set('surface_color', 'teal', N); cmd.set('surface_color', 'deep', C); cmd.set('surface_color', 'rust', AC)
for k, v in dict(cartoon_fancy_helices=1, cartoon_oval_length=1.3, cartoon_oval_width=.28, cartoon_loop_radius=.25, cartoon_sampling=10,
                 transparency=.72, solvent_radius=1.6, transparency_mode=1, ray_transparency_oblique=1.0, ray_transparency_oblique_power=5,
                 ray_transparency_specular=.9, ray_transparency_shadows=0, ray_transparency_contrast=1.3, ray_shadows=0, specular=1.0,
                 spec_power=160, spec_reflect=1.6, shininess=70, ambient=.28, direct=.45, reflect=.55, light_count=3, depth_cue=1,
                 ray_trace_fog=1, fog=.9, fog_start=.45, ray_opaque_background=1, orthoscopic=0, field_of_view=FOV, hash_max=400,
                 sphere_scale=.85, sphere_quality=3).items():
    cmd.set(k, v)
cmd.set('cartoon_highlight_color', 'grey30')
cmd.set('surface_quality', 1 if A.quality == 'hi' else 0); cmd.set('antialias', 2 if A.quality == 'hi' else 1)
for o, c in (('gef', 'cyan'), ('atp', 'amber'), ('thr', 'grey90'), ('met', 'pink')):
    cmd.show('spheres', o); cmd.color(c, o)
cmd.color('sulf', 'met and name SD'); cmd.color('pink', 'thr and name OG1'); cmd.set('sphere_scale', 1.0, 'met or thr')

g = lambda s: np.array(cmd.centerofmass(s))
up = g('mut and resi 696-790') - g('mut and resi 796-1010'); up /= np.linalg.norm(up)
ctr = g('mut'); z = g('gef') - ctr; z -= up * z.dot(up); z /= np.linalg.norm(z); x = np.cross(up, z)
pocket = g('gef'); C0 = {o: cmd.get_coords(o) for o in ('gef', 'atp')}
vg = x * .8 + z * .55 - up * .15; vg /= np.linalg.norm(vg)
va = x * .65 + z * .6 + up * .45; va /= np.linalg.norm(va)
ANCH = dict(res=g('met'), nlobe=g('mut and resi 720-730'), clobe=g('mut and resi 900-930'))

def show(o, a):
    if a < .02: cmd.hide('spheres', o)
    else: cmd.show('spheres', o); cmd.set('sphere_transparency', 1 - a, o)

def frame(i):
    s = state(i / FPS)
    a = np.radians(s['ang']); t = np.radians(6 * s['zoom'])
    X = x * np.cos(a) - z * np.sin(a); Z = x * np.sin(a) + z * np.cos(a)
    Y = up * np.cos(t) - Z * np.sin(t); Z = up * np.sin(t) + Z * np.cos(t)
    org = ctr + (pocket - ctr) * .85 * s['zoom']; dist = 218 - 44 * s['zoom']
    cmd.set_view([X[0], Y[0], Z[0], X[1], Y[1], Z[1], X[2], Y[2], Z[2], 0, 0, -dist, *org, dist - HALF, dist + HALF, 0])
    show('thr', 1 - s['met']); show('met', s['met']); cmd.set('sphere_scale', 1.0 + .25 * s['pulse'], 'met')
    dg = vg * 30 * s['gef_out']; da = va * 28 * (1 - s['atp_in'])
    cmd.load_coords(C0['gef'] + dg, 'gef'); cmd.load_coords(C0['atp'] + da, 'atp')
    show('gef', s['gef_a']); show('atp', s['atp_a'])
    tmp = os.path.join(A.out, 'tmp_%04d.png' % i)
    cmd.png(tmp, width=W, height=H, ray=1, quiet=1); os.rename(tmp, os.path.join(A.out, 'f%04d.png' % i))
    # screen positions of label anchors, in fractions of width/height
    F = (H / 2) / np.tan(np.radians(FOV / 2)); P = dict(ANCH, gef=C0['gef'].mean(0) + dg, atp=C0['atp'].mean(0) + da)
    out = {}
    for k, p in P.items():
        d = p - org; depth = dist - d.dot(Z)
        out[k] = [float((W / 2 + F * d.dot(X) / depth) / W), float((H / 2 - F * d.dot(Y) / depth) / H)]
    json.dump(out, open(os.path.join(A.out, 'f%04d.json' % i), 'w'))

todo = [int(v) for v in A.frames.split(',')] if A.frames else [i for i in range(NF) if i % A.of == A.shard]
for i in todo:
    t0 = time.time(); frame(i); print('frame', i, round(time.time() - t0, 1), 's', flush=True)
missing = [i for i in todo if not (os.path.exists(os.path.join(A.out, 'f%04d.png' % i)) and os.path.exists(os.path.join(A.out, 'f%04d.json' % i)))]
assert not missing, missing
