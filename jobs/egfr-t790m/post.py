"""Composite glow, floating labels and captions over the ray-traced frames and encode.
Usage: post.py FRAMES_DIR OUT.mp4   |   post.py FRAMES_DIR --test 30,150"""
import json, os, shutil, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from timeline import FPS, NF, ss, state

IN = sys.argv[1]
W, H = Image.open(os.path.join(IN, sorted(f for f in os.listdir(IN) if f.endswith('.png'))[0])).size
S = W / 720.0
px = lambda v: int(round(v * S))
F = lambda w, s: ImageFont.truetype('fonts/FiraSans-%s.ttf' % w, px(s))
PINK, AMBER, CYAN, WHITE, GREY = (255, 110, 170), (255, 184, 82), (120, 222, 246), (255, 255, 255), (200, 208, 220)
win = lambda t, a, b, f=.35: ss(a, a + f, t) * (1 - ss(b - f, b, t))

def spaced(d, xy, txt, font, fill, sp):
    x, y = xy
    for ch in txt:
        d.text((x, y), ch, font=font, fill=fill); x += d.textlength(ch, font=font) + sp
    return x
def spaced_w(d, txt, font, sp): return sum(d.textlength(ch, font=font) + sp for ch in txt) - sp

# --- static layer: scrims, title, legend -------------------------------------
yy = np.arange(H) / S
g = np.zeros((H, W, 4), np.uint8)
g[..., 3] = (np.clip(1 - yy / 300, 0, 1) ** 1.5 * 200 + np.clip((yy - 990) / 290, 0, 1) ** 0.9 * 250)[:, None].astype(np.uint8)
ST = Image.fromarray(g); d = ImageDraw.Draw(ST)
spaced(d, (px(36), px(54)), '03  KINASE', F('Medium', 15), GREY + (255,), px(5))
d.text((px(34), px(78)), 'T790M', font=F('ExtraLight', 76), fill=WHITE + (255,))
d.text((px(36), px(172)), 'EGFR gatekeeper mutation', font=F('Light', 22), fill=(214, 220, 228, 255))
x, y = px(36), px(1222)
for col, txt in ((PINK, 'residue 790'), (AMBER, 'ATP'), (CYAN, 'gefitinib')):
    d.ellipse((x, y + px(4), x + px(12), y + px(16)), fill=col + (255,)); d.text((x + px(20), y - px(2)), txt, font=F('Regular', 16), fill=(214, 220, 228, 255))
    x += px(20) + d.textlength(txt, font=F('Regular', 16)) + px(20)
d.text((x, y - px(2)), '|  PDB 2JIT · 2ITY · 2ITX', font=F('Light', 16), fill=(150, 160, 174, 255))

# --- captions ----------------------------------------------------------------
CAPS = [(0.3, 4.0, '1 / 3', 'BEFORE', 'Gefitinib blocks the ATP pocket.', 'Thr790 sits at the back of it.'),
        (4.3, 7.7, '2 / 3', 'MUTATION', 'T790M: Thr790 becomes Met790.', 'A bulkier gatekeeper.'),
        (8.0, 14.5, '3 / 3', 'RESISTANCE', 'ATP now binds tighter.', 'Gefitinib is outcompeted.')]
def cap_layer(c):
    L = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(L)
    x = spaced(d, (px(36), px(1068)), c[2], F('Medium', 15), WHITE + (255,), px(3)) + px(14)
    spaced(d, (x, px(1068)), c[3], F('Medium', 15), GREY + (255,), px(5))
    d.text((px(36), px(1098)), c[4], font=F('Light', 31), fill=WHITE + (255,))
    d.text((px(36), px(1140)), c[5], font=F('Light', 31), fill=(190, 198, 210, 255))
    return L
CAPL = [cap_layer(c) for c in CAPS]

# --- floating labels ---------------------------------------------------------
def labels(t, s):
    intro = win(t, 0.5, 3.3)
    return [  # anchor, title, subtitle, colour, offset(px @720), alpha, leader?
        ('nlobe', 'N-LOBE', None, GREY, (150, -150), intro, False),
        ('clobe', 'C-LOBE', None, GREY, (-60, 150), intro, False),
        ('res', 'Thr790', 'gatekeeper residue', WHITE, (-70, -205), win(t, 0.9, 4.4), True),
        ('res', 'Met790', 'bulkier gatekeeper', PINK, (-70, -205), win(t, 4.5, 13.5), True),
        ('gef', 'Gefitinib', 'EGFR inhibitor', CYAN, (70, 150), win(t, 0.9, 9.0) * s['gef_a'], True),
        ('atp', 'ATP', 'binds tighter', AMBER, (150, -120), win(t, 8.0, 13.5) * s['atp_a'], True),
    ]
def draw_label(base, pos, title, sub, col, off, alpha, leader):
    if alpha < .02: return base
    L = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(L)
    ax, ay = pos[0] * W, pos[1] * H
    if leader:
        ft, fs = F('Medium', 28), F('Light', 20)
        tw = max(d.textlength(title, font=ft), d.textlength(sub, font=fs)); th = px(28 + 8 + 22)
    else:
        ft = F('Medium', 14); tw = spaced_w(d, title, ft, px(5)); th = px(16)
    pad = px(11); bw, bh = tw + 2 * pad, th + 2 * pad
    cx = min(max(ax + px(off[0]), px(28) + bw / 2), W - px(28) - bw / 2); cy = min(max(ay + px(off[1]), px(250) + bh / 2), px(1030) - bh / 2)
    x0, y0 = cx - bw / 2, cy - bh / 2
    if leader:
        ex = min(max(ax, x0 + pad), x0 + bw - pad); ey = y0 + bh if ay > cy else y0
        d.line((ax, ay, ex, ey), fill=col + (220,), width=max(2, px(1.6)))
        r = px(5); d.ellipse((ax - r, ay - r, ax + r, ay + r), fill=col + (255,), outline=(0, 0, 0, 160), width=px(1))
    d.rounded_rectangle((x0, y0, x0 + bw, y0 + bh), radius=px(9), fill=(6, 10, 16, 185), outline=col + (150,), width=max(1, px(1)))
    if leader:
        d.text((x0 + pad, y0 + pad - px(3)), title, font=ft, fill=col + (255,))
        d.text((x0 + pad, y0 + pad + px(34)), sub, font=fs, fill=(214, 220, 228, 255))
    else:
        spaced(d, (x0 + pad, y0 + pad - px(2)), title, ft, col + (255,), px(5))
    L.putalpha(L.getchannel('A').point(lambda v: int(v * alpha)))
    return Image.alpha_composite(base, L)

def comp(i):
    t = i / FPS; s = state(t)
    a = np.asarray(Image.open(os.path.join(IN, 'f%04d.png' % i)).convert('RGB')).astype(np.float32) / 255
    hi = a * np.clip((a.max(axis=2, keepdims=True) - .62) / .38, 0, 1)
    b = np.asarray(Image.fromarray((hi * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(14 * S))).astype(np.float32) / 255
    out = Image.fromarray((np.clip(1 - (1 - a) * (1 - .9 * b), 0, 1) * 255).astype(np.uint8)).convert('RGBA')
    out = Image.alpha_composite(out, ST)
    P = json.load(open(os.path.join(IN, 'f%04d.json' % i)))
    for k, title, sub, col, off, al, leader in labels(t, s):
        out = draw_label(out, P[k], title, sub, col, off, al, leader)
    for c, L in zip(CAPS, CAPL):
        al = win(t, c[0], c[1], .4)
        if al > .01:
            cc = L.copy(); cc.putalpha(cc.getchannel('A').point(lambda v: int(v * al))); out = Image.alpha_composite(out, cc)
    fade = ss(0, .5, t) * (1 - ss(NF / FPS - .45, NF / FPS - .05, t))
    return (np.asarray(out.convert('RGB')).astype(np.float32) * fade).astype(np.uint8)

if sys.argv[2] == '--test':
    for i in map(int, sys.argv[3].split(',')): Image.fromarray(comp(i)).save(os.path.join(IN, 'comp_%04d.jpg' % i), quality=92)
else:
    ff = shutil.which('ffmpeg')
    if not ff:
        import imageio_ffmpeg; ff = imageio_ffmpeg.get_ffmpeg_exe()
    out = sys.argv[2]; tmp = out[:-4] + '.tmp.mp4'
    p = subprocess.Popen([ff, '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '%dx%d' % (W, H), '-r', str(FPS), '-i', '-',
                          '-c:v', 'libx264', '-crf', '16', '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', tmp], stdin=subprocess.PIPE)
    for i in range(NF): p.stdin.write(comp(i).tobytes())
    p.stdin.close(); assert p.wait() == 0; os.rename(tmp, out)
