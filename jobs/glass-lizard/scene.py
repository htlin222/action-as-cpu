#!/usr/bin/env python3
"""Glass lizard glowing in the dark - Blender Cycles (path traced) procedural shot."""
import bpy, numpy as np, math, sys, os, time, argparse
from mathutils import Vector

FPS = 24
NF = 120
T_GAIT = 1.35
STRIDE = 0.30
SW = 0.32
V = STRIDE / T_GAIT

# ---------------------------------------------------------------- tunables
CFG = dict(
    res=(1024, 576), samples=56, noise=0.02, mnee=False,
    lens=40.0, fstop=3.6,
    key_power=2300.0, rim_power=65.0, top_power=4.0,
    core_strength=3.2, spot_strength=1.5, particle_strength=14.0,
    belly_power=4.0, bloom=0.26, bloom_thresh=2.0, scale_bump=0.35, coat=0.10, disp=0.0, ior=1.5, wall=0.016,
    exposure=0.0,
)

# quality overrides from the environment (used by the action-as-cpu pipeline)
if os.environ.get('LIZ_RES'): CFG['res'] = tuple(int(v) for v in os.environ['LIZ_RES'].lower().split('x'))
if os.environ.get('LIZ_SAMPLES'): CFG['samples'] = int(os.environ['LIZ_SAMPLES'])
if os.environ.get('LIZ_NOISE'): CFG['noise'] = float(os.environ['LIZ_NOISE'])
if os.environ.get('LIZ_MNEE'): CFG['mnee'] = os.environ['LIZ_MNEE'] == '1'

# x, zc, ry (half width), rz (half height)
SPINE = np.array([
    [1.06, 0.157, 0.030, 0.026],
    [1.01, 0.162, 0.062, 0.048],
    [0.94, 0.169, 0.094, 0.070],
    [0.86, 0.177, 0.120, 0.090],
    [0.78, 0.183, 0.134, 0.102],
    [0.70, 0.175, 0.130, 0.096],
    [0.60, 0.158, 0.098, 0.082],
    [0.42, 0.158, 0.135, 0.110],
    [0.20, 0.150, 0.175, 0.135],
    [0.00, 0.145, 0.190, 0.142],
    [-0.20, 0.148, 0.175, 0.135],
    [-0.40, 0.155, 0.140, 0.112],
    [-0.56, 0.140, 0.105, 0.092],
    [-0.80, 0.115, 0.080, 0.072],
    [-1.10, 0.090, 0.056, 0.052],
    [-1.40, 0.068, 0.036, 0.034],
    [-1.70, 0.050, 0.020, 0.019],
    [-1.95, 0.036, 0.007, 0.007]])
_sx = SPINE[::-1, 0]


def sp_zc(x): return np.interp(x, _sx, SPINE[::-1, 1])
def sp_ry(x): return np.interp(x, _sx, SPINE[::-1, 2])
def sp_rz(x): return np.interp(x, _sx, SPINE[::-1, 3])


def smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def norm(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def ik2(S, F, l1, l2, pole):
    d0 = np.linalg.norm(F - S)
    dirv = (F - S) / d0
    d = min(d0, 0.985 * (l1 + l2))
    a = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
    h = math.sqrt(max(l1 * l1 - a * a, 0.0))
    perp = norm(pole - np.dot(pole, dirv) * dirv)
    return S + a * dirv + h * perp, S + dirv * d


def rot_between(u, v):
    u = norm(u); v = norm(v)
    c = float(np.dot(u, v)); ax = np.cross(u, v); s = np.linalg.norm(ax)
    if s < 1e-9:
        return np.eye(3)
    ax = ax / s
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


# ---------------------------------------------------------------- legs (rest)
LEGS = []
for name, side, front, phase in [('LF', 1, True, 0.0), ('RF', -1, True, 0.5),
                                 ('LH', 1, False, 0.55), ('RH', -1, False, 0.05)]:
    if front:
        S = np.array([0.42, side * 0.13, 0.165]); W = np.array([0.50, side * 0.37, 0.036])
        l1, l2 = 0.20, 0.20; pole = np.array([-0.85, side * 0.25, 0.55]); tl = 0.105; fan0 = 30
    else:
        S = np.array([-0.40, side * 0.13, 0.160]); W = np.array([-0.47, side * 0.40, 0.036])
        l1, l2 = 0.21, 0.22; pole = np.array([0.85, side * 0.2, 0.5]); tl = 0.125; fan0 = 40
    E, W2 = ik2(S, W, l1, l2, pole)
    toes = []
    for k in range(5):
        ang = math.radians(side * (fan0 + (k - 2) * 36))
        d = np.array([math.cos(ang), math.sin(ang), 0.0])
        ln = tl * (1.0 - 0.12 * abs(k - 2.3))
        mid = W + d * ln * 0.55 + np.array([0, 0, -0.008])
        tip = W + d * ln + np.array([0, 0, -0.016])
        toes.append((mid, tip))
    LEGS.append(dict(name=name, side=side, front=front, phase=phase, S=S, E=E, W=W, l1=l1, l2=l2,
                     pole=pole, toes=toes))

EYES = [np.array([0.84, s * 0.082, 0.220]) for s in (1, -1)]
EYE_R = 0.025

# ---------------------------------------------------------------- motion
XS = np.linspace(-2.2, 1.4, 361)
DX = XS[1] - XS[0]
I0 = int(round((0.42 - XS[0]) / DX))


def theta(x, t):
    ph = 2 * np.pi * t / T_GAIT
    A = np.interp(x, [-2.0, -1.2, -0.4, 0.0, 0.42, 0.7, 1.2], [0.60, 0.42, 0.17, 0.13, 0.09, 0.03, 0.02])
    kap = np.interp(x, [-2.0, -0.4, 0.42, 1.0], [np.pi + 3.6, np.pi, 0.0, -0.2])
    th = A * np.sin(ph - 2 * np.pi * 0.55 - kap)
    # slow head look-around
    th = th + smooth(0.52, 0.74, x) * (0.26 * np.sin(2 * np.pi * t / 5.2 + 0.6) + 0.06 * np.sin(2 * np.pi * t / 1.9))
    # tail extra slow curl
    th = th + smooth(-0.7, -1.9, x) * 0.5 * np.sin(2 * np.pi * t / 4.3 + 2.0)
    return th


def dzfun(x, t):
    head = smooth(0.35, 1.1, x) * (0.07 + 0.03 * np.sin(2 * np.pi * t / 4.1 + 1.0))
    tail = smooth(-1.0, -1.95, x) * (0.09 + 0.04 * np.sin(2 * np.pi * t / 2.7))
    bob = 0.008 * np.sin(4 * np.pi * t / T_GAIT) * smooth(-1.0, -0.5, x)
    return head + tail + bob


def spine_tables(t):
    """t: array (M,) -> cx, cy, th, dz arrays (M, len(XS)) in body-local frame."""
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    th = theta(XS[None, :], t[:, None])
    cx = np.cumsum(np.cos(th), axis=1) * DX
    cy = np.cumsum(np.sin(th), axis=1) * DX
    cx = cx - cx[:, I0:I0 + 1] + 0.42
    cy = cy - cy[:, I0:I0 + 1]
    dz = dzfun(XS[None, :], t[:, None])
    return cx, cy, th, dz


def warp(P, tab, breath=0.0):
    cx, cy, th, dz = tab
    x = P[:, 0]
    C = np.interp(x, XS, cx); Cy = np.interp(x, XS, cy); T = np.interp(x, XS, th); Z = np.interp(x, XS, dz)
    y = P[:, 1]; z = P[:, 2]
    if breath != 0.0:
        k = 1.0 + breath * np.exp(-((x - 0.05) / 0.3) ** 2)
        zc = sp_zc(x)
        y = y * k; z = zc + (z - zc) * k
    out = np.empty_like(P)
    out[:, 0] = C - y * np.sin(T)
    out[:, 1] = Cy + y * np.cos(T)
    out[:, 2] = z + Z
    return out


def pose(t):
    tab = tuple(a[0] for a in spine_tables(t))
    legs = []
    for L in LEGS:
        S1 = warp(L['S'][None, :], tab)[0]
        p = t / T_GAIT + L['phase']
        n = math.floor(p); f = p - n
        base = L['W'][0] - STRIDE * (1 - SW) / 2
        if f < SW:
            e = f / SW; ez = e * e * (3 - 2 * e)
            fx = base + STRIDE * (n + ez); lift = 0.075 * math.sin(math.pi * e); sw = math.sin(math.pi * e)
        else:
            fx = base + STRIDE * (n + 1); lift = 0.0; sw = 0.0
        fx_local = fx - STRIDE * (t / T_GAIT + L['phase']) + STRIDE * L['phase'] - V * t + V * t
        # world foot x = fx - STRIDE*phase ; local = world - V t
        fx_local = (fx - STRIDE * L['phase']) - V * t
        F = np.array([fx_local, L['W'][1], L['W'][2] + lift])
        E1, W1 = ik2(S1, F, L['l1'], L['l2'], L['pole'])
        R1 = rot_between(L['E'] - L['S'], E1 - S1)
        R2 = rot_between(L['W'] - L['E'], W1 - E1)
        yaw = 0.5 * float(np.interp(L['S'][0], XS, tab[2]))
        pitch = 0.55 * sw
        cz, sz = math.cos(yaw), math.sin(yaw); cp, spp = math.cos(pitch), math.sin(pitch)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        Ry = np.array([[cp, 0, spp], [0, 1, 0], [-spp, 0, cp]])
        legs.append(dict(S=S1, E=E1, W=W1, R=(R1, R2, Rz @ Ry)))
    breath = 0.035 * math.sin(2 * math.pi * t / 2.4)
    return dict(tab=tab, legs=legs, breath=breath, t=t)


# ---------------------------------------------------------------- skin weights
def seg_dist(P, a, b):
    ab = b - a
    tt = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(P - (a + tt[:, None] * ab), axis=1), tt


def compute_weights(P, force_leg=None):
    N = len(P)
    wleg = np.zeros((N, 4)); wb = np.zeros((N, 4, 3)); dl = np.full((N, 4), 1e9)
    x = P[:, 0]
    nd = np.sqrt((P[:, 1] / sp_ry(x)) ** 2 + ((P[:, 2] - sp_zc(x)) / sp_rz(x)) ** 2)
    for li, L in enumerate(LEGS):
        d1, t1 = seg_dist(P, L['S'], L['E']); d2, t2 = seg_dist(P, L['E'], L['W'])
        darm = np.minimum(d1, d2)
        u = np.where(d1 <= d2, t1 * L['l1'], L['l1'] + t2 * L['l2'])
        dtoe = np.full(N, 1e9)
        for mid, tip in L['toes']:
            dtoe = np.minimum(dtoe, seg_dist(P, L['W'], mid)[0]); dtoe = np.minimum(dtoe, seg_dist(P, mid, tip)[0])
        b = 0.04
        wf = np.where(dtoe < darm, 1.0, smooth(L['l1'] + L['l2'] - b, L['l1'] + L['l2'], u))
        wl = smooth(L['l1'] - b, L['l1'] + b, u) * (1 - wf)
        wb[:, li, 0] = 1 - wl - wf; wb[:, li, 1] = wl; wb[:, li, 2] = wf
        dl[:, li] = np.minimum(darm, dtoe)
    if force_leg is not None:
        for li in range(4):
            wleg[:, li] = (force_leg == li)
    else:
        best = np.argmin(dl, axis=1)
        dmin = dl[np.arange(N), best]
        w = smooth(1.05, 1.6, nd) * (1 - smooth(0.08, 0.12, dmin))
        wleg[np.arange(N), best] = w
    return wleg, wb


def deform(P, W, ps):
    wleg, wb = W
    out = warp(P, ps['tab'], ps['breath']) * (1 - wleg.sum(1))[:, None]
    for li, L in enumerate(LEGS):
        m = wleg[:, li] > 1e-5
        if not m.any():
            continue
        Q = P[m]; lg = ps['legs'][li]
        R1, R2, R3 = lg['R']
        q = (wb[m, li, 0:1] * (lg['S'] + (Q - L['S']) @ R1.T) +
             wb[m, li, 1:2] * (lg['E'] + (Q - L['E']) @ R2.T) +
             wb[m, li, 2:3] * (lg['W'] + (Q - L['W']) @ R3.T))
        out[m] += wleg[m, li:li + 1] * q
    return out


# ---------------------------------------------------------------- mesh builders
def new_obj(name, verts, edges, faces, smooth_shade=True):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, edges, faces)
    me.update()
    if smooth_shade and len(me.polygons):
        me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def build_body_rest(swap=False, mw=None, mh=None):
    verts = []; edges = []; radii = []
    for i, (x, zc, ry, rz) in enumerate(SPINE):
        if mw is not None: ry = ry * mw[i]; rz = rz * mh[i]
        verts.append((x, 0, zc)); radii.append((rz, ry) if swap else (ry, rz))
        if i: edges.append((i - 1, i))
    for L in LEGS:
        root = 7 if L['front'] else 11
        base = len(verts)
        verts += [tuple(L['S']), tuple(L['E']), tuple(L['W'])]
        radii += [(0.080, 0.080), (0.060, 0.060), (0.046, 0.046)]
        edges += [(root, base), (base, base + 1), (base + 1, base + 2)]
        for mid, tip in L['toes']:
            k = len(verts)
            verts += [tuple(mid), tuple(tip)]
            radii += [(0.0125, 0.0125), (0.0165, 0.013)]
            edges += [(base + 2, k), (k, k + 1)]
    ob = new_obj('skel', verts, edges, [], False)
    sk = ob.modifiers.new('skin', 'SKIN')
    sk.branch_smoothing = 0.6
    sv = ob.data.skin_vertices[0].data
    for i, r in enumerate(radii):
        sv[i].radius = r; sv[i].use_root = (i == 9)
    ob.modifiers.new('s1', 'SUBSURF').levels = 1
    ob.modifiers['s1'].render_levels = 1
    sm = ob.modifiers.new('sm', 'SMOOTH'); sm.iterations = 8; sm.factor = 0.5
    s2 = ob.modifiers.new('s2', 'SUBSURF'); s2.levels = 2; s2.render_levels = 2
    dg = bpy.context.evaluated_depsgraph_get(); dg.update()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    bpy.data.objects.remove(ob)
    return me


def tube(path, rad, nseg=10):
    path = np.asarray(path); n = len(path)
    tang = norm(np.gradient(path, axis=0))
    up = np.array([0, 0, 1.0])
    side = norm(np.cross(up, tang)); up2 = np.cross(tang, side)
    ang = np.linspace(0, 2 * np.pi, nseg, endpoint=False)
    ring = (np.cos(ang)[None, :, None] * side[:, None, :] + np.sin(ang)[None, :, None] * up2[:, None, :])
    V_ = path[:, None, :] + ring * np.asarray(rad)[:, None, None]
    faces = []
    for i in range(n - 1):
        for j in range(nseg):
            a = i * nseg + j; b = i * nseg + (j + 1) % nseg
            faces.append((a, b, b + nseg, a + nseg))
    faces.append(tuple(range(nseg))[::-1]); faces.append(tuple(range((n - 1) * nseg, n * nseg)))
    return V_.reshape(-1, 3), faces


OCTA = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float64)
OCTA_F = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]

# ---------------------------------------------------------------- node helper
class NT:
    def __init__(self, mat):
        mat.use_nodes = True
        self.t = mat.node_tree
        self.t.nodes.clear()

    def n(self, kind, **kw):
        nd = self.t.nodes.new(kind)
        for k, v in kw.items():
            if k.startswith('_'):
                setattr(nd, k[1:], v)
            else:
                key = int(k[1:]) if (k[0] == 'i' and k[1:].isdigit()) else k.replace('_', ' ')
                sock = nd.inputs[key]
                if hasattr(v, 'is_output') or isinstance(v, bpy.types.NodeSocket):
                    self.t.links.new(v, sock)
                else:
                    sock.default_value = v
        return nd

    def math(self, op, a, b=None, c=None, clamp=False):
        nd = self.t.nodes.new('ShaderNodeMath'); nd.operation = op; nd.use_clamp = clamp
        for i, v in enumerate((a, b, c)):
            if v is None: continue
            if isinstance(v, bpy.types.NodeSocket): self.t.links.new(v, nd.inputs[i])
            else: nd.inputs[i].default_value = v
        return nd.outputs[0]

    def ramp(self, fac, stops, interp='LINEAR'):
        nd = self.t.nodes.new('ShaderNodeValToRGB'); cr = nd.color_ramp; cr.interpolation = interp
        while len(cr.elements) < len(stops): cr.elements.new(0.5)
        for e, (p, c) in zip(cr.elements, stops):
            e.position = p; e.color = c
        self.t.links.new(fac, nd.inputs[0])
        return nd.outputs[0]

    def maprange(self, v, a, b, c=0.0, d=1.0, mode='SMOOTHSTEP'):
        nd = self.t.nodes.new('ShaderNodeMapRange'); nd.interpolation_type = mode
        self.t.links.new(v, nd.inputs[0])
        for i, val in zip((1, 2, 3, 4), (a, b, c, d)): nd.inputs[i].default_value = val
        return nd.outputs[0]

    def mix(self, fac, a, b):
        nd = self.t.nodes.new('ShaderNodeMix'); nd.data_type = 'RGBA'
        for key, v in (('Factor', fac), ('A', a), ('B', b)):
            socks = [s for s in nd.inputs if s.name == key and (key == 'Factor' and s.type == 'VALUE' or s.type == 'RGBA')]
            s = socks[0]
            if isinstance(v, bpy.types.NodeSocket): self.t.links.new(v, s)
            else: s.default_value = v
        return [o for o in nd.outputs if o.type == 'RGBA'][0]

    def link(self, a, b): self.t.links.new(a, b)


HUE = [(0.0, (1.0, 0.08, 0.55, 1)), (0.3, (0.55, 0.15, 1.0, 1)), (0.55, (0.1, 0.35, 1.0, 1)),
       (0.8, (0.0, 0.85, 1.0, 1)), (1.0, (0.15, 1.0, 0.65, 1))]


def hue_np(x):
    u = np.clip((x + 1.95) / 2.95, 0, 1)
    ps = [h[0] for h in HUE]
    return np.stack([np.interp(u, ps, [h[1][k] for h in HUE]) for k in range(3)], -1)


def pulse_np(x, t):
    # bright wave travelling head -> tail every 2 s
    ph = (x * 0.9 + t / 2.0) % 1.0
    return 0.55 + 1.5 * np.exp(-((ph - 0.5) / 0.11) ** 2)


# ---------------------------------------------------------------- materials
def mat_body():
    m = bpy.data.materials.new('glass_lizard'); nt = NT(m)
    at = nt.n('ShaderNodeAttribute', _attribute_name='rest')
    rv = at.outputs['Vector']
    sep = nt.n('ShaderNodeSeparateXYZ', i0=rv)
    geo = nt.n('ShaderNodeNewGeometry')
    nsep = nt.n('ShaderNodeSeparateXYZ', i0=geo.outputs['Normal'])
    # --- scales: one F1 voronoi lookup, analytic dome normal (no finite-difference bump)
    SCL = 44.0
    vor = nt.n('ShaderNodeTexVoronoi', _feature='F1', Vector=rv, Scale=SCL)
    sp = nt.n('ShaderNodeVectorMath', _operation='SCALE', i0=rv, Scale=SCL)
    dv = nt.n('ShaderNodeVectorMath', _operation='SUBTRACT', i0=sp.outputs[0], i1=vor.outputs['Position'])
    dn_ = nt.n('ShaderNodeVectorMath', _operation='DOT_PRODUCT', i0=dv.outputs[0], i1=geo.outputs['Normal'])
    proj = nt.n('ShaderNodeVectorMath', _operation='SCALE', i0=geo.outputs['Normal'], Scale=dn_.outputs['Value'])
    tang = nt.n('ShaderNodeVectorMath', _operation='SUBTRACT', i0=dv.outputs[0], i1=proj.outputs[0])
    tsc = nt.n('ShaderNodeVectorMath', _operation='SCALE', i0=tang.outputs[0], Scale=CFG['scale_bump'])
    nadd = nt.n('ShaderNodeVectorMath', _operation='ADD', i0=geo.outputs['Normal'], i1=tsc.outputs[0])
    nscale = nt.n('ShaderNodeVectorMath', _operation='NORMALIZE', i0=nadd.outputs[0])
    zm = nt.math('ADD', 0.150, nt.math('MULTIPLY', 0.30, nt.math('MAXIMUM', 0.0, nt.math('SUBTRACT', 0.84, sep.outputs[0]))))
    dzm = nt.math('DIVIDE', nt.math('SUBTRACT', sep.outputs[2], zm), 0.0035)
    groove = nt.math('MULTIPLY', nt.math('EXPONENT', nt.math('MULTIPLY', nt.math('MULTIPLY', dzm, dzm), -1.0)),
                     nt.maprange(sep.outputs[0], 0.72, 0.77, 0.0, 1.0))
    ay = nt.n('ShaderNodeCombineXYZ', X=sep.outputs[0], Y=nt.math('ABSOLUTE', sep.outputs[1]), Z=sep.outputs[2])
    dn = nt.n('ShaderNodeVectorMath', _operation='DISTANCE', i0=ay.outputs[0], i1=(1.005, 0.024, 0.182))
    nos = nt.maprange(dn.outputs['Value'], 0.003, 0.008, 1.0, 0.0)
    hgt = nt.math('MULTIPLY', nt.math('ADD', nt.math('MULTIPLY', groove, 1.6), nt.math('MULTIPLY', nos, 1.8)), -1.0)
    bump = nt.n('ShaderNodeBump', Strength=0.5, Distance=0.004, Height=hgt)
    # --- bioluminescent dorsal spots
    csep = nt.n('ShaderNodeSeparateColor', i0=vor.outputs['Color'])
    spot = nt.math('MULTIPLY', nt.maprange(csep.outputs[0], 0.74, 0.80, 0.0, 1.0),
                   nt.maprange(vor.outputs['Distance'], 0.12, 0.42, 1.0, 0.0))
    dors = nt.maprange(nsep.outputs[2], 0.15, 0.75, 0.0, 1.0)
    tval = nt.n('ShaderNodeValue'); tval.name = 'TIME'
    ph = nt.math('FRACT', nt.math('ADD', nt.math('MULTIPLY', sep.outputs[0], 0.9), nt.math('MULTIPLY', tval.outputs[0], 0.5)))
    g = nt.math('DIVIDE', nt.math('SUBTRACT', ph, 0.5), 0.09)
    pul = nt.math('ADD', 0.45, nt.math('MULTIPLY', 1.6, nt.math('EXPONENT', nt.math('MULTIPLY', nt.math('MULTIPLY', g, g), -1.0))))
    hx = nt.maprange(sep.outputs[0], -1.95, 1.0, 0.0, 1.0, 'LINEAR')
    hue = nt.ramp(hx, HUE)
    spot = nt.math('MULTIPLY', spot, nt.maprange(sep.outputs[0], 0.50, 0.64, 1.0, 0.0))
    estr = nt.math('MULTIPLY', nt.math('MULTIPLY', spot, dors), nt.math('MULTIPLY', pul, CFG['spot_strength']))
    pr = nt.n('ShaderNodeBsdfPrincipled', Base_Color=(1.0, 1.0, 1.0, 1), Roughness=0.0, IOR=CFG['ior'],
              Transmission_Weight=1.0, Normal=bump.outputs[0], Emission_Color=hue, Emission_Strength=estr,
              Coat_Weight=CFG['coat'], Coat_Roughness=0.035, Coat_IOR=1.7, Coat_Normal=nscale.outputs[0])
    try:
        pr.inputs['Transmission Dispersion Scale'].default_value = CFG['disp']
        pr.inputs['Transmission Dispersion Abbe Number'].default_value = 30.0
    except Exception as e:
        print('no dispersion', e)
    vol = nt.n('ShaderNodeVolumeAbsorption', Color=(0.5, 0.9, 0.95, 1), Density=1.1)
    out = nt.n('ShaderNodeOutputMaterial', Surface=pr.outputs[0], Volume=vol.outputs[0])
    m.cycles.emission_sampling = 'NONE'
    return m


def mat_emit_attr(name, attr, strength):
    m = bpy.data.materials.new(name); nt = NT(m)
    at = nt.n('ShaderNodeAttribute', _attribute_name=attr)
    em = nt.n('ShaderNodeEmission', Color=at.outputs['Color'], Strength=strength)
    nt.n('ShaderNodeOutputMaterial', Surface=em.outputs[0])
    try: m.cycles.emission_sampling = 'NONE'
    except Exception as e: print(e)
    return m


def mat_eye():
    m = bpy.data.materials.new('eye'); nt = NT(m)
    tc = nt.n('ShaderNodeTexCoord')
    sc_ = nt.n('ShaderNodeVectorMath', _operation='SCALE', i0=tc.outputs['Object'], Scale=1.0 / EYE_R)  # unscaled object space
    sep = nt.n('ShaderNodeSeparateXYZ', i0=sc_.outputs[0])
    rr = nt.math('SQRT', nt.math('ADD', nt.math('MULTIPLY', sep.outputs[0], sep.outputs[0]), nt.math('MULTIPLY', sep.outputs[2], sep.outputs[2])))
    front = nt.maprange(sep.outputs[1], 0.0, 0.2, 0.0, 1.0)
    iris = nt.math('MULTIPLY', nt.maprange(rr, 0.70, 0.82, 1.0, 0.0), front)
    slit = nt.math('MULTIPLY', nt.maprange(nt.math('ABSOLUTE', sep.outputs[0]), 0.07, 0.15, 1.0, 0.0),
                   nt.maprange(nt.math('ABSOLUTE', sep.outputs[2]), 0.5, 0.62, 1.0, 0.0))
    glow = nt.math('MULTIPLY', iris, nt.math('SUBTRACT', 1.0, slit))
    noi = nt.n('ShaderNodeTexNoise', Vector=sc_.outputs[0], Scale=7.0, Detail=3.0)
    col = nt.ramp(nt.math('ADD', nt.math('MULTIPLY', rr, 0.8), nt.math('MULTIPLY', noi.outputs[0], 0.3)),
                  [(0.15, (1.0, 0.75, 0.12, 1)), (0.6, (1.0, 0.28, 0.02, 1)), (0.9, (0.5, 0.05, 0.0, 1))])
    pr = nt.n('ShaderNodeBsdfPrincipled', Base_Color=(0.004, 0.004, 0.005, 1), Roughness=0.04,
              Emission_Color=col, Emission_Strength=nt.math('MULTIPLY', glow, 3.5), Coat_Weight=1.0, Coat_Roughness=0.01)
    nt.n('ShaderNodeOutputMaterial', Surface=pr.outputs[0])
    return m


def fbm_tile(n, beta, seed):
    rng = np.random.default_rng(seed)
    f = np.fft.fft2(rng.normal(size=(n, n)))
    kx = np.fft.fftfreq(n)[:, None]; ky = np.fft.fftfreq(n)[None, :]
    k = np.sqrt(kx * kx + ky * ky); k[0, 0] = 1
    g = np.real(np.fft.ifft2(f / k ** beta))
    g -= g.min(); return g / g.max()


def bake_ground(n=2048, tile=6.0):
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(11)
    rock = fbm_tile(n, 1.5, 1); fine = fbm_tile(n, 0.9, 2); big = fbm_tile(n, 2.6, 5)
    npts = 170
    pts = rng.random((npts, 2))
    allp = np.vstack([pts + np.array([i, j]) for i in (-1, 0, 1) for j in (-1, 0, 1)])
    yy, xx = np.mgrid[0:n, 0:n] / n
    q = np.stack([xx.ravel(), yy.ravel()], 1)
    q = q + 0.012 * np.stack([fbm_tile(n, 1.8, 8).ravel() - 0.5, fbm_tile(n, 1.8, 9).ravel() - 0.5], 1)
    d, idx = cKDTree(allp).query(q, k=2)
    edge = (d[:, 1] - d[:, 0]).reshape(n, n)
    cell = (idx[:, 0] % npts).reshape(n, n)
    crack = 1 - smooth(0.0, 0.012, edge)
    slab = rng.random(npts)[cell]
    pud = smooth(0.40, 0.47, big)
    h = (0.55 * rock + 0.10 * fine + 0.25 * slab - 0.9 * crack) * (1 - pud) + 0.15 * (1 - pud)
    px = tile / n
    gx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) / (2 * px) * 0.035
    gy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) / (2 * px) * 0.035
    nrm = np.stack([-gx, -gy, np.ones_like(h)], -1); nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    alb = (0.02 + 0.07 * rock + 0.03 * slab) * (1 - 0.6 * crack)
    alb = 0.17 * alb * (1 - pud) + 0.003 * pud
    rgh = (0.05 + 0.24 * rock + 0.06 * fine) * (1 - pud) + 0.012 * pud + 0.2 * crack * (1 - pud)
    def img(name, arr3):
        im = bpy.data.images.new(name, n, n, alpha=False, float_buffer=True, is_data=True)
        px4 = np.concatenate([arr3, np.ones((n, n, 1))], -1).astype(np.float32)
        im.pixels.foreach_set(px4.ravel()); im.pack()
        return im
    return img('g_nrm', nrm * 0.5 + 0.5), img('g_ar', np.stack([alb, rgh, pud], -1))


def mat_ground(pebble=False):
    m = bpy.data.materials.new('wet_stone' + ('_p' if pebble else '')); nt = NT(m)
    tc = nt.n('ShaderNodeTexCoord')
    if pebble:
        co = tc.outputs['Object']
        rock = nt.n('ShaderNodeTexNoise', Vector=co, Scale=5.0, Detail=3.0, Roughness=0.68)
        rough = nt.maprange(rock.outputs[0], 0.3, 0.7, 0.10, 0.38)
        col = nt.ramp(rock.outputs[0], [(0.25, (0.012, 0.014, 0.018, 1)), (0.8, (0.07, 0.075, 0.085, 1))])
        pr = nt.n('ShaderNodeBsdfPrincipled', Base_Color=col, Roughness=rough, Coat_Weight=0.5, Coat_Roughness=0.04)
    else:
        inrm, iar = bake_ground()
        mp = nt.n('ShaderNodeMapping', Vector=tc.outputs['UV']); mp.inputs['Scale'].default_value = (80 / 6.0, 80 / 6.0, 1)
        tn = nt.n('ShaderNodeTexImage', Vector=mp.outputs[0]); tn.image = inrm
        ta = nt.n('ShaderNodeTexImage', Vector=mp.outputs[0]); ta.image = iar
        nm = nt.n('ShaderNodeNormalMap', Color=tn.outputs['Color'], Strength=0.6)
        sp = nt.n('ShaderNodeSeparateColor', i0=ta.outputs['Color'])
        cc = nt.n('ShaderNodeCombineColor', i0=sp.outputs[0], i1=nt.math('MULTIPLY', sp.outputs[0], 1.05), i2=nt.math('MULTIPLY', sp.outputs[0], 1.22))
        pr = nt.n('ShaderNodeBsdfPrincipled', Base_Color=cc.outputs[0], Roughness=sp.outputs[1], Normal=nm.outputs[0],
                  Coat_Weight=0.5, Coat_Roughness=0.02)
    nt.n('ShaderNodeOutputMaterial', Surface=pr.outputs[0])
    return m


# ---------------------------------------------------------------- particles
RNG = np.random.default_rng(7)
NA = 150
A_base = np.stack([RNG.uniform(-3.5, 4.5, NA), RNG.uniform(-3.2, 0.35, NA), RNG.uniform(0.04, 1.25, NA) ** 1.5], 1)
A_ph = RNG.uniform(0, 2 * np.pi, (NA, 4))
A_size = np.clip(0.0011 * np.exp(RNG.normal(0, 0.55, NA)), 0.0006, 0.0045)
A_col = hue_np(RNG.uniform(-1.95, 1.0, NA))
NS = 520
S_life = RNG.uniform(1.2, 2.8, NS)
S_b = RNG.uniform(0, 2.8, NS)
S_x = -1.45 + 2.45 * RNG.random(NS) ** 0.8
S_ang = RNG.uniform(-1.3, 1.3, NS)
S_vel = np.stack([RNG.normal(0, 0.05, NS), RNG.normal(0, 0.06, NS), RNG.uniform(0.06, 0.30, NS)], 1)
S_ph = RNG.uniform(0, 2 * np.pi, (NS, 3))
S_size = np.clip(0.0010 * np.exp(RNG.normal(0, 0.5, NS)), 0.0005, 0.0034)
NFst = 150
F_life = RNG.uniform(0.45, 0.85, NFst)
F_b = RNG.uniform(0, 1.6, NFst)
F_x = -1.3 + 2.2 * RNG.random(NFst)
F_vel = np.stack([RNG.normal(0, 0.30, NFst), RNG.normal(0.12, 0.34, NFst), RNG.uniform(0.5, 1.2, NFst)], 1)
F_size = RNG.uniform(0.0008, 0.0018, NFst)


def _born(x, ang, tb):
    n = len(x)
    cx, cy, th, dz = spine_tables(tb)
    idx = np.clip((x - XS[0]) / DX, 0, len(XS) - 2); i0 = idx.astype(int); fr = idx - i0
    r = np.arange(n)
    lerp = lambda a: a[r, i0] * (1 - fr) + a[r, i0 + 1] * fr
    Cx, Cy, Th, Dz = lerp(cx), lerp(cy), lerp(th), lerp(dz)
    ly = np.sin(ang) * sp_ry(x); lz = sp_zc(x) + np.cos(ang) * sp_rz(x)
    return np.stack([Cx - ly * np.sin(Th) + V * tb, Cy + ly * np.cos(Th), lz + Dz], 1)


def particles(t):
    """returns centers, radius, colour, streak vector (world units)"""
    pa = A_base.copy()
    pa[:, 0] += 0.07 * np.sin(0.5 * t + A_ph[:, 0]); pa[:, 1] += 0.07 * np.cos(0.43 * t + A_ph[:, 1])
    pa[:, 2] += 0.03 * t + 0.04 * np.sin(0.7 * t + A_ph[:, 2])
    tw = 0.2 + 0.8 * (0.5 + 0.5 * np.sin(2.4 * t + A_ph[:, 3])) ** 3
    ca = A_col * tw[:, None] * 0.8; sa = A_size; va = np.zeros((NA, 3))
    # embers shed from the skin
    age = (t - S_b) % S_life; tb = t - age
    born = _born(S_x, S_ang, tb)
    a = age[:, None]
    swirl = np.stack([0.05 * np.sin(2.2 * age + S_ph[:, 0]), 0.05 * np.cos(1.9 * age + S_ph[:, 1]),
                      0.02 * np.sin(2.5 * age + S_ph[:, 2])], 1) * np.minimum(age, 1.0)[:, None]
    ps = born + S_vel * a * (1.0 - 0.12 * a) + swirl
    k = age / S_life
    inten = smooth(0.0, 0.05, k) * (1 - k) ** 1.5 * (0.55 + 0.45 * np.sin(9 * age + S_ph[:, 0]))
    cs = hue_np(S_x) * (inten * 1.5)[:, None] * pulse_np(S_x, tb)[:, None] ** 0.6
    ss = S_size * (0.4 + 0.9 * (1 - k)); vs = S_vel * (1.0 - 0.24 * a) / 30 * 0.5
    # fast ballistic sparks (streaked by a half-frame shutter)
    agf = (t - F_b) % 1.6; tbf = t - agf
    bf = _born(F_x, np.zeros(NFst), tbf)
    g = np.array([0, 0, -2.6])
    vz = F_vel[:, 2]; z0 = bf[:, 2] - 0.004
    tl = (vz + np.sqrt(vz * vz + 2 * 2.6 * z0)) / 2.6            # landing time
    tt = np.minimum(agf, tl)
    pf = bf + F_vel * tt[:, None] + 0.5 * g * tt[:, None] ** 2
    landed = agf >= tl
    vf = (F_vel + g * tt[:, None]) * (~landed)[:, None]
    kf = np.clip(agf / (tl + 0.95), 0, 1)
    cf = hue_np(F_x) * ((1 - kf) ** 1.0 * 2.4)[:, None] * np.where(landed, 2.2, 1.0)[:, None]
    sf = F_size * (kf < 1) * np.where(landed, 2.3, 1.0)
    return (np.vstack([pa, ps, pf]), np.concatenate([sa, ss, sf]), np.vstack([ca, cs, cf]),
            np.vstack([va, vs, vf / 30 * 0.5]))


# ---------------------------------------------------------------- scene
S = {}


def build():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    c = sc.cycles
    c.device = 'CPU'
    c.samples = CFG['samples']; c.use_adaptive_sampling = True; c.adaptive_threshold = CFG['noise']
    c.adaptive_min_samples = 12
    c.use_denoising = True; c.denoiser = 'OPENIMAGEDENOISE'
    try:
        c.denoising_input_passes = 'RGB_ALBEDO_NORMAL'; c.denoising_prefilter = 'ACCURATE'
    except Exception as e: print(e)
    c.max_bounces = 8; c.transmission_bounces = 8; c.glossy_bounces = 4; c.diffuse_bounces = 2
    c.transparent_max_bounces = 8; c.volume_bounces = 0
    c.caustics_reflective = False; c.caustics_refractive = True
    c.sample_clamp_indirect = 6.0; c.sample_clamp_direct = 0.0; c.blur_glossy = 0.6
    sc.render.resolution_x, sc.render.resolution_y = CFG['res']; sc.render.resolution_percentage = 100
    sc.render.threads_mode = 'AUTO'
    sc.render.image_settings.file_format = 'PNG'; sc.render.image_settings.color_mode = 'RGB'
    sc.render.film_transparent = False
    sc.view_settings.view_transform = 'AgX'
    for look in ('AgX - Punchy', 'Punchy', 'AgX - Medium High Contrast', 'Medium High Contrast'):
        try:
            sc.view_settings.look = look; break
        except Exception: pass
    print('look:', sc.view_settings.look)
    sc.view_settings.exposure = CFG['exposure']
    w = bpy.data.worlds.new('w'); sc.world = w; w.use_nodes = True
    w.node_tree.nodes['Background'].inputs[0].default_value = (0.006, 0.012, 0.022, 1)
    w.node_tree.nodes['Background'].inputs[1].default_value = 0.6

    # ---- body
    def getP(me):
        P = np.empty(len(me.vertices) * 3); me.vertices.foreach_get('co', P); return P.reshape(-1, 3)
    def measure(P):
        xb = SPINE[:, 0]; hw = []; hh = []
        for x in xb:
            q = P[np.abs(P[:, 0] - x) < 0.015]; zc = sp_zc(x)
            if len(q) < 4: hw.append(np.nan); hh.append(np.nan); continue
            hw.append(np.abs(q[:, 1]).max()); hh.append((q[:, 2].max() - q[:, 2].min()) / 2)
        hw = np.array(hw); hh = np.array(hh)
        ok = ~(((xb > 0.26) & (xb < 0.64)) | ((xb > -0.64) & (xb < -0.20))) & np.isfinite(hw)
        o = np.argsort(xb[ok])
        return np.interp(xb, xb[ok][o], hw[ok][o]), np.interp(xb, xb[ok][o], hh[ok][o])
    best = None
    for swap in (False, True):
        mw = np.ones(len(SPINE)); mh = np.ones(len(SPINE))
        for it in range(4):
            me = build_body_rest(swap, mw, mh); P = getP(me)
            hw, hh = measure(P)
            err = float(np.abs(hw / SPINE[:, 2] - 1)[1:-1].mean() + np.abs(hh / SPINE[:, 3] - 1)[1:-1].mean())
            print('fit swap', swap, 'it', it, 'err %.3f' % err, flush=True)
            if best is None or err < best[0]:
                if best is not None: bpy.data.meshes.remove(best[1])
                best = (err, me)
            else:
                bpy.data.meshes.remove(me)
            mw = np.clip(mw * (SPINE[:, 2] / hw) ** 0.9, 0.3, 4.0); mh = np.clip(mh * (SPINE[:, 3] / hh) ** 0.9, 0.3, 4.0)
    me = best[1]; P = getP(me)
    for i, side in enumerate((1, -1)):
        x0 = 0.84; axp = np.array([x0, 0, sp_zc(x0)])
        dd = np.array([0, side * math.sin(math.radians(74)), math.cos(math.radians(74))])
        mm = np.abs(P[:, 0] - x0) < 0.02
        rel = P[mm] - axp; pr_ = rel @ dd; perp = np.linalg.norm(rel - pr_[:, None] * dd, axis=1)
        rs = pr_[perp < 0.025].max()
        EYES[i] = axp + dd * (rs - 0.62 * EYE_R)
        print('eye', i, EYES[i], 'surf r', rs)
    for e in EYES:   # brow ridge above/behind each eye
        d = np.linalg.norm(P - (e + np.array([-0.012, 0, 0.026])), axis=1)
        P += (norm(P - np.array([e[0], 0, sp_zc(e[0])])) * (0.014 * np.exp(-(d / 0.035) ** 2))[:, None])
    zc = sp_zc(P[:, 0]); rz = sp_rz(P[:, 0])
    belly = (np.abs(P[:, 1]) < sp_ry(P[:, 0]) * 1.05) & (P[:, 2] < zc)
    P[belly, 2] = zc[belly] + (P[belly, 2] - zc[belly]) * 0.82
    x = P[:, 0]
    crest = (0.030 * np.exp(-(P[:, 1] / 0.017) ** 2) * (0.25 + 0.75 * np.abs(np.cos(x * 34)) ** 3)
             * smooth(0.70, 0.5, x) * smooth(-1.5, -0.7, x) * (P[:, 2] > zc))
    P[:, 2] += crest * (0.6 + 0.4 * np.exp(-((x - 0.1) / 0.5) ** 2))
    # ---- hollow blown-glass shell: outer skin + inward-offset inner wall
    W0 = compute_weights(P)
    me.vertices.foreach_set('co', P.ravel()); me.update()
    nr = np.empty(len(P) * 3); me.vertices.foreach_get('normal', nr); nr = nr.reshape(-1, 3)
    xx = P[:, 0]
    th_t = np.minimum(CFG['wall'], 0.38 * np.minimum(sp_ry(xx), sp_rz(xx)))
    wl = W0[0].sum(1)
    th_l = (W0[1] * W0[0][:, :, None]).sum(1) @ np.array([0.010, 0.008, 0.0035])
    th = th_t * (1 - wl) + th_l
    Pin = P - nr * th[:, None]
    faces = [tuple(p.vertices) for p in me.polygons]
    N0 = len(P)
    faces2 = faces + [tuple(v + N0 for v in f[::-1]) for f in faces]
    P = np.vstack([P, Pin])
    bpy.data.meshes.remove(me)
    me = bpy.data.meshes.new('lizard'); me.from_pydata(P.tolist(), [], faces2); me.update()
    S['Wpre'] = (np.vstack([W0[0], W0[0]]), np.vstack([W0[1], W0[1]]))
    me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
    ra = me.attributes.new(name='rest', type='FLOAT_VECTOR', domain='POINT')
    ra.data.foreach_set('vector', np.vstack([P[:N0], P[:N0]]).ravel())
    body = bpy.data.objects.new('lizard', me); sc.collection.objects.link(body)
    body.data.materials.append(mat_body())
    body.visible_diffuse = False
    body.cycles.is_caustics_caster = CFG['mnee']
    S['body'] = body; S['P'] = P; S['W'] = S['Wpre']
    print('body verts', len(P), 'bbox', P.min(0), P.max(0))

    # ---- luminous core (spine + limb bones)
    xs = np.linspace(0.70, -1.90, 760)
    rad = np.interp(xs, [-1.9, -0.5, 0.7], [0.07, 0.115, 0.115]) * sp_rz(xs) * (0.03 + 0.97 * np.abs(np.sin(np.pi * xs / 0.105)) ** 2.2) * smooth(0.72, 0.62, xs)
    rad = np.maximum(rad, 0.0025)
    cv, cf = tube(np.stack([xs, 0 * xs, sp_zc(xs) + 0.55 * sp_rz(xs)], 1), rad, 10)
    allv = [cv]; allf = list(cf); force = [np.full(len(cv), -1)]; hx = [cv[:, 0].copy()]; dim = [np.ones(len(cv))]
    for li, L in enumerate(LEGS):
        pts = np.vstack([np.linspace(L['S'], L['E'], 8), np.linspace(L['E'], L['W'], 8)[1:]])
        ii = np.arange(len(pts))
        rr = 0.0032 + 0.0045 * (np.exp(-(ii / 1.1) ** 2) + np.exp(-((ii - 7) / 1.1) ** 2) + np.exp(-((ii - 14) / 1.1) ** 2))
        lv, lf = tube(pts, rr, 8)
        off = sum(len(a) for a in allv)
        allf += [tuple(i + off for i in f) for f in lf]
        allv.append(lv); force.append(np.full(len(lv), li)); hx.append(np.full(len(lv), L['S'][0]) - np.linalg.norm(lv - L['S'], axis=1)); dim.append(np.full(len(lv), 0.35))
    for li, L in enumerate(LEGS):
        for mid_, tip in L['toes']:
            pts = np.vstack([np.linspace(L['W'], mid_, 4), np.linspace(mid_, tip, 4)[1:]])
            lv, lf = tube(pts, 0.0016 + 0.0016 * np.abs(np.cos(np.arange(len(pts)) * np.pi / 3)), 6)
            off = sum(len(a_) for a_ in allv)
            allf += [tuple(i + off for i in f) for f in lf]
            allv.append(lv); force.append(np.full(len(lv), li)); hx.append(np.full(len(lv), L['S'][0] - 0.45)); dim.append(np.full(len(lv), 0.30))
    for xr in np.linspace(0.40, -0.36, 11):
        for sd in (1, -1):
            aa = np.linspace(0.30, 2.25, 16)
            pts = np.stack([xr - 0.05 * np.sin(aa), sd * 0.74 * sp_ry(xr) * np.sin(aa), sp_zc(xr) + 0.72 * sp_rz(xr) * np.cos(aa) + 0.0 * aa], 1)
            lv, lf = tube(pts, np.linspace(0.0042, 0.0022, len(pts)), 6)
            off = sum(len(a_) for a_ in allv)
            allf += [tuple(i + off for i in f) for f in lf]
            allv.append(lv); force.append(np.full(len(lv), -1)); hx.append(np.full(len(lv), xr)); dim.append(np.repeat(np.linspace(0.62, 0.18, len(pts)), 6))
    def add_wire(pts, r0, r1, d0, d1):
        lv, lf = tube(pts, np.linspace(r0, r1, len(pts)), 6)
        off = sum(len(a_) for a_ in allv)
        allf.extend([tuple(i + off for i in f) for f in lf])
        allv.append(lv); force.append(np.full(len(lv), -1)); hx.append(lv[:, 0].copy())
        dim.append(np.repeat(np.linspace(d0, d1, len(pts)), 6))
    xh = np.linspace(0.70, 0.92, 16)
    add_wire(np.stack([xh, 0 * xh, sp_zc(xh) + 0.60 * sp_rz(xh)], 1), 0.0035, 0.0015, 0.7, 0.12)       # cranial ridge
    xj = np.linspace(0.70, 0.90, 60)
    for sd in (1, -1):
        lvn = sum(len(a_) for a_ in allv)
        add_wire(np.stack([xj, sd * 0.60 * sp_ry(xj), sp_zc(xj) - 0.40 * sp_rz(xj)], 1), 0.0017, 0.0015, 0.30, 0.10)  # jaw + teeth
        allv[-1][:] = (allv[-1].reshape(-1, 6, 3) - np.stack([xj, sd * 0.60 * sp_ry(xj), sp_zc(xj) - 0.40 * sp_rz(xj)], 1)[:, None, :]).reshape(-1, 3)
        tooth = (0.8 + 0.5 * np.abs(np.sin(np.pi * xj / 0.028)) ** 6 * smooth(0.74, 0.80, xj))
        allv[-1][:] = (allv[-1].reshape(-1, 6, 3) * tooth[:, None, None] + np.stack([xj, sd * 0.60 * sp_ry(xj), sp_zc(xj) - 0.40 * sp_rz(xj)], 1)[:, None, :]).reshape(-1, 3)
        for xr in (0.74, 0.90):                                                                     # orbit arcs
            aa = np.linspace(0.2, 2.05, 12)
            add_wire(np.stack([xr + 0 * aa, sd * 0.62 * sp_ry(xr) * np.sin(aa), sp_zc(xr) + 0.62 * sp_rz(xr) * np.cos(aa)], 1), 0.0026, 0.0018, 0.55, 0.3)
    CV = np.vstack(allv); force = np.concatenate(force)
    core = new_obj('core', CV.tolist(), [], allf)
    core.data.color_attributes.new('glow', 'FLOAT_COLOR', 'POINT')
    core.data.materials.append(mat_emit_attr('core', 'glow', CFG['core_strength']))
    core.visible_diffuse = False; core.visible_shadow = False
    Wc = compute_weights(CV, force_leg=force)
    Wc[0][force < 0] = 0
    S['core'] = core; S['CP'] = CV; S['CW'] = Wc; S['Chx'] = np.concatenate(hx); S['Cdim'] = np.concatenate(dim)

    # ---- eyes
    S['eyes'] = []
    em = mat_eye()
    for i, e in enumerate(EYES):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=EYE_R)
        ob = bpy.context.object; ob.name = 'eye%d' % i
        bpy.ops.object.shade_smooth()
        ob.data.materials.append(em)
        ob.scale = (0.62, 0.62, 0.62)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=EYE_R)
        co = bpy.context.object; co.name = 'cornea%d' % i
        bpy.ops.object.shade_smooth()
        if i == 0:
            cmat = bpy.data.materials.new('cornea'); cn = NT(cmat)
            cp = cn.n('ShaderNodeBsdfPrincipled', Base_Color=(1, 1, 1, 1), Roughness=0.0, IOR=1.45, Transmission_Weight=1.0)
            cv_ = cn.n('ShaderNodeVolumeAbsorption', Color=(1.0, 0.45, 0.08, 1), Density=28.0)
            cn.n('ShaderNodeOutputMaterial', Surface=cp.outputs[0], Volume=cv_.outputs[0])
        co.data.materials.append(cmat); co.visible_diffuse = False; co.visible_shadow = False
        co.parent = ob; co.scale = (1 / 0.62, 1 / 0.62, 1 / 0.62)
        S['eyes'].append(ob)

    # ---- particles
    npart = NA + NS + NFst
    pv = np.zeros((npart * 6, 3))
    pf = [tuple(i * 6 + k for k in f) for i in range(npart) for f in OCTA_F]
    part = new_obj('particles', pv.tolist(), [], pf, False)
    part.data.color_attributes.new('pcol', 'FLOAT_COLOR', 'POINT')
    part.data.materials.append(mat_emit_attr('particles', 'pcol', CFG['particle_strength']))
    part.visible_diffuse = False; part.visible_shadow = False
    S['part'] = part

    # ---- ground + pebbles
    bpy.ops.mesh.primitive_plane_add(size=80)
    g = bpy.context.object; g.name = 'ground'; g.data.materials.append(mat_ground())
    g.cycles.is_caustics_receiver = True
    rng = np.random.default_rng(3)
    pvs = []; pfs = []
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=1)
    ico = bpy.context.object
    iv = np.array([v.co[:] for v in ico.data.vertices]); ifc = [tuple(p.vertices) for p in ico.data.polygons]
    bpy.data.objects.remove(ico)
    for k in range(230):
        x = rng.uniform(-3.5, 6.0); y = rng.uniform(0.55, 4.0) * rng.choice([-1, 1])
        if rng.random() < 0.25: y = rng.uniform(4, 9) * rng.choice([-1, 1])
        s = rng.uniform(0.03, 0.10)
        if abs(y) < 1.6: s = min(s, 0.045)
        if y > 0: s = rng.uniform(0.012, 0.035)
        sc3 = s * np.array([rng.uniform(0.8, 1.5), rng.uniform(0.8, 1.3), rng.uniform(0.35, 0.7)])
        lump = 1 + 0.18 * np.sin(iv @ rng.normal(0, 2.2, 3) + rng.uniform(0, 6)) + 0.1 * np.sin(iv @ rng.normal(0, 4, 3))
        a = rng.uniform(0, np.pi); R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        v = (iv * lump[:, None] * sc3) @ R.T + np.array([x, y, sc3[2] * 0.35])
        off = len(pvs) * len(iv)
        pfs += [tuple(i + off for i in f) for f in ifc]; pvs.append(v)
    peb = new_obj('pebbles', np.vstack(pvs).tolist(), [], pfs)
    peb.data.materials.append(mat_ground(True))
    peb.cycles.is_caustics_receiver = True

    # ---- lights
    def light(name, kind, energy, color, **kw):
        ld = bpy.data.lights.new(name, kind); ld.energy = energy; ld.color = color
        for k, v in kw.items(): setattr(ld, k, v)
        ob = bpy.data.objects.new(name, ld); sc.collection.objects.link(ob)
        return ob
    key = light('key', 'SPOT', CFG['key_power'], (0.72, 0.9, 1.0), spot_size=math.radians(34), spot_blend=0.9, shadow_soft_size=0.015)
    key.data.cycles.is_caustics_light = True; key.visible_glossy = False
    rim = light('rim', 'AREA', CFG['rim_power'], (0.3, 0.85, 1.0), shape='ELLIPSE', size=0.16, size_y=3.0)
    rim2 = light('rim2', 'AREA', CFG['rim_power'] * 0.55, (0.75, 0.3, 1.0), shape='ELLIPSE', size=0.16, size_y=2.6)
    top = light('top', 'AREA', CFG['top_power'], (0.25, 0.4, 1.0), shape='DISK', size=2.4)
    for o in (rim, rim2, top): o.visible_camera = False
    try:
        lc = bpy.data.collections.new('rimrecv'); lc.objects.link(S['body'])
        for o in (rim, rim2): o.light_linking.receiver_collection = lc
    except Exception as e: print('light linking failed', e)
    kick = light('kick', 'AREA', 20.0, (0.45, 0.95, 1.0), shape='DISK', size=1.1); kick.visible_camera = False; kick.visible_transmission = False
    try: kick.light_linking.receiver_collection = lc
    except Exception as e: print(e)
    S['lights'] = dict(key=key, rim=rim, rim2=rim2, top=top, kick=kick)
    S['belly'] = []
    for i, x in enumerate(np.linspace(0.40, -1.5, 7)):
        ob = light('belly%d' % i, 'POINT', CFG['belly_power'], (1, 1, 1), shadow_soft_size=0.05)
        ob.visible_glossy = False; ob.visible_transmission = False
        ob.data.use_shadow = False
        S['belly'].append((x, ob))

    # ---- camera
    cd = bpy.data.cameras.new('cam'); cd.lens = CFG['lens']; cd.sensor_width = 36
    cd.dof.use_dof = True; cd.dof.aperture_fstop = CFG['fstop']; cd.dof.aperture_blades = 7
    cd.clip_start = 0.05; cd.clip_end = 200
    cam = bpy.data.objects.new('cam', cd); sc.collection.objects.link(cam); sc.camera = cam
    S['cam'] = cam

    # ---- compositor bloom
    try:
        t = bpy.data.node_groups.new('comp', 'CompositorNodeTree')
        t.interface.new_socket(name='Image', in_out='OUTPUT', socket_type='NodeSocketColor')
        rl = t.nodes.new('CompositorNodeRLayers')
        gl = t.nodes.new('CompositorNodeGlare')
        gl.inputs['Type'].default_value = 'Bloom'; gl.inputs['Quality'].default_value = 'High'
        gl.inputs['Threshold'].default_value = CFG['bloom_thresh']; gl.inputs['Strength'].default_value = CFG['bloom']
        gl.inputs['Size'].default_value = 0.55; gl.inputs['Smoothness'].default_value = 0.4
        go = t.nodes.new('NodeGroupOutput')
        t.links.new(rl.outputs['Image'], gl.inputs['Image']); t.links.new(gl.outputs['Image'], go.inputs[0])
        sc.compositing_node_group = t
        sc.render.use_compositing = True
    except Exception as e:
        print('compositor setup failed:', e)
    return sc


def look_at(ob, pos, tgt, roll=0.0):
    ob.location = pos
    d = Vector(tgt) - Vector(pos)
    q = d.to_track_quat('-Z', 'Y')
    ob.rotation_euler = q.to_euler()


def cam_path(t):
    u = t / (NF / FPS); e = u * u * u * (u * (6 * u - 15) + 10)
    tgt = np.array([V * t + 0.55 - 0.38 * e, 0.0, 0.20 - 0.10 * e])
    az = math.radians(26 + 8 * e); r = 1.55 + 0.55 * e; h = 0.22 + 0.62 * e
    pos = tgt + np.array([r * math.cos(az), r * math.sin(az), h - 0.20])
    return pos, tgt, e


def set_frame(f):
    t = f / FPS
    ps = pose(t)
    off = np.array([V * t, 0, 0])
    body = S['body']
    Q = deform(S['P'], S['W'], ps)
    body.data.vertices.foreach_set('co', Q.ravel()); body.data.update()
    body.location = off
    body.data.materials[0].node_tree.nodes['TIME'].outputs[0].default_value = t
    core = S['core']
    Qc = deform(S['CP'], S['CW'], ps)
    core.data.vertices.foreach_set('co', Qc.ravel())
    hx = S['Chx']
    col = hue_np(hx) * pulse_np(hx, t)[:, None]
    col = col * S['Cdim'][:, None]
    rgba = np.concatenate([col, np.ones((len(col), 1))], 1)
    core.data.color_attributes['glow'].data.foreach_set('color', rgba.ravel()); core.data.update()
    core.location = off
    for e, ob in zip(EYES, S['eyes']):
        p = warp(e[None, :], ps['tab'])[0]
        th = float(np.interp(e[0], XS, ps['tab'][2]))
        ob.location = p + off
        side = 1 if e[1] > 0 else -1
        ob.rotation_euler = (0.2, 0.0, th + (-0.5 if side > 0 else math.pi + 0.5))
    for x, ob in S['belly']:
        p = warp(np.array([[x, 0, 0.022]]), ps['tab'])[0]; p[2] = 0.035
        ob.location = p + off
        c3 = hue_np(np.array([x]))[0]
        ob.data.color = tuple(c3); ob.data.energy = CFG['belly_power'] * float(pulse_np(np.array([x]), t)[0])
    # particles
    pp, sz, pc, pvel = particles(t)
    L = np.linalg.norm(pvel, axis=1)
    ax = np.where(L[:, None] > 1e-6, pvel / np.maximum(L, 1e-9)[:, None], np.array([[0, 0, 1.0]]))
    ref = np.where(np.abs(ax[:, 2:3]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
    bx = norm(np.cross(ax, ref)); cx_ = np.cross(ax, bx)
    la = (sz + L * 0.5)[:, None]; s1 = sz[:, None]
    pv = np.stack([pp + ax * la, pp - ax * la, pp + bx * s1, pp - bx * s1, pp + cx_ * s1, pp - cx_ * s1], 1).reshape(-1, 3)
    part = S['part']
    part.data.vertices.foreach_set('co', pv.ravel())
    rgba = np.concatenate([np.repeat(pc, 6, 0), np.ones((len(pv), 1))], 1)
    part.data.color_attributes['pcol'].data.foreach_set('color', rgba.ravel()); part.data.update()
    # camera + lights follow
    pos, tgt, e = cam_path(t)
    cam = S['cam']; look_at(cam, pos, tgt)
    head = warp(np.array([[0.86 - 0.5 * e, 0.0, 0.2]]), ps['tab'])[0] + off
    vd = (tgt - pos) / np.linalg.norm(tgt - pos)
    cam.data.dof.focus_distance = float(np.dot(head - pos, vd))
    cam.data.dof.aperture_fstop = CFG['fstop'] + 4.0 * e
    S['dbg'] = [((warp(np.array([[xx, 0.0, 0.15]]), ps['tab'])[0] + off - pos)) for xx in (1.06, -1.95)]
    rt = norm(np.cross(vd, [0, 0, 1.0])); upv = np.cross(rt, vd)
    S['dbg'] = [(round(float(np.dot(d, rt) / np.dot(d, vd) / 0.45), 2), round(float(np.dot(d, upv) / np.dot(d, vd) / 0.253), 2)) for d in S['dbg']]
    c = np.array([V * t, 0, 0.12])
    L = S['lights']
    look_at(L['key'], c + np.array([-1.3, -2.3, 2.5]), c + np.array([0.1, 0.1, 0]))
    look_at(L['rim'], c + np.array([-2.4, -1.4, 0.55]), c)
    look_at(L['rim2'], c + np.array([1.8, -2.2, 0.7]), c)
    look_at(L['top'], c + np.array([0.3, 0.4, 2.6]), c)
    hd = warp(np.array([[0.9, 0.0, 0.2]]), ps['tab'])[0] + off
    look_at(L['kick'], hd + np.array([0.5, -1.0, 0.55]), hd)
    bpy.context.view_layer.update()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', default='0'); ap.add_argument('--out', default='prev')
    ap.add_argument('--scale', type=int, default=100); ap.add_argument('--samples', type=int, default=0)
    ap.add_argument('--topview', action='store_true'); ap.add_argument('--threads', type=int, default=0); ap.add_argument('--clay', action='store_true'); ap.add_argument('--cam', default='')
    ap.add_argument('--nodof', action='store_true'); ap.add_argument('--nomnee', action='store_true'); ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    if a.samples: CFG['samples'] = a.samples
    sc = build()
    sc.render.resolution_percentage = a.scale
    if a.threads:
        sc.render.threads_mode = 'FIXED'; sc.render.threads = a.threads
    os.makedirs(a.out, exist_ok=True)
    if ':' in a.frames:
        p = [int(v) for v in a.frames.split(':')]; frames = range(p[0], p[1], p[2] if len(p) > 2 else 1)
    else:
        frames = [int(v) for v in a.frames.split(',')]
    for f in frames:
        fn = os.path.join(a.out, 'f%04d.png' % f)
        if os.path.exists(fn): continue
        t0 = time.time()
        set_frame(f)
        if a.topview:
            c = np.array([V * f / FPS - 0.4, 0, 0])
            look_at(S['cam'], c + np.array([0.0, 0.01, 4.2]), c); S['cam'].data.dof.use_dof = False
        if a.cam:
            az, r, h, tx = [float(v) for v in a.cam.split(',')]
            c = np.array([V * f / FPS + tx, 0, 0.15])
            look_at(S['cam'], c + np.array([r * math.cos(math.radians(az)), r * math.sin(math.radians(az)), h]), c)
            S['cam'].data.dof.focus_distance = r
        if a.nodof or a.clay: S['cam'].data.dof.use_dof = False
        if a.nomnee: S['body'].cycles.is_caustics_caster = False
        if a.dry:
            print('DRY', f, S['dbg']); continue
        if a.clay:
            cm = bpy.data.materials.new('clay'); cm.use_nodes = True
            cm.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.5, 0.5, 0.5, 1)
            S['body'].data.materials[0] = cm; S['body'].visible_diffuse = True
            S['part'].hide_render = True; S['core'].hide_render = True
            sc.world.node_tree.nodes['Background'].inputs[0].default_value = (0.5, 0.5, 0.5, 1)
            sc.world.node_tree.nodes['Background'].inputs[1].default_value = 1.0
            sc.render.use_compositing = False
            for o in S['lights'].values(): o.data.energy *= 0.15
        sc.render.filepath = os.path.abspath(fn[:-4] + '.tmp.png')
        bpy.ops.render.render(write_still=True)
        os.rename(fn[:-4] + '.tmp.png', fn)
        print('FRAME', f, 'time %.1fs' % (time.time() - t0), 'ndc head/tail', S.get('dbg'), flush=True)
