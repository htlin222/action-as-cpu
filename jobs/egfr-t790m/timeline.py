# Shared timing for render (work.py) and compositing (post.py).
FPS = 30
DUR = 14.0
NF = int(FPS * DUR)
def ss(a, b, t):
    u = min(1.0, max(0.0, (t - a) / (b - a)))
    return u * u * (3 - 2 * u)
def state(t):
    return dict(
        zoom=ss(1.5, 3.6, t) - ss(11.4, 13.3, t),
        ang=-20 + 40 * ss(0, DUR, t),
        met=ss(4.2, 5.4, t),                      # Thr -> Met crossfade
        pulse=ss(5.0, 5.7, t) * (1 - ss(5.7, 6.6, t)),
        gef_out=ss(6.4, 8.6, t),                  # gefitinib leaves
        gef_a=1 - ss(7.8, 8.6, t),
        atp_in=ss(8.0, 10.2, t),                  # ATP arrives
        atp_a=ss(7.9, 8.7, t),
    )
