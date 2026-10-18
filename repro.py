#!/usr/bin/env python3
# Reproduction for the "silent identity layer" case study.
#
# A Lorentz-boost geometry layer whose boost parameter was reset to zero
# silently reduced to a fixed affine map:
#
#     out = 0.5 * (0.5 * x / ||x||) + 0.5 * x
#
# Because expmap0 and logmap0 are exact inverses, the curvature cancels
# out of the round-trip, so d(loss)/d(curvature) is identically zero.
#
# Standard library only. Run: python3 repro.py

import math
import random


# ---------------- manifold ops (Lorentz model) ----------------

def softplus(x):
    return math.log1p(math.exp(-abs(x))) + max(x, 0.0)


def curvature(raw):
    # NOTE: raw = 0.01 yields c = 0.699, not 0.01
    return softplus(raw) + 1e-3


def vnorm(v):
    return math.sqrt(sum(t * t for t in v))


def vmax(v):
    return max(abs(t) for t in v)


def vsub(a, b):
    return [x - y for x, y in zip(a, b)]


def expmap0(u, c):
    r = math.sqrt(c) * vnorm(u)
    scale = math.sinh(r) / r if r >= 1e-6 else 1.0 + r * r / 6.0
    return [math.cosh(r) / math.sqrt(c)] + [scale * t for t in u]


def logmap0(x, c):
    x0, xs = x[0], x[1:]
    n = vnorm(xs)
    arg = max(math.sqrt(c) * x0, 1.0 + 1e-6)
    ac = math.log(arg + math.sqrt(arg * arg - 1.0))
    sc = 0.0 if n < 1e-6 else ac / (math.sqrt(c) * n + 1e-12)
    return [sc * t for t in xs]


def lorentz_boost(h, phi, a):
    # phi = 0 => identity, for any unit vector a
    x0, xs = h[0], h[1:]
    adx = sum(p * q for p, q in zip(a, xs))
    ch, sh = math.cosh(phi), math.sinh(phi)
    return ([ch * x0 + sh * adx]
            + [xs[i] + (ch - 1.0) * adx * a[i] + sh * x0 * a[i]
               for i in range(len(xs))])


def layer_forward(x, raw_curv, phi, gate_logit, a):
    c = curvature(raw_curv)
    u = [t / vnorm(x) * 0.5 for t in x]        # normalise to radius 0.5
    h = lorentz_boost(expmap0(u, c), phi, a)
    v = logmap0(h, c)
    g = 1.0 / (1.0 + math.exp(-gate_logit))    # sigmoid
    return [g * v[i] + (1.0 - g) * x[i] for i in range(len(x))]


# ---------------- checks ----------------

def main():
    random.seed(0)
    dim = 8
    x = [random.gauss(0.0, 1.0) for _ in range(dim)]
    a = [random.gauss(0.0, 1.0) for _ in range(dim)]
    an = vnorm(a)
    a = [t / an for t in a]

    print("=" * 66)
    print(" silent identity layer - reproduction")
    print("=" * 66)

    raw = 0.01
    c = curvature(raw)
    print("\n[0] parameter mapping")
    print("    raw_curvature = %s  ->  curvature = %.6f" % (raw, c))
    print("    (the config comment said 'curvature_init = 0.01')")

    u = [t / vnorm(x) * 0.5 for t in x]
    h = expmap0(u, c)
    hb = lorentz_boost(h, 0.0, a)
    d1 = vmax(vsub(hb, h))
    print("\n[1] boost with phi = 0")
    print("    max|boost(h) - h| = %.3e   -> %s" % (d1, "identity" if d1 < 1e-12 else "not identity"))

    vr = logmap0(h, c)
    d2 = vmax(vsub(vr, u))
    print("\n[2] round trip logmap(expmap(u))")
    print("    max|logmap(expmap(u)) - u| = %.3e   -> %s" % (d2, "exact inverse" if d2 < 1e-9 else "not exact"))

    out = layer_forward(x, 0.01, 0.0, 0.0, a)
    xn = vnorm(x)
    expected = [0.5 * (0.5 * t / xn) + 0.5 * t for t in x]
    d3 = vmax(vsub(out, expected))
    print("\n[3] layer output with phi = 0, gate_logit = 0")
    print("    max|out - (0.25*x/||x|| + 0.5*x)| = %.3e" % d3)
    print("    -> the layer is a fixed affine map; no geometry involved")

    outs = [layer_forward(x, r, 0.0, 0.0, a) for r in (0.01, 0.5, 2.0, 5.0)]
    d4 = max(vmax(vsub(o, outs[0])) for o in outs)
    print("\n[4] curvature sensitivity (phi = 0)")
    print("    curvature 0.01 -> 5.0 changes output by %.3e" % d4)
    print("    -> d(loss)/d(curvature) == 0; curvature can never learn")

    outs2 = [layer_forward(x, r, 0.8, 1.0, a) for r in (0.01, 0.5, 2.0, 5.0)]
    d5 = max(vmax(vsub(o, outs2[0])) for o in outs2)
    print("\n[5] control: phi = 0.8, gate_logit = 1.0")
    print("    curvature 0.01 -> 5.0 changes output by %.3e" % d5)
    print("    -> with a non-zero boost the curvature is live")

    print("\n[6] boost strength vs phi")
    for phi in (0.0, 0.01, 0.05, 0.1, 0.5, 0.8):
        hb = lorentz_boost(h, phi, a)
        print("    phi = %4.2f   relative change = %.4f" % (phi, vmax(vsub(hb, h))))

    print("\n" + "=" * 66)


if __name__ == "__main__":
    main()
