import numpy as np
from common import load, Results

X, y, v = load()
R = Results()
n = len(v)
mean = v.mean(); sd = v.std(ddof=1); se = sd / np.sqrt(n)
R.put("n", n); R.put("mean_value", mean); R.put("sd_value", sd); R.put("se_value", se)
R.put("ci95_low", mean - 1.96 * se); R.put("ci95_high", mean + 1.96 * se)
z = (mean - 8.0) / se
R.put("z_stat", z); R.put("reject_h0", abs(z) > 1.96)
nreq = int(np.ceil((1.96 * sd / 0.5) ** 2 - 1e-12))
R.put("n_required", nreq)

x4 = X[:, 3]
def simple(x, yy):
    xm, ym = x.mean(), yy.mean()
    cov = ((x - xm) * (yy - ym)).sum() / (len(x) - 1)
    var = ((x - xm) ** 2).sum() / (len(x) - 1)
    s = cov / var; i = ym - s * xm
    f = i + s * x
    tss = ((yy - ym) ** 2).sum(); ess = ((f - ym) ** 2).sum(); rss = ((yy - f) ** 2).sum()
    return s, i, f, tss, ess, rss

def halves(f, resid):
    o = np.argsort(f, kind="stable"); h = len(f) // 2
    return resid[o[:h]].std(ddof=1), resid[o[h:]].std(ddof=1)

s, i, f, tss, ess, rss = simple(x4, v)
R.put("reg_slope", s); R.put("reg_intercept", i)
R.put("tss", tss); R.put("ess", ess); R.put("rss", rss)
R.put("tss_equals_ess_plus_rss", abs(tss - ess - rss) < 1e-6 * tss)
R.put("r2", 1 - rss / tss); R.put("rmse", np.sqrt(rss / n)); R.put("mae", np.abs(v - f).mean())

# 1.5 gradient descent
sx = x4.std(ddof=1); mx = x4.mean()
zz = (x4 - mx) / sx
m = c = 0.0
J = lambda: np.mean((v - (m * zz + c)) ** 2)
R.put("gd_J0", J())
for t in range(1, 501):
    e = v - (m * zz + c)
    gm = -2 / n * (zz * e).sum(); gc = -2 / n * e.sum()
    m -= 0.1 * gm; c -= 0.1 * gc
    if t in (1, 10, 100, 500):
        R.put("gd_J%d" % t, J())
gs = m / sx; gi = c - m * mx / sx
R.put("gd_slope", gs); R.put("gd_intercept", gi); R.put("gd_slope_gap", abs(gs - s))

# 1.6
lo, up = halves(f, v - f)
R.put("resid_sd_lower_raw", lo); R.put("resid_sd_upper_raw", up); R.put("resid_sd_ratio_raw", up / lo)
lv, lx = np.log(v), np.log(x4)
s2, i2, f2, t2, e2, r2s = simple(lx, lv)
R.put("loglog_intercept", i2); R.put("loglog_slope", s2)
R.put("loglog_r2", 1 - r2s / t2); R.put("loglog_rmse", np.sqrt(r2s / n))
lo, up = halves(f2, lv - f2)
R.put("resid_sd_lower_log", lo); R.put("resid_sd_upper_log", up); R.put("resid_sd_ratio_log", up / lo)

# 1.7
A = np.column_stack([np.ones(n), lx, X[:, 0], X[:, 1], X[:, 2], X[:, 4], X[:, 5], X[:, 6]])
beta = np.linalg.solve(A.T @ A, A.T @ lv)
for k, b in enumerate(beta):
    R.put("mr_beta_%d" % k, b)
res = lv - A @ beta
r2 = 1 - (res ** 2).sum() / ((lv - lv.mean()) ** 2).sum()
R.put("mr_r2", r2); R.put("mr_adj_r2", 1 - (1 - r2) * (n - 1) / (n - 7 - 1))
r2l = 1 - r2s / t2
R.put("loglog_adj_r2", 1 - (1 - r2l) * (n - 1) / (n - 1 - 1))
R.save("outputs/part1_results.txt")
