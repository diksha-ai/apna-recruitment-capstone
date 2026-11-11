import numpy as np
from common import *

X, y, v = load()
R = Results()
X5 = X[:, :5]
n = len(X5)
Z = (X5 - X5.mean(axis=0)) / X5.std(axis=0, ddof=1)

def kmeans(Z, K):
    step = len(Z) // K
    C = Z[[i * step for i in range(K)]].copy()
    lab = None; passes = 0
    while True:
        d = ((Z[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        new = np.argmin(d, axis=1)       # ties -> lower index
        passes += 1
        if lab is not None and (new == lab).all():
            break
        lab = new
        for k in range(K):
            m = lab == k
            if m.any():
                C[k] = Z[m].mean(axis=0)
    w = ((Z - C[lab]) ** 2).sum()
    return lab, w, passes

W = {}; labs = {}
for K in range(2, 7):
    labs[K], W[K], p = kmeans(Z, K)
    R.put("wcss_k%d" % K, W[K])
elbow = None
for K in (3, 4, 5):
    if W[K] - W[K + 1] < 0.5 * (W[K - 1] - W[K]):
        elbow = K; break
R.put("elbow_k", elbow if elbow is not None else 0)

lab = labs[3]
order = sorted(range(3), key=lambda k: X5[lab == k, 2].mean())
seg = np.zeros(n, dtype=int)
rows = []
for s, k in enumerate(order):
    m = lab == k; seg[m] = s
    rows.append(["ABC"[s], int(m.sum())] + [X5[m, j].mean() for j in range(5)] + [y[m].mean(), v[m].mean()])
    R.put("size_segment_" + "ABC"[s], int(m.sum()))
write_csv("outputs/part3_segments.csv",
          ["segment", "size", "mean_X1", "mean_X2", "mean_X3", "mean_X4", "mean_X5", "outcome_rate", "mean_value"], rows)

# 3.4 hierarchical
names = ["E1", "E2", "E3", "E4", "E5", "E6"]
D = np.array([[0, 7.9, 7.4, 2.9, 2.9, 6.1],
              [7.9, 0, 1.4, 5.9, 5.3, 2.0],
              [7.4, 1.4, 0, 5.8, 4.6, 1.3],
              [2.9, 5.9, 5.8, 0, 2.8, 4.5],
              [2.9, 5.3, 4.6, 2.8, 0, 3.4],
              [6.1, 2.0, 1.3, 4.5, 3.4, 0]])
def link(a, b, how):
    vals = [D[i, j] for i in a for j in b]
    return min(vals) if how == "single" else max(vals) if how == "complete" else sum(vals) / len(vals)
rows = []
for how in ("single", "complete", "average"):
    cl = [(i,) for i in range(6)]; heights = []
    for m in range(1, 6):
        best = None
        for a in range(len(cl)):
            for b in range(a + 1, len(cl)):
                d = link(cl[a], cl[b], how)
                if best is None or d < best[0]:
                    best = (d, a, b)
        d, a, b = best
        ca, cb = cl[a], cl[b]
        if min(cb) < min(ca): ca, cb = cb, ca
        txt = lambda c: "+".join(names[i] for i in sorted(c))
        rows.append([how, m, txt(ca), txt(cb), d]); heights.append(d)
        cl = [c for i, c in enumerate(cl) if i not in (a, b)] + [tuple(sorted(ca + cb))]
    gaps = [heights[i + 1] - heights[i] for i in range(4)]
    g = int(np.argmax(gaps))            # cut between merge g+1 and g+2 -> g+1 merges below
    R.put(how + "_clusters_at_cut", 6 - (g + 1))
write_csv("outputs/part3_hierarchical.csv", ["linkage", "merge_no", "cluster_a", "cluster_b", "height"], rows)

# 3.5 PCA
Xc = Z - Z.mean(axis=0)
S = Xc.T @ Xc / (n - 1)
ev, V = np.linalg.eigh(S)
o = np.argsort(-ev); ev = ev[o]; V = V[:, o]
for j in range(5):
    k = np.argmax(np.abs(V[:, j]))
    if V[k, j] < 0: V[:, j] = -V[:, j]
tot = ev.sum()
R.put("eigenvalue_sum", tot)
for j in range(5): R.put("eigenvalue_%d" % (j + 1), ev[j])
for j in range(5): R.put("explained_ratio_%d" % (j + 1), ev[j] / tot)
cum = np.cumsum(ev) / tot
for j in range(5): R.put("cumulative_ratio_%d" % (j + 1), cum[j])
R.put("components_for_80pct", int(np.argmax(cum >= 0.8 - 1e-12)) + 1)
for c in (0, 1):
    for j in range(5): R.put("pc%d_loading_X%d" % (c + 1, j + 1), V[j, c])
P = Xc @ V[:, :2]
for r in range(3):
    R.put("row%d_pc1" % (r + 1), P[r, 0]); R.put("row%d_pc2" % (r + 1), P[r, 1])
for s in range(3):
    R.put("mean_pc1_segment_" + "ABC"[s], P[seg == s, 0].mean())
R.save("outputs/part3_results.txt")
