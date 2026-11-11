import numpy as np
from common import *

Xtr, ytr, Xte, yte, Str, Ste = split_std()
R = Results()

def ent(p, n):
    if n == 0: return 0.0
    q = p / n
    return -sum(a * np.log2(a) for a in (q, 1 - q) if a > 0)
def gini(p, n):
    if n == 0: return 0.0
    q = p / n
    return 1 - q * q - (1 - q) ** 2

N = len(ytr); P = int(ytr.sum())
R.put("root_entropy", ent(P, N)); R.put("root_gini", gini(P, N))
conds = {"S1": Xtr[:, 5] == 1, "S2": Xtr[:, 6] == 1, "S3": Xtr[:, 2] > 0.5, "S4": Xtr[:, 0] <= 24}
rows = []; ig = {}; gg = {}
for s, m in conds.items():
    ny, py = int(m.sum()), int(ytr[m].sum()); nn, pn = N - ny, P - py
    ig[s] = ent(P, N) - (ny / N * ent(py, ny) + nn / N * ent(pn, nn))
    gg[s] = gini(P, N) - (ny / N * gini(py, ny) + nn / N * gini(pn, nn))
    rows.append([s, ny, py, nn, pn, ig[s], gg[s]])
write_csv("outputs/part4_splits.csv", ["split", "n_yes", "positives_yes", "n_no", "positives_no", "info_gain", "gini_gain"], rows)
bi = max(ig, key=lambda s: (ig[s], -int(s[1]))); bg = max(gg, key=lambda s: (gg[s], -int(s[1])))
R.put("best_split_by_info_gain", bi); R.put("best_split_by_gini_gain", bg)

# stump
m = conds[bi]
r_yes, r_no = ytr[m].mean(), ytr[~m].mean()
R.put("stump_rate_yes", r_yes); R.put("stump_rate_no", r_no)
col, thr, op = {"S1": (5, 1, "eq"), "S2": (6, 1, "eq"), "S3": (2, 0.5, "gt"), "S4": (0, 24, "le")}[bi]
xc = Xte[:, col]
yes_te = (xc == thr) if op == "eq" else (xc > thr) if op == "gt" else (xc <= thr)
p_stump = np.where(yes_te, r_yes, r_no)

w, b, _ = train_logistic(Str, ytr, 0.1, 1000)
p_log = sigmoid(Ste @ w + b)
K = 5
# best K from part 2 results file
for line in open("outputs/part2_results.txt"):
    if line.startswith("knn_best_k"):
        K = int(line.split("=")[1])
kp, p_knn = knn_pred(Str, ytr, Ste, K)
e = 1e-12
pred_log = p_log >= 0.3 - e
pred_stump = p_stump >= 0.3 - e
votes = pred_log.astype(int) + kp.astype(int) + pred_stump.astype(int)
hard = votes >= 2
soft = (p_log + p_knn + p_stump) / 3 >= 0.3 - e
rows = []
for nm, pr in (("logistic", pred_log), ("knn", kp), ("stump", pred_stump), ("hard_vote", hard), ("soft_vote", soft)):
    c = confusion(pr, yte); a, pre, rec, f1, _ = metrics(*c)
    rows.append([nm, *c, pre, rec, f1])
write_csv("outputs/part4_voting.csv", ["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"], rows)

# AdaBoost
y = np.array([1, 1, 1, -1, -1, -1, 1, -1], float)
h = [np.array([1, 1, -1, -1, -1, -1, -1, -1], float), np.array([-1, 1, 1, -1, 1, -1, 1, -1], float)]
wt = np.full(8, 1 / 8); rows = []; alphas = []
for t in (0, 1):
    eps = wt[h[t] != y].sum()
    al = 0.5 * np.log((1 - eps) / eps)
    un = wt * np.exp(-al * y * h[t]); new = un / un.sum()
    R.put("round%d_error" % (t + 1), eps); R.put("round%d_alpha" % (t + 1), al)
    for i in range(8):
        rows.append([t + 1, i + 1, wt[i], un[i], new[i]])
    alphas.append(al); wt = new
write_csv("outputs/part4_adaboost.csv", ["round", "point", "weight_before", "weight_unnormalised", "weight_after"], rows)
ens = np.sign(alphas[0] * h[0] + alphas[1] * h[1])
R.put("ensemble_accuracy", float((ens == y).mean()))
R.put("oob_fraction", (1 - 1 / 560) ** 560)

# architecture
def conv_out(i, f, p, s): return (i - f + 2 * p) // s + 1
m1, m2, m3 = 784 * 16 + 16, 16 * 16 + 16, 16 * 10 + 10
s1 = conv_out(28, 5, 2, 1); c1 = 5 * 5 * 1 * 6 + 6
s2 = conv_out(s1, 2, 0, 2)
s3 = conv_out(s2, 5, 0, 1); c2 = 5 * 5 * 6 * 16 + 6 * 0 + 16
s4 = conv_out(s3, 2, 0, 2)
flat = s4 * s4 * 16; d1 = flat * 32 + 32; d2 = 32 * 10 + 10
write_csv("outputs/part4_architecture.csv", ["layer", "parameters"],
          [["mlp_hidden1", m1], ["mlp_hidden2", m2], ["mlp_output", m3], ["cnn_conv1", c1],
           ["cnn_conv2", c2], ["cnn_dense1", d1], ["cnn_dense2", d2]])
R.put("mlp_total", m1 + m2 + m3); R.put("cnn_total", c1 + c2 + d1 + d2)
R.put("fc_layer_params", 128 * 128 * 3 * 1024 + 1024)
R.put("conv_layer_params", 3 * 3 * 3 * 64 + 64)
R.put("rnn_total", 128 * 64 + 128 * 128 + 16 * 128 + 128 + 16)
R.save("outputs/part4_results.txt")
print("shapes", s1, s2, s3, s4, flat)
