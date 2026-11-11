import numpy as np
from common import *

Xtr, ytr, Xte, yte, Str, Ste = split_std()
R = Results()
R.put("train_rows", len(ytr)); R.put("train_positives", int(ytr.sum()))
R.put("test_rows", len(yte)); R.put("test_positives", int(yte.sum()))

w, b, rec = train_logistic(Str, ytr, 0.1, 1000, record=(0, 1, 10, 100, 1000))
for k in (0, 1, 10, 100, 1000):
    R.put("loss_epoch_%d" % k, rec[k])
for j in range(7):
    R.put("weight_X%d" % (j + 1), w[j])
R.put("bias", b)
R.put("test_bce", bce(Ste, yte, w, b))

for eta in (0.01, 0.1, 1.0):
    _, _, r = train_logistic(Str, ytr, eta, 200, record=(200,))
    R.put("loss_epoch_200_eta_%s" % eta, r[200])
for j in range(7):
    R.put("odds_ratio_X%d" % (j + 1), np.exp(w[j]))

pte = sigmoid(Ste @ w + b); ptr = sigmoid(Str @ w + b)
for tau in (0.5, 0.3):
    c = confusion(pte >= tau, yte)
    for nme, val in zip(("tp", "fp", "tn", "fn"), c):
        R.put("tau_%s_%s" % (tau, nme), val)
    for nme, val in zip(("accuracy", "precision", "recall", "f1", "specificity"), metrics(*c)):
        R.put("tau_%s_%s" % (tau, nme), val)
R.put("blind_accuracy", 1 - yte.mean())

rows = []
for k in range(1, 10):
    tau = k / 10
    tp, fp, tn, fn = confusion(pte >= tau - 1e-12, yte)
    rows.append([tau, tp, fp, tn, fn, tp / (tp + fn), fp / (fp + tn)])
write_csv("outputs/part2_thresholds.csv", ["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"], rows)

# 2.7 KNN CV
fold = np.arange(len(ytr)) % 5
rows = []
for K in range(1, 22, 2):
    f1s = []
    for f in range(5):
        ref = fold != f
        pred, _ = knn_pred(Str[ref], ytr[ref], Str[~ref], K)
        f1s.append(metrics(*confusion(pred, ytr[~ref]))[3])
    rows.append([K, float(np.mean(f1s))])
write_csv("outputs/part2_knn_cv.csv", ["K", "mean_cv_f1"], rows)
best = max(rows, key=lambda r: (round(r[1], 12), -r[0]))[0]
R.put("knn_best_k", best)
pred, _ = knn_pred(Str, ytr, Ste, best)
c = confusion(pred, yte)
for nme, val in zip(("tp", "fp", "tn", "fn"), c):
    R.put("knn_" + nme, val)
for nme, val in zip(("accuracy", "precision", "recall", "f1"), metrics(*c)[:4]):
    R.put("knn_" + nme, val)

# 2.8 cost
def cost(c): return 50 * c[1] + 400 * c[3]
costs = [(cost(confusion(ptr >= k / 20 - 1e-12, ytr)), k / 20) for k in range(1, 20)]
best_cost, tau = min(costs, key=lambda t: (t[0], t[1]))
R.put("cost_tau", tau); R.put("cost_train_at_tau", best_cost)
R.put("cost_test_at_tau", cost(confusion(pte >= tau - 1e-12, yte)))
R.put("cost_test_at_0.5", cost(confusion(pte >= 0.5, yte)))
R.put("cost_test_flag_nobody", 400 * int(yte.sum()))
R.save("outputs/part2_results.txt")
