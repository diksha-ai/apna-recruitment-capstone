"""Shared helpers (importing this file runs nothing)."""
import csv
import numpy as np


def load():
    with open("data/records.csv") as f:
        r = list(csv.reader(f))
    a = np.array([[float(x) for x in row[1:]] for row in r[1:]])
    return a[:, :7], a[:, 7].astype(int), a[:, 8]   # X1..X7, outcome, value


def fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, str):
        return v
    return "%.4f" % v


class Results:
    def __init__(self):
        self.lines = []

    def put(self, k, v):
        self.lines.append("%s = %s" % (k, fmt(v)))

    def save(self, path):
        with open(path, "w") as f:
            f.write("\n".join(self.lines) + "\n")


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(fmt(v) for v in r) + "\n")


def split_std():
    X, y, _ = load()
    tr, te = [], []
    for c in (0, 1):
        idx = np.where(y == c)[0]
        k = (7 * len(idx)) // 10
        tr += list(idx[:k]); te += list(idx[k:])
    tr = np.array(sorted(tr)); te = np.array(sorted(te))
    mu = X[tr].mean(axis=0); sd = X[tr].std(axis=0, ddof=1)
    return X[tr], y[tr], X[te], y[te], (X[tr] - mu) / sd, (X[te] - mu) / sd


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def bce(Xs, y, w, b):
    p = sigmoid(Xs @ w + b)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def train_logistic(Xs, y, eta, steps, record=()):
    w = np.zeros(Xs.shape[1]); b = 0.0; n = len(y)
    rec = {0: bce(Xs, y, w, b)} if 0 in record else {}
    for s in range(1, steps + 1):
        e = sigmoid(Xs @ w + b) - y
        w = w - eta * (Xs.T @ e) / n
        b = b - eta * e.sum() / n
        if s in record:
            rec[s] = bce(Xs, y, w, b)
    return w, b, rec


def confusion(pred, y):
    pred = np.asarray(pred).astype(bool); y = np.asarray(y).astype(bool)
    return (int((pred & y).sum()), int((pred & ~y).sum()),
            int((~pred & ~y).sum()), int((~pred & y).sum()))


def metrics(tp, fp, tn, fn):
    d = lambda a, b: a / b if b else 0.0
    acc = d(tp + tn, tp + fp + tn + fn)
    pr = d(tp, tp + fp); rc = d(tp, tp + fn)
    f1 = d(2 * pr * rc, pr + rc)
    return acc, pr, rc, f1, d(tn, tn + fp)


def knn_prob(ref, ref_y, query, K):
    """share of positive neighbours among K nearest (stable tie-break by file order)."""
    out = np.empty(len(query))
    for i, q in enumerate(query):
        d = ((ref - q) ** 2).sum(axis=1)
        o = np.argsort(d, kind="stable")[:K]
        out[i] = ref_y[o].mean()
    return out


def knn_pred(ref, ref_y, query, K):
    p = knn_prob(ref, ref_y, query, K)
    return p * K >= (K + 1) / 2 - 1e-9, p
