import numpy as np

N_IN = 120
N_H1 = 64
N_H2 = 32
N_H3 = 16
N_OUT = 7


def random_brain():
    s = 0.4
    return [
        np.random.randn(N_IN, N_H1).astype(np.float32) * s,
        np.random.randn(N_H1, N_H2).astype(np.float32) * s,
        np.random.randn(N_H2, N_H3).astype(np.float32) * s,
        np.random.randn(N_H3, N_OUT).astype(np.float32) * s,
    ]


def forward(weights, x):
    h = np.tanh(x @ weights[0])
    h = np.tanh(h @ weights[1])
    h = np.tanh(h @ weights[2])
    return np.tanh(h @ weights[3])


def batch_forward(all_weights, all_inputs):
    if len(all_weights) == 0:
        return np.zeros((0, N_OUT), dtype=np.float32)
    w0 = np.stack([w[0] for w in all_weights])
    w1 = np.stack([w[1] for w in all_weights])
    w2 = np.stack([w[2] for w in all_weights])
    w3 = np.stack([w[3] for w in all_weights])
    h = np.tanh(np.einsum('ni,nij->nj', all_inputs, w0))
    h = np.tanh(np.einsum('ni,nij->nj', h, w1))
    h = np.tanh(np.einsum('ni,nij->nj', h, w2))
    return np.tanh(np.einsum('ni,nij->nj', h, w3))


def mutate(weights, rate=0.08, noise=0.15):
    return [
        w + (np.random.rand(*w.shape) < rate) * np.random.randn(*w.shape).astype(np.float32) * noise
        for w in weights
    ]


def crossover(a, b):
    return [np.where(np.random.rand(*wa.shape) < 0.5, wa, wb) for wa, wb in zip(a, b)]


def brain_stats(weights):
    flat = np.concatenate([w.ravel() for w in weights])
    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "max": float(np.max(flat)),
        "min": float(np.min(flat)),
        "norm": float(np.linalg.norm(flat)),
    }
