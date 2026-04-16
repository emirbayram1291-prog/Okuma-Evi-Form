# Okuma-Evi-Form
Okuma evini tasarlarken izlediğim yolların bir kısmını burada paylaşıyorum
# ===============================
# LIBS
# ===============================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# -------------------------------
# PARAMETERS
# -------------------------------
width, depth = 25, 10
N = 1200                 # nokta sayısı
k_nn = 8                 # komşu sayısı

A, k, w = 1.8, 0.18, 1.0
cx, cy = width/2, depth/2

T = 0.15                 # crack start threshold
T_stop = 0.05            # stop threshold
step = 0.15              # adım boyu
max_len = 200            # max adım

branch_prob = 0.15       # dallanma olasılığı

# -------------------------------
# POINT CLOUD (uniform random; istersen Poisson yap)
# -------------------------------
pts = np.column_stack([
    np.random.rand(N)*width,
    np.random.rand(N)*depth
])

# -------------------------------
# FIELD
# -------------------------------
def field(p):
    dx = p[:,0]-cx
    dy = p[:,1]-cy
    r = np.sqrt(dx*dx + dy*dy) + 1e-6
    return A*np.exp(-k*r)*np.cos(w*r)

F = field(pts)

# -------------------------------
# NEIGHBORS (for gradient approx)
# -------------------------------
nbrs = NearestNeighbors(n_neighbors=k_nn, algorithm='kd_tree').fit(pts)
dists, idx = nbrs.kneighbors(pts)

def grad_at(i):
    # local plane fit: z = ax + by + c
    P = pts[idx[i]]
    Z = F[idx[i]]
    X = np.column_stack([P[:,0], P[:,1], np.ones(len(P))])
    # least squares
    a,b,c0 = np.linalg.lstsq(X, Z, rcond=None)[0]
    return np.array([a, b])

grads = np.array([grad_at(i) for i in range(N)])
gmag = np.linalg.norm(grads, axis=1)

# -------------------------------
# SEEDS
# -------------------------------
seed_ids = np.where(gmag > T)[0]

# -------------------------------
# TRACE CRACKS
# -------------------------------
def trace(seed_i, direction=1):
    path = []
    p = pts[seed_i].copy()
    g = grads[seed_i]
    # tangent (perpendicular to gradient)
    t = np.array([-g[1], g[0]])
    if np.linalg.norm(t) < 1e-6:
        return path
    t = t/np.linalg.norm(t) * direction

    for _ in range(max_len):
        path.append(p.copy())

        # nearest point index for local gradient
        _, ii = nbrs.kneighbors(p.reshape(1,-1), n_neighbors=1)
        i = ii[0][0]
        g = grads[i]
        if np.linalg.norm(g) < T_stop:
            break

        # update tangent with slight rotation towards new perpendicular
        t_new = np.array([-g[1], g[0]])
        if np.linalg.norm(t_new) > 1e-6:
            t_new = t_new/np.linalg.norm(t_new)
            t = 0.7*t + 0.3*t_new
            t = t/np.linalg.norm(t)

        # step
        p = p + step * t

        # bounds check
        if not (0 <= p[0] <= width and 0 <= p[1] <= depth):
            break

        # branching
        if np.random.rand() < branch_prob and np.linalg.norm(g) > 2*T:
            angle = np.deg2rad(np.random.choice([-40, 40]))
            R = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle),  np.cos(angle)]])
            t = (R @ t)
            t = t/np.linalg.norm(t)

    return np.array(path)

cracks = []
for sid in seed_ids[:200]:   # hepsini değil, seçerek
    c1 = trace(sid, 1)
    c2 = trace(sid, -1)
    if len(c1)>2: cracks.append(c1)
    if len(c2)>2: cracks.append(c2)

# -------------------------------
# PLOT
# -------------------------------
plt.figure(figsize=(10,4))
plt.scatter(pts[:,0], pts[:,1], s=2, alpha=0.2)

for c in cracks:
    plt.plot(c[:,0], c[:,1], linewidth=1)

plt.xlim(0, width); plt.ylim(0, depth)
plt.gca().set_aspect('equal', adjustable='box')
plt.title("Field-driven crack network (no grid)")
plt.show()
