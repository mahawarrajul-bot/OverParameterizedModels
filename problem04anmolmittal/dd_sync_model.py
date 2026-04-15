import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


print("Loading CIFAR-10")
(x_tr, y_tr), (x_te, y_te) = keras.datasets.cifar10.load_data()
y_tr = y_tr.ravel()
y_te = y_te.ravel()

# keep only cat=3, dog=5
train_mask = (y_tr == 3) | (y_tr == 5)
test_mask  = (y_te == 3) | (y_te == 5)

x_tr, y_tr = x_tr[train_mask], y_tr[train_mask]
x_te, y_te = x_te[test_mask],  y_te[test_mask]

# remap labels: cat→0, dog→1
y_tr = (y_tr == 5).astype(np.int32)
y_te = (y_te == 5).astype(np.int32)

# downsample images to 8×8 using tf
def resize_images(x):
    x = tf.image.rgb_to_grayscale(x)        # → (N, H, W, 1)
    x = tf.image.resize(x, [8, 8]).numpy()  # → (N, 8, 8, 1)
    return x.reshape(len(x), -1).astype(np.float32)  # → (N, 64)

d = 64

x_tr = resize_images(x_tr)
x_te = resize_images(x_te)

# take n=960 training samples
n = 960
x_tr, y_tr = x_tr[:n], y_tr[:n]

# standardise
scaler = StandardScaler()
x_tr   = scaler.fit_transform(x_tr)
x_te   = scaler.transform(x_te)

K = 2    # number of classes
d = 64   # input dimension (8×8)
print(f"Train: {x_tr.shape}, Test: {x_te.shape}")
print(f"Interpolation threshold at n·K = {n*K} parameters")

# ─────────────────────────────────────────────
# 2. One-hot labels
# ─────────────────────────────────────────────
def one_hot(y, K=2):
    oh = np.zeros((len(y), K), dtype=np.float32)
    oh[np.arange(len(y)), y] = 1.0
    return oh

Y_tr_oh = one_hot(y_tr, K)
Y_te_oh = one_hot(y_te, K)

# ─────────────────────────────────────────────
# 3. Build & train a single hidden layer net
#    for a given H (hidden units)
# ─────────────────────────────────────────────
def num_params(H, d=64, K=2):
    return (d + 1) * H + (H + 1) * K

def build_and_train(H, x_tr, y_tr_oh, x_te, y_te,
                    epochs=2000, lr=1e-3, seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = keras.Sequential([
        keras.layers.Input(shape=(d,)),
        keras.layers.Dense(H, activation='relu'),
        keras.layers.Dense(K, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss='mse',               # squared loss
        metrics=['accuracy']
    )

    model.fit(
        x_tr, y_tr_oh,
        epochs=epochs,
        batch_size=min(256, len(x_tr)),
        verbose=1
    )

    # 
    tr_sq  = model.evaluate(x_tr, y_tr_oh,  verbose=0)[0]
    te_sq  = model.evaluate(x_te, Y_te_oh,  verbose=0)[0]

    # ── zero-one loss ──
    tr_pred = np.argmax(model.predict(x_tr, verbose=0), axis=1)
    te_pred = np.argmax(model.predict(x_te, verbose=0), axis=1)
    tr_01   = np.mean(tr_pred != y_tr) * 100
    te_01   = np.mean(te_pred != y_te) * 100

    return tr_sq, te_sq, tr_01, te_01


threshold = n * K   # 1920 parameters

# generate H values that give params spread from ~50 to ~650000
# solve (d+1)*H + (H+1)*K = P  →  H = (P - K) / (d + 1 + K)
target_params = np.unique(np.round(
    np.concatenate([
        np.linspace(50,    threshold,    20),
        np.linspace(threshold, 650_000, 20)
    ])
).astype(int))

H_values = np.unique(np.maximum(1,
    ((target_params - K) / (d + 1 + K)).astype(int)
))

print(f"\nSweeping {len(H_values)} hidden sizes …")

param_counts = []
tr_sq_list, te_sq_list = [], []
tr_01_list, te_01_list = [], []

for i, H in enumerate(H_values):
    P = num_params(H)
    print(f"  H={H:5d}  params={P:7d}  ({i+1}/{len(H_values)})", end="\r")

    tr_sq, te_sq, tr_01, te_01 = build_and_train(H, x_tr, Y_tr_oh, x_te, y_te)

    param_counts.append(P)
    tr_sq_list.append(tr_sq)
    te_sq_list.append(te_sq)
    tr_01_list.append(tr_01)
    te_01_list.append(te_01)

print("\nDone!")

# ─────────────────────────────────────────────
# 5. Plot — simple matplotlib, no ax objects
# ─────────────────────────────────────────────
plt.figure(figsize=(7, 8))
plt.suptitle(f"Double Descent — FC Net on CIFAR-10\n"
             f"(n={n}, d={d}, K={K}, single hidden layer)", fontsize=11)

# ── top: zero-one loss ──
plt.subplot(2, 1, 1)
plt.plot(param_counts, te_01_list, 'o-', color='steelblue',
         label='Test', markersize=4)
plt.plot(param_counts, tr_01_list, 'o-', color='darkorange',
         label='Train', markersize=4)
plt.axvline(threshold, color='black', linestyle='--', linewidth=1.2)
plt.xscale('log')
plt.ylabel("Zero-one loss (%)")
plt.legend(fontsize=9)
plt.title("Zero-one loss")

# ── bottom: squared loss ──
plt.subplot(2, 1, 2)
plt.plot(param_counts, te_sq_list, 'o-', color='steelblue',
         label='Test', markersize=4)
plt.plot(param_counts, tr_sq_list, 'o-', color='darkorange',
         label='Train', markersize=4)
plt.axvline(threshold, color='black', linestyle='--', linewidth=1.2)
plt.xscale('log')
plt.ylabel("Squared loss")
plt.xlabel("Number of parameters/weights")
plt.legend(fontsize=9)
plt.title("Squared loss")

plt.tight_layout()
plt.savefig("dd_syn_model.png", dpi=150, bbox_inches='tight')
print("Saved to dd_syn_model.png")