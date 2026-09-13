# %% [markdown]
# # A No-nonsense Introduction to AI — Companion Notebook
#
# This notebook reproduces every experiment in the article. It runs on a
# laptop CPU in under ten minutes. The running example is MNIST, chosen
# because handwritten digits are intuitive enough to inspect by eye but
# rich enough to expose training, overfitting, distribution shift, and
# adversarial examples.
#
# **Design principles.**
#
# 1. From scratch first, framework second. We implement logistic
#    regression and backpropagation in NumPy before switching to PyTorch.
# 2. Every section produces a plot. The plots double as the article's
#    figures.
# 3. One experiment changes one variable. Each result isolates its cause.
# 4. Show the model being wrong. Error galleries and adversarial examples
#    are as important as accuracy numbers.
#
# **How to run.** Execute cells in order. Total runtime is roughly eight
# minutes on a free Colab CPU. Seeds are fixed so results are
# reproducible.

# %%
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

np.random.seed(0)
plt.rcParams['figure.dpi'] = 110
plt.rcParams['figure.figsize'] = (7, 4)
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

print(f"numpy {np.__version__}")

# %% [markdown]
# ## Part 1 — Look at the data before modeling
#
# We load MNIST and inspect it. Before any model, we should know what the
# data looks like. The per-pixel mean and standard deviation already tell
# us something important: the border pixels carry almost no information.

# %%
from torchvision import datasets

train = datasets.MNIST(root='./data', train=True, download=True)
test  = datasets.MNIST(root='./data', train=False, download=True)

# Flatten to (N, 784) and normalize to [0, 1]
X_train = train.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
y_train = train.targets.numpy().astype(np.int64)
X_test  = test.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
y_test  = test.targets.numpy().astype(np.int64)

print(f"train: {X_train.shape}, test: {X_test.shape}")
print(f"label distribution (train): {np.bincount(y_train)}")

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# 25 sample digits
grid = np.zeros((5 * 28, 5 * 28))
for i in range(25):
    r, c = divmod(i, 5)
    grid[r*28:(r+1)*28, c*28:(c+1)*28] = X_train[i].reshape(28, 28)
axes[0].imshow(grid, cmap='gray')
axes[0].set_title('25 training digits')
axes[0].axis('off')

axes[1].imshow(X_train.mean(axis=0).reshape(28, 28), cmap='viridis')
axes[1].set_title('Per-pixel mean')
axes[1].axis('off')

axes[2].imshow(X_train.std(axis=0).reshape(28, 28), cmap='viridis')
axes[2].set_title('Per-pixel std. dev.')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('fig01_data.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# The mean image shows the average shape of a digit; the standard
# deviation shows which pixels vary most across examples. Note that the
# border pixels are dark in both — they carry almost no information. This
# is a preview of feature selection: not all pixels matter equally.

# %% [markdown]
# ## Part 2 — Baseline with no learning
#
# Before any model, establish what a trivial baseline achieves. We train
# a logistic regression with scikit-learn, which is fast and gives us a
# reference point.

# %%
from sklearn.linear_model import LogisticRegression

t0 = time.time()
baseline = LogisticRegression(max_iter=200, n_jobs=-1)
baseline.fit(X_train, y_train)
acc = baseline.score(X_test, y_test)
print(f"scikit-learn logistic regression: test acc = {acc:.4f} "
      f"({time.time()-t0:.1f}s)")

# %% [markdown]
# Around 92%. Every subsequent model should beat this by a meaningful
# margin. If a sophisticated model does not, the sophistication is not
# buying anything.

# %% [markdown]
# ## Part 3 — Logistic regression from scratch in NumPy
#
# This is the heart of the notebook. Every model in the rest of the
# article — convolutional networks, transformers, language models — is a
# refinement of the loop we write here in twenty lines.
#
# The model: `logits = X @ W + b`, then `softmax`.
# The loss: cross-entropy.
# The gradient: `softmax(logits) - onehot(y)`, backpropagated to `W`.

# %%
def softmax(z):
    """Numerically stable softmax along the last axis."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def loss_and_grad(W, b, X, y):
    """Cross-entropy loss and gradients for a linear classifier."""
    N = X.shape[0]
    logits = X @ W + b
    p = softmax(logits)
    loss = -np.log(p[np.arange(N), y] + 1e-12).mean()
    # Gradient of loss w.r.t. logits is (p - onehot(y)) / N
    P = p.copy()
    P[np.arange(N), y] -= 1
    gW = X.T @ P / N
    gb = P.mean(axis=0)
    return loss, gW, gb


def accuracy(W, b, X, y):
    logits = X @ W + b
    return (logits.argmax(axis=1) == y).mean()

# %%
# Training loop
W = np.zeros((784, 10), dtype=np.float32)
b = np.zeros(10, dtype=np.float32)
lr = 0.5
n_steps = 500

losses = []
for step in range(n_steps):
    loss, gW, gb = loss_and_grad(W, b, X_train, y_train)
    W -= lr * gW
    b -= lr * gb
    losses.append(loss)
    if (step + 1) % 100 == 0:
        acc = accuracy(W, b, X_test, y_test)
        print(f"step {step+1:4d}  loss {loss:.4f}  test acc {acc:.4f}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(losses)
axes[0].set_xlabel('step')
axes[0].set_ylabel('training loss')
axes[0].set_title('Training loss (logistic regression)')
axes[0].set_yscale('log')

# Learned weight images: each column of W is a 28x28 template
grid = np.zeros((2 * 28, 5 * 28))
for i in range(10):
    r, c = divmod(i, 5)
    grid[r*28:(r+1)*28, c*28:(c+1)*28] = W[:, i].reshape(28, 28)
axes[1].imshow(grid, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
axes[1].set_title('Learned weights (one per digit class)')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('fig02_logreg.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# Each column of `W` is a learned template for one digit. Positive
# (red) pixels push the input toward that class; negative (blue) pixels
# push away. The templates are interpretable: the "0" template looks
# roughly like a 0.
#
# **This is what learning is.** Gradient descent nudged 7,850 numbers
# until the templates classified digits well. There is no rule about
# which pixels matter; the rules emerged from the data.

# %% [markdown]
# ## Part 4 — Break it on purpose
#
# Three experiments, each changing one variable. This is where the
# reader builds intuition for what can go wrong.

# %% [markdown]
# ### 4.1 — Learning-rate sweep

# %%
def train_logreg(lr, n_steps=500, X=X_train, y=y_train):
    W = np.zeros((784, 10), dtype=np.float32)
    b = np.zeros(10, dtype=np.float32)
    losses = []
    for _ in range(n_steps):
        loss, gW, gb = loss_and_grad(W, b, X, y)
        W -= lr * gW
        b -= lr * gb
        losses.append(loss)
    return losses

fig, ax = plt.subplots()
for lr in [0.001, 0.5, 50.0]:
    losses = train_logreg(lr)
    ax.plot(losses, label=f'lr = {lr}')
ax.set_xlabel('step')
ax.set_ylabel('training loss')
ax.set_yscale('log')
ax.set_title('Learning-rate sweep')
ax.legend()
plt.savefig('fig03_lr_sweep.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# - `lr = 0.001`: training is slow — loss barely moves in 500 steps.
# - `lr = 0.5`: stable convergence.
# - `lr = 50.0`: diverges — loss oscillates and grows.
#
# This is why the learning rate is the single most important
# hyperparameter.

# %% [markdown]
# ### 4.2 — No normalization

# %%
X_train_raw = X_train * 255.0
X_test_raw  = X_test * 255.0

losses_raw = train_logreg(0.5, n_steps=500, X=X_train_raw, y=y_train)
losses_norm = train_logreg(0.5, n_steps=500)

fig, ax = plt.subplots()
ax.plot(losses_raw, label='raw pixels [0, 255]')
ax.plot(losses_norm, label='normalized [0, 1]')
ax.set_xlabel('step')
ax.set_ylabel('training loss')
ax.set_yscale('log')
ax.set_title('Effect of input normalization')
ax.legend()
plt.savefig('fig04_normalization.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# Same learning rate, same number of steps, same architecture. Only the
# input scale differs. Unnormalized inputs make the loss surface
# ill-conditioned and training stalls. Normalization is not a detail;
# it is the difference between working and not working.

# %% [markdown]
# ### 4.3 — Too few steps vs. too many

# %%
fig, ax = plt.subplots()
for n in [10, 100, 2000]:
    losses = train_logreg(0.5, n_steps=n)
    ax.plot(losses, label=f'{n} steps')

# We need a longer curve for the "2000" label to show fully
losses_long = train_logreg(0.5, n_steps=2000)
ax.clear()
for n, c in zip([10, 100, 2000], ['C0', 'C1', 'C2']):
    ax.plot(losses_long[:n], color=c, label=f'{n} steps')
ax.set_xlabel('step')
ax.set_ylabel('training loss')
ax.set_yscale('log')
ax.set_title('Underfitting vs. converged')
ax.legend()
plt.savefig('fig05_steps.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Part 5 — Why nonlinearity matters
#
# We add one hidden layer to the from-scratch model. Accuracy jumps.
# Then we remove the activation function and show the network collapses
# back to a linear model.

# %%
def init_mlp(hidden=128):
    rng = np.random.default_rng(0)
    # He initialization
    W1 = rng.normal(0, np.sqrt(2/784), (784, hidden)).astype(np.float32)
    b1 = np.zeros(hidden, dtype=np.float32)
    W2 = rng.normal(0, np.sqrt(2/hidden), (hidden, 10)).astype(np.float32)
    b2 = np.zeros(10, dtype=np.float32)
    return W1, b1, W2, b2


def mlp_forward(params, X, activation='relu'):
    W1, b1, W2, b2 = params
    H = X @ W1 + b1
    if activation == 'relu':
        A = np.maximum(H, 0)
    elif activation == 'linear':
        A = H
    logits = A @ W2 + b2
    return H, A, logits


def mlp_loss_and_grad(params, X, y, activation='relu'):
    W1, b1, W2, b2 = params
    N = X.shape[0]
    H, A, logits = mlp_forward(params, X, activation)
    p = softmax(logits)
    loss = -np.log(p[np.arange(N), y] + 1e-12).mean()

    # Backward pass
    dlogits = p.copy()
    dlogits[np.arange(N), y] -= 1
    dlogits /= N
    gW2 = A.T @ dlogits
    gb2 = dlogits.sum(axis=0)
    dA = dlogits @ W2.T
    if activation == 'relu':
        dH = dA * (H > 0)
    else:
        dH = dA
    gW1 = X.T @ dH
    gb1 = dH.sum(axis=0)
    return loss, (gW1, gb1, gW2, gb2)


def train_mlp(activation='relu', n_steps=2000, lr=0.5, batch_size=128):
    params = init_mlp()
    losses = []
    rng = np.random.default_rng(0)
    for step in range(n_steps):
        idx = rng.choice(len(X_train), batch_size, replace=False)
        loss, grads = mlp_loss_and_grad(params, X_train[idx], y_train[idx], activation)
        for p, g in zip(params, grads):
            p -= lr * g
        losses.append(loss)
    return params, losses

# %%
print("Training MLP with ReLU (2000 steps, minibatch 128)...")
t0 = time.time()
params_relu, losses_relu = train_mlp('relu', n_steps=2000)
acc_relu = accuracy(params_relu[2], params_relu[3],
                    np.maximum(X_test @ params_relu[0] + params_relu[1], 0), y_test)
print(f"  MLP + ReLU:      test acc = {acc_relu:.4f}  ({time.time()-t0:.1f}s)")

print("Training MLP with no activation (linear)...")
t0 = time.time()
params_lin, losses_lin = train_mlp('linear', n_steps=2000)
_, A_test, logits = mlp_forward(params_lin, X_test, 'linear')
acc_lin = (logits.argmax(axis=1) == y_test).mean()
print(f"  MLP + no act.:   test acc = {acc_lin:.4f}  ({time.time()-t0:.1f}s)")

# %%
fig, ax = plt.subplots()
ax.plot(losses_relu, label='ReLU')
ax.plot(losses_lin, label='no activation (linear)')
ax.set_xlabel('step')
ax.set_ylabel('training loss (minibatch)')
ax.set_yscale('log')
ax.set_title('A single nonlinearity changes everything')
ax.legend()
plt.savefig('fig06_activation.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# Removing the activation drops accuracy back to the logistic regression
# baseline. This is the essential fact about deep networks: **without
# nonlinearity, depth is an illusion.** A composition of linear functions
# is itself linear.

# %% [markdown]
# ## Part 6 — Switch to PyTorch
#
# Same architecture, expressed in a framework. Compare the code length.
# Then build a small CNN and show the accuracy jump.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device: {device}")

# Convert to torch tensors
Xtr = torch.tensor(X_train).to(device)
ytr = torch.tensor(y_train).to(device)
Xte = torch.tensor(X_test).to(device)
yte = torch.tensor(y_test).to(device)

train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=128, shuffle=True)

# %%
class MLP(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, 10)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def train_torch(model, loader, n_epochs=5, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(n_epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
    return losses

t0 = time.time()
mlp = MLP().to(device)
losses_torch = train_torch(mlp, train_loader, n_epochs=5)
with torch.no_grad():
    acc = (mlp(Xte).argmax(1) == yte).float().mean().item()
print(f"PyTorch MLP: test acc = {acc:.4f}  ({time.time()-t0:.1f}s)")

# %%
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        return self.fc(x)

# Need to rebuild the loader for image-shaped input
Xtr_img = Xtr.view(-1, 1, 28, 28)
Xte_img = Xte.view(-1, 1, 28, 28)
train_loader_img = DataLoader(TensorDataset(Xtr_img, ytr), batch_size=128, shuffle=True)

t0 = time.time()
cnn = SmallCNN().to(device)
losses_cnn = train_torch(cnn, train_loader_img, n_epochs=3)
with torch.no_grad():
    acc_cnn = (cnn(Xte_img).argmax(1) == yte).float().mean().item()
print(f"Small CNN: test acc = {acc_cnn:.4f}  ({time.time()-t0:.1f}s)")

# %%
fig, ax = plt.subplots()
ax.plot(losses_torch, label='MLP')
ax.plot(np.linspace(0, len(losses_torch), len(losses_cnn)), losses_cnn, label='CNN')
ax.set_xlabel('step')
ax.set_ylabel('loss')
ax.set_title('PyTorch models: MLP vs CNN')
ax.legend()
plt.savefig('fig07_pytorch.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Part 7 — See what the network learned

# %% [markdown]
# ### 7.1 — What the first convolutional layer detects

# %%
filters = cnn.conv1.weight.detach().cpu()
fig, axes = plt.subplots(2, 8, figsize=(12, 3.5))
for i, ax in enumerate(axes.flat):
    ax.imshow(filters[i, 0], cmap='RdBu_r', vmin=-1, vmax=1)
    ax.axis('off')
fig.suptitle('First-layer CNN filters (16 of them)')
plt.tight_layout()
plt.savefig('fig08_filters.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 7.2 — Embeddings cluster by class

# %%
# Extract penultimate features for 2000 test digits
with torch.no_grad():
    idx = np.random.choice(len(Xte_img), 2000, replace=False)
    x_sample = Xte_img[idx]
    # MLP features
    mlp_features = F.relu(mlp.fc1(x_sample.view(-1, 784))).cpu().numpy()
    y_sample = yte[idx].cpu().numpy()

from sklearn.decomposition import PCA
proj = PCA(n_components=2).fit_transform(mlp_features)

fig, ax = plt.subplots(figsize=(7, 6))
sc = ax.scatter(proj[:, 0], proj[:, 1], c=y_sample, cmap='tab10', s=4)
plt.colorbar(sc, label='digit')
ax.set_title('PCA of penultimate-layer features (MLP)')
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
plt.savefig('fig09_embeddings.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# Digits cluster by class without any supervision at the clustering
# level. The model was only told which digit each image represents; the
# geometry of the representation — that 4s and 9s live near each other,
# that 1s form a tight group — emerged from training. This is what
# representation learning means.

# %% [markdown]
# ## Part 8 — Where it fails

# %% [markdown]
# ### 8.1 — Confusion matrix

# %%
from sklearn.metrics import confusion_matrix

with torch.no_grad():
    preds = mlp(Xte).argmax(1).cpu().numpy()
cm = confusion_matrix(y_test, preds)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')
plt.colorbar(im, label='count')
ax.set_xlabel('predicted')
ax.set_ylabel('true')
ax.set_xticks(range(10)); ax.set_xticklabels(range(10))
ax.set_yticks(range(10)); ax.set_yticklabels(range(10))
for i in range(10):
    for j in range(10):
        if cm[i, j] > 20:
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > 200 else 'black', fontsize=8)
ax.set_title('Confusion matrix')
plt.savefig('fig10_confusion.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# The 4/9 confusion is the largest off-diagonal entry. These are
# genuinely ambiguous in some handwriting.

# %% [markdown]
# ### 8.2 — Error gallery: most confidently wrong

# %%
with torch.no_grad():
    probs = F.softmax(mlp(Xte), dim=1).cpu().numpy()

preds = probs.argmax(1)
wrong = preds != y_test
confidence = probs[np.arange(len(preds)), preds]
confidence[~wrong] = -1  # only look at wrong predictions

top_wrong = np.argsort(-confidence)[:20]

fig, axes = plt.subplots(2, 10, figsize=(14, 3.5))
for ax, idx in zip(axes.flat, top_wrong):
    ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    ax.set_title(f'true {y_test[idx]}\npred {preds[idx]} ({probs[idx, preds[idx]]:.2f})',
                 fontsize=8)
    ax.axis('off')
fig.suptitle('Twenty most confidently wrong predictions')
plt.tight_layout()
plt.savefig('fig11_errors.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# Several of these are genuinely ambiguous. Some are not. The model is
# confident and wrong — a failure mode that is more consequential in
# deployment than aggregate accuracy suggests.

# %% [markdown]
# ### 8.3 — Calibration

# %%
# Bucket predicted confidence and compare to actual accuracy
bins = np.linspace(0, 1, 11)
bin_accs = []
bin_confs = []
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (probs.max(1) >= lo) & (probs.max(1) < hi)
    if mask.sum() > 0:
        bin_accs.append((preds[mask] == y_test[mask]).mean())
        bin_confs.append(probs.max(1)[mask].mean())

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot([0, 1], [0, 1], 'k--', label='perfect calibration')
ax.plot(bin_confs, bin_accs, 'o-', label='MLP')
ax.set_xlabel('predicted confidence')
ax.set_ylabel('actual accuracy')
ax.set_title('Calibration plot')
ax.legend()
plt.savefig('fig12_calibration.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# The model is overconfident: predictions made with 95% confidence are
# correct less than 95% of the time. This matters anywhere the
# probability itself is used as a decision input.

# %% [markdown]
# ## Part 9 — Distribution shift and adversarial examples
#
# Four experiments, each a distinct way the model can fail on inputs that
# a human finds trivial.

# %% [markdown]
# ### 9.1 — Rotation

# %%
from scipy.ndimage import rotate

def rotate_dataset(X, angle):
    out = np.zeros_like(X)
    for i, x in enumerate(X):
        out[i] = rotate(x.reshape(28, 28), angle, reshape=False).flatten()
    return out

X_rot = rotate_dataset(X_test, 30)
with torch.no_grad():
    acc_rot = (mlp(torch.tensor(X_rot).to(device)).argmax(1) == yte).float().mean().item()
print(f"accuracy on 30°-rotated test: {acc_rot:.4f}")
print(f"accuracy on original test:    {acc:.4f}")

# %%
fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for ax, i in zip(axes, range(5)):
    ax.imshow(X_rot[i].reshape(28, 28), cmap='gray')
    ax.set_title(f'true {y_test[i]}\npred {mlp(torch.tensor(X_rot[i:i+1]).to(device)).argmax(1).item()}')
    ax.axis('off')
fig.suptitle('Model on rotated digits')
plt.tight_layout()
plt.savefig('fig13_rotation.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 9.2 — Gaussian noise

# %%
noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5]
accs = []
for sigma in noise_levels:
    X_noisy = np.clip(X_test + sigma * np.random.randn(*X_test.shape), 0, 1).astype(np.float32)
    with torch.no_grad():
        a = (mlp(torch.tensor(X_noisy).to(device)).argmax(1) == yte).float().mean().item()
    accs.append(a)

fig, ax = plt.subplots()
ax.plot(noise_levels, accs, 'o-')
ax.set_xlabel('noise std. dev.')
ax.set_ylabel('test accuracy')
ax.set_title('Accuracy under Gaussian noise')
plt.savefig('fig14_noise.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 9.3 — FGSM adversarial example

# %%
def fgsm(model, x, y, epsilon):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    return (x + epsilon * x.grad.sign()).clamp(0, 1).detach()

epsilon = 0.15
idx = 42
x_orig = Xte[idx:idx+1]
y_true = yte[idx:idx+1]

x_adv = fgsm(mlp, x_orig, y_true, epsilon)
with torch.no_grad():
    p_orig = F.softmax(mlp(x_orig), 1)[0].cpu().numpy()
    p_adv  = F.softmax(mlp(x_adv),  1)[0].cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
axes[0].imshow(x_orig[0].cpu().numpy().reshape(28, 28), cmap='gray')
axes[0].set_title(f'original\ntrue {y_true.item()}, pred {p_orig.argmax()} ({p_orig.max():.2f})')
axes[0].axis('off')

axes[1].imshow((x_adv - x_orig)[0].cpu().numpy().reshape(28, 28), cmap='RdBu_r', vmin=-0.2, vmax=0.2)
axes[1].set_title(f'perturbation\nε = {epsilon}')
axes[1].axis('off')

axes[2].imshow(x_adv[0].cpu().numpy().reshape(28, 28), cmap='gray')
axes[2].set_title(f'adversarial\npred {p_adv.argmax()} ({p_adv.max():.2f})')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('fig15_fgsm.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# The perturbation is invisible to a human reader. The model
# confidently misclassifies the digit. This is what the article means
# when it says the model's decision boundary is not aligned with the
# human concept: the model is correct on easy cases for the wrong
# reasons, and we can prove it by manipulating the reasons.

# %% [markdown]
# ## Part 10 — Overfitting lab
#
# Train on 100 examples. Watch training and validation loss diverge.
# Then apply regularization one technique at a time.

# %%
# Split off a small training set and a validation set
rng = np.random.default_rng(0)
small_idx = rng.choice(len(X_train), 100, replace=False)
X_small = torch.tensor(X_train[small_idx]).to(device)
y_small = torch.tensor(y_train[small_idx]).to(device)

# Validation: 2000 test examples
X_val = Xte[:2000]; y_val = yte[:2000]

small_loader = DataLoader(TensorDataset(X_small, y_small), batch_size=32, shuffle=True)

# %%
def train_with_reg(model, loader, X_val, y_val, n_epochs=100, lr=1e-3, weight_decay=0.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_losses, val_losses = [], []
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(xb)
        train_losses.append(epoch_loss / len(loader.dataset))
        model.eval()
        with torch.no_grad():
            val_losses.append(F.cross_entropy(model(X_val), y_val).item())
    return train_losses, val_losses

# %%
fig, ax = plt.subplots(figsize=(7, 4))
for wd, label in zip([0.0, 1e-3], ['no regularization', 'weight decay 1e-3']):
    torch.manual_seed(0)
    m = MLP(hidden=256).to(device)
    tr, vl = train_with_reg(m, small_loader, X_val, y_val, n_epochs=100, weight_decay=wd)
    ax.plot(tr, label=f'train, {label}', linestyle='--')
    ax.plot(vl, label=f'val, {label}')
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
ax.set_title('Overfitting on 100 examples (with and without weight decay)')
ax.legend(fontsize=8)
plt.savefig('fig16_overfit.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# Training loss keeps falling. Validation loss falls, then rises. The
# gap is overfitting, and it is visible within a hundred epochs on 100
# examples. Weight decay delays but does not eliminate it.

# %% [markdown]
# ## Part 11 — Generation (brief)
#
# The same machinery that classifies can also generate. A tiny
# autoencoder is enough to make the point: the model learns a compact
# representation, and samples from that representation look like digits.

# %%
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(784, 64), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(64, 784), nn.Sigmoid())
    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z

ae = Autoencoder().to(device)
opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
for epoch in range(10):
    for xb, _ in train_loader:
        opt.zero_grad()
        xh, _ = ae(xb)
        loss = F.mse_loss(xh, xb)
        loss.backward()
        opt.step()

# Sample from the latent distribution
with torch.no_grad():
    ae.eval()
    # Encode a batch, compute mean and std, sample from a Gaussian
    _, z_all = ae(Xte[:1000])
    z_mean, z_std = z_all.mean(0), z_all.std(0)
    samples = torch.randn(20, 64).to(device) * z_std + z_mean
    generated = ae.dec(samples).cpu().numpy()

fig, axes = plt.subplots(2, 10, figsize=(14, 3))
for ax, img in zip(axes.flat, generated):
    ax.imshow(img.reshape(28, 28), cmap='gray')
    ax.axis('off')
fig.suptitle('Samples generated from latent space')
plt.tight_layout()
plt.savefig('fig17_generation.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Part 12 — What MNIST does not teach
#
# MNIST is images. The dominant modern application of AI is text. The
# training loop is identical; the data pipeline is not. Below, a
# character-level MLP with a context window of five characters learns to
# predict the next character of Shakespeare. Same loss, same optimizer,
# different representation.

# %%
import urllib.request

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not Path("shakespeare.txt").exists():
    urllib.request.urlretrieve(URL, "shakespeare.txt")
text = Path("shakespeare.txt").read_text()[:200_000]  # first 200 KB

chars = sorted(set(text))
c2i = {c: i for i, c in enumerate(chars)}
i2c = {i: c for c, i in c2i.items()}
vocab_size = len(chars)
print(f"corpus: {len(text):,} chars, vocab: {vocab_size}")

# %%
context = 5
seqs = np.array([[c2i[c] for c in text[i:i+context]]
                 for i in range(len(text) - context - 1)])
targets = np.array([c2i[text[i+context]] for i in range(len(text) - context - 1)])
print(f"sequences: {seqs.shape}")

# %%
def one_hot(idx, n):
    out = np.zeros((len(idx), n), dtype=np.float32)
    out[np.arange(len(idx)), idx] = 1.0
    return out

Xc = one_hot(seqs.reshape(-1), vocab_size).reshape(len(seqs), -1)
yc = targets

# %%
class CharMLP(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(context * vocab_size, hidden)
        self.fc2 = nn.Linear(hidden, vocab_size)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

Xc_t = torch.tensor(Xc).to(device)
yc_t = torch.tensor(yc).to(device)
char_loader = DataLoader(TensorDataset(Xc_t, yc_t), batch_size=256, shuffle=True)

char_model = CharMLP().to(device)
opt = torch.optim.Adam(char_model.parameters(), lr=1e-3)
t0 = time.time()
for epoch in range(3):
    total, n = 0, 0
    for xb, yb in char_loader:
        opt.zero_grad()
        loss = F.cross_entropy(char_model(xb), yb)
        loss.backward()
        opt.step()
        total += loss.item() * len(xb); n += len(xb)
    print(f"epoch {epoch+1}: loss = {total/n:.4f}")
print(f"training time: {time.time()-t0:.1f}s")

# %%
def generate(seed, n_chars=200, temperature=0.8):
    out = seed
    for _ in range(n_chars):
        ctx = out[-context:]
        x = torch.tensor([[c2i[c] for c in ctx]]).reshape(1, -1)
        x_oh = torch.zeros(1, context * vocab_size).to(device)
        x_oh[0] = torch.tensor(one_hot(np.array([[c2i[c] for c in ctx]]).reshape(-1), vocab_size).reshape(-1)).to(device)
        with torch.no_grad():
            logits = char_model(x_oh) / temperature
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        next_char = i2c[np.random.choice(len(probs), p=probs)]
        out += next_char
    return out

print(generate("ROMEO: "))
print()
print(generate("To be or "))

# %% [markdown]
# The output is recognisably English-shaped but semantically empty.
# Local fluency, no global meaning. That gap — between predicting the
# next character well and understanding what the text is about — is
# where the discussion of large language models begins, and where this
# notebook ends.
#
# The training loop, loss, and optimizer are identical to the digit
# classifier. Only the representation of the data changed. This is the
# single most important lesson from this notebook: modern AI is one
# mechanism applied to many representations.

# %% [markdown]
# ## Summary
#
# This notebook has reproduced, in miniature, the core experiments of
# the article:
#
# | Section | Concept | Notebook part |
# |---|---|---|
# | 4 | Foundations | Parts 1–3 |
# | 5 | How learning works | Parts 3–5 |
# | 6 | Model families | Parts 6, 11 |
# | 7 | Mechanics | Parts 5, 6, 12 |
# | 8 | Lifecycle | Parts 4, 9 |
# | 9 | Limits | Parts 8, 9, 10 |
#
# The experiments are designed to be modified. Change the learning rate.
# Remove the activation. Rotate the digits by 90 degrees. Train on 10
# examples. The point of the notebook is not the numbers it produces,
# but the intuition it builds for what can go wrong.