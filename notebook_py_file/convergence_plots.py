import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Define your log file path
log_file = "../results/run_0511183535933370/logs/vein_grower_fl_128.txt"

rel_save_thresh = 0.0

# load errors
total_train_losses, total_val_losses, learning_rates = [], [], []
with open(log_file, "r") as f:
    for i, line in enumerate(f):
        if i == 0:
            continue
        line = line.split(",")
        total_train_losses.append(float(line[1]))
        total_val_losses.append(float(line[2]))
        learning_rates.append(float(line[3]))

# find where errors decreased
train_idx, train_loss, val_idx, val_loss = [], [], [], []
best_train, best_val = 1e12, 1e12
for i in range(len(total_train_losses)):
    rel_diff = best_train - total_train_losses[i]
    rel_diff /= best_train
    if rel_diff > rel_save_thresh:
        best_train = total_train_losses[i]
        train_idx.append(i)
        train_loss.append(best_train)
    rel_diff = best_val - total_val_losses[i]
    rel_diff /= best_val
    if rel_diff > rel_save_thresh:
        best_val = total_val_losses[i]
        val_idx.append(i)
        val_loss.append(best_val)
idx = np.argmin(val_loss)

best_epoch = val_idx[idx] + 1
best_loss = val_loss[idx]
print(f"Best epoch: {best_epoch}")
print(f"Best validation loss: {best_loss:.6f}")
print(f"Learning rate at best epoch: {learning_rates[best_epoch - 1]:.6e}")

# ===== FIGURE 1: Convergence + Improvements + Learning Rate =====
fig = plt.figure(figsize=(18, 5))

ax = fig.add_subplot(1, 3, 1)
plt.plot(total_train_losses, "b")
plt.plot(total_val_losses, "r")
plt.plot(val_idx[idx], val_loss[idx], "ko")
plt.legend([r"Train error", r"Val error", "Best model"])
plt.xlabel(r"Epochs")
plt.ylabel(r"Total Loss")
plt.title(r"Convergence")
plt.grid()

ax = fig.add_subplot(1, 3, 2)
plt.plot(train_idx, train_loss, "b.-")
plt.plot(val_idx, val_loss, "r.-")
plt.legend([r"Train error", r"Val error"])
plt.xlabel("Epochs")
plt.ylabel(r"Total Loss")
plt.title(r"Improvements")
plt.grid()

ax = fig.add_subplot(1, 3, 3)
plt.plot(learning_rates, "g", linewidth=2)
plt.xlabel("Epochs")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Evolution")
plt.grid()

plt.tight_layout(h_pad=2, w_pad=2)
plt.show()

# ===== FIGURE 2: Log-scaled errors =====
fig = plt.figure(figsize=(15, 5))

ax = fig.add_subplot(1, 2, 1)
plt.semilogy(total_train_losses, "b")
plt.semilogy(total_val_losses, "r")
plt.semilogy(val_idx[idx], val_loss[idx], "ko")
plt.legend([r"Train error", r"Val error", "Best model"])
plt.xlabel(r"Epochs")
plt.ylabel(r"Total Loss")
plt.title(r"Log Convergence")
plt.grid()

ax = fig.add_subplot(1, 2, 2)
plt.semilogy(train_idx, train_loss, "b.-")
plt.semilogy(val_idx, val_loss, "r.-")
plt.legend([r"Train error", r"Val error"])
plt.xlabel("Epochs")
plt.ylabel(r"Total Loss")
plt.title(r"Log Improvements")
plt.grid()

plt.tight_layout(h_pad=2, w_pad=2)
plt.show()
