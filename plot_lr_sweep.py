#!/usr/bin/env python3
"""Plot loss curves for LR sweep (500 steps each)."""
import re, matplotlib, matplotlib.pyplot as plt

matplotlib.use('Agg')  # Non-interactive backend

def parse_log(path):
    with open(path) as f:
        text = f.read()
    matches = re.findall(r'step=(\d+) loss=([0-9.]+)', text)
    return [(int(s), float(l)) for s, l in matches]

# Parse all three logs
data = {}
for lr in ['1e-4', '5e-5', '1e-5']:
    path = f'/home/joefox/workspace/tmp/recursivemas-original/logs/lr_sweep_long_{lr}.log'
    data[lr] = parse_log(path)

# Create plot
fig, ax = plt.subplots(figsize=(14, 8))

colors = {'1e-4': '#e74c3c', '5e-5': '#3498db', '1e-5': '#2ecc71'}
markers = {'1e-4': 'o', '5e-5': 's', '1e-5': '^'}

for lr, points in data.items():
    steps = [p[0] for p in points]
    losses = [p[1] for p in points]
    ax.plot(steps, losses, color=colors[lr], marker=markers[lr], markersize=4,
            linewidth=2, label=f'LR={lr} (min={min(losses):.4f}, final={losses[-1]:.4f})',
            alpha=0.8)

# Add moving average (window=10) for smoother view
window = 10
for lr, points in data.items():
    losses = [p[1] for p in points]
    steps = [p[0] for p in points]
    if len(losses) >= window:
        ma = [sum(losses[max(0, i-window+1):i+1]) / min(i+1, window) for i in range(len(losses))]
        ax.plot(steps, ma, color=colors[lr], linestyle='--', linewidth=1.5, alpha=0.4)

ax.set_xlabel('Training Step', fontsize=14)
ax.set_ylabel('Loss', fontsize=14)
ax.set_title('Outer Adapter Training: Loss vs Step (500 steps, Aligned Init)', fontsize=16)
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 520)

# Add min loss markers
for lr, points in data.items():
    losses = [p[1] for p in points]
    min_loss = min(losses)
    min_step = points[losses.index(min_loss)][0]
    ax.annotate(f'min={min_loss:.4f}', xy=(min_step, min_loss),
                xytext=(min_step+30, min_loss+0.05),
                fontsize=10, color=colors[lr],
                arrowprops=dict(arrowstyle='->', color=colors[lr], lw=1.5))

plt.tight_layout()
plt.savefig('/home/joefox/workspace/tmp/recursivemas-original/logs/lr_sweep_long.png', dpi=150)
print("Saved to /home/joefox/workspace/tmp/recursivemas-original/logs/lr_sweep_long.png")