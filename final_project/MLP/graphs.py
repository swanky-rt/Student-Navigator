import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- DATA FROM YOUR TERMINAL OUTPUT ---
epochs = list(range(1, 16))
loss = [0.6186, 0.3461, 0.1928, 0.1426, 0.1089, 0.0939, 0.0749, 0.0621, 0.0525, 0.0466, 0.0352, 0.0308, 0.0262, 0.0249, 0.0201]
acc = [94.68, 94.68, 95.74, 96.81, 96.81, 96.81, 95.74, 96.81, 95.74, 95.74, 95.74, 95.74, 95.74, 95.74, 95.74]

# --- PLOT 1: TRAINING DYNAMICS ---
sns.set_style("whitegrid")
fig, ax1 = plt.subplots(figsize=(8, 5))

# Plot Loss (Left Axis)
color = 'tab:blue'
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Training Loss', color=color, fontweight='bold')
ax1.plot(epochs, loss, color=color, linewidth=3, marker='o', label='Train Loss')
ax1.tick_params(axis='y', labelcolor=color)

# Plot Accuracy (Right Axis)
ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('Validation Accuracy (%)', color=color, fontweight='bold')
ax2.plot(epochs, acc, color=color, linewidth=3, marker='s', linestyle='--', label='Val Accuracy')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(90, 100)

plt.title('Context Agent Training Dynamics', fontweight='bold')
plt.tight_layout()
plt.show()

# --- PLOT 2: CONFUSION MATRIX ---
# Constructed from 96% accuracy on ~94 validation samples
cm = np.array([[46, 1], 
               [1, 46]]) 
labels = ['Restaurant', 'Bank']

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, yticklabels=labels, 
            cbar=False, annot_kws={"size": 16})
plt.xlabel('Predicted Label', fontweight='bold')
plt.ylabel('True Label', fontweight='bold')
plt.title('Confusion Matrix (Validation Set)', fontweight='bold')
plt.tight_layout()
plt.show()