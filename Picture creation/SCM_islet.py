import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

np.random.seed(0)

# Generate concentric circles data
n_points = 40

# Inner circle (class 0)
r_inner = 1
theta_inner = 2 * np.pi * np.random.rand(n_points)
X_inner = np.c_[r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)]

# Outer circle (class 1)
r_outer = 3
theta_outer = 2 * np.pi * np.random.rand(n_points)
X_outer = np.c_[r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)]

# Stack data
X = np.vstack((X_inner, X_outer))
y = np.hstack((np.zeros(n_points), np.ones(n_points)))

# Define classifiers
clf_linear = svm.SVC(kernel='linear', C=1)
clf_poly = svm.SVC(kernel='poly', degree=10, C=1)

# Fit classifiers
clf_linear.fit(X, y)
clf_poly.fit(X, y)

# Create meshgrid for decision function
xx, yy = np.meshgrid(np.linspace(-4, 4, 500),
                     np.linspace(-4, 4, 500))
XY = np.c_[xx.ravel(), yy.ravel()]

# Get decision function values
Z_linear = clf_linear.decision_function(XY).reshape(xx.shape)
Z_poly = clf_poly.decision_function(XY).reshape(xx.shape)

# Plot both classifiers
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

titles = ['Linear Kernel', 'Polynomial Kernel (degree=10)']
Zs = [Z_linear, Z_poly]

for ax, Z, title in zip(axes, Zs, titles):
    ax.contourf(xx, yy, Z > 0, alpha=0.2, cmap=plt.cm.coolwarm)
    ax.contour(xx, yy, Z, colors=['k', 'k', 'k'], levels=[-1, 0, 1],
               linestyles=['--', '-', '--'])

    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=40, edgecolors='k')

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

    lines = [plt.Line2D([0], [0], color='k', linestyle='-', label='Decision Boundary'),
             plt.Line2D([0], [0], color='k', linestyle='--', label='Margins')]
    ax.legend(handles=lines, loc='upper right')

plt.tight_layout()
plt.show()
