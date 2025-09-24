import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility (optional)
np.random.seed(42)

# Generate pairs of uniform random numbers
n_points = 1000  # Number of points to generate
x = np.random.uniform(0, 1, n_points)  # Random numbers between 0 and 1
y = np.random.uniform(0, 1, n_points)  # Random numbers between 0 and 1

# Create the scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(x, y, alpha=0.6, s=20, c='blue', edgecolors='none')

# Add labels and title
plt.xlabel('X (Uniform Random)', fontsize=12)
plt.ylabel('Y (Uniform Random)', fontsize=12)
plt.title(f'Scatter Plot of {n_points} Pairs of Uniform Random Numbers', fontsize=14)

# Set equal aspect ratio and grid
plt.axis('equal')
plt.grid(True, alpha=0.3)

# Set axis limits to show the full [0,1] x [0,1] square
plt.xlim(0, 1)
plt.ylim(0, 1)

# Show the plot
plt.tight_layout()
plt.show()

# Optional: Print some statistics
print(f"Generated {n_points} random pairs")
print(f"X range: [{x.min():.3f}, {x.max():.3f}]")
print(f"Y range: [{y.min():.3f}, {y.max():.3f}]")
print(f"X mean: {x.mean():.3f}, X std: {x.std():.3f}")
print(f"Y mean: {y.mean():.3f}, Y std: {y.std():.3f}")
