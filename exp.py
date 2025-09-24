import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility (optional)
np.random.seed(42)

# Parameters
n_samples = 1000
lambda_param = 10

# Generate exponentially distributed random numbers
# Note: numpy uses scale parameter (1/lambda) instead of rate parameter (lambda)
scale = 1 / lambda_param
exponential_samples = np.random.exponential(scale, n_samples)

# Calculate statistics
sample_mean = np.mean(exponential_samples)
sample_std = np.std(exponential_samples, ddof=1)  # Using sample standard deviation (n-1)

# Theoretical values for comparison
theoretical_mean = 1 / lambda_param
theoretical_std = 1 / lambda_param

# Print statistics
print(f"Exponential Distribution (λ = {lambda_param})")
print(f"Number of samples: {n_samples}")
print("-" * 40)
print(f"Sample mean:      {sample_mean:.4f}")
print(f"Theoretical mean: {theoretical_mean:.4f}")
print(f"Sample std dev:   {sample_std:.4f}")
print(f"Theoretical std:  {theoretical_std:.4f}")
print("-" * 40)
print(f"Min value: {np.min(exponential_samples):.4f}")
print(f"Max value: {np.max(exponential_samples):.4f}")

# Create histogram
plt.figure(figsize=(12, 5))

# Subplot 1: Histogram with theoretical PDF overlay
plt.subplot(1, 2, 1)
n_bins = 30
counts, bins, patches = plt.hist(exponential_samples, bins=n_bins, density=True, 
                                alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)

# Overlay theoretical exponential PDF
x_theory = np.linspace(0, np.max(exponential_samples), 1000)
y_theory = lambda_param * np.exp(-lambda_param * x_theory)
plt.plot(x_theory, y_theory, 'r-', linewidth=2, label=f'Theoretical PDF (λ={lambda_param})')

plt.xlabel('Value', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title(f'Histogram of {n_samples} Exponentially Distributed Random Numbers\n(λ = {lambda_param})', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Empirical CDF vs Theoretical CDF
plt.subplot(1, 2, 2)
sorted_samples = np.sort(exponential_samples)
empirical_cdf = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
theoretical_cdf = 1 - np.exp(-lambda_param * sorted_samples)

plt.plot(sorted_samples, empirical_cdf, 'b-', linewidth=1.5, label='Empirical CDF')
plt.plot(sorted_samples, theoretical_cdf, 'r--', linewidth=2, label='Theoretical CDF')
plt.xlabel('Value', fontsize=12)
plt.ylabel('Cumulative Probability', fontsize=12)
plt.title('Empirical vs Theoretical CDF', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional analysis: percentage of values in first few intervals
intervals = [0.1, 0.2, 0.3, 0.5]
print(f"\nDistribution analysis:")
for interval in intervals:
    percentage = (exponential_samples <= interval).mean() * 100
    theoretical_percentage = (1 - np.exp(-lambda_param * interval)) * 100
    print(f"Values ≤ {interval:.1f}: {percentage:.1f}% (theoretical: {theoretical_percentage:.1f}%)")
