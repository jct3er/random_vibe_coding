#!/usr/bin/env python3
"""
norm2.py - Correlated Normal Distributions Visualization

Generates pairs of normally distributed random numbers with specified correlation
and visualizes them as heatmaps.

X ~ N(0, 1)  - mean=0, std=1
Y ~ N(0, 2)  - mean=0, std=2
Correlation coefficients: ρ = 0.5, 0, 1, -1

Author: Generated with Claude
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import seaborn as sns

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("viridis")

def generate_correlated_normal_pairs(n_samples, mean1, std1, mean2, std2, correlation):
    """
    Generate pairs of correlated normally distributed random numbers.
    
    Parameters:
    -----------
    n_samples : int
        Number of random pairs to generate
    mean1, mean2 : float
        Means of the two distributions
    std1, std2 : float
        Standard deviations of the two distributions
    correlation : float
        Correlation coefficient ρ between -1 and 1
        
    Returns:
    --------
    x, y : numpy arrays
        Arrays of correlated random numbers
    """
    # Create covariance matrix
    cov_xy = correlation * std1 * std2
    covariance_matrix = np.array([[std1**2, cov_xy],
                                 [cov_xy, std2**2]])
    
    # Mean vector
    mean_vector = np.array([mean1, mean2])
    
    # Generate correlated samples
    samples = np.random.multivariate_normal(mean_vector, covariance_matrix, n_samples)
    
    return samples[:, 0], samples[:, 1]

def create_heatmap(x, y, correlation, ax, title=None):
    """
    Create a heatmap of the 2D distribution.
    
    Parameters:
    -----------
    x, y : numpy arrays
        Random number pairs
    correlation : float
        Correlation coefficient
    ax : matplotlib axis
        Subplot axis to plot on
    title : str, optional
        Custom title for the plot
    """
    # Create 2D histogram
    hist, xedges, yedges = np.histogram2d(x, y, bins=50, density=True)
    
    # Create meshgrid for plotting
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    
    # Create heatmap
    im = ax.pcolormesh(X, Y, hist.T, cmap='viridis', shading='auto')
    
    # Customize plot
    if title is None:
        title = f'ρ = {correlation}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X ~ N(0, 1)', fontsize=12)
    ax.set_ylabel('Y ~ N(0, 2)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('auto')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Density')
    
    # Add correlation info as text
    ax.text(0.02, 0.98, f'ρ = {correlation}\nn = 10,000', 
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='white', alpha=0.8))

def main():
    """
    Main function to generate and plot correlated normal distributions.
    """
    # Parameters
    n_samples = 10000
    mean1, std1 = 0, 1      # X ~ N(0, 1)
    mean2, std2 = 0, 2.0    # Y ~ N(0, 2)
    correlations = [0.5, 0, 1, -1]
    
    # Set up the figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Correlated Normal Distributions: X ~ N(0,1), Y ~ N(0,2)', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Flatten axes array for easier indexing
    axes_flat = axes.flatten()
    
    # Generate and plot for each correlation
    for i, rho in enumerate(correlations):
        print(f"Generating data for ρ = {rho}...")
        
        # Generate correlated random pairs
        x, y = generate_correlated_normal_pairs(n_samples, mean1, std1, mean2, std2, rho)
        
        # Verify actual correlation (for validation)
        actual_corr = np.corrcoef(x, y)[0, 1]
        print(f"  Theoretical ρ: {rho}, Actual ρ: {actual_corr:.4f}")
        
        # Create heatmap
        create_heatmap(x, y, rho, axes_flat[i])
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Add some statistics as text
    stats_text = (f"Sample size: {n_samples:,}\n"
                 f"X: μ={mean1}, σ={std1}\n" 
                 f"Y: μ={mean2}, σ={std2}")
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    print("Displaying plots...")
    plt.savefig("norm2_canvas.png")
    plt.show()

def verify_correlation_theory():
    """
    Optional function to verify the correlation theory with a simple example.
    """
    print("\n" + "="*60)
    print("CORRELATION VERIFICATION")
    print("="*60)
    
    n_test = 100000
    correlations_test = [0, 0.3, 0.7, 1.0, -0.5, -1.0]
    
    for rho in correlations_test:
        x, y = generate_correlated_normal_pairs(n_test, 0, 1, 0, 2, rho)
        actual_rho = np.corrcoef(x, y)[0, 1]
        error = abs(rho - actual_rho)
        print(f"Target ρ: {rho:5.1f}, Actual ρ: {actual_rho:7.4f}, Error: {error:.6f}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("Starting Correlated Normal Distributions Analysis...")
    print("="*60)
    
    # Run main analysis
    main()
    
    # Optional: Run verification
    verify_correlation_theory()