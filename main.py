"""
Main Script: MNIST Denoising Pipeline
Pipeline:
1. Load MNIST data
2. Inject white noise
3. Denoise using PCA (standard & incremental)
4. Denoise using VAE
5. Evaluate and compare
"""

import numpy as np
import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import custom modules
from denoising.data_loader import load_mnist_data, flatten_images, unflatten_images
from denoising.noise_injection import inject_gaussian_noise, visualize_noise_injection
from denoising.pca_denoiser import StandardPCADenoiser, IncrementalPCADenoiser
from denoising.vae_denoiser import VAEDenoiser
from denoising.evaluation import evaluate_denoising, compare_methods


def create_results_dir():
    """Create results directory"""
    os.makedirs('results', exist_ok=True)
    print("Created results directory")


def visualize_results(clean, noisy, denoised_dict, num_samples=8):
    """
    Visualize denoising results
    
    Args:
        clean: original images (N, H, W)
        noisy: noisy images (N, H, W)
        denoised_dict: {method_name: denoised_images}
        num_samples: number of samples to show
    """
    num_methods = len(denoised_dict) + 2  # +2 for clean and noisy
    fig, axes = plt.subplots(num_methods, num_samples, figsize=(16, 3*num_methods))
    
    # Row 0: Clean images
    for i in range(num_samples):
        axes[0, i].imshow(clean[i], cmap='gray')
        axes[0, i].set_title(f'Clean {i}')
        axes[0, i].axis('off')
    
    # Row 1: Noisy images
    for i in range(num_samples):
        axes[1, i].imshow(noisy[i], cmap='gray')
        axes[1, i].set_title(f'Noisy {i}')
        axes[1, i].axis('off')
    
    # Rows 2+: Denoised by different methods
    for row, (method_name, denoised) in enumerate(denoised_dict.items(), start=2):
        for i in range(num_samples):
            axes[row, i].imshow(denoised[i], cmap='gray')
            axes[row, i].set_title(f'{method_name} {i}')
            axes[row, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/denoising_comparison.png', dpi=100, bbox_inches='tight')
    print("Saved visualization to results/denoising_comparison.png")
    plt.close()


def main():
    """
    Main pipeline
    """
    print("="*80)
    print("MNIST DENOISING PIPELINE")
    print("="*80)
    
    # Create results directory
    create_results_dir()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # =================================================================
    # Step 1: Load MNIST Data
    # =================================================================
    print("\n" + "="*80)
    print("STEP 1: Loading MNIST Data")
    print("="*80)
    
    clean_images = load_mnist_data(train=True, num_samples=1000)
    clean_flat = flatten_images(clean_images)
    
    print(f"Clean images shape: {clean_images.shape}")
    print(f"Clean flattened shape: {clean_flat.shape}")
    
    # =================================================================
    # Step 2: Inject White Noise
    # =================================================================
    print("\n" + "="*80)
    print("STEP 2: Injecting White Gaussian Noise")
    print("="*80)
    
    noise_level = 0.15
    noisy_images = inject_gaussian_noise(clean_images, noise_level=noise_level, seed=42)
    noisy_flat = flatten_images(noisy_images)
    
    print(f"Noisy images shape: {noisy_images.shape}")
    print(f"Noisy flattened shape: {noisy_flat.shape}")
    
    # Visualize noise injection
    visualize_noise_injection(clean_images, noisy_images, num_samples=8)
    
    # =================================================================
    # Step 3: PCA Standard Denoising
    # =================================================================
    print("\n" + "="*80)
    print("STEP 3: PCA Standard Denoising")
    print("="*80)
    
    pca_standard = StandardPCADenoiser(n_components=100)
    denoised_pca_standard_flat = pca_standard.fit_denoise(noisy_flat, clean_flat)
    denoised_pca_standard = unflatten_images(denoised_pca_standard_flat)
    
    print(f"PCA Standard denoised shape: {denoised_pca_standard.shape}")
    
    # =================================================================
    # Step 4: PCA Incremental Denoising
    # =================================================================
    print("\n" + "="*80)
    print("STEP 4: PCA Incremental Denoising")
    print("="*80)
    
    pca_incremental = IncrementalPCADenoiser(n_components=100, batch_size=100)
    denoised_pca_incremental_flat = pca_incremental.fit_denoise(noisy_flat, clean_flat)
    denoised_pca_incremental = unflatten_images(denoised_pca_incremental_flat)
    
    print(f"PCA Incremental denoised shape: {denoised_pca_incremental.shape}")
    
    # =================================================================
    # Step 5: VAE Denoising
    # =================================================================
    print("\n" + "="*80)
    print("STEP 5: VAE Denoising")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    vae_denoiser = VAEDenoiser(
        input_dim=784,
        latent_dim=32,
        hidden_dim=256,
        learning_rate=1e-3,
        device=device
    )
    
    # Train VAE
    vae_denoiser.train(
        noisy_flat,
        clean_flat,
        epochs=20,
        batch_size=64,
        verbose=True
    )
    
    # Denoise with VAE
    denoised_vae_flat = vae_denoiser.denoise(noisy_flat, batch_size=64)
    denoised_vae = unflatten_images(denoised_vae_flat)
    
    print(f"VAE denoised shape: {denoised_vae.shape}")
    
    # Save VAE model
    vae_denoiser.save_model('results/vae_model.pth')
    
    # =================================================================
    # Step 6: Evaluation
    # =================================================================
    print("\n" + "="*80)
    print("STEP 6: Evaluation and Comparison")
    print("="*80)
    
    denoised_dict = {
        'PCA Standard': denoised_pca_standard,
        'PCA Incremental': denoised_pca_incremental,
        'VAE': denoised_vae
    }
    
    results = compare_methods(clean_images, noisy_images, denoised_dict)
    
    # =================================================================
    # Step 7: Visualization
    # =================================================================
    print("\n" + "="*80)
    print("STEP 7: Visualization")
    print("="*80)
    
    visualize_results(clean_images, noisy_images, denoised_dict, num_samples=8)
    
    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\nBest performing methods by metric:")
    
    # Find best by PSNR
    psnr_scores = {name: r['psnr_denoised']['mean'] for name, r in results.items()}
    best_psnr = max(psnr_scores, key=psnr_scores.get)
    print(f"Best PSNR: {best_psnr} ({psnr_scores[best_psnr]:.2f} dB)")
    
    # Find best by SSIM
    ssim_scores = {name: r['ssim_denoised']['mean'] for name, r in results.items()}
    best_ssim = max(ssim_scores, key=ssim_scores.get)
    print(f"Best SSIM: {best_ssim} ({ssim_scores[best_ssim]:.4f})")
    
    # Find best by MSE
    mse_scores = {name: r['mse_denoised']['mean'] for name, r in results.items()}
    best_mse = min(mse_scores, key=mse_scores.get)
    print(f"Best MSE: {best_mse} ({mse_scores[best_mse]:.6f})")
    
    print("\n" + "="*80)
    print("Pipeline completed successfully!")
    print(f"Results saved to results/ directory")
    print("="*80)


if __name__ == "__main__":
    main()
