# mnist-denoising-pca-vae
MNIST denoising project using PCA (standard + incremental) and VAE with white noise injection
├── README.md
├── requirements.txt
├── main.py (script principal)
├── denoising/
│   ├── __init__.py
│   ├── data_loader.py (charger MNIST)
│   ├── noise_injection.py (ajouter bruit blanc)
│   ├── pca_denoiser.py (PCA standard + incrémental)
│   ├── vae_denoiser.py (autoencodeur variationnel)
│   └── evaluation.py (MSE/PSNR/SSIM)
├── notebooks/
│   └── visualization.ipynb (visualiser avant/après)
└── results/ (sauvegarder les résultats)
