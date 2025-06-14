# A Multi-branch anomalous sound detection model

## Date: 2025-06-08
## Model Structure:
Simplified Anomalous Sound Detection model for DCASE2025 Task 2
================================================================

Architecture
------------
Raw Audio → Log‑Mel Spectrogram → **Fine‑tuned AST Encoder** → latent **z**  
                                                               ↘            
                                                       **Decoder** → Reconstructed Spectrogram  

Anomaly scoring:
* **Whitened latent distance** normalises each dimension by its own standard deviation before computing the L2 norm.
* **Reconstruction MSE** measures low‑level signal mismatch and is scaled via MAD.
* Combined score `S = 0.7 · M(z) + 0.3 · E(x,\hat x)`.

This single‑branch model keeps the strengths of discriminative and generative
approaches while remaining easy to train on a single GPU.
personalpages.surrey.ac.uk

# Train the model
```bash 01_train_2025t2.sh -d```
# Resume training from the last checkpoint
```bash 01_train_2025t2.sh --restart```
# Freeze the AST encoder for N warm-up epochs
```bash 01_train_2025t2.sh --warm_up_epochs N```
# Test on the development set
```bash 02a_test_2025t2.sh -d```

# Debugging
Use `debug_freeze.py` to freeze a single branch and inspect the training behaviour. Example:

```bash
python debug_freeze.py --freeze b3 --epochs 10
```

The script logs `debug_freeze_<branch>.csv` in the `logs` directory and plots loss and gradient norms via `tools/plot_loss_curve.py`.

Use `debug_recon_split.py` to compare raw reconstruction MSE between normal and
anomalous splits:

```bash
python debug_recon_split.py --model_ckpt your_model.pth
```

It prints the mean and standard deviation of frame MSE for the two splits so you
can verify that anomalies yield higher reconstruction error.
