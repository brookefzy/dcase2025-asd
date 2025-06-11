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
* **Mahalanobis distance** in latent space measures global deviation of *z* from the normal cluster.
* **Reconstruction MSE** measures low‑level signal mismatch.
* Combined score`S = α · M(z) + (1 –α) · E(x,\hat x)`with α∈[0,1].

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
