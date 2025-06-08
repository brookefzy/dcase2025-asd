# A Multi-branch anomalous sound detection model

## Date: 2025-05-18
## Model Structure:
```

                +-------------------+
                |   Raw Audio WAV   |
                +-------------------+
                          ↓
                +-------------------+
                | Log-Mel Spectrogram|
                +-------------------+
                          ↓
┌───────────┐    ┌───────────┐   ┌──────────────┐   ┌─────────────┐
│  Branch 1 │    │  Branch 2 │   │  Branch 3    │   │  Branch 4   │
│  Pretrained│   │ Transformer│   │ Contrastive │   │ Diffusion   │
│  Models   │    │  Autoencoder│   │  Embedding  │   │  Generator  │
└───────────┘    └───────────┘   └──────────────┘   └─────────────┘
        ↓                ↓                ↓                   ↓
  Embeddings z        Latent z       Contrastive           Denoised
                       & Recon        Score via             Reconstruction
                    Error Score        InfoNCE               Error
        ↓                ↓                ↓                   ↓
             +---------------------------------------------+
             |      Normalizing Flow on z (Branch 5)      |
             +---------------------------------------------+
                          ↓
             +---------------------------------------------+
             |   Score Fusion & Learnable Attention Head   |
             +---------------------------------------------+
                          ↓
             +---------------------------------------------+
             |          Meta-Learner & Classifier           |
             +---------------------------------------------+
                          ↓
             +---------------------------------------------+
             |      Final Anomaly Score & Decisions        |
             +---------------------------------------------+
```

# Explanation
1. Branch modules under models/:

* branch_pretrained.py (AST via Hugging Face) [Hugging Face](https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer?utm_source=chatgpt.com)

* branch_transformer_ae.py (spectrogram Transformer AE) [GitHub](https://arxiv.org/abs/2203.16691)

* branch_contrastive.py (Machine-ID contrastive pretraining) [personalpages.surrey.ac.uk arXiv](https://arxiv.org/abs/2304.03588)

* branch_diffusion.py (ASD-Diffusion denoiser) [arXiv](https://arxiv.org/pdf/2409.15957)

* branch_flow.py (RealNVP normalizing flow) [GitHub](https://github.com/AxelNathanson/pytorch-normalizing-flows)

2. Fusion & decision modules:

* fusion_attention.py (learnable score fusion)

* meta_learner.py (MAML wrapper via Learn2Learn) 
arXiv

3. Data / training scripts following the DCASE baseline structure, with hooks for each branch.

4. Pre-trained weights you can download:

* AST: ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593") Google Colab

* SSAST (optional): ASTModel.from_pretrained("facebook/ssast-conv1d")

* MAML code (for meta­-training): https://github.com/cbfinn/maml 
GitHub

* CLAR / Machine-ID contrastive pretraining: re-implement based on the method in [Guan et al.](https://personalpages.surrey.ac.uk/w.wang/papers/Zhang%20et%20al_ICASSP_2024.pdf?utm_source=chatgpt.com), ICASSP’23 
personalpages.surrey.ac.uk

# Train the model
```bash train_ae.sh -d```
# Debugging
Use `debug_freeze.py` to freeze a single branch and inspect the training behaviour. Example:

```bash
python debug_freeze.py --freeze b3 --epochs 10
```

The script logs `debug_freeze_<branch>.csv` in the `logs` directory and plots loss and gradient norms via `tools/plot_loss_curve.py`.
