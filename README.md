# Anastrophic Regularization in PyTorch
A Data-Free, Spectral Solution to Catastrophic Forgetting in Continual Learning.

Anastrophic Regularization ($\mathcal{R}_{ana}$) is a novel technique derived from Anastrophic Theory that stabilizes neural network training. Instead of blindly penalizing weight magnitudes, it preserves the structural "Harmonic Memory" of learned tasks.

## Why Choose This Over EWC?
Traditional methods like Elastic Weight Consolidation (EWC) attempt to prevent forgetting by anchoring individual parameters using the diagonal of the Fisher Information Matrix. This over-constrains the network and limits its plasticity for new tasks. $\mathcal{R}_{ana}$ solves this by offering three major advantages:

* Maximum Plasticity: Individual weights are free to change as long as the global periodic functional invariants of the previous task are maintained. 
* 100% Data-Free & Privacy Preserving: EWC requires computing gradients over old datasets, posing significant privacy and storage issues. $\mathcal{R}_{ana}$ operates purely in the spectral domain, requiring absolutely zero old data.
* Computationally Efficient: It leverages fast Fourier transformations (FFTs) to seamlessly minimize harmonic frustration without needing complex state-space reconstructions.

## How It Works
The regularizer guides weights along Fisher-Rao geodetic paths by optimizing two key spectral metrics:

$$\mathcal{R}_{ana}(W) = \lambda(1 - \Phi(Spec(W))) + \eta BB(W, W_{prev})$$

* Spectral Coherence ($\Phi$): A geometric measure of phase alignment on the spectral torus.
* Anastrophic Beta ($BB$): Measures the harmonic tension between current and previous weight matrices.

## Quick Start
See `main.py` for a plug-and-play PyTorch implementation demonstrating $\mathcal{R}_{ana}$ on a Split-MNIST benchmark using a CNN architecture.

Read the full theoretical framework on Zenodo: [https://zenodo.org/records/18699347]
