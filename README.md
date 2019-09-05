Code accompanying paper titled "Algorithmic Guarantees for Inverse Imaging with Untrained Network Priors", Advances in Neural Information Processing Systems (NeurIPS), 2019.

For details please refer to the preprint available at: https://arxiv.org/abs/1906.08763

Run the python notebook compressive_imaging.ipynb. The code works in 3 modes:

* Mode 1: model fitting. Find the best network weights w that approximate image x as G(w,z), with fixed code z. Solves the obective function min_w || x - G(w,z) ||2^2. Can also be used for denoising type applications such as super-resolution, inpainting, denoising.
* Mode 2: inverting linear compressive measurements. Reconstruct an image x from linear compressive measurements y=Ax. Solves the objective function min_x ||y - Ax||2^2 such that x = G(w,z).
* Mode 3: inverting magnitude-only compressive measurements. Reconstruct an image x from magnitude-only compressive measurements y=|Ax|. Solves the objective function min_x ||y - |Ax|||2^2 such that x = G(w,z).

Datasets used: MNIST and CelebA

The untrained generative network G(w,z) can assume a decoder architecture (set decodetype='decoder')described in "Deep Decoder: Concise Image Representations from Untrained Non-convolutional Networks", ICLR 2019, https://arxiv.org/abs/1810.03982; or can also assume a DCGAN architecture (set decodetype='transposeconv').
