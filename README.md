# Synthetic Time Series Generation using GANs and Transformers

This repository contains the implementation of a novel approach for synthetic time series generation. The method combines Generative Adversarial Networks (GANs) with Transformer architectures to produce high-quality synthetic time series data. This project is designed to address the challenges of generating realistic and diverse time series data for various applications.

# Key Features

GAN Architecture: Utilizes a Wasserstein GAN with Gradient Penalty (WGAN-GP) to ensure stable training and high-quality outputs.

Transformer Models: Leverages the attention mechanisms of Transformers to capture complex temporal dependencies in time series data.

Synthetic Dataset: Trained on a dataset of sinusoidal waves, providing a controlled environment for evaluating the model’s performance.

PyTorch Implementation: Built entirely using PyTorch, ensuring modularity, flexibility, and scalability.

# Usage

**Main function (Generate Synthetic Data):**

*python trainGAN.py*

**Load training data function:**

*python dataLoader.py*

**GAN Generator and Discriminator architecture:**

*python GANModels.py*

**Training loop function:**

*python functions.py*

**Sinusoidal waves dataset generator:**

*python utils/sinusoidalWavesDataset.py*

**Configuration file:**

*json cfg.json*

# Future Work

* Validation metrics
* Conditional outputs based on metadata
* Configurable output length of signal (currenly fixed length output)
* Tokenization / embeding of time series 

# License

This project is licensed under the MIT License. See the LICENSE file for details.

# Acknowledgments

Xiaomin Li, Vangelis Metsis, Huangyingrui Wang, Anne Hee Hiong Ngu: [TTS-GAN: A Transformer-based Time-Series Generative Adversarial Network](https://arxiv.org/abs/2202.02691)
