# Training a Cursive Transformer

Sam Greydanus & Zach Wimpee

## Abstract

*To be completed.*

## Introduction

Handwriting synthesis has long been a topic of interest in the fields of computer vision and pattern recognition, with applications ranging from personalized font generation to assisting individuals with motor impairments. Traditional methods have leveraged statistical models and recurrent neural networks to capture the sequential nature of handwriting. However, these approaches often struggle with long-term dependencies and lack the capacity to model complex temporal dynamics effectively.

The advent of Transformer architectures [Vaswani et al., 2017], which rely solely on attention mechanisms, has revolutionized sequence modeling tasks, particularly in natural language processing. Transformers have demonstrated remarkable performance in capturing long-range dependencies without relying on recurrent connections. Despite their success in language tasks, their application to handwriting synthesis remains underexplored.

In this paper, we introduce the **Cursive Transformer (CT)**, a novel Transformer-based architecture tailored for handwriting synthesis. The CT model leverages a custom tokenization scheme to represent pen strokes and incorporates cross-attention mechanisms between textual input and stroke sequences. Our approach addresses the limitations of previous methods by effectively modeling the sequential and spatial dependencies inherent in handwriting data.

## Related Work

### Handwriting Synthesis

Early approaches to handwriting synthesis utilized Hidden Markov Models (HMMs) [Plamondon and Srihari, 2000] and statistical techniques to model pen stroke sequences. With the rise of deep learning, Recurrent Neural Networks (RNNs), particularly Long Short-Term Memory (LSTM) networks, became the standard for sequential data modeling.

Graves [2013] introduced a generative model for handwriting synthesis using RNNs with Mixture Density Networks (MDNs). This model could generate realistic cursive handwriting by predicting the parameters of a mixture of Gaussians at each time step. However, RNN-based models often suffer from limitations in capturing long-term dependencies and are computationally intensive due to their sequential nature.

### Transformer Architectures

Transformers, introduced by Vaswani et al. [2017], have transformed the landscape of sequence modeling by utilizing self-attention mechanisms to capture dependencies between elements in a sequence, regardless of their distance. This paradigm shift has led to state-of-the-art results in various NLP tasks.

Recent works have explored applying Transformers to non-textual sequential data, such as music generation [Huang et al., 2018] and time series forecasting [Zhou et al., 2021]. However, the application of Transformers to handwriting synthesis is not straightforward due to the continuous and bi-dimensional nature of handwriting data.

### Handwriting Modeling with Transformers

Some attempts have been made to apply attention mechanisms to handwriting recognition [Bluche et al., 2016], but these primarily focus on recognition rather than synthesis. The challenge lies in effectively encoding the spatial information of pen strokes and integrating textual input to guide the synthesis process.

## Methodology

In this section, we present the Cursive Transformer architecture, detailing the custom tokenization scheme, the model architecture, and the training procedure.

### Custom Tokenization Scheme

Handwriting data consists of sequences of pen positions $(x_t, y_t)$ and pen states $p_t$ (pen-down or pen-up) at each time step $t$. To utilize Transformer architectures, which are designed for discrete token sequences, we need to convert continuous stroke data into a discrete token representation.

#### Stroke Representation

We represent each handwriting sample as a sequence of pen stroke offsets:

$$
\Delta x_t = x_t - x_{t-1}, \quad \Delta y_t = y_t - y_{t-1}, \quad s_t = p_t.
$$

The sequence $\{ (\Delta x_t, \Delta y_t, s_t) \}_{t=1}^T$ captures the movement of the pen at each time step.

#### Conversion to Polar Coordinates

To decouple magnitude and direction, we convert the Cartesian offsets to polar coordinates:

$$
r_t = \sqrt{(\Delta x_t)^2 + (\Delta y_t)^2}, \quad \theta_t = \arctan2(\Delta y_t, \Delta x_t).
$$

#### Discretization

We discretize $r_t$ and $\theta_t$ into fixed bins to create a finite vocabulary. Let $N_r$ and $N_\theta$ be the number of bins for magnitude and angle, respectively.

- **Magnitude bins**: We define bin edges $\{ b_{r_i} \}_{i=0}^{N_r}$ and assign each $r_t$ to a bin index $k_t^r$ such that $b_{r_{k_t^r}} \leq r_t < b_{r_{k_t^r+1}}$.
- **Angle bins**: Similarly, we define bin edges $\{ b_{\theta_i} \}_{i=0}^{N_\theta}$ for $\theta_t$ and assign each $\theta_t$ to a bin index $k_t^\theta$.

#### Token Assignment

Each time step is represented by a tuple of tokens:

$$
\text{Token}_t = (k_t^r, k_t^\theta, s_t).
$$

This discrete representation allows us to model the handwriting sequence as a sequence of tokens compatible with the Transformer architecture.

### Model Architecture

The Cursive Transformer model comprises the following components:

#### Input Embeddings

We use separate embedding layers for each type of token:

- **Magnitude embedding**: $E_r \in \mathbb{R}^{N_r \times d}$
- **Angle embedding**: $E_\theta \in \mathbb{R}^{N_\theta \times d}$
- **Pen state embedding**: $E_s \in \mathbb{R}^{2 \times d}$ (since $s_t \in \{ 0, 1 \}$)

The embeddings for each time step are combined:

$$
e_t = E_r(k_t^r) + E_\theta(k_t^\theta) + E_s(s_t) + E_{\text{pos}}(t),
$$

where $E_{\text{pos}}(t)$ is the positional encoding.

#### Textual Input Encoding

The textual input (ASCII sequence) is tokenized and embedded using a character-level embedding:

$$
c_i = E_{\text{char}}(w_i) + E_{\text{pos}}(i),
$$

where $w_i$ is the $i$-th character in the input text.

#### Transformer Layers

The model consists of $L$ Transformer layers. Each layer includes:

1. **Self-Attention over Stroke Tokens**

   The stroke embeddings $\{ e_t \}$ attend to each other to capture temporal dependencies.

2. **Cross-Attention with Text Tokens**

   The stroke embeddings attend to the text embeddings to incorporate information from the input text.

#### Layer Operations

For each Transformer layer $l$, we compute:

1. **Self-Attention**

   $$ 
   \begin{align*}
   Q_t^{(l)} &= W_Q^{(l)} h_t^{(l-1)}, \\
   K_t^{(l)} &= W_K^{(l)} h_t^{(l-1)}, \\
   V_t^{(l)} &= W_V^{(l)} h_t^{(l-1)},
   \end{align*}
   $$

   where $h_t^{(l-1)}$ is the hidden state from the previous layer (or input embedding for $l=1$).

   The self-attention output is:

   $$
   \tilde{h}_t^{(l)} = \text{Softmax}\left( \frac{Q_t^{(l)} (K^{(l)})^\top}{\sqrt{d}} \right) V^{(l)}.
   $$

2. **Cross-Attention**

   We compute cross-attention between the stroke tokens and text tokens:

   $$
   \begin{align*}
   Q_t^{(l)} &= W_Q^{(l)} \tilde{h}_t^{(l)}, \\
   K_i^{(l)} &= W_K^{(l)} c_i, \\
   V_i^{(l)} &= W_V^{(l)} c_i,
   \end{align*}
   $$

   The cross-attention output is:

   $$
   \hat{h}_t^{(l)} = \text{Softmax}\left( \frac{Q_t^{(l)} (K^{(l)}_{\text{text}})^\top}{\sqrt{d}} \right) V^{(l)}_{\text{text}},
   $$

   where $K^{(l)}_{\text{text}}$ and $V^{(l)}_{\text{text}}$ are the keys and values from the text embeddings.

3. **Feed-Forward Network**

   The combined output is passed through a feed-forward network (FFN):

   $$
   h_t^{(l)} = \text{LayerNorm}\left( \hat{h}_t^{(l)} + \text{FFN}\left( \hat{h}_t^{(l)} \right) \right).
   $$

#### Output Layer

The final hidden states are projected to logits for each token type:

- **Magnitude logits**: $z_t^r = W_r h_t^{(L)} + b_r$
- **Angle logits**: $z_t^\theta = W_\theta h_t^{(L)} + b_\theta$
- **Pen state logits**: $z_t^s = W_s h_t^{(L)} + b_s$

The model outputs a probability distribution over the possible tokens at each time step.

### Training Objective

We use the cross-entropy loss for each token type:

$$
\mathcal{L} = \sum_{t=1}^T \left( \mathcal{L}_{r_t} + \mathcal{L}_{\theta_t} + \mathcal{L}_{s_t} \right),
$$

where:

$$
\mathcal{L}_{r_t} = -\log p(k_t^r | \text{previous tokens}),
$$

$$
\mathcal{L}_{\theta_t} = -\log p(k_t^\theta | \text{previous tokens}),
$$

$$
\mathcal{L}_{s_t} = -\log p(s_t | \text{previous tokens}).
$$

### Differences from Traditional Transformers

- **Custom Tokenization**: Unlike standard Transformers that operate on word or subword tokens, we use a custom tokenization scheme tailored for continuous handwriting data.
- **Cross-Attention**: We incorporate cross-attention layers between stroke tokens and text tokens to allow the model to align pen strokes with the input text effectively.
- **Spatial Data Handling**: The model is designed to handle spatial (bi-dimensional) data, extending the Transformer architecture beyond its typical use in NLP tasks.

### Differences from Historical Handwriting Synthesis Methods

- **Non-Recurrent Architecture**: Traditional methods often rely on RNNs to model sequential dependencies. Our approach leverages the Transformer architecture to capture long-range dependencies without recurrence.
- **Direct Text-to-Stroke Modeling**: Previous models typically generate handwriting conditioned on latent variables or style embeddings. The CT model directly conditions on the input text through cross-attention mechanisms.
- **Scalability**: The use of attention mechanisms allows for parallel computation over sequences, improving training efficiency compared to recurrent models.

## Experiments

*To be completed.*

## Conclusion

*To be completed.*

## References

- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). *Attention is All You Need*. In Advances in Neural Information Processing Systems (NIPS).
- Graves, A. (2013). *Generating Sequences With Recurrent Neural Networks*. arXiv preprint arXiv:1308.0850.
- Bluche, T., Messina, R., Louradour, J., & Kermorvant, C. (2016). *Scan, Attend and Read: End-to-End Handwritten Paragraph Recognition with MDLSTM Attention*. In International Conference on Frontiers in Handwriting Recognition (ICFHR).
- Plamondon, R., & Srihari, S. N. (2000). *Online and off-line handwriting recognition: a comprehensive survey*. IEEE Transactions on Pattern Analysis and Machine Intelligence.
---