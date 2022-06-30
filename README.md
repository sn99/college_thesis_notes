# transformers-notes

- Bert, gpt-3, t5
- Type of neural network architecture


- RNN
  - Sequential
  - Hard to train
  - Cannot retain information

**Main idea for transformers:**
- Positional encoding
  - Store position data in the data itself and not the network structure
- Attention
  - Look at input data while making output data
- Self Attention
  - While making output data look at output data itself too

## [Original paper: Vaswani et al. Attention Is All You Need. In NIPS, 2017.](https://arxiv.org/abs/1706.03762)

#### Abstract:

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an 
encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention 
mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, 
dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models 
to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model 
achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, 
including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new 
single-model state-of-the-art BLEU score of 41.0 after training for 3.5 days on eight GPUs, a small fraction of the 
training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by 
applying it successfully to English constituency parsing both with large and limited training data.

#### Highlights:

- Get away from RNN 
- Positional encoding using trigonometric functions(mainly sin waves)
- Three "attention" mainly
  - One for input
    - Discovers interesting things and builds (key, value) pairs
    - Keys are ways to index value
  - Two for decoder
    - One normal
      - Builds queries
      - Queries are questions and align with key 
    - Another one which is fed by input
- Attention(Q, K, V) = softmax((QK^T)/dk.sqrt()) * V, dot.product of QK
  - Keys are just vectors in space and values in table
  - Query is vector in space too and multiplied with each key, the dot product of maximum giving out the closest key
  - Softmax is exponential function of numbers (basically make them big)
    - Kind of an indexing scheme
- Attention reduces path length, retain information as long path loses information
- Use hidden state of encoder with hidden state of decoder using `key, value, query` pair
- Every step in production is one training sample

## [Big Transfer (BiT): General Visual Representation Learning](https://arxiv.org/abs/1912.11370)

#### Abstract
Transfer of pre-trained representations improves sample efficiency and simplifies hyperparameter tuning when training 
deep neural networks for vision. We revisit the paradigm of pre-training on large supervised datasets and fine-tuning 
the model on a target task. We scale up pre-training, and propose a simple recipe that we call Big Transfer (BiT). By 
combining a few carefully selected components, and transferring using a simple heuristic, we achieve strong performance 
on over 20 datasets. BiT performs well across a surprisingly wide range of data regimes -- from 1 example per class to 
1M total examples. BiT achieves 87.5% top-1 accuracy on ILSVRC-2012, 99.4% on CIFAR-10, and 76.3% on the 19 task Visual 
Task Adaptation Benchmark (VTAB). On small datasets, BiT attains 76.8% on ILSVRC-2012 with 10 examples per class, and 
97.0% on CIFAR-10 with 10 examples per class. We conduct detailed analysis of the main components that lead to high 
transfer performance.

#### Highlights:

- 