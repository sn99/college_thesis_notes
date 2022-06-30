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
- Application/Learning paper for community
- Train on big dataset and fine tune on small data set
  - Hope some overlap between data
- Database (google)
  - L -> 300M JFT, is not released publicly
  - M -> 14M imageNet21k, kinda funky
  - S -> 1.3M imageNet
- All models are pretty much residual networks, RN152 * 4
- Nothing really new but iterates what exactly matters and what to do
- State of the art on full dataset
  - Do not achieve state of the art in all specialist models
- Remove all images that appear in downstream models in big dataset
- 2 parts
  - How to pretrain
    - First component is scale
      - The larger the model the better the performance but also need more data to make actual difference, in small architecture hurts to increase data
      - Might be belkiins double descend curve, No. of parameter : Number of Datapoint
    - Group normalization and weight normalization - Per sample
      - Batch norm bad - calculate mean and standard variance and depends on batch size, when we distribute bath size gets very short
      - Very fast
  - How to fine tune
    - Rule to select Hyper-parameter (BiT HyperRule)
      - Hyper hyper-parameters - basically look up table
      - Decide on training schedule length, resolution and use-or-not MixUp regularization
      - Basically a stand training form
- Outperform all the generalist models, do not outperform all specialist 
- VTAB (Visual task adaptation benchmark) (19 tasks)
  - Biggest gain in natural (7 tasks)
  - Little gain Specialized (4 tasks)
  - Very minor gainStructured (8 tasks)
- Invest more computation time in bigger data 
  - Weight decay should not be low (questionable) - decrease learning rate
  - Standard time is 8gpu weeks but use 8gpu months

## [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

#### Abstract: 
While the Transformer architecture has become the de-facto standard for natural language processing tasks, its 
applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional
networks, or used to replace certain components of convolutional networks while keeping their overall structure in 
place. We show that this reliance on CNNs is not necessary and a pure transformer can perform very well on image 
classification tasks when applied directly to sequences of image patches. When pre-trained on large amounts of data and 
transferred to multiple recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc), Vision Transformer attains excellent 
results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources 
to train.

#### Highlights:
- First point of comparison is BiT
- Do something aka attention and is n^2, so only do local attention
- Global attention over image patches
  - 16 by 16 patches
    - Roll them up in a set and combine them with positional embeddings
    - 11, 12, 13, 21 not work just number 1,2,3,4 and index to a vector table
    - Unroll 16x16 ito 256 operation vector but first multiply into matrix E (embedding)
    - Patch + position embeddings
  - Transformers have no idea about what is where
    - In a way generalization of nlp in a feed forward network
      - Connection between nodes with fixed weigh
      - IN transformers `w` is not fixed, computes on fly
- ViT is better than BiT and costs less to train 
- It learns filters like CNN but not exactly, learns same thing like CNN but not programmed to do so