# Deep Learning

**Nvidia Driver Install**

[https://www.nvidia.com/Download/index.aspx?lang=en-us](https://www.nvidia.com/Download/index.aspx?lang=en-us)

After you have downloaded the file NVIDIA-Linux-x86_64-xxx.xx.run, change to the directory containing the downloaded file, and as the **root user** run the executable:

```shell
sh NVIDIA-Linux-x86_64-xxx.xx.run
nvidia-smi
```

**Nvidia Driver Cuda Version Match**

![](img/nvidia-driver-cuda-version-match.png)

[https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)

**CUDA Download**

[https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive)

**CUDA Install**

[https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

**cuDNN Download**

[https://developer.nvidia.com/rdp/cudnn-archive](https://developer.nvidia.com/rdp/cudnn-archive)

**cuDNN Install**

[https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html)

**tensorflow cuda cudnn version**

![](img/tensorflow-cuda-cudnn-version.png)

[https://www.tensorflow.org/install/source](https://www.tensorflow.org/install/source)

# Papers

## Algorithms

[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)

[ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION](https://arxiv.org/abs/1412.6980)

[Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407)

[Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://jmlr.org/papers/v15/srivastava14a.html)

[Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a.html)

[Practical Recommendations for Gradient-Based Training of Deep Architectures](https://arxiv.org/abs/1206.5533)

## CNN

[Gradient-based learning applied to document recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)

[ImageNet classification with deep convolutional neural networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

[Network in network](https://arxiv.org/abs/1312.4400)

[Visualizing and understanding convolutional networks](https://arxiv.org/abs/1311.2901)

[Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)

[Very deep convolutional networks for large-scale image recognition](https://arxiv.org/abs/1409.1556)

[Deep residual networks for image recognition](https://arxiv.org/abs/1512.03385)

A guide to convolution arithmetic for deep learning

Is the deconvolution layer the same as a convolutional layer?

Deep Inside Convolutional Networks: Visualizing Image Classification Models and Saliency Maps

Understanding Neural Networks Through Deep Visualization

[Learning Deep Features for Discriminative Localization](http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf)

DenseNet: Densely Connected Convolutional Networks

### Object Detection

#### YOLO

[You Only Look Once: Unified real-time object detection](https://arxiv.org/abs/1506.02640)

[YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)

YOLOv3: An Incremental Improvement

[OverFeat: Integrated recognition, localization and detection using convolutional networks](https://arxiv.org/abs/1312.6229)

#### R-CNN

[Fast R-CNN](https://arxiv.org/abs/1504.08083)

[Faster R-CNN: Towards real-time object detection with region proposal networks](https://arxiv.org/abs/1506.01497)

[Mask R-CNN](https://arxiv.org/abs/1703.06870)

[R-FCN: Object Detection via Region-based Fully Convolutional Networks ](https://arxiv.org/abs/1605.06409)

#### Semantic Segmentation

[Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524)

Fully Convolutional Networks for Semantic Segmentation

#### SSD

[SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)

### Face Recognize

[FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)

[DeepFace: Closing the gap to human-level performance in face verification](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf)

[DeepFace Reading](https://www.cv-foundation.org/openaccess/content_cvpr_2014/html/Taigman_DeepFace_Closing_the_2014_CVPR_paper.html)

#### Open Source

FaceNet: https://github.com/davidsandberg/facenet

OpenFace: https://github.com/iwantooxxoox/Keras-OpenFace

### Art Generation

[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)

#### Readings

[Harish Narayanan, Convolutional neural networks for artistic style transfer](https://harishnarayanan.org/writing/artistic-style-transfer/)

[Log0, TensorFlow Implementation of "A Neural Algorithm of Artistic Style"](http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style)

## RNN

Cho et al., 2014. On the properties of neural machine translation: Encoder-decoder approaches

Chung et al., 2014. Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling

Hochreiter & Schmidhuber 1997. Long short-term memory

van der Maaten and Hinton., 2008. Visualizing data using t-SNE

Mikolov et. al., 2013, Linguistic regularities in continuous space word representations

Bengio et. al., 2003, A neural probabilistic language model

Bolukbasi et. al., 2016. Man is to computer programmer as woman is to homemaker? Debiasing word embeddings

Sutskever et al., 2014. Sequence to sequence learning with neural networks

Cho et al., 2014. Learning phrase representations using RNN encoder-decoder for statistical machine translation

Mao et. al., 2014. Deep captioning with multimodal recurrent neural networks

Vinyals et. al., 2014. Show and tell: Neural image caption generator

Karpathy and Li, 2015. Deep visual-semantic alignments for generating image descriptions

Papineni et. al., 2002. Bleu: A method for automatic evaluation of machine translation

Bahdanau et. al., 2014. Neural machine translation by jointly learning to align and translate

Xu et. al., 2015. Show, attend and tell: Neural image caption generation with visual attention

Graves et al., 2006. Connectionist Temporal Classification: Labeling unsegmented sequence data with recurrent neural networks

Multiple Object Recognition with Visual Attention

DRAW: A Recurrent Neural Network For Image Generation

## Deep Reinforcement Learning

Silver, Schrittwieser, Simonyan et al. (2017): Mastering the game of Go without human knowledge

Mnih, Kavukcuoglu, Silver et al. (2015): Human Level Control through Deep Reinforcement Learning

Francisco S. Melo: Convergence of Q-learning: a simple proof

[Video credits to Two minute papers: Google DeepMind's Deep Q-learning playing Atari Breakout](https://www.youtube.com/watch?v=V1eYniJ0Rnk)

Mnih, Kavukcuoglu, Silver et al. (2015): Human Level Control through Deep Reinforcement Learning

[Credits: DeepMind, DQN Breakout](https://www.youtube.com/watch?v=TmPfTpjtdgg)

Ho et al. (2016): Generative Adversarial Imitation Learning

Schulman et al. (2017): Trust Region Policy Optimization

Schulman et al. (2017): Proximal Policy Optimization

Bansal et al. (2017): Emergent Complexity via multi-agent competition

OpenAI Blog: Competitive self-play

[alphago-zero-learning-scratch](https://deepmind.com/blog/alphago-zero-learning-scratch/)

Silver, Schrittwieser, Simonyan et al. (2017): Mastering the game of Go without human knowledge

Finn et al. (2017): Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks

Bellemare et al. (2017):Unifying Count-Based Exploration and Intrinsic Motivation

[Human-level control through deep reinforcement learning](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)

[Mastering the Game of Go without Human Knowledge](https://deepmind.com/documents/119/agz_unformatted_nature.pdf)

## Recommendation System

Koren, Yehuda, Robert Bell, and Chris Volinsky：Matrix factorization techniques for recommender systems

Sedhain, Suvash, et al. AutoRec: Autoencoders meet collaborative filtering

# Readings

[Demystifying Deep Convolutional Neural Networks](http://scs.ryerson.ca/~aharley/neural-networks/)

http://mccormickml.com/2018/06/15/applying-word2vec-to-recommenders-and-advertising/

Bjorck N, Gomes C P, Selman B, et al. Understanding batch normalization[C]//Advances in Neural Information Processing Systems. 2018: 7705-7716.

Santurkar S, Tsipras D, Ilyas A, et al. How does batch normalization help optimization?[C]//Advances in Neural Information Processing Systems. 2018: 2488-2498.
