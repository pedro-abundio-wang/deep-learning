---
# Page settings
layout: default
keywords:
comments: false

# Hero section
title: Introduction to Tensorflow (Graph, Session, Nodes and variable scope)
description: 

# Micro navigation
micro_nav: true

# Page navigation
page_nav:
    prev:
        content: Previous page
        url: '/blog/tensorflow'
    next:
        content: Next page
        url: '/blog/datapipeline'
---

This post follows the [main post](/blog/tips) announcing the release of the CS230 code examples. We will explain here the TensorFlow part of the code, in our [github repository](https://github.com/cs230-stanford/cs230-code-examples/tree/master/tensorflow).

```python
tensorflow/
    vision/
    nlp/
```

This tutorial is among a series explaining how to structure a deep learning project:
- [installation, get started with the code for the projects](/blog/tips)
- **this post: (TensorFlow) explain the global structure of the code**
- [(TensorFlow) how to build the data pipeline](/blog/datapipeline)
- [(Tensorflow) how to build the model and train it](/blog/createtrainmodel) 

## **Goals of this tutorial**
- learn more about TensorFlow
- learn an example of how to correctly structure a deep learning project in TensorFlow
- fully understand the code to be able to use it for your own projects 

## **Resources**

For an official **introduction** to the Tensorflow concepts of `Graph()` and `Session()`, check out the [official introduction on tensorflow.org](https://www.tensorflow.org/tutorials/#tensorflow_core_tutorial).

For a **simple example on MNIST**, read the [official tutorial](https://www.tensorflow.org/tutorials/), but keep in mind that some of the techniques are not recommended for big projects (they use `placeholders` instead of the new `tf.data` pipeline, they don’t use `tf.layers`, etc.).

For a more **detailed tour** of Tensorflow, reading the [programmer’s guide](https://www.tensorflow.org/guide/) is definitely worth the time. You’ll learn more about Tensors, Variables, Graphs and Sessions, as well as the saving mechanism or how to import data.

For a **more advanced use** with concrete examples and code, we recommend reading [the relevant tutorials](https://www.tensorflow.org/tutorials/) for your project. You’ll find good code and explanations, going from [sequence-to-sequence in Tensorflow](https://www.tensorflow.org/tutorials/) to an [introduction to TF layers for convolutionnal Neural Nets](https://www.tensorflow.org/tutorials/estimators/cnn#getting_started).

You might also be interested in [Stanford’s CS20 class: Tensorflow for Deep Learning Research](http://web.stanford.edu/class/cs20si/) and its [github repo](https://github.com/chiphuyen/stanford-tensorflow-tutorials) containing some cool examples.

## **Structure of the code**

The code for each Tensorflow example shares a common structure:

```python
data/
experiments/
model/
    input_fn.py
    model_fn.py
    utils.py
    training.py
    evaluation.py
train.py
search_hyperparams.py
synthesize_results.py
evaluate.py
```

Here is each `model/` file purpose:
- `model/input_fn.py`: where you define the input data pipeline
- `model/model_fn.py`: creates the deep learning model
- `model/utils.py`: utility functions for handling hyperparams / logging
- `model/training.py`: utility functions to train a model
- `model/evaluation.py`: utility functions to evaluate a model

We recommend reading through `train.py` to get a high-level overview.

Once you get the high-level idea, depending on your task and dataset, you might want to modify
- `model/model_fn.py` to change the model’s architecture, i.e. how you transform your input into your prediction as well as your loss, etc.
- `model/input_fn` to change the process of feeding data to the model.
- `train.py` and `evaluate.py` to change the story-line (maybe you need to change the filenames, load a vocabulary, etc.)

Once you get something working for your dataset, feel free to edit any part of the code to suit your own needs.

## **Graph, Session and nodes**

When designing a Model in Tensorflow, there are [basically 2 steps](https://www.tensorflow.org/tutorials/#tensorflow_core_tutorial)

1. building the computational graph, the nodes and operations and how they are connected to each other
2. evaluating / running this graph on some data

As an example of **step 1**, if we define a TF constant (=a graph node), when we print it, we get a Tensor object (= a node) and not its value

```python
x = tf.constant(1., dtype=tf.float32, name="my-node-x")
print(x)
> Tensor("my-node-x:0", shape=(), dtype=float32)
```

Now, let’s get to **step 2**, and evaluate this node. We’ll need to create a `tf.Session` that will take care of actually evaluating the graph

```python
with tf.Session() as sess:
    print(sess.run(x))
> 1.0
```

In the code examples,
- **step 1** `model/input_fn.py` and `model/model_fn`
- **step 2** `model/training.py` and `model/evaluation.py`

## **A word about [variable scopes](https://www.tensorflow.org/guide/#the_problem)**

When creating a node, Tensorflow will have a name for it. You can add a prefix to the nodes names. This is done with the `variable_scope` mechanism

```python
with tf.variable_scope('model'):
    x1 = tf.get_variable('x', [], dtype=tf.float32) # get or create variable with name 'model/x:0'
    print(x1)
> <tf.Variable 'model/x:0' shape=() dtype=float32_ref>
```

What happens if I instantiate `x` twice ?

```python
with tf.variable_scope('model'):
    x2 = tf.get_variable('x', [], dtype=tf.float32)
> ValueError: Variable model/x already exists, disallowed.
```

When trying to create a new variable named `model/x`, we run into an Exception as a variable with the same name already exists. Thanks to this naming mechanism, you can actually control which value you give to the different nodes, and at different points of your code, decide to have 2 python objects correspond to the same node !

```python
with tf.variable_scope('model', reuse=True):
    x2 = tf.get_variable('x', [], dtype=tf.float32)
    print(x2)
> <tf.Variable 'model/x:0' shape=() dtype=float32_ref>
```

We can check that they indeed have the same value

```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # Initialize the Variables
    sess.run(tf.assign(x1, tf.constant(1.)))    # Change the value of x1
    sess.run(tf.assign(x2, tf.constant(2.)))    # Change the value of x2
    print("x1 = ", sess.run(x1), " x2 = ", sess.run(x2))

> x1 =  2.0  x2 =  2.0
```

## **How we deal with different Training / Evaluation Graphs**

Code examples design choice: theoretically, the graphs you define for training and inference can be different, but they still need to share their weights. To remedy this issue, there are two possibilities
1. re-build the graph, create a new session and reload the weights from some file when we switch between training and inference.
2. create all the nodes for training and inference in the graph and make sure that the python code does not create the nodes twice by using the `reuse=True` trick explained above.

We decided to go for this option. As you’ll notice in `train.py` we give an extra argument when we build our graphs

```python
train_model_spec = model_fn('train', train_inputs, params)
eval_model_spec = model_fn('eval', eval_inputs, params, reuse=True)
```

When we create the graph for the evaluation (`eval_model_spec`), the `model_fn` will encapsulate all the nodes in a `tf.variable_scope("model", reuse=True)` so that the nodes that have the same names than in the training graph share their weights !

For those interested in the problem of making training and eval graphs coexist, you can read this [discussion](https://www.tensorflow.org/tutorials/) which advocates for the other option.

As a side note, option 1 is also the one used in [`tf.Estimator`](https://www.tensorflow.org/guide/estimators).

Next page, you'll see how you can input data to your model.


