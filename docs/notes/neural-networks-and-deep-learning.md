---
# Page settings
layout: default
keywords:
comments: false

# Hero section
title: Neural Networks and Deep Learning
description:

# Micro navigation
micro_nav: true

# Page navigation
page_nav:
    prev:
        content:
        url: '#'
    next:
        content:
        url: '#'
---

## Introduction to deep learning

### What is Neural Network

Let’s start with the house price prediction example. Suppose that you have a dataset with six houses and we know the price and the size of these houses. We want to fit a function to predict the price of these houses with respect to its size.

{% include image.html image="notes/neural-networks-and-deep-learning/12.png" %}

We will put a straight line through these data points. Since we know that our prices cannot be negative, we end up with a horizontal line that passes through 0.

The blue line is the function for predicting the price of the house as a function of its size. You can think of this function as a very simple neural network. The input to the neural network is the size of a house, denoted by $$x$$, which goes into a single neuron and then outputs the predicted price, which we denote by $$y$$.

{% include image.html image="notes/neural-networks-and-deep-learning/14.png" %}

If this is a neural network with a single neuron, a much larger neural network is formed by taking many of the single neurons and stacking them together.

A basic Neural Network with more features is ilustrated in the following image.

{% include image.html image="notes/neural-networks-and-deep-learning/15.png" %}

### Supervised learning with neural networks

In supervised learning, we have some input $$x$$, and we want to learn a function mapping to some output $$y$$. Just like in the house price prediction application our input were some features of a home and our goal was to estimate the price of a home $$y$$.

Here are some other fields where neural networks have been applied very effectively.

{% include image.html image="notes/neural-networks-and-deep-learning/16.png" %}

We might input an image and want to output an index from one to a thousand, trying to tell if this picture might be one of a thousand different image classes. This can be used for photo tagging.

The recent progress in speech recognition has also been very exciting. Now you can input an audio clip to a neural network and can have it output a text transcript.

Machine translation has also made huge strikes thanks to deep learning where now you can have a neural network input an English sentence and directly output a Chinese sentence.

Different types of neural networks are useful for different applications.

{% include image.html image="notes/neural-networks-and-deep-learning/17.png" %}

- In the real estate application, we use a universally **Standard Neural Network** architecture.
- For image applications we’ll often use **Convolutional Neural Network (CNN)**.
- Audio is most naturally represented as a one-dimensional time series or as a one-dimensional temporal sequence. Hence, for a sequence data, we often use **Recurrent Neural Network (RNN)**.
- Language, English and Chinese, the alphabets or the words come one at a time and language is also represented as a sequence data. **Recurrent Neural Network (RNN)** are often used for these applications.

Machine learning is applied to both **Structured Data and Unstructured Data**.

Structured Data means basically databases of data. In house price prediction, you might have a database or the column that tells you the size and the number of bedrooms.

In predicting whether or not a user will click on an ad, we might have information about the user, such as the age, some information about the ad, and then labels that you’re trying to predict.

{% include image.html image="notes/neural-networks-and-deep-learning/18.png" %}

Structured data means, that each of the features, such as a size of the house, the number of bedrooms, or the age of a user, have a very well-defined meaning. In contrast, unstructured data refers to things like audio, raw audio, or images where you might want to recognize what’s in the image or text. Here, the features might be the pixel values in an image or the individual words in a piece of text.

{% include image.html image="notes/neural-networks-and-deep-learning/19.png" %}

Neural networks, computers are now much better at interpreting unstructured data as compared to just a few years ago. This creates opportunities for many new exciting applications that use speech recognition, image recognition, natural language processing of text.

### Why is deep learning taking off

Many of the ideas of deep learning (neural networks) have been around for decades. Why are these ideas taking off now?

{% include image.html image="notes/neural-networks-and-deep-learning/11.png" %}

In detail, even as you accumulate more data, usually the performance of older learning algorithms, such as logistic regression, **plateaus**. This means its learning curve flattens out, and the algorithm stops improving even as you give it more data. It was as if the older algorithms didn't know what to do with all the data we now have.

If you train a small neural network (NN) on the same supervised learning task, you might get slightly better performance. Finally, if you train larger and larger neural networks, you can obtain even better performance:

The diagram shows NNs doing better in the regime of small datasets. This effect is less consistent than the effect of NNs doing well in the regime of huge datasets. In the small data regime, traditional algorithms may or may not do better. For example, if you only have 20 training examples, it might not matter much whether you use logistic regression or a neural network; the features-engineering will have a bigger effect than the choice of algorithm. But if you have one-million examples, I would favor the neural network.

Biggest drivers of recent progress have been:

- **Data availability**: People are now spending more time on digital devices (laptops, mobile devices). Their digital activities generate huge amounts of data that we can feed to our learning algorithms.
- **Computational scale**: We started just a few years ago, techniques (like GPUs/Powerful CPUs/Distributed computing) to be able to train neural networks that are big enough to take advantage of the huge datasets we now have.
- **Algorithm**: Creative algorithms has appeared that changed the way NN works.

{% include image.html image="notes/neural-networks-and-deep-learning/20.png" %}

To conclude, often you have an idea for a neural network architecture and you want to implement it in code. Fast computation is important because the process of training a neural network is very iterative and can be time-consuming. Implementing our idea then lets us run an experiment which tells us how well our neural network does. Then, by looking at it, you go back to change the details of our neural network and then you go around this circle over and over, until we get the desired performance.

## Neural Networks Basics

### Binary classification

Binary classification is the task of classifying elements of a given set into two classification.

{% include image.html image="notes/neural-networks-and-deep-learning/21.png" %}

A binary classification problem:

- We have an input image $$x$$ and the output $$y$$ is a label to recognize the image.
- 1 means cat is on an image, 0 means that a non-cat object is on an image.

In binary classification, our goal is to learn a classifier that can input an image represented by its feature vector $$x$$ and predict whether the corresponding label is 1 or 0. That is, whether this is a cat image or a non-cat image.

The computer stores 3 separate matrices corresponding to the red, green and blue (RGB) color channels of the image. If the input image is 64 by 64 pixels, then we would have three 64 by 64 matrices corresponding to the red, green and blue pixel intensity values for our image. For a 64 by 64 image - the total dimension of this vector will be 64 * 64 * 3 = 12288.

{% include image.html image="notes/neural-networks-and-deep-learning/22.png" %}

Notation that we will follow is shown in the table below:

{% include image.html image="notes/neural-networks-and-deep-learning/23.png" %}

### Logistic regression

**Logistic regression** is a supervised learning algorithm that we can use when labels are either 0 or 1 and this is the so-called **Binary Classification Problem**. An input feature vector $$x$$ may correspond to an image that we want to recognize as either a cat picture (1) or a non-cat picture (0). That is, we want an algorithm to output the prediction which is an estimate of $$y$$:

$$ \hat{y} = P(y=1|x) $$

$$ x \in R^{n_x} $$

$$ Parameters: w \in R^{n_x}, b \in R $$

More formally, we want $$\hat{y}$$ to be the chance that $$\hat{y}$$ is equal to 1, given the input features $$x$$. In other words, if $$x$$ is a picture, we want $$y$$ to tell us what is the chance that this is a cat picture.

The $$x$$ is an $$n^x$$ - dimensional vector. The parameters of logistic regression are $$w$$, which is also an $$n^x$$ - dimensional vector together with $$b$$ wich is a real number.

Given an input $$x$$ and the parameters $$w$$ and $$b$$, how do we generate the output $$\hat{y}$$. One thing we could try, that doesn’t work, would be to have: $$\hat{y} = w^T x + b$$ which is a linear function of the input and in fact, this is what we use if we were doing **Linear Regression**.

However, this is not a very good algorithm for binary classification, because we want $$\hat{y}$$ to be the chance that $$y$$ is equal to 1, so $$\hat{y}$$ should be between 0 and 1.

It is difficult to enforce this because $$w^T x + b$$ can be much bigger than 1 or can even be negative which doesn’t make sense for a probability that we want to be in a range between 0 and 1. We can conclude that we need a function which will transform $$\hat{y} = w^T x + b$$ to be in a range between 0 and 1.

Let’s see one function that can help us do that. In logistic regression, the output is going to be the **Sigmoid Function**. We can see that it goes smoothly from 0 up to 1.

{% include image.html image="notes/neural-networks-and-deep-learning/25.jpeg" %}

- use $$z$$ to denote the following quantity $$w^T x + b$$.
- we have: $$\hat{y} = \sigma(w^T x + b)$$.
- if $$z$$ is a large positive number: $$\sigma(z) = 1$$
- if $$z$$ is a large negative number: $$\sigma(z) = 0$$

When we implement logistic regression, our job is to try to learn parameters $$w$$ and $$b$$, so that $$\hat{y}$$ becomes a good estimate of the chance of $$y$$ being equal to 1.

### Logistic regression cost function

First, to train parameters $$w$$ and $$b$$ of a logistic regression model we need to define a **cost function**.

Given a training set of $$m$$ training examples, we want to find parameters $$w$$ and $$b$$, so that $$\hat{y}$$ is as close to $$y$$ (ground truth).

Here, we will use $$(i)$$ superscript to index different training examples.

Henceforth, we will use **loss (error) function** to measure how well our algorithm is doing. The loss function is applied only to a single training sample, and commonly used loss function is a **squared error**:

$$L(\hat{y}, y) = \frac{1}{2} (\hat{y} - y)^2$$

In logistic regression squared error loss function is not an optimal choice. It results in an optimization problem which is not convex, and the gradient descent algorithm may not work well, it may not converge optimally.

In terms of a surface, the surface is convex if, loosely speaking, it looks like a parabola. If you have a ball and let it roll along the surface, that surface is convex if that ball is guaranteed to always end up at the same point in the end. However, if the surface has bumps, then, depending on where you drop the ball from, it might get stuck somewhere else. That surface is then non-convex.

{% include image.html image="notes/neural-networks-and-deep-learning/27.png" %}

To be sure that we will get to the global optimum, we will use following loss function:

$$L(\hat{y}, y) = - (y log\hat{y} + (1-y)log(1-\hat{y}))$$

It will give us a convex optimization problem and it is therefore much easier to be optimized.

To understand why this is a good choice, let’s see these two cases:

if $$y=1$$ then $$L(\hat{y}, y) = -log\hat{y} \to log\hat{y}$$ should be large, so we want $$\hat{y}$$ large (as close as possible to 1)

if $$y=0$$ then $$L(\hat{y}, y) = -log(1-\hat{y}) \to log(1-\hat{y})$$ should be large, so we want $$\hat{y}$$ small (as close as possible to 0)

A cost function measures how well our parameters $$w$$ and $$b$$ are doing on the entire training set :

{% include image.html image="notes/neural-networks-and-deep-learning/30.png" %}

- Cost function $$J$$ is defined as an average of a sum of loss functions of all training examples.
- Cost function is a function of parameters $$w$$ and $$b$$.

In cost function diagram, the horizontal axes represent our spatial parameters, $$w$$ and $$b$$. In practice, $$w$$ can be of a much higher dimension, but for the purposes of plotting, we will illustrate $$w$$ and $$b$$ as scalars.

The cost function $$J(w,b)$$ is then some surface above these horizontal axes $$w$$ and $$b$$. So, the height of the surface represents the value of $$J(w,b)$$ at a certain point. Our goal will be to minimize function $$J$$, and to find parameters $$w$$ and $$b$$.

### Gradient Descent

Gradient Descent is an algorithm that tries to minimize the cost function $$J(w,b)$$ and to find optimal values for $$w$$ and $$b$$.

For the purpose of illustration we will use $$J(w)$$, function that we want to minimize, as a function of one variable. To make this easier to draw, we are going to ignore $$b$$ for now, just to make this a one-dimensional plot instead of a high-dimensional plot.

Gradient Descent starts at an initial parameter and begins to take values in the steepest downhill direction. Function $$J(w,b)$$ is convex, so no matter where we initialize, we should get to the same point or roughly the same point.

After a single step, it ends up a little bit down and closer to a global otpimum because it is trying to take a step downhill in the direction of steepest descent or quickly down low as possible.

After a fixed number of iterations of Gradient Descent, hopefully, will converge to the global optimum or get close to the global optimum.

The **learning rate** $$\alpha$$ controls how big step we take on each iteration of Gradient Descent.

{% include image.html image="notes/neural-networks-and-deep-learning/31.jpeg" %}

If the derivative is positive, $$w$$ gets updated as $$w$$ minus a learning rate $$\alpha$$ times the derivative $$dw$$.

We know that the derivative is positive, so we end up subtracting from $$w$$ and taking a step to the left. Here, Gradient Descent would make your algorithm slowly decrease the parameter if you have started off with this large value of $$w$$.

Next, when the derivative is negative (left side of the convex function), the Gradient Descent update would subtract $$\alpha$$ times a negative number, and so we end up slowly increasing $$w$$ and we are making $$w$$ bigger and bigger with each successive iteration of Gradient Descent.

So, whether you initialize $$w$$ on the left or on the right, Gradient Descent would move you towards this global minimum.

$$ w = w -\alpha \frac{dJ(w,b)}{dw} $$

$$ b = b - \alpha \frac{dJ(w,b)}{db} $$

### Computation graph

Let’s say that we’re trying to compute a function $$J$$, which is a function of three variables $$a$$, $$b$$, and $$c$$ and let’s say that function $$J$$ is $$3(a + bc)$$.

{% include image.html image="notes/neural-networks-and-deep-learning/40.png" %}

Computation of this function has actually three distinct steps:

- Compute $$bc$$ and store it in the variable $$u$$, so $$u = bc$$
- Compute $$v = a + u$$
- Output $$J$$ is $$3v$$

Let’s summarize:

$$J(a,b,c) = 3(a + bc)$$

$$u = bc$$

$$v = a + u$$

$$J = 3v$$

In this simple example we see that, through a left-to-right pass, you can compute the value of $$J$$.

### Derivatives with Computation Graph

How to figure out derivative calculations of the function $$J$$.

Now we want using a computation graph to compute the derivative of $$J$$ with respect to $$v$$. Let’s get back to our picture, but with concrete parameters.

{% include image.html image="notes/neural-networks-and-deep-learning/42.png" %}

First, let’s see the final change of value $$J$$ if we change $$v$$ value a little bit:

$$J = 3v$$

$$ \frac{dJ}{dv} = 3$$

We emphasize that calculation of $$dJ/dv$$ is one step of a back propagation. The following picture depicts **forward propagation** as well as **backward propagation**:

{% include image.html image="notes/neural-networks-and-deep-learning/45.png" %}

Next, what is $$dJ/da$$. One way to break this down is to say that if we change $$a$$, that would change $$v$$ and through changing $$v$$ that would change $$J$$. By increasing $$a$$, how much $$J$$ changed is also determined by $$dv/da$$. This is called a **chain rule** in calculus:

$$ \frac{dJ}{da} = \frac{dJ}{dv} \frac{dv}{da} = 3 * 1 $$

Now, let’s calculate derivative $$dJ/du$$.

$$ \frac{dJ}{du} = \frac{dJ}{dv} \frac{dv}{du} = 3 * 1 $$

Finally, we have to find the most important values: value of $$dJ/db$$ and $$dJ/dc$$. Let’s calculate them:

$$ \frac{dJ}{db} = \frac{dJ}{du} \frac{du}{db} = 3 * 2 $$

$$ \frac{dJ}{dc} = \frac{dJ}{du} \frac{du}{dc} = 3 * 3 $$

### Logistic Regression Gradient Descent

Why do we need a computation graph? To answer this question, we have to check how the computation for our neural network is organized. There are two important principles in neural network computation:

  - Forward pass or forward propagation step
  - Backward pass or backpropagation step

During NN’s **forward propagation step** we compute the output of our neural network. In a binary classification case, our neural network output is defined by a variable and it can have any value from [0,1] interval.

In order to actually train our neural network (find parameters 𝑤 and 𝑏 as local optima of our cost function) we have to conduct a **backpropagation step**. In this way, we can compute gradients or compute derivatives. With this information, we are able to implement gradient descent algorithm for finding optimal values of 𝑤 and 𝑏. That way we can train our neural network and expect that it will do well on a classification task.

A computation graph is a systematic and easy way to represent our neural network and it is used to better understand (or compute) derivatives or neural network output.

The computation graph of a logistic regression looks like the following:

{% include image.html image="notes/neural-networks-and-deep-learning/04.png" %}

In this example, we have features $$x_1$$ and $$x_2$$. In order to compute $$z$$, we will need to input $$w_1$$, $$w_1$$ and $$b$$ in addition to the feature values $$x_1$$ and $$x_2$$

$$ z = w_1 x_1 + w_2 x_2 + b $$

After that, we can compute our $$\hat{y}$$ (equals sigma of $$z$$)

$$ \hat{y} = \sigma(z) $$

$$ a = \sigma(z) $$

Finally, we are able to compute our loss function.

$$ L(a,y) = - (yloga + (1-y)log(1-a)) $$

To reduce our loss function (remember right now we are talking only about one data sample) we have to update our $$w$$ and $$b$$ parameters. So, first we have to compute the loss using forward propagation step. After this, we go in the opposite direction (backward propagation step) to compute the derivatives.

$$da = \frac{dL(a,y)}{da}$$

$$da = -\frac{y}{a} + \frac{1-y}{1-a}$$

Having computed $$da$$, we can go backwards and compute $$dz$$:

$$dz = \frac{dL(a,y)}{dz}$$

$$dz = \frac{dL(a,y)}{da} \frac{da}{dz}$$

$$\frac{dL(a,y)}{da} = -\frac{y}{a} + \frac{1-y}{1-a}$$

$$\frac{da}{dz} = a(1-a)$$

$$dz = a - y$$

The final step in back propagation is to go back to compute amount of change of our parameters $$w$$ and $$b$$:

$$dw_1 = \frac{dL(a,y)}{dw_1} = x_1 dz$$

$$dw_2 = \frac{dL(a,y)}{dw_w} = x_2 dz$$

$$db = \frac{dL(a,y)}{db} = dz$$

To conclude, if we want to do gradient descent with respect to just this one training example, we would do the following updates

$$w_1 = w_1 - \alpha dw_1$$

$$w_2 = w_2 - \alpha dw_2$$

$$b = b - \alpha db$$

### Gradient Descent on training set

The cost function is the average of our loss function, when the algorithm outputs $$a^{(i)}$$ for the pair $$(x^{(i)},y^{(i)})$$.

$$J(w,b) = \frac{1}{m} \sum_{i=1}^{m} L(a^{(i)},y^{(i)})$$

Here $$a^{(i)}$$ is the prediction on the $$i$$ th training example which is sigmoid of $$z^{(i)}$$, were $$z^{(i)} = w^T x^{(i)} + b$$

$$ a^{(i)} = \hat{y}^{(i)} = \sigma(z^{(i)}) =  \sigma(w^T x^{(i)} + b) $$

The derivative with respect to $$w_1$$ of the overall cost function, is the average of derivatives with respect to $$w_1$$ of the individual loss term,

$$\frac{d}{dw_1} J(w,b) = \frac{1}{m} \sum_{i=1}^{m} \frac{d}{dw_1} L(a^{(i)}, y^{(i)})$$

and to calculate the derivative $$dw_1$$ we compute,

$$dw_1 = \frac{d}{dw_1} J(w,b)$$

This gives us the overall gradient that we can use to implement logistic regression.

To implement Logistic Regression, here is what we can do, if $$n=2$$, were $$n$$ is our number of features and $$m$$ is a number of samples.

{% include image.html image="notes/neural-networks-and-deep-learning/54.png" %}

After leaving the inner for loop, we need to divide $$J$$, $$dw_1$$, $$dw_2$$ and $$b$$ by $$m$$, because we are computing their average.

$$J /= m, dw_1 /= m, dw_2 /= m, db /= m$$

After finishing all these calculations, to implement one step of a gradient descent, we need to update our parameters $$w_1$$, $$w_2$$, and $$b$$.

$$w_1 = w_1 - \alpha dw_1$$

$$w_2 = w_2 - \alpha dw_2$$

$$b = b - \alpha db$$

It turns out there are two weaknesses with our calculations as we’ve implemented it here.

To implement logistic regression this way, we need to write two for loops (loop over $$m$$ training samples and $$n$$ features).

When implementing deep learning algorithms, having explicit for loops makes our algorithm run less efficient. Especially on larger datasets, which we must avoid. For this, we use what we call vectorization.

The above code should run for some iterations to minimize error. So there will be two inner loops to implement the logistic regression. Vectorization is so important on deep learning to reduce loops. In the last code we can make the whole loop in one step using vectorization!

### Vectorization

A vectorization is basically the art of getting rid of explicit for loops whenever possible. With the help of vectorization, operations are applied to whole arrays instead of individual elements. The rule of thumb to remember is to avoid using explicit loops in your code. Deep learning algorithms tend to shine when trained on large datasets, so it’s important that your code runs quickly. Otherwise, your code might take a long time to get your result.

{% include image.html image="notes/neural-networks-and-deep-learning/58.png" %}

### Vectorizing Logistic Regression

When we are programming Logistic Regression or Neural Networks we should avoid explicit for-loops. It’s not always possible, but when we can, we should use built-in functions or find some other ways to compute it. Vectorizing the implementation of Logistic Regression  makes the code highly efficient. We will see how we can use this technique to compute gradient descent without using even a single for-loop.

Now, we will examine the forward propagation step of logistic regression. If we have $$m$$ training examples, to make a prediction on the first example we need to compute $$z$$ and the activation function $$a$$ as follows:

$$z^{(1)} = w^T x^{(1)} + b$$

$$a^{(1)} = \sigma(z^{(1)})$$

To make prediction on the second training example we need to compute this:

$$z^{(2)} = w^T x^{(2)} + b$$

$$a^{(2)} = \sigma(z^{(2)})$$

The same is with prediction of third training example:

$$z^{(3)} = w^T x^{(3)} + b$$

$$a^{(3)} = \sigma(z^{(3)})$$

So if we have $$m$$ training examples we need to do these calculations 𝑚 times. In order to carry out the forward propagation step, which means to compute these predictions for all $$m$$ training examples, there is a way to do this without needing an explicit for loop.

We will stack all training examples horizontally in a matrix $$X$$, so that every column in matrix $$X$$ represents one training example:

{% include image.html image="notes/neural-networks-and-deep-learning/63.png" %}

Notice that matrix $$w$$ is a $$n_x * 1$$ matrix (or a column vector), so when we transpose it we get $$w^T$$ which is a $$1 * n_x$$ matrix (or a row vector) so multiplying $$w^T$$ with $$X$$ we get a $$1 * m$$ matrix. Then we add a $$1 * m$$ matrix $$b$$ to obtain $$Z$$.

We will define matrix $$Z$$ by placing all $$z^{(i)}$$ values in a row vector:

$$ Z = [z^{(1)}, z^{(2)}, \dots, z^{(m)}] = w^T X + b = [w^T x^{(1)} + b, w^T x^{(2)} + b, \dots, w^T x^{(m)} + b] $$

We can easily implement the calculation of a matrix $$Z$$:

```python
Z = np.dot(w.T, X) + b
```

As we can see $$b$$ is defined as a scalar. When you add this vector to this real number, Python automatically takes this real number $$b$$ and expands it out to the $$1 * m$$ row vector. This operation is called **broadcasting**.

Matrix $$A$$ is defined as a $$1 * m$$, wich we also got by stacking horizontaly values $$a^{(i)}$$ as we did with matrix $$Z$$:

$$ A = [a^{(1)}, a^{(2)}, \dots, a^{(m)}] = \sigma(Z) $$

In Python, we can also calculate matrix $$A$$ with one line of code as follows (if we have defined sigmoid function as above):

$$ A = sigmoid(Z) $$

For the gradient computation we had to compute detivative $$dz$$ for every training example:

$$ \begin{align*}
dz^{(1)} &= a^{(1)} - y^{(1)} \\
dz^{(2)} &= a^{(2)} - y^{(2)} \\
\vdots \\
dz^{(m)} &= a^{(m)} - y^{(2)} \\
\end{align*} $$

In the same way, we have defined previous variables, now we will define matrix $$dZ$$, where we will stack all $$dz^{(i)}$$ variables horizontally, dimension of this matrix $$dZ$$ is $$1 * m$$ or alternativly a $$m$$ dimensional row vector.

$$ dZ = [dz^{(1)}, dz^{(2)}, \dots, dz^{(m)}] $$

As we know that matrices $$A$$ and $$Y$$ are defined as follows:

$$ A = [a^{(1)}, a^{(2)}, \dots, a^{(m)}] $$

$$ Y = [y^{(1)}, y^{(2)}, \dots, y^{(m)}] $$

We can see that $$dZ$$ below, all values in $$dZ$$ can be computed at the same time.

$$ dZ = A - Y = [a^{(1)} - y^{(1)}, a^{(2)} - y^{(2)}, \dots, a^{(m)} - y^{(m)},] $$

To implement Logistic Regression on code we did this:

{% include image.html image="notes/neural-networks-and-deep-learning/72.png" %}

This code was non-vectorized and highly inefficent so we need to transform it. First, using vectorization, we can transform equations (∗) and (∗∗) into one equation:

$$dw += x^{(i)} dz^{(i)}$$

The cost function is:

$$ J = - \frac{1}{m} \sum_{i=1}^{m} (y^{(i)} loga^{(i)} + (1-y^{(i)})log(1-a^{(i)})) $$

The derivatives are:

$$\frac{dJ}{dw} = dw = \frac{1}{m} X (A-Y)^T$$

$$\frac{dJ}{db} = db = \frac{1}{m} \sum_{i=1}^{m} (a^{(i)} - y^{(i)})^T$$

To calculate $$w$$ and $$b$$ we will still need following for loop.

{% include image.html image="notes/neural-networks-and-deep-learning/76.png" %}

We don’t need to loop through entire training set, but still we need to loop through number of iterations and that’s a for loop that we can’t get rid off.

### Broadcasting in Python

The term broadcasting describes how NumPy treats arrays with different shapes during arithmetic operations. The simplest broadcasting example occurs when an array and a scalar value are combined in an operation. If we have a matrix $$A$$ and scalar value $$b$$ then scalar $$b$$ is being stretched during the arithmetic operation into an array which is the same shape as $$A$$, but that stretch is only conceptual. Numpy uses the original scalar value without making copies, so that broadcasting operations are as memory and computationally efficient as possible.

Adding a scalar to a row vector:

{% include image.html image="notes/neural-networks-and-deep-learning/77.png" %}

Adding a scalar to a column vector:

{% include image.html image="notes/neural-networks-and-deep-learning/78.png" %}

Adding a row vector to a matrix:

{% include image.html image="notes/neural-networks-and-deep-learning/79.png" %}

Adding a column vector to a matrix:

{% include image.html image="notes/neural-networks-and-deep-learning/80.png" %}

### Explanation of Logistic Regression Cost Function

The way to motivate logistic regression with the **cross entropy loss function** is to the following

We can now write out the **likelihood estimators** of seeing a particular $$y$$ for a given $$x$$ via

$$ \begin{align*}
P(y=1 \mid x) &= a \\
P(y=0 \mid x) &= 1 - a \\
\end{align*} $$

so that

$$ P(y \mid x) = (a)^y (1 - a)^{1-y} $$

Now, according to the **maximum likelihood principle**, the best values of $$b$$ and $$w$$ are those that maximize the likelihood of the entire dataset:

$$ \begin{align*}
P(Y \mid X) &= \prod_{i=1}^{m} P(y^{(i)} \mid x^{(i)}) \\
&= \prod_{i=1}^{m} (a^{(i)})^{y^{(i)}} (1 - a^{(i)})^{1-y^{(i)}} \\
\end{align*} $$

Estimators chosen according to the maximum likelihood principle are called **Maximum Likelihood Estimators**. While, maximizing the product of many exponential functions, might look difficult, we can simplify things significantly, without changing the objective, by maximizing the **log** of the likelihood instead.

$$ log \text{ } P(Y \mid X) = \sum_{i=1}^{m} \bigg( y^{(i)} log \text{ } a^{(i)} + (1 - y^{(i)}) log(1 - a^{(i)}) \bigg) $$

## Shallow neural networks

### Neural Networks Overview

Logistic Regression model

{% include image.html image="notes/neural-networks-and-deep-learning/86.png" %}

corresponds to the following computation graph:

{% include image.html image="notes/neural-networks-and-deep-learning/87.png" %}

We have a feature vector $$x$$, parameters $$w$$ and $$b$$ as the inputs to the computation graph. That allows us to compute $$z$$ which is then used to compute $$a$$ and we use $$a$$ interchangeably with the output $$\hat{y}$$. Finally, we can compute a loss function. A circle we draw in a Logistic Regression model, we will call a node in the Neural Networks representation. The output of every node in a Neural Network is calculated in two steps: the first compute $$z$$ value and the second computes an $$a$$ value as we can see in the picture below:

{% include image.html image="notes/neural-networks-and-deep-learning/88.png" %}

A neural network is shown in the picture below. We can see we can form a neural network is created by stacking together several node units. One stack of nodes we will call a layer.

{% include image.html image="notes/neural-networks-and-deep-learning/89.png" %}

The first stack of nodes we will call Layer 1, and the second we will call Layer 2. We have two types of calculations in every node in the Layer 1, as well as in the Layer 2 ( which consists of just one node). We will use a superscript square bracket with a number of particular layer to refer to an activation function or a node that belongs to that layer. So, a superscript [1] refers to the quantities associated with the first stack of nodes, called Layer 1. The same is with a superscript [2] which refers to the second layer. Remember also that $$x^{(i)}$$ refers to an individual training example.

The computation graph that corresponds to this Neural Network looks like this:

{% include image.html image="notes/neural-networks-and-deep-learning/89.png" %}

So after computing $$z^{[1]}$$, similarly to the logistic regression, there is a computation of $$a^{[1]}$$ and that’s sigmoid of $$z^{[1]}$$. Next, we compute $$z^{[2]}$$ using another linear equation and then compute $$a^{[2]}$$ which is the final output of the neural network. Let’s remind ourselves once more that $$a^{[2]} = \hat{y}$$. The key intuition to take away is that, whereas for Logistic Regression we had $$z$$ followed by $$a$$ calculation, and in this Neural Network we just do it multiple times.

In the same way, in a Neural Network we’ll end up doing a backward calculation that looks like this:

{% include image.html image="notes/neural-networks-and-deep-learning/91.png" %}

### Neural Network Representation

We will now represent a single layer Neural Network. It is a Neural network with one input layer, one hidden layer and the output layer, which is a single node layer, and it is responsible for generating the predicted value $$\hat{y}$$.

{% include image.html image="notes/neural-networks-and-deep-learning/92.png" %}

We have the following parts of the neural network:

- $$x_1$$, $$x_2$$ and $$x_3$$ are inputs of a Neural Network. These elements are scalars and they are stacked vertically. This also represents an input layer.
- Variables in a hidden layer are not seen in the input set. Thus, it is called a hidden layer.
- The output layer consists of a single neuron only and $$\hat{y}$$ is the output of the neural network.

In the training set we see what the inputs are and we see what the output should be. But the things in the hidden layer are not seen in the training set, so the name hidden layer just means you don’t see it in the training set. An alternative notation for the values of the input features will be $$a^{[0]}$$ and the term $$a$$ also stands for activations. Refers to the values that different layers of the neural network are passing on to the subsequent layers.

The input layer passes on the value $$x$$ to the hidden layer and we're going to call that the activations of the input layer $$a^{[0]}$$. The next layer, the hidden layer will in turn generate some set of activations which we will denote as $$a^{[1]}$$, so in particular, this first unit or this first node will generate the value $$a_1^{[1]}$$, the second node will generate the value $$a_2^{[1]}$$ and so on.

$$a^{[1]}$$ is a 4 × 1 matrix. $$a^{[2]}$$ will be a single value scalar and this is the analogous to the output of the sigmoid function in the logistic regression.

When we count layers in a neural network we do not count an input layer. Therefore, this is a 2-layer neural network. The first hidden layer is associated with parameters $$w^{[1]}$$ and $$b^{[1]}$$. The dimensions of these matrices are:

- $$w^{[1]}$$ is (4,3) matrix
- $$b^{[1]}$$ is (4,1) matrix

Parameters $$w^{[2]}$$ and $$b^{[2]}$$ are associeted with the second layer or actually with the output layer. The dimensions of parameters in the output layer are:

- $$w^{[2]}$$ is (1,4) matrix
- $$b^{[2]}$$ is a real number

### Computing Neural Network Output

Computing an output of a Neural Network is like computing an output in Logistic Regression, but repeating it multiple times. We have said that circle in Logistic Regression, or one node in Neural Network, represents two steps of calculations. We have also said that Logistic Regression is the simplest Neural Network.

{% include image.html image="notes/neural-networks-and-deep-learning/94.png" %}

We will show how to compute the output of the following neural network

{% include image.html image="notes/neural-networks-and-deep-learning/05-a.png" %}

If we look at the first node and write equations for that node, and the same we will do with the second node.

{% include image.html image="notes/neural-networks-and-deep-learning/95.png" %}

{% include image.html image="notes/neural-networks-and-deep-learning/99.png" %}

Calculations for the third and fourth node look the same. Now, we will put all these equations together:

{% include image.html image="notes/neural-networks-and-deep-learning/05-b.png" %}

Calculating all these equations with for loop is highly inefficient so we will to vectorize this.

{% include image.html image="notes/neural-networks-and-deep-learning/05-c.png" %}

So we can define these matrices:

{% include image.html image="notes/neural-networks-and-deep-learning/96.png" %}

To compute the output of a Neural Network we need the following four equations. For the first layer of a Neural network we need these equations:

{% include image.html image="notes/neural-networks-and-deep-learning/97.png" %}

Calculating the output of the Neural Network is like calculating a Logistic Regression with parameters $$W^{[2]}$$ as $$w^T$$ and $$b^{[2]}$$ as $$b$$.

{% include image.html image="notes/neural-networks-and-deep-learning/98.png" %}

### Vectorizing across multiple examples

Logistic Regression Equations

{% include image.html image="notes/neural-networks-and-deep-learning/100.png" %}

These equations tell us how, when given an input feature vector $$x$$, we can generate predictions.

{% include image.html image="notes/neural-networks-and-deep-learning/101.png" %}

If we have $$m$$ training examples we need to repeat this proces $$m$$ times. For each training example, or for each feature vector that looks like this:

{% include image.html image="notes/neural-networks-and-deep-learning/102.png" %}

The notation 𝑎<sup>[2] (𝑖)</sup> means that we are talking about activation in the second layer that comes from 𝑖<sup>𝑡ℎ</sup> training example. In the square parentheses we write number of a layer, and number in the  parentheses reffers to the particular training example.

We will now see eguations for one hidden layer neural network which is presented in the following picture.

{% include image.html image="notes/neural-networks-and-deep-learning/103.png" %}

To do calculations written above, we need a for loop that would look like this:

{% include image.html image="notes/neural-networks-and-deep-learning/104.png" %}

Now our task is to vectorize all these equations and get rid of this for loop.

We will recall definitions of some matrices. Martix 𝐗 was defined as we have put all feature vectors in columns of a matrix, actually we stacked feature vectors horizontally. Every column in matrix 𝐗 is a feature vector for one training example, so the dimension of this matrix is(**number of features in every vector, number of training examples**). Matrix 𝐗 is defined as follows:

{% include image.html image="notes/neural-networks-and-deep-learning/105.png" %}

In the same way we can get the 𝐙<sup>[1]</sup> matrix, as we stack horizontally values 𝑧<sup>[1] (1)</sup> ... 𝑧<sup>[1] (𝑚)</sup>:

{% include image.html image="notes/neural-networks-and-deep-learning/106.png" %}

Similiar is with  values 𝑎<sup>[1] (1)</sup> ... 𝑎<sup>[1] (𝑚)</sup> which are the activations in the first node for paritcular training example:

{% include image.html image="notes/neural-networks-and-deep-learning/107.png" %}

An element in the first row and in the first column of a matrix 𝐀<sup>[1]</sup> is an activation of the first hidden unit and the first training example. In the first row of this matrix there are activations in the first hidden unit among all training examples. The same is with another rows in this matrix. Next element, element in the first row and the second column, is an activation of the first unit from second training element and so on.

{% include image.html image="notes/neural-networks-and-deep-learning/108.png" %}

To conclude, in matrix 𝐀<sup>[1]</sup> there are activation of the first hidden layer of a Neural Network. In every column there are activations for each training example, so number of columns in this matrix is equal to the number of training examples. In the first row of this matrix there are activations first hidden unit among all training examples.

Vectorized version of previous calculations looks like this:

{% include image.html image="notes/neural-networks-and-deep-learning/109.png" %}

In the following picture we can see comparation of vectorized and non-vectorized version.

{% include image.html image="notes/neural-networks-and-deep-learning/110.png" %}

### Explanation For Vectorized Implementation

Let’s go through part of a forward  propagation calculation for a few examples. 𝑥<sup>(1)</sup>, 𝑥<sup>(2)</sup> and 𝑥<sup>(3)</sup> are input vectors, those are three examples of feature vectors or three training examples.

{% include image.html image="notes/neural-networks-and-deep-learning/111.png" %}

We will ignore 𝑏<sup>[1]</sup> values, to simplify these calculations, so we have following equations:

{% include image.html image="notes/neural-networks-and-deep-learning/112.png" %}

So when we multiply matrix 𝐖<sup>[1]</sup> with each training example we get following calculation:

{% include image.html image="notes/neural-networks-and-deep-learning/113.png" %}

So when we multiply matrix 𝐖<sup>[1]</sup> with each training example we get following calculation:

{% include image.html image="notes/neural-networks-and-deep-learning/114.png" %}

And if we multiply 𝐖<sup>[1]</sup> with matrix 𝐗 we will get:

{% include image.html image="notes/neural-networks-and-deep-learning/115.png" %}

If we now put back the value of 𝑏<sup>[1]</sup> in equations values are still correct. What actully happens when we add 𝑏<sup>[1]</sup> values is that we end up with Python broadcasting.

With these equations we have justified that 𝐙<sup>[1]</sup> = 𝐖<sup>[1]</sup>𝐗 + 𝑏<sup>[1]</sup> is a correct vectorization.

### Activation functions

When we build a neural network, one of the choices we have to make is what activation functions to use in the hidden layers as well as at the output unit of the Neural Network. So far, we’ve just been using the sigmoid activation function but sometimes other choices can work much better. Let’s take a look at some of the  options.

**sigmoid activation function**

In the forward propagation steps for Neural Network we use sigmoid function as the activation function.

{% include image.html image="notes/neural-networks-and-deep-learning/116.png" %}

**tanh activation function**

An activation function that almost always goes better than sigmoid function is tanh function. The graphic of this function is the following one:

{% include image.html image="notes/neural-networks-and-deep-learning/117.png" %}

This function is a shifted version of a 𝑠𝑖𝑔𝑚𝑜𝑖𝑑 function but scaled between -1 and 1. If we use a tanh as the activation function it almost always works better then sigmoid function because the mean of all possible values of this function is zero. Actually, it has an effect of centering the data so that the mean of the data is close to zero rather than to 0.5 and it also makes learning easier for the next layers.

When solving a binary classification problem it is better to use sigmoid function because it is more natural choice because if output labels 𝑦 ∈ {0,1} then it makes sence that 𝑦̂ ∈ [0,1].

An activation function may be different for different layers through Neural Network, but in one layer there must be one - the same activation function. We use superscripts is squar parentheses [] to denote to wich layer of a Neural Network belongs each activation function. For example, activation function 𝑔<sup>[1]</sup> is the activation function of the first layer of the Neural Network and 𝑔<sup>[2]</sup> is the activation function of the second layer, as presented in the following picture.

{% include image.html image="notes/neural-networks-and-deep-learning/129.png" %}

When talking about 𝜎(𝑧) and tanh(𝑧) activation functions, one of their downsides is that derivatives of these functions are very small for higher values of 𝑧 and this can slow down gradient descent.

**ReLU and LeakyReLU activation function**

One other choice that is well known in Machine Learning is ReLU function. This function is commonly used activation function nowadays.

{% include image.html image="notes/neural-networks-and-deep-learning/123.png" %}

There is one more function, and it is modification of ReLU function. It is a  LeakyReLU function. LeakyReLU usually works better then ReLU function. Here is a graphical representation of this function:

{% include image.html image="notes/neural-networks-and-deep-learning/124.png" %}

### Why Non-linear Activation Functions

For this shallow Neural Network:

{% include image.html image="notes/neural-networks-and-deep-learning/05-a.png" %}

we have following propagation steps:

{% include image.html image="notes/neural-networks-and-deep-learning/130.png" %}

If we want our activation functions to be linear functions, so that we have 𝑔<sup>[1]</sup> = 𝑧<sup>[1]</sup> and 𝑔<sup>[2]</sup> = 𝑧<sup>[2]</sup>, then these equations above become:

{% include image.html image="notes/neural-networks-and-deep-learning/128.png" %}

Now, it’s clear that if we use a linear activation function (identity activation function), then the Neural Network will output linear output of the input. This loses much of the representational power of the neural network as often times the output that we are trying to predict has a non-linear relationship with the inputs. It can be shown that if we use a linear activation function for a hidden layer and sigmoid function for an output layer, our model becomes logistic regression model. Due to the fact that a composition of two linear functions is linear function, our area of implementing such Neural Network reduces rapidly. Rare implementation example can be solving regression problem in machine learning (where we use linear activation function in hidden layer). Recommended usage of linear activation function is to be implemented in output layer in case of regression.

### Derivatives of activation functions

**Derivative of sigmoid function**

{% include image.html image="notes/neural-networks-and-deep-learning/119.png" %}

We denote an activation function with 𝑎, so we have:

{% include image.html image="notes/neural-networks-and-deep-learning/120.png" %}

**Derivative of a tahn function**

{% include image.html image="notes/neural-networks-and-deep-learning/121.png" %}

{% include image.html image="notes/neural-networks-and-deep-learning/122.png" %}

**Derivatives of ReLU and LeakyReLU activation functions**

A derivative of a ReLU function is:

{% include image.html image="notes/neural-networks-and-deep-learning/125.png" %}

The derivative of a ReLU function is undefined at 0, but we can say that derivative of this function at zero is either 0 or 1. Both solution would work when they are implemented in software. The same solution works for LeakyReLU function.

{% include image.html image="notes/neural-networks-and-deep-learning/126.png" %}

Derivative of LeakyReLU function is :

{% include image.html image="notes/neural-networks-and-deep-learning/127.png" %}

### Gradient descent for Neural Networks

we will see how to implement gradient descent for one hidden layer Neural Network as presented in the picture below.

{% include image.html image="notes/neural-networks-and-deep-learning/131.png" %}

Parameters for one hidden layer Neural Network are 𝐖<sup>[1]</sup>, 𝑏<sup>[1]</sup>, 𝐖<sup>[2]</sup> and 𝑏<sup>[2]</sup>. Number of unitis in each layer are:

  - input of a Neural Network is feature vector ,so the length of “zero” layer 𝑎<sup>[0]</sup> is the size of an input feature vector 𝑛<sub>𝑥</sub> = 𝑛<sup>[0]</sup>
  - number of hidden units in a hidden layer is 𝑛<sup>[1]</sup>
  - number of units in output layer is 𝑛<sup>[2]</sup>, so far we had one unit in an output layer so 𝑛<sup>[2]</sup>

  As we have defined a number of units in hidden layers we can now tell what are dimension of the following matrices:

  - 𝐖<sup>[1]</sup> is (𝑛<sup>[1]</sup>,𝑛<sup>[0]</sup>) matrix
  - 𝑏<sup>[1]</sup> is (𝑛<sup>[1]</sup>,1) matrix or a column vector
  - 𝐖<sup>[2]</sup> is (𝑛<sup>[2]</sup>,𝑛<sup>[1]</sup>) matrix
  - 𝑏<sup>[2]</sup> is (𝑛<sup>[2]</sup>,1) , so far 𝑏<sup>[2]</sup> is a scalar

Notation:

{% include image.html image="notes/neural-networks-and-deep-learning/141.png" %}

Equations for one example 𝑥<sup>(𝑖)</sup>:

{% include image.html image="notes/neural-networks-and-deep-learning/132.png" %}

Assuming that we are doing a binary classification, and assuming that we have 𝑚 training examples, the cost function 𝐽 is:

{% include image.html image="notes/neural-networks-and-deep-learning/133.png" %}

To train parameters of our algorithm we need to perform gradient descent. When training neural network, it is important to initialize the parameters randomly rather then to all zeros. So after initializing the paramethers we get into gradient descent which looks like this:

{% include image.html image="notes/neural-networks-and-deep-learning/134.png" %}

So we need equations to calculate these derivatives.

Forward propagation equations (remember that if we are doing a binary classification then the activation function in the output layer is a sigmoid function):

{% include image.html image="notes/neural-networks-and-deep-learning/135.png" %}

Now we will show equations in the backpropagation step:

{% include image.html image="notes/neural-networks-and-deep-learning/136.png" %}

Sign ∗ stands for element  wise multiplication.

### Backpropagation Intuition

We will now the relation between a computation graph and these equations.

{% include image.html image="notes/neural-networks-and-deep-learning/142.png" %}

{% include image.html image="notes/neural-networks-and-deep-learning/137.png" %}

We have defined a loss function the actual loss when the ground truth label is 𝑦, and our output is 𝑎:

{% include image.html image="notes/neural-networks-and-deep-learning/138.png" %}

And corresponding derivatives are:

{% include image.html image="notes/neural-networks-and-deep-learning/139.png" %}

Backprpagation grapf is a graph that describes which calculations do we need to make when we want to calculate various derivatives and do the parameters update. In the following graph we can see that it is similar to the Logistic Regression grapf except that we do those calculations twice.

{% include image.html image="notes/neural-networks-and-deep-learning/140.png" %}

Firstly, we calculate 𝑑𝑎<sup>[2]</sup>, 𝑑𝑧<sup>[2]</sup> and these calculations allows us to calculate 𝐝𝐖<sup>[2]</sup> and 𝑑𝑏<sup>[2]</sup>. Then, as we go deeper in the backpropagation step, we calculate 𝑑𝑎<sup>[1]</sup>, 𝑑𝑧<sup>[1]</sup> which allows us to calculate 𝐝𝐖<sup>[1]</sup> and 𝑑𝑏<sup>[1]</sup>.

### Random Initialization

If we have for example this shallow Neural Network:

{% include image.html image="notes/neural-networks-and-deep-learning/145.png" %}

{% include image.html image="notes/neural-networks-and-deep-learning/149.png" %}

Even if we have a lot of hidden units in the hidden layer they all are symetric if we initialize corresponding parameters to zeros. To solve this problem we need to initialize randomly rather then with zeros. We can do it in the following way (we consider the same shallow neural network with 2 hidden units in the hidden layer as above):

{% include image.html image="notes/neural-networks-and-deep-learning/146.png" %}

And then we can initialize 𝑏<sub>1</sub> with zeros, because initialization of 𝑊<sub>1</sub> breaks the symmetry, and unit1 and unit2 will not output the same value even if we initialize 𝑏<sub>1</sub> to zero. So we have:

{% include image.html image="notes/neural-networks-and-deep-learning/147.png" %}

For the output layer we have:

{% include image.html image="notes/neural-networks-and-deep-learning/148.png" %}

Why do we multipy with 0.01 rather then multiplying with 100 for example? What happens if we initialize parameters randomly but with big random values?

If we are doing a binary classification and the activation in the output layer is sigmoid function or if use tanh activation function in the hidden layers then for a not so high input value these functions get saturated, for a not so big inputs they become constant (they output 0 or 1 for sigmoid or -1 or 1 for tanh function).

So, we do the initialization of parameters 𝐖<sup>[1]</sup> and 𝐖<sup>[2]</sup> with small random values, hence we multipy with 0.01.

Random initialization is used to break symmetry and make sure different hidden units can learn different things.

We can conclude that we must initialize our parameters with small random values.

Well chosen initialization values of parameters leads to:

  - Speed up convergence of gradient descent.
  - Increase the likelihood of gradient descent to find lower training error rates

## Deep Neural Networks

### Deep L-layer neural network

Let's make a Neural Network overview. We will see what is the simplest representation of a Neural Network and how deep representation of a Neural Network looks like.

We have defined a Logistic Regression as a single unit that uses sigmoid activation function. Both of these simple Neural Networks we also call shallow neural networks and they are only reasonable to be applied when classifying linearly separable classes.

{% include image.html image="notes/neural-networks-and-deep-learning/150.png" %}

Slightly more complex neural network is a two layer neural network (it is a neural network with one hidden layer). This shallow neural network can classify two datasets that are not linaearly separable, but it is not good at classifying more compelex datasets.

{% include image.html image="notes/neural-networks-and-deep-learning/151.png" %}

A little bit more complex model than previous one is a tree layer neural network (it is a neural network with two nidden layers):

{% include image.html image="notes/neural-networks-and-deep-learning/152.png" %}

Even more complex neural network, which we can call **deep neural** network, is for example, a six layer neural network (or neural network with five hidden layers):

{% include image.html image="notes/neural-networks-and-deep-learning/153.png" %}

When counting layers in a neural network we count hidden layers as well as the output layer, **but we don’t count an input layer**.

{% include image.html image="notes/neural-networks-and-deep-learning/154.jpg" %}

Here is the notation overview that we will use to describe deep neural networks:

Here is a four layer neural network, so it is a neural network with three hidden layers. Notation we will use for this neural network is:

  - 𝐿 to denote the number of layers in a neural network
    - in this neural network 𝐿=4
  - 𝑛<sup>[𝑙]</sup> to denote a number of layers in the 𝑙<sup>𝑡ℎ</sup> layer
    - 𝑛<sup>[1]</sup>=4, there are four units in the first layer
    - 𝑛<sup>[2]</sup>=4, there are four units in the second layer
    - 𝑛<sup>[3]</sup>=3, there are three units in the thirs layer
    - 𝑛<sup>[4]</sup>=1, this neural network outputs a scalar value
    - 𝑛<sup>[0]</sup>=𝑛<sub>𝑥</sub>=3 because input vector, feature vector, has three features
  - 𝑎<sup>[𝑙]</sup>=𝑔(𝑧<sup>[𝑙]</sup>) to denote activation functions in the 𝑙<sup>𝑡ℎ</sup> layer
    - 𝑥=𝑎<sup>[0]</sup>
  - 𝐖<sup>[𝑙]</sup> to denote weights for computing 𝑧<sup>[𝑙]</sup>

### Forward Propagation in a Deep Network

Once again we will see how the forward propagation equations look like. We will show equation for the neural network ilustrated above. In addition, below every two equations we will show the dimensions of vectors or matrices used in the calculations.

A vectorized version of these equations, equations considering all input examples, and correspodnding dimensions of these matrices (which are printed in gray as above) are:

{% include image.html image="notes/neural-networks-and-deep-learning/155.png" %}

From equations we have written, we can see that generalized equations for layer 𝑙:

{% include image.html image="notes/neural-networks-and-deep-learning/156.png" %}

In case that you are thinking how can we add 𝑏<sup>[𝑙]</sup>

Notice that, when making a calculation for the first layer, we can also write 𝑧<sup>[1]</sup>=𝐖<sup>[1]</sup>𝑎<sup>[0]</sup>+𝑏<sup>[1]</sup>. So, instead of using 𝑥 we use 𝑎<sup>[0]</sup> as an activations in the input layer. 𝑔<sup>[1]</sup> is activation function in the first layer. Remember that we can choose different activation functions in a Neural Network, but in a single layer we must use the same activation function, so in the output layer we have 𝑔<sup>[2]</sup> as the activation function and so on.

Matrix Ŷ is a matrix of predictions for all input examples, so it is the output of a neural network when the input is matrix 𝐗, matrix of all input examples (or a feature matrix).

We can see that there must be a 𝑓𝑜𝑟 loop, going through all layers in a neural network and calculating all 𝑍<sup>[𝑙]</sup> and 𝐴<sup>[𝑙]</sup> values (where 𝑙 is a number of the layer). Here, it is prefectly fine to use an explicit for loop.

### Getting your matrix dimensions right

forward propagation matrix dimensions check

{% include image.html image="notes/neural-networks-and-deep-learning/158.png" %}

From equations we have written, we can see that generalized equations for layer 𝑙:

{% include image.html image="notes/neural-networks-and-deep-learning/159.png" %}

### Why deep representations

We’ve heard that neural networks work really well for a lot of problems. However, neural networks doesn’t need only to be big. Neural Networks also need to be deep or to have a lot hidden layers.

If we are, for example, building a system for an image classification, here is what a deep neural network could be computing. The input of a neural network is a picture of a face. The first layer of the neural network could be a feature detector, or an edge detector. So, the first layer can look at the pictures and find out where are the edges in the picture. Then, in next layer those detected edges could be grouped together to form parts of faces. By putting a lot of edges it can start to detect different parts of faces. For example, we might have a low neurons trying to see if it’s finding an eye or a different neuron trying to find  part of a nose. Finally, putting together eyes, nose etc. it can recognise different faces.

{% include image.html image="notes/neural-networks-and-deep-learning/170.png" %}

To conclude, earlier layers of a neural network detects simpler functions (like edges), and composing them together, in the later layers of a neural network, deep neural network can compute more complex functions.

In case of trying to build a speech recognition system, the first layer could detect if a tone is going up or down or is it a white noise or a slithering sound or some other low level wave of features. In the following layer by composing low level wave forms, nural network might be abe to learn to detect basic units of sound – phonems. In the word cat phonemes are c, a and t. Composing all this together a deep neural network might be able to recognize words and maybe sentences.

So the general intuition behind everything we have said is that earlier layers learn lower level simple features and then later deep layers put together the simpler things it has detected in order to detect more complex things, so that a deep neural network can do some really complex things.

### Building blocks of deep neural networks

Let's see what are the building blocks of a Deep Neural Network.

We will pick one layer, for example layer 𝑙 of a deep neural network and we will focus on computatons for that layer. For layer 𝑙 we have parameters 𝐖<sup>[𝑙]</sup> and 𝑏<sup>[𝑙]</sup>. Calculation of the forward pass for layer 𝑙 we get as we input activations from the previous layer and as the output we get  activations of the current layer, layer 𝑙.

{% include image.html image="notes/neural-networks-and-deep-learning/171.png" %}

Equations for this calculation step are:

{% include image.html image="notes/neural-networks-and-deep-learning/172.png" %}

where 𝑔(𝑧<sup>[𝑙]</sup>) is an activation function in the layer 𝑙.

It is good to cache the value of 𝑧<sup>[𝑙]</sup> for calculations in backwardpass.

Backward pass is done as we input 𝑑𝑎<sup>[𝑙]</sup> and we get the output 𝑑𝑎<sup>[𝑙−1]</sup>, as presented in the following graph. We will always draw backward passes in red.

{% include image.html image="notes/neural-networks-and-deep-learning/173.png" %}

In the following picture we can see a diagram of both a forward and a backward pass in the layer 𝑙. So, to calulate values in the backward pass we need cached values. Here we just draw 𝑧<sup>[𝑙]</sup> as a cached value, but indeed we will need to cache also 𝑊<sup>[𝑙]</sup> and 𝑏<sup>[𝑙]</sup>.

{% include image.html image="notes/neural-networks-and-deep-learning/174.png" %}

If we implement these two calculations as presented in a graph above, the computation for an 𝐿 layer neural network will be as follows. We will get 𝑎[0], which is our feature vector, feed it in, and that will compute the activations of the first layer. The same thing we will do with next layers.

{% include image.html image="notes/neural-networks-and-deep-learning/08.png" %}

Having all derivative terms we can update parameters:

{% include image.html image="notes/neural-networks-and-deep-learning/175.png" %}

In our programming implementation of this algorithm, when we cache 𝑧<sup>[𝑙]</sup> for backpropagation calculations we will cache also 𝐖<sup>[𝑙]</sup> and 𝑏<sup>[𝑙]</sup>, and 𝑎<sup>[𝑙−1]</sup>.

### Forward and Backward Propagation

Here, we will see equations for calculating the forward step.

{% include image.html image="notes/neural-networks-and-deep-learning/176.png" %}

Here we will see equations for caluculating backward step:

{% include image.html image="notes/neural-networks-and-deep-learning/157.png" %}

Remember that ∗ represents an element wise multiplication.

A multi layer neural network is presented in the picture below:

{% include image.html image="notes/neural-networks-and-deep-learning/177.png" %}

{% include image.html image="notes/neural-networks-and-deep-learning/180.png" %}

{% include image.html image="notes/neural-networks-and-deep-learning/178.png" %}

and for vectorized version we have:

{% include image.html image="notes/neural-networks-and-deep-learning/179.png" %}

### Parameters vs Hyperparameters

For building a deep neural network it is very important to organize both parameters and hyperparameters. Parameters of a deep neural network are 𝐖<sup>[1]</sup>,𝑏<sup>[1]</sup>,𝐖<sup>[2]</sup>,𝑏<sup>[2]</sup>,𝐖<sup>[3]</sup>,𝑏<sup>[3]</sup> ... and deep neural network also has other parameters which are crucial for our algorithm. Those parameters are :

  - a learning rate 𝛼
  - a number of iteration
  - a number of layers 𝐿
  - a number of hidden units 𝑛<sup>[1]</sup>,𝑛<sup>[2]</sup>,…𝑛<sup>[𝐿]</sup>
  - a choice of activation function

These are parameters that control our parameters 𝐖<sup>[1]</sup>,𝑏<sup>[1]</sup>,𝐖<sup>[2]</sup>,𝑏<sup>[2]</sup>,𝐖<sup>[3]</sup>,𝑏<sup>[3]</sup> ... and we call them **hyperparameters**. In deep learning there are also these parameters: momentum, bach size, number of epochs etc ...

We can see that **hyperparameters** are the variables that determines the network structure and the variables which determine how the network will be trained. Notice also that **hyperparameters** are set before training or before optimizing weights and biases.

To conclude, model parameters are estimated from data during the training step and model hyperparameters are manually set earlier and are used  to assist estimation model parameter.
