# Neural Networks and Deep Learning

* [Neural Networks and Deep Learning](#neural-networks-and-deep-learning)
    * [Introduction to deep learning](#introduction-to-deep-learning)
      * [What is a Neural Network](#what-is-a-neural-network)
      * [Supervised learning with neural networks](#supervised-learning-with-neural-networks)
      * [Scale drives machine learning progress](#scale-drives-machine-learning-progress)
   * [Neural Networks Basics](#neural-networks-basics)
      * [Binary classification](#binary-classification)
      * [Logistic regression](#logistic-regression)
      * [Logistic regression cost function](#logistic-regression-cost-function)
      * [Gradient Descent](#gradient-descent)
      * [Computation graph](#computation-graph)
      * [Derivatives with a Computation Graph](#derivatives-with-a-computation-graph)
      * [Logistic Regression Gradient Descent](#logistic-regression-gradient-descent)
      * [Gradient Descent on m Examples](#gradient-descent-on-m-examples)
      * [Vectorization](#vectorization)
      * [Vectorizing Logistic Regression](#vectorizing-logistic-regression)
      * [Broadcasting in Python](#Broadcasting-in-Python)
      * [Explanation of Logistic Regression Cost Function](#Explanation-of-Logistic-Regression-Cost-Function)
   * [Shallow neural networks](#shallow-neural-networks)
      * [Neural Networks Overview](#neural-networks-overview)
      * [Neural Network Representation](#neural-network-representation)
      * [Computing a Neural Network's Output](#computing-a-neural-networks-output)
      * [Vectorizing across multiple examples](#vectorizing-across-multiple-examples)
      * [Activation functions](#activation-functions)
      * [Why do you need non-linear activation functions?](#why-do-you-need-non-linear-activation-functions)
      * [Derivatives of activation functions](#derivatives-of-activation-functions)
      * [Gradient descent for Neural Networks](#gradient-descent-for-neural-networks)
      * [Random Initialization](#random-initialization)
   * [Deep Neural Networks](#deep-neural-networks)
      * [Deep L-layer neural network](#deep-l-layer-neural-network)
      * [Forward Propagation in a Deep Network](#forward-propagation-in-a-deep-network)
      * [Getting your matrix dimensions right](#getting-your-matrix-dimensions-right)
      * [Why deep representations?](#why-deep-representations)
      * [Building blocks of deep neural networks](#building-blocks-of-deep-neural-networks)
      * [Forward and Backward Propagation](#forward-and-backward-propagation)
      * [Parameters vs Hyperparameters](#parameters-vs-hyperparameters)
      * [What does this have to do with the brain](#what-does-this-have-to-do-with-the-brain)

## Introduction to deep learning

### What is a Neural Network

Let’s start with the house price prediction example. Suppose that you have a dataset with six houses and we know the price and the size of these houses. We want to fit a function to predict the price of these houses with respect to its size.

<div align="center">
  <img src="Images/12.png">
</div>

We will put a straight line through these data points. Since we know that our prices cannot be negative, we end up with a horizontal line that passes through 0.

<div align="center">
  <img src="Images/13.png">
</div>

The blue line is the function for predicting the price of the house as a function of its size. You can think of this function as a very simple neural network.

The input to the neural network is the size of a house, denoted by 𝑥, which goes into a single neuron and then outputs the predicted price, which we denote by 𝑦.


<div align="center">
  <img src="Images/14.png">
</div>

If this is a neural network with a single neuron, a much larger neural network is formed by taking many of the single neurons and stacking them together.

A basic Neural Network with more features is ilustrated in the following image.

![](Images/15.jpg)

### Supervised learning with neural networks

In supervised learning, we have some input 𝑥, and we want to learn a function mapping to some output 𝑦. Just like in the house price prediction application our input were some features of a home and our goal was to estimate the price of a home 𝑦.

Here are some other fields where neural networks have been applied very effectively.

![](Images/16.png)

We might input an image and want to output an index from one to a thousand, trying to tell if this picture might be one of a thousand different image classes. This can be used for photo tagging.

The recent progress in speech recognition has also been very exciting. Now you can input an audio clip to a neural network and can have it output a text transcript.

Machine translation has also made huge strikes thanks to deep learning where now you can have a neural network input an English sentence and directly output a Chinese sentence.

Different types of neural networks are useful for different applications.

![](Images/17.png)

  - In the real estate application, we use a universally **Standard Neural Network** architecture.
  - For image applications we’ll often use **Convolutional Neural Network (CNN)**.
  - Audio is most naturally represented as a one-dimensional time series or as a one-dimensional temporal sequence. Hence, for a sequence data, we often use **Recurrent Neural Network (RNN)**.
  - Language, English and Chinese, the alphabets or the words come one at a time and language is also represented as a sequence data. **Recurrent Neural Network (RNN)** are often used for these applications.

Structured and Unstructured Data

Machine learning is applied to both Structured Data and Unstructured Data.

Structured Data means basically databases of data. In house price prediction, you might have a database or the column that tells you the size and the number of bedrooms.

In predicting whether or not a user will click on an ad, we might have information about the user, such as the age, some information about the ad, and then labels that you’re trying to predict.

<div align="center">
  <img src="Images/18.png">
</div>

Structured data means, that each of the features, such as a size of the house, the number of bedrooms, or the age of a user, have a very well-defined meaning. In contrast, unstructured data refers to things like audio, raw audio, or images where you might want to recognize what’s in the image or text. Here, the features might be the pixel values in an image or the individual words in a piece of text.

<div align="center">
  <img src="Images/19.png">
</div>

Neural networks, computers are now much better at interpreting unstructured data as compared to just a few years ago. This creates opportunities for many new exciting applications that use speech recognition, image recognition, natural language processing of text.

### Scale drives machine learning progress

Many of the ideas of deep learning (neural networks) have been around for decades. Why are these ideas taking off now?

If we plot the performance of traditional learning algorithms such as Support Vector Machine or Logistic Regression as a function of the amount of data. We will get the following curve. In detail, even as you accumulate more data, usually the performance of traditional learning algorithms, plateaus. This means its learning curve flattens out, and the algorithm stops improving even as you give it more data. It was as if the traditional learning algorithms didn’t know what to do with all the data we now have.

<div align="center">
  <img src="Images/11.png">
</div>

With neural networks, it turns out that if you train a very large neural network then its performance often keeps getting better and better.

Three of the biggest drivers of recent progress have been:

  - Data availability:
    - People are now spending more time on digital devices (laptops, mobile devices). Their digital activities generate huge amounts of data that we can feed to our learning algorithms.

  - Computational scale:
    - We started just a few years ago, techniques (like GPUs/Powerful CPUs/Distributed computing) to be able to train neural networks that are big enough to take advantage of the huge datasets we now have.

  - Algorithm:
    - Creative algorithms has appeared that changed the way NN works. Using **RELU function** is so much better than using **Sigmoid function** in training a NN.

To conclude, often you have an idea for a neural network architecture and you want to implement it in code. Fast computation is important because the process of training a neural network is very iterative and can be time-consuming. Implementing our idea then lets us run an experiment which tells us how well our neural network does. Then, by looking at it, you go back to change the details of our neural network and then you go around this circle over and over, until we get the desired performance.

<div align="center">
  <img src="Images/20.png">
</div>

## Neural Networks Basics

### Binary classification

Binary classification is the task of classifying elements of a given set into two classification.

![](Images/21.png)

A binary classification problem:

  - We have an input image 𝑥 and the output 𝑦 is a label to recognize the image.
  - 1 means cat is on an image, 0 means that a non-cat object is on an image.

In binary classification, our goal is to learn a classifier that can input an image represented by its feature vector 𝑥 and predict whether the corresponding label is 1 or 0. That is, whether this is a cat image or a non-cat image.

Image representation in a computer

The computer stores 3 separate matrices corresponding to the red, green and blue (RGB) color channels of the image. If the input image is 64 by 64 pixels, then we would have three 64 by 64 matrices corresponding to the red, green and blue pixel intensity values for our image. For a 64 by 64 image – the total dimension of this vector will be 64 * 64 * 3 = 12288.

![](Images/22.png)

Notation that we will follow is shown in the table below:

![](Images/23.png)

### Logistic regression

Logistic regression is a supervised learning algorithm that we can use when labels are either 0 or 1 and this is the so-called **Binary Classification Problem**. An input feature vector 𝑥 may correspond to an image that we want to recognize as either a cat picture (1) or a non-cat picture (0). That is, we want an algorithm to output the prediction which is an estimate of 𝑦:

![](Images/24.png)

More formally, we want 𝑦̂ to be the chance that 𝑦 is equal to 1, given the input features 𝑥. In other words, if 𝑥 is a picture, we want 𝑦 to tell us what is the chance that this is a cat picture.

The 𝑥 is an 𝑛<sup>𝑥</sup> – dimensional vector. The parameters of logistic regression are 𝑤, which is also an 𝑛<sup>𝑥</sup> – dimensional vector together with 𝑏 wich is a real number.

Given an input 𝑥 and the parameters  𝑤 and 𝑏, how do we generate the output 𝑦̂, One thing we could try, that doesn’t work, would be to have: 𝑦̂ = 𝑤<sup>𝑇</sup>𝑥 + 𝑏 which is a linear function of the input and in fact, this is what we use if we were doing Linear Regression.

However, this is not a very good algorithm for binary classification, because we want 𝑦̂ to be the chance that 𝑦 is equal to 1, so 𝑦̂ should be between 0 and 1.

It is difficult to enforce this because 𝑤<sup>𝑇</sup>𝑥 + 𝑏 can be much bigger than 1 or can even be negative which doesn’t make sense for a probability that we want to be in a range between 0 and 1. We can conclude that we need a function which will transform 𝑦̂ = 𝑤<sup>𝑇</sup>𝑥 + 𝑏 to be in a range between 0 and 1.

Let’s see one function that can help us do that. In logistic regression, the output is going to be the **Sigmoid Function**. We can see that it goes smoothly from  0 up to 1.

<div align="center">
  <img src="Images/25.jpeg">
</div>

  - use 𝑧 to denote the following quantity 𝑤<sup>𝑇</sup>𝑥 + 𝑏.
  - we have: 𝑦̂ = 𝜎(𝑤<sup>𝑇</sup>𝑥 + 𝑏).
  - if 𝑧 is a large positive number: 𝜎(𝑧) ≈ 1
  - if 𝑧 is a large negative number: 𝜎(𝑧) ≈ 0

When we implement logistic regression, our job is to try to learn parameters 𝑤 and 𝑏, so that 𝑦̂  becomes a good estimate of the chance of 𝑦 being equal to 1.

### Logistic regression cost function

First, to train parameters 𝑤 and 𝑏 of a logistic regression model we need to define a **cost function**.

Given a training set of 𝑚 training examples, we want to find parameters 𝑤 and 𝑏, so that 𝑦̂ is as close to 𝑦 (ground truth).

Here, we will use (𝑖) superscript to index different training examples.

Henceforth, we will use **loss (error) function** to measure how well our algorithm is doing. The loss function is applied only to a single training sample, and commonly used loss function is a **squared error**:

![](Images/26.png)

In logistic regression squared error loss function is not an optimal choice. It results in an optimization problem which is not convex, and the gradient descent algorithm may not work well, it may not converge optimally.

In terms of a surface, the surface is convex if, loosely speaking, it looks like a parabola. If you have a ball and let it roll along the surface, that surface is convex if that ball is guaranteed to always end up at the same point in the end. However, if the surface has bumps, then, depending on where you drop the ball from, it might get stuck somewhere else. That surface is then non-convex.

![](Images/27.png)

To be sure that we will get to the global optimum, we will use following loss function:

![](Images/28.png)

It will give us a convex optimization problem and it is therefore much easier to be optimized.

To understand why this is a good choice, let’s see these two cases:

![](Images/29.png)

A cost function measures how well our parameters 𝑤 and 𝑏 are doing on the entire training set :

![](Images/30.png)

  - Cost function 𝐽 is defined as an average of a sum of loss functions of all training examples.
  - Cost function is a function of parameters 𝑤 and 𝑏.

In cost function diagram, the horizontal axes represent our spatial parameters, 𝑤 and 𝑏. In practice, 𝑤 can be of a much higher dimension, but for the purposes of plotting, we will illustrate 𝑤 and 𝑏 as scalars.

The cost function 𝐽(𝑤,𝑏) is then some surface above these horizontal axes 𝑤 and 𝑏. So, the height of the surface represents the value of 𝐽(𝑤,𝑏) at a certain point. Our goal will be to minimize function 𝐽, and to find parameters 𝑤 and 𝑏.

### Gradient Descent

Gradient Descent is an algorithm that tries to minimize the cost function 𝐽(𝑤,𝑏) and to find optimal values for 𝑤 and 𝑏.

For the purpose of illustration we will use 𝐽(𝑤), function that we want to minimize, as a function of one variable. To make this easier to draw, we are going to ignore 𝑏 for now, just to make this a one-dimensional plot instead of a high-dimensional plot.

Gradient Descent starts at an initial parameter and begins to take values in the steepest downhill direction. Function 𝐽(𝑤,𝑏) is convex, so no matter where we initialize, we should get to the same point or roughly the same point.

After a single step, it ends up a little bit down and closer to a global otpimum because it is trying to take a step downhill in the direction of steepest descent or quickly down low as possible.

After a fixed number of iterations of Gradient Descent, hopefully, will converge to the global optimum or get close to the global optimum.

The learning rate 𝛼 controls how big step we take on each iteration of Gradient Descent.

![](Images/31.jpeg)

If the derivative is positive, 𝑤 gets updated as 𝑤 minus a learning rate 𝛼 times the derivative 𝑑𝑤.

We know that the derivative is positive, so we end up subtracting from 𝑤 and taking a step to the left. Here, Gradient Descent would make your algorithm slowly decrease the parameter if you have started off with this large value of 𝑤.

Next, when the derivative is negative (left side of the convex function),  the Gradient Descent update would subtract 𝛼 times a negative number, and so we end up slowly increasing 𝑤 and we are making 𝑤 bigger and bigger with each successive iteration of Gradient Descent.

So, whether you initialize 𝑤 on the left or on the right, Gradient Descent would move you towards this global minimum.

![](Images/32.png)

### Computation graph

Let’s say that we’re trying to compute a function 𝐽, which is a function of three variables 𝑎, 𝑏, and 𝑐 and let’s say that function 𝐽 is 3(𝑎 + 𝑏𝑐).

![](Images/40.png)

Computation of this function has actually three distinct steps:

  - Compute 𝑏𝑐 and store it in the variable 𝑢, so 𝑢 = 𝑏𝑐
  - Compute 𝑣 = 𝑎 + 𝑢,
  - Output 𝐽 is 3𝑣.

Let’s summarize:

![](Images/41.png)

In this simple example we see that, through a left-to-right pass, you can compute the value of 𝐽.

### Derivatives with a Computation Graph

How to figure out derivative calculations of the function 𝐽.

Now we want using a computation graph to compute the derivative of 𝐽 with respect to 𝑣. Let’s get back to our picture, but with concrete parameters.

![](Images/42.png)

First, let’s see the final change of value 𝐽 if we change 𝑣 value a little bit:

![](Images/43.png)

We can get the same result if we know calculus:

![](Images/44.png)

We emphasize that calculation of d𝐽/d𝑣 is one step of a back propagation. The following picture depicts **forward propagation** as well as **backward propagation**:

![](Images/45.png)

Next, what is d𝐽/d𝑎. If we increase 𝑎 from 5 to 5.001, 𝑣 will increase to 11.001 and 𝐽 will increase to 33.003. So, the increase to 𝐽 is the three times the increase to 𝑎 so that means this derivative is equal to 3.

![](Images/46.png)

One way to break this down is to say that if we change 𝑎, that would change 𝑣 and through changing 𝑣 that would change 𝐽. By increasing 𝑎, how much 𝐽 changed is also determined by d𝑣/d𝑎. This is called a **chain rule** in calculus:

![](Images/49.png)

Now, let’s calculate derivative d𝐽/d𝑢.

![](Images/47.png)

Finally, we have to find the most important values: value of d𝐽/d𝑏 and d𝐽/d𝑐. Let’s calculate them:

![](Images/48.png)

### Logistic Regression Gradient Descent

Why do we need a computation graph? To answer this question, we have to check how the computation for our neural network is organized. There are two important principles in neural network computation:

  - Forward pass or forward propagation step
  - Backward pass or backpropagation step

During NN’s **forward propagation step** we compute the output of our neural network. In a binary classification case, our neural network output is defined by a variable and it can have any value from [0,1] interval.

In order to actually train our neural network (find parameters 𝑤 and 𝑏 as local optima of our cost function) we have to conduct a **backpropagation step**. In this way, we can compute gradients or compute derivatives. With this information, we are able to implement gradient descent algorithm for finding optimal values of 𝑤 and 𝑏. That way we can train our neural network and expect that it will do well on a classification task.

A computation graph is a systematic and easy way to represent our neural network and it is used to better understand (or compute) derivatives or neural network output.

The computation graph of a logistic regression looks like the following:

![](Images/04.png)

In this example, we only have two features 𝑥<sub>1</sub> and 𝑥<sub>2</sub>. In order to compute 𝑧, we will need to input 𝑤<sub>1</sub>, 𝑤<sub>2</sub> and 𝑏 in addition to the feature values 𝑥<sub>1</sub> and 𝑥<sub>2</sub>

![](Images/33.png)

After that, we can compute our 𝑦̂ (equals sigma of 𝑧)

![](Images/34.png)

Finally, we are able to compute our loss function.

![](Images/39.png)

To reduce our loss function (remember right now we are talking only about one data sample) we have to update our 𝑤 and 𝑏 parameters. So, first we have to compute the loss using forward propagation step. After this, we go in the opposite direction (backward propagation step) to compute the derivatives.

![](Images/35.png)

Having computed 𝑑𝑎, we can go backwards and compute 𝑑𝑧:

![](Images/36.png)

The final step in back propagation is to go back to compute amount of change of our parameters 𝑤 and 𝑏:

![](Images/37.png)

To conclude, if we want to do gradient descent with respect to just this one training example, we would do the following updates

![](Images/38.png)

### Gradient Descent on m Examples

The cost function is the average of our loss function, when the algorithm outputs 𝑎<sup>(𝑖)</sup> for the pair (𝑥<sup>(𝑖)</sup>,𝑦<sup>(𝑖)</sup>).

![](Images/50.png)

Here 𝑎<sup>(𝑖)</sup> is the prediction on the 𝑖-th training example which is sigmoid of 𝑧<sup>(𝑖)</sup>, were 𝑧<sup>(𝑖)</sup> = 𝑤<sup>𝑇</sup>𝑥<sup>(𝑖)</sup> + 𝑏

![](Images/51.png)

The derivative with respect to 𝑤<sub>1</sub> of the overall cost function, is the average of derivatives with respect to 𝑤<sub>1</sub> of the individual loss term,

![](Images/52.png)

and to calculate the derivative d𝑤<sub>1</sub> we compute,

![](Images/53.png)

This gives us the overall gradient that we can use to implement logistic regression.

To implement Logistic Regression, here is what we can do, if 𝑛=2, were 𝑛 is our number of features and 𝑚 is a number of samples.

![](Images/54.png)

After leaving the inner for loop, we need to divide 𝐽, d𝑤<sub>1</sub>, d𝑤<sub>2</sub> and 𝑏 by 𝑚, because we are computing their average.

![](Images/55.png)

After finishing all these calculations, to implement one step of a gradient descent, we need to update our parameters 𝑤<sub>1</sub>, 𝑤<sub>2</sub>, and 𝑏.

![](Images/57.png)

It turns out there are two weaknesses with our calculations as we’ve implemented it here.

To implement logistic regression this way, we need to write two for loops (loop over 𝑚 training samples and 𝑛 features).

When implementing deep learning algorithms, having explicit for loops makes our algorithm run less efficient. Especially on larger datasets, which we must avoid. For this, we use what we call vectorization.

The above code should run for some iterations to minimize error. So there will be two inner loops to implement the logistic regression. Vectorization is so important on deep learning to reduce loops. In the last code we can make the whole loop in one step using vectorization!

### Vectorization

A vectorization is basically the art of getting rid of explicit for loops whenever possible. With the help of vectorization, operations are applied to whole arrays instead of individual elements. The rule of thumb to remember is to avoid using explicit loops in your code. Deep learning algorithms tend to shine when trained on large datasets, so it’s important that your code runs quickly. Otherwise, your code might take a long time to get your result.

![](Images/58.png)

### Vectorizing Logistic Regression

When we are programming Logistic Regression or Neural Networks we should avoid explicit 𝑓𝑜𝑟 loops. It’s not always possible, but when we can, we should use built-in functions or find some other ways to compute it. Vectorizing the implementation of Logistic Regression  makes the code highly efficient. We will see how we can use this technique to compute gradient descent without using even a single 𝑓𝑜𝑟 loop.

Now, we will examine the forward propagation step of logistic regression. If we have 𝑚 training examples, to make a prediction on the first example we need to compute 𝑧 and the activation function 𝑎 as follows:

![](Images/60.png)

To make prediction on the second training example we need to compute this:

![](Images/61.png)

The same is with prediction of third training example:

![](Images/62.png)

So if we have 𝑚 training examples we need to do these calculations 𝑚 times. In order to carry out the forward propagation step, which means to compute these predictions for all 𝑚 training examples, there is a way to do this without needing an explicit for loop.

We will stack all training examples horizontally in a matrix 𝐗, so that every column in matrix 𝐗 represents one training example:

![](Images/63.png)

Notice that matrix 𝜔 is a 𝑛<sub>𝑥</sub> × 1 matrix (or a column vector), so when we transpose it we get 𝜔<sup>𝑇</sup> which is a 1 × 𝑛<sub>𝑥</sub> matrix (or a row vector) so multiplying  𝜔<sup>𝑇</sup> with 𝐗 we get a 1 × 𝑚 matrix. Then we add a 1 × 𝑚 matrix 𝑏 to obtain 𝐙.

We will define matrix 𝐙 by placing all 𝑧<sup>(𝑖)</sup> values in a row vector:

![](Images/64.png)

In Python, we can easily implement the calculation of a matrix 𝐙:

![](Images/65.png)

As we can see 𝑏 is defined as a scalar. When you add this vector to this real number, Python automatically takes this real number 𝑏 and expands it out to the 1 × 𝑚 row vector. This operation is called **broadcasting**.

Matrix 𝐀 is defined as a 1 × 𝑚, wich we also got by stacking horizontaly values 𝑎<sup>(𝑖)</sup> as we did with matrix 𝐙:

![](Images/66.png)

In Python, we can also calculate matrix 𝐀 with one line of code as follows (if we have defined sigmoid function as above):

![](Images/67.png)

For the gradient computation we had to compute detivative 𝑑𝑧 for every training example:

![](Images/68.png)

In the same way, we have defined previous variables, now we will define matrix 𝐝𝐙, where we will stack all 𝑑𝑧<sup>(𝑖)</sup> variables horizontally, dimension of this matrix 𝐝𝐙 is 1 × 𝑚 or alternativly a 𝑚 dimensional row vector.

![](Images/69.png)

As we know that matrices 𝐀 and 𝐘 are defined as follows:

![](Images/70.png)

We can see that 𝐝𝐙 below, all values in 𝐝𝐙 can be computed at the same time.

![](Images/71.png)

To implement Logistic Regression on code we did this:

![](Images/72.png)

This code was non-vectorized and highly inefficent so we need to transform it. First, using vectorization, we can transform equations (∗) and (∗∗) into one equation:

<div align="center">
  <img src="Images/73.png">
</div>

The cost function is:

![](Images/74.png)

The derivatives are:

![](Images/75.png)

To calculate 𝑤 and 𝑏 we will still need following 𝑓𝑜𝑟 loop.

![](Images/76.png)

We don’t need to loop through entire training set, but still we need to loop through number of iterations and that’s a 𝑓𝑜𝑟 loop that we can’t get rid off.

### Broadcasting in Python

The term broadcasting describes how NumPy treats arrays with different shapes during arithmetic operations. The simplest broadcasting example occurs when an array and a scalar value are combined in an operation. If we have a matrix 𝐀 and scalar value 𝑏 then scalar 𝑏 is being stretched during the arithmetic operation into an array which is the same shape as 𝐀, but that stretch is only conceptual. Numpy uses the original scalar value without making copies, so that broadcasting operations are as memory and computationally efficient as possible.

Adding a scalar to a row vector:

![](Images/77.png)

Adding a scalar to a column vector:

![](Images/78.png)

Adding a row vector to a matrix:

![](Images/79.png)

Adding a column vector to a matrix:

![](Images/80.png)

### Explanation of Logistic Regression Cost Function

One way to motivate linear regression with the mean squared error loss function is to formally assume that observations arise from noisy observations, where the noise is normally distributed as follows

![](Images/81.png)

Thus, we can now write out the **likelihood estimators** of seeing a particular 𝑦 for a given 𝑥 via

![](Images/82.png)

Now, according to the **maximum likelihood principle**, the best values of 𝑏 and 𝑤 are those that maximize the likelihood of the entire dataset:

![](Images/83.png)

Estimators chosen according to the maximum likelihood principle are called **Maximum Likelihood Estimators**. While, maximizing the product of many exponential functions, might look difficult, we can simplify things significantly, without changing the objective, by maximizing the **log** of the likelihood instead.

![](Images/84.png)

Now we just need one more assumption: that 𝜎 is some fixed constant. Thus we can ignore the first term because it doesn’t depend on 𝑤 or 𝑏. Now the second term is identical to the **squared error** objective, but for the multiplicative constant 1/𝜎<sup>2</sup>. Fortunately, the solution does not depend on 𝜎. It follows that minimizing squared error is equvalent to maximum likelihood estimation of a linear model under the assumption of additive Gaussian noise.

![](Images/85.png)

## Shallow neural networks

> Learn to build a neural network with one hidden layer, using forward propagation and backpropagation.

### Neural Networks Overview

- In logistic regression we had:

  ```
  X1  \  
  X2   ==>  z = XW + B ==> a = Sigmoid(z) ==> l(a,Y)
  X3  /
  ```

- In neural networks with one layer we will have:

  ```
  X1  \  
  X2   =>  z1 = XW1 + B1 => a1 = Sigmoid(z1) => z2 = a1W2 + B2 => a2 = Sigmoid(z2) => l(a2,Y)
  X3  /
  ```


- `X` is the input vector `(X1, X2, X3)`, and `Y` is the output variable `(1x1)`
- NN is stack of logistic regression objects.

### Neural Network Representation

- We will define the neural networks that has one hidden layer.
- NN contains of input layers, hidden layers, output layers.
- Hidden layer means we cant see that layers in the training set.
- `a0 = x` (the input layer)
- `a1` will represent the activation of the hidden neurons.
- `a2` will represent the output layer.
- We are talking about 2 layers NN. The input layer isn't counted.

### Computing a Neural Network's Output

- Equations of Hidden layers:
  - ![](Images/05-a.png)
  - ![](Images/05-b.png)
  - ![](Images/05-c.png)
- Here are some informations about the last image:
  - `noOfHiddenNeurons = 4`
  - `Nx = 3`
  - Shapes of the variables:
    - `W1` is the matrix of the first hidden layer, it has a shape of `(noOfHiddenNeurons,nx)`
    - `b1` is the matrix of the first hidden layer, it has a shape of `(noOfHiddenNeurons,1)`
    - `z1` is the result of the equation `z1 = W1*X + b`, it has a shape of `(noOfHiddenNeurons,1)`
    - `a1` is the result of the equation `a1 = sigmoid(z1)`, it has a shape of `(noOfHiddenNeurons,1)`
    - `W2` is the matrix of the second hidden layer, it has a shape of `(1,noOfHiddenNeurons)`
    - `b2` is the matrix of the second hidden layer, it has a shape of `(1,1)`
    - `z2` is the result of the equation `z2 = W2*a1 + b`, it has a shape of `(1,1)`
    - `a2` is the result of the equation `a2 = sigmoid(z2)`, it has a shape of `(1,1)`

### Vectorizing across multiple examples

- Pseudo code for forward propagation for the 2 layers NN:

  ```
  for i = 1 to m
    z[1, i] = W1*x[i] + b1      # shape of z[1, i] is (noOfHiddenNeurons,1)
    a[1, i] = sigmoid(z[1, i])  # shape of a[1, i] is (noOfHiddenNeurons,1)
    z[2, i] = W2*a[1, i] + b2   # shape of z[2, i] is (1,1)
    a[2, i] = sigmoid(z[2, i])  # shape of a[2, i] is (1,1)
  ```

- Lets say we have `X` on shape `(Nx,m)`. So the new pseudo code:

  ```
  Z1 = W1X + b1     # shape of Z1 (noOfHiddenNeurons,m)
  A1 = sigmoid(Z1)  # shape of A1 (noOfHiddenNeurons,m)
  Z2 = W2A1 + b2    # shape of Z2 is (1,m)
  A2 = sigmoid(Z2)  # shape of A2 is (1,m)
  ```

- If you notice always m is the number of columns.
- In the last example we can call `X` = `A0`. So the previous step can be rewritten as:

  ```
  Z1 = W1A0 + b1    # shape of Z1 (noOfHiddenNeurons,m)
  A1 = sigmoid(Z1)  # shape of A1 (noOfHiddenNeurons,m)
  Z2 = W2A1 + b2    # shape of Z2 is (1,m)
  A2 = sigmoid(Z2)  # shape of A2 is (1,m)
  ```

### Activation functions

- So far we are using sigmoid, but in some cases other functions can be a lot better.
- Sigmoid can lead us to gradient decent problem where the updates are so low.
- Sigmoid activation function range is [0,1]
  `A = 1 / (1 + np.exp(-z)) # Where z is the input matrix`
- Tanh activation function range is [-1,1]   (Shifted version of sigmoid function)
  - In NumPy we can implement Tanh using one of these methods:
    `A = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z)) # Where z is the input matrix`

    Or
    `A = np.tanh(z)   # Where z is the input matrix`
- It turns out that the tanh activation usually works better than sigmoid activation function for hidden units because the mean of its output is closer to zero, and so it centers the data better for the next layer.
- Sigmoid or Tanh function disadvantage is that if the input is too small or too high, the slope will be near zero which will cause us the gradient decent problem.
- One of the popular activation functions that solved the slow gradient decent is the RELU function.
  `RELU = max(0,z) # so if z is negative the slope is 0 and if z is positive the slope remains linear.`
- So here is some basic rule for choosing activation functions, if your classification is between 0 and 1, use the output activation as sigmoid and the others as RELU.
- Leaky RELU activation function different of RELU is that if the input is negative the slope will be so small. It works as RELU but most people uses RELU.
  `Leaky_RELU = max(0.01z,z)  #the 0.01 can be a parameter for your algorithm.`
- In NN you will decide a lot of choices like:
  - No of hidden layers.
  - No of neurons in each hidden layer.
  - Learning rate.       (The most important parameter)
  - Activation functions.
  - And others..
- It turns out there are no guide lines for that. You should try all activation functions for example.

### Why do you need non-linear activation functions?

- If we removed the activation function from our algorithm that can be called linear activation function.
- Linear activation function will output linear activations
  - Whatever hidden layers you add, the activation will be always linear like logistic regression (So its useless in a lot of complex problems)
- You might use linear activation function in one place - in the output layer if the output is real numbers (regression problem). But even in this case if the output value is non-negative you could use RELU instead.

### Derivatives of activation functions

- Derivation of Sigmoid activation function:

  ```
  g(z)  = 1 / (1 + np.exp(-z))
  g'(z) = (1 / (1 + np.exp(-z))) * (1 - (1 / (1 + np.exp(-z))))
  g'(z) = g(z) * (1 - g(z))
  ```

- Derivation of Tanh activation function:

  ```
  g(z)  = (e^z - e^-z) / (e^z + e^-z)
  g'(z) = 1 - np.tanh(z)^2 = 1 - g(z)^2
  ```

- Derivation of RELU activation function:

  ```
  g(z)  = np.maximum(0,z)
  g'(z) = {
    0  if z < 0
    1  if z >= 0
  }
  ```

- Derivation of leaky RELU activation function:

  ```
  g(z)  = np.maximum(0.01 * z, z)
  g'(z) = {
    0.01  if z < 0
    1     if z >= 0
  }
  ```

### Gradient descent for Neural Networks
- In this section we will have the full back propagation of the neural network (Just the equations with no explanations).
- Gradient descent algorithm:
  - NN parameters:
    - `n[0] = Nx`
    - `n[1] = NoOfHiddenNeurons`
    - `n[2] = NoOfOutputNeurons = 1`
    - `W1` shape is `(n[1],n[0])`
    - `b1` shape is `(n[1],1)`
    - `W2` shape is `(n[2],n[1])`
    - `b2` shape is `(n[2],1)`
  - Cost function `I =  I(W1, b1, W2, b2) = (1/m) * Sum(L(Y,A2))`
  - Then Gradient descent:

    ```
    Repeat:
    		Compute predictions (y'[i], i = 0,...m)
    		Get derivatives: dW1, db1, dW2, db2
    		Update: W1 = W1 - LearningRate * dW1
    				b1 = b1 - LearningRate * db1
    				W2 = W2 - LearningRate * dW2
    				b2 = b2 - LearningRate * db2
    ```

- Forward propagation:

  ```
  Z1 = W1A0 + b1    # A0 is X
  A1 = g1(Z1)
  Z2 = W2A1 + b2
  A2 = Sigmoid(Z2)      # Sigmoid because the output is between 0 and 1
  ```

- Backpropagation (derivations):   
  ```
  dZ2 = A2 - Y      # derivative of cost function we used * derivative of the sigmoid function
  dW2 = (dZ2 * A1.T) / m
  db2 = Sum(dZ2) / m
  dZ1 = (W2.T * dZ2) * g'1(Z1)  # element wise product (*)
  dW1 = (dZ1 * A0.T) / m   # A0 = X
  db1 = Sum(dZ1) / m
  # Hint there are transposes with multiplication because to keep dimensions correct
  ```
- How we derived the 6 equations of the backpropagation:   
  ![](Images/06-a.png)
  ![](Images/06-b.png)

### Random Initialization

- In logistic regression it wasn't important to initialize the weights randomly, while in NN we have to initialize them randomly.

- If we initialize all the weights with zeros in NN it won't work (initializing bias with zero is OK):
  - all hidden units will be completely identical (symmetric) - compute exactly the same function
  - on each gradient descent iteration all the hidden units will always update the same

- To solve this we initialize the W's with a small random numbers:

  ```
  W1 = np.random.randn((2,2)) * 0.01    # 0.01 to make it small enough
  b1 = np.zeros((2,1))                  # its ok to have b as zero, it won't get us to the symmetry breaking problem
  ```

- We need small values because in sigmoid (or tanh), for example, if the weight is too large you are more likely to end up even at the very start of training with very large values of Z. Which causes your tanh or your sigmoid activation function to be saturated, thus slowing down learning. If you don't have any sigmoid or tanh activation functions throughout your neural network, this is less of an issue.

- Constant 0.01 is alright for 1 hidden layer networks, but if the NN is deep this number can be changed but it will always be a small number.

## Deep Neural Networks

> Understand the key computations underlying deep learning, use them to build and train deep neural networks, and apply it to computer vision.

### Deep L-layer neural network

- Shallow NN is a NN with one or two layers.
- Deep NN is a NN with three or more layers.
- We will use the notation `L` to denote the number of layers in a NN.
- `n[l]` is the number of neurons in a specific layer `l`.
- `n[0]` denotes the number of neurons input layer. `n[L]` denotes the number of neurons in output layer.
- `g[l]` is the activation function.
- `a[l] = g[l](z[l])`
- `w[l]` weights is used for `z[l]`
- `x = a[0]`, `a[l] = y'`
- These were the notation we will use for deep neural network.
- So we have:
  - A vector `n` of shape `(1, NoOfLayers+1)`
  - A vector `g` of shape `(1, NoOfLayers)`
  - A list of different shapes `w` based on the number of neurons on the previous and the current layer.
  - A list of different shapes `b` based on the number of neurons on the current layer.

### Forward Propagation in a Deep Network

- Forward propagation general rule for one input:

  ```
  z[l] = W[l]a[l-1] + b[l]
  a[l] = g[l](a[l])
  ```

- Forward propagation general rule for `m` inputs:

  ```
  Z[l] = W[l]A[l-1] + B[l]
  A[l] = g[l](A[l])
  ```

- We can't compute the whole layers forward propagation without a for loop so its OK to have a for loop here.
- The dimensions of the matrices are so important you need to figure it out.

### Getting your matrix dimensions right

- The best way to debug your matrices dimensions is by a pencil and paper.
- Dimension of `W` is `(n[l],n[l-1])` . Can be thought by right to left.
- Dimension of `b` is `(n[l],1)`
- `dw` has the same shape as `W`, while `db` is the same shape as `b`
- Dimension of `Z[l],` `A[l]`, `dZ[l]`, and `dA[l]`  is `(n[l],m)`

### Why deep representations?

- Why deep NN works well, we will discuss this question in this section.
- Deep NN makes relations with data from simpler to complex. In each layer it tries to make a relation with the previous layer. E.g.:
  - 1) Face recognition application:
      - Image ==> Edges ==> Face parts ==> Faces ==> desired face
  - 2) Audio recognition application:
      - Audio ==> Low level sound features like (sss,bb) ==> Phonemes ==> Words ==> Sentences
- Neural Researchers think that deep neural networks "think" like brains (simple ==> complex)
- When starting on an application don't start directly by dozens of hidden layers. Try the simplest solutions (e.g. Logistic Regression), then try the shallow neural network and so on.

### Building blocks of deep neural networks

- Forward and back propagation for a layer l:
  - ![Untitled](Images/10.png)
- Deep NN blocks:
  - ![](Images/08.png)

### Forward and Backward Propagation

- Pseudo code for forward propagation for layer l:

  ```
  Input  A[l-1]
  Z[l] = W[l]A[l-1] + b[l]
  A[l] = g[l](Z[l])
  Output A[l], cache(Z[l])
  ```

- Pseudo  code for back propagation for layer l:

  ```
  Input da[l], Caches
  dZ[l] = dA[l] * g'[l](Z[l])
  dW[l] = (dZ[l]A[l-1].T) / m
  db[l] = sum(dZ[l])/m                # Dont forget axis=1, keepdims=True
  dA[l-1] = w[l].T * dZ[l]            # The multiplication here are a dot product.
  Output dA[l-1], dW[l], db[l]
  ```

- If we have used our loss function then:

  ```
  dA[L] = (-(y/a) + ((1-y)/(1-a)))
  ```

### Parameters vs Hyperparameters

- Main parameters of the NN is `W` and `b`
- Hyper parameters (parameters that control the algorithm) are like:
  - Learning rate.
  - Number of iteration.
  - Number of hidden layers `L`.
  - Number of hidden units `n`.
  - Choice of activation functions.
- You have to try values yourself of hyper parameters.
- In the earlier days of DL and ML learning rate was often called a parameter, but it really is (and now everybody call it) a hyperparameter.
- On the next course we will see how to optimize hyperparameters.

### What does this have to do with the brain

- The analogy that "It is like the brain" has become really an oversimplified explanation.
- There is a very simplistic analogy between a single logistic unit and a single neuron in the brain.
- No human today understand how a human brain neuron works.
- No human today know exactly how many neurons on the brain.
- Deep learning in Andrew's opinion is very good at learning very flexible, complex functions to learn X to Y mappings, to learn input-output mappings (supervised learning).
- The field of computer vision has taken a bit more inspiration from the human brains then other disciplines that also apply deep learning.
- NN is a small representation of how brain work. The most near model of human brain is in the computer vision (CNN)








- Hidden layers predicts connection between inputs automatically, thats what deep learning is good at.
  - ![](Images/01.jpg)
- Deep Neural Network consists of more hidden layers
  - ![](Images/07.png)
- Each Input will be connected to the hidden layer and the NN will decide the connections.


- [computation graph](https://colah.github.io/posts/2015-08-Backprop/)
- Its a graph that organizes the computation from bottom to top.
  - ![](Images/02.png)

- ![](Images/03.png)

- So we have:
  ![](Images/09.png)
