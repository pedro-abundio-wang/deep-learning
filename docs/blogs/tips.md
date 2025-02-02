---
# Page settings
layout: default
keywords:
comments: false

# Hero section
title: Introduction to Project Framework
description:

# Micro navigation
micro_nav: true
---

## **Introduction**

The code contains TensorFlow and PyTorch, in vision and NLP. The structure is the following:

```python
pytorch/
    vision/     -- (SIGNS dataset classification)
    nlp/        -- (Named Entity Recognition)
tensorflow/    
    vision/     -- (SIGNS dataset classification)
    nlp/        -- (Named Entity Recognition)
```

This post will help you familiarize with the Project Framework, and introduces a series of posts explaining how to structure a deep learning project:

**Goals of the project framework**

- explain and demonstrate the best practices for structuring a deep learning project
- in each tensorflow and pytorch, give projects: one for a vision task, one for a NLP task

## **Installation**

It is a good practice to have multiple virtual environments to work on different projects. Here we will use `python3` and install the requirements in the file `requirements.txt`.

**Installing Python 3**: To use `python3`, make sure to install version 3.5 or 3.6 on your local machine.

**Virtual environment**: If we don’t have it already, install `virtualenv` by typing `sudo pip install virtualenv` (or `pip install --user virtualenv` if you don’t have sudo) in your terminal. Here we create a virtual environment named `.env`. Navigate inside each repo and run the following command, for instance in `tensorflow/nlp`

```python
virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirements.txt
```

Run `deactivate` if you want to leave the virtual environment. Next time you want to work on the project, just re-run `source .env/bin/activate` after navigating to the correct directory.

## **Use GPU**

- for tensorflow, just run `pip install tensorflow-gpu`. When both `tensorflow` and `tensorflow-gpu` are installed, if a GPU is available, `tensorflow` will automatically use it, making it transparent for you to use.
- for PyTorch, follow the instructions [here](https://pytorch.org).

Note that your GPU needs to be set up first (drivers, CUDA and CuDNN).

## **Download the data**

**You’ll find descriptions of the tasks** in `tensorflow/vision/README.md`, `tensorflow/nlp/README.md` etc.

**Computer Vision**

All instructions can be found in the `tensorflow/vision/README.md`.

For the vision example, we will used the SIGNS dataset created for the Deep Learning Specialization. The dataset is hosted on google drive, download it [here](https://drive.google.com/file/d/1ufiR6hUKhXoAyiBNsySPkUwlvE_wfEHC/view).

This will download the SIGNS dataset (~1.1 GB) containing photos of hands signs representing numbers between 0 and 5. Here is the structure of the data:

```python
SIGNS/
    train_signs/
        0_IMG_5864.jpg
        ...
    test_signs/
        0_IMG_5942.jpg
        ...
```

The images are named following `{label}_IMG_{id}.jpg` where the label is in `[0, 5]`. The training set contains 1,080 images and the test set contains 120 images.

Once the download is complete, move the dataset into the `data/SIGNS` folder. Run the script python `build_dataset.py` which will resize the images to size (64, 64). The new resized dataset will be located by default in `data/64x64_SIGNS`.

**Natural Language Processing (NLP)**

All instructions can be found in the `tensorflow/nlp/README.md`.

We provide a small subset of the kaggle dataset (30 sentences) for testing in `data/small` but you are encouraged to download the original version on the [Kaggle](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/data) website.

**Download the dataset** `ner_dataset.csv` on [Kaggle](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/data) and save it under the `nlp/data/kaggle` directory. Make sure you download the simple version `ner_dataset.csv` and NOT the full version `ner.csv`.

**Build the dataset** Run the following script

```python
python build_kaggle_dataset.py
```

It will extract the sentences and labels from the dataset, split it into train / test / dev and save it in a convenient format for our model. Here is the structure of the data

```python
kaggle/
    train/
        sentences.txt
        labels.txt
    test/
        sentences.txt
        labels.txt
    dev/
        sentences.txt
        labels.txt
```

Debug If you get some errors, check that you downloaded the right file and saved it in the right directory.

**Build the vocabulary** For both datasets, `data/small` and `data/kaggle` you need to build the vocabulary, with

```python
python build_vocab.py --data_dir  data/small
```

or

```python
python build_vocab.py --data_dir data/kaggle
```

## **Structure of the code**

The code shares a common structure:

```python
data/{dataset_name}
    train/
    dev/
    test/
experiments/
model/
    *.py
build_dataset.py
train.py
search_hyperparams.py
synthesize_results.py
evaluate.py
```

Here is each file or directory’s purpose:

- `data/`: will contain all the data of the project (generally not stored on github), with an explicit train/dev/test split
- `experiments`: contains the different experiments (will be explained in the following section)
- `model/`: module defining the model and functions used in train or eval.
- `build_dataset.py`: creates or transforms the dataset, build the split into train/dev/test
- `train.py`: train the model on the input data, and evaluate each epoch on the dev set
- `search_hyperparams.py`: run `train.py` multiple times with different hyperparameters
- `synthesize_results.py`: explore different experiments in a directory and display a nice table of the results
- `evaluate.py`: evaluate the model on the test set (should be run once at the end of your project)

## **Running experiments**

Now that you have understood the structure of the code, we can try to train a model on the data, using the `train.py` script:

```python
python train.py --model_dir experiments/base_model
```

We need to pass the model directory in argument, where the hyperparameters are stored in a json file named `params.json`. Different experiments will be stored in different directories, each with their own `params.json` file. Here is an example:

`experiments/base_model/params.json`:

```python
{
"learning_rate": 1e-3,
"batch_size": 32,
"num_epochs": 20
}
```

The structure of `experiments` after running a few different models might look like this (try to give meaningful names to the directories depending on what experiment you are running):

```python
experiments/
    base_model/
        params.json
        ...
    learning_rate/
        lr_0.1/
            params.json
        lr_0.01/
            params.json
    batch_norm/
        params.json
```

Each directory after training will contain multiple things:

- `params.json`: the list of hyperparameters, in json format
- `train.log`: the training log (everything we print to the console)
- `train_summaries`: train summaries for TensorBoard (TensorFlow only)
- `eval_summaries`: eval summaries for TensorBoard (TensorFlow only)
- `last_weights`: weights saved from the 5 last epochs
- `best_weights`: best weights (based on dev accuracy)

**Training and evaluation**

We can now train an example model with the parameters provided in the configuration file `experiments/base_model/params.json`:

```python
python train.py --model_dir experiments/base_model
```

Once training is done, we can evaluate on the test set:

```python
python evaluate.py --model_dir experiments/base_model
```

**Hyperparameters search**

We provide an example that will call `train.py` with different values of learning rate. We first create a directory

```python
experiments/
    learning_rate/
        params.json
```

with a `params.json` file that contains the other hyperparameters. Then, by calling

```python
python search_hyperparams.py --parent_dir experiments/learning_rate
```

It will train and evaluate a model with different values of learning rate defined in `search_hyperparams.py` and create a new directory for each experiment under `experiments/learning_rate/`, like

```python
experiments/
    learning_rate/
        learning_rate_0.001/
            metrics_eval_best_weights.json
        learning_rate_0.01/
            metrics_eval_best_weights.json
        ...
```

**Display the results of multiple experiments**

If you want to aggregate the metrics computed in each experiment (`metrics_eval_best_weights.json`), simply run

```python
python synthesize_results.py --parent_dir experiments/learning_rate
```

It will display a table synthesizing the results like this that is compatible with markdown:

```python
|                                               |   accuracy |      loss |
|:----------------------------------------------|-----------:|----------:|
| experiments/base_model                        |   0.989    | 0.0550    |
| experiments/learning_rate/learning_rate_0.01  |   0.939    | 0.0324    |
| experiments/learning_rate/learning_rate_0.001 |   0.979    | 0.0623    |
```

## **Tensorflow or PyTorch ?** 

Both framework have their pros and cons:

**Tensorflow**

- mature, most of the models and layers are already implemented in the library.
- documented and plenty of code / tutorials online
- the Deep Learning Specialization teaches you how to use Tensorflow
- built for large-scale deployment and used by a lot of companies
- has some very useful tools like Tensorboard for visualization (though you can also use [Tensorboard with PyTorch](https://github.com/lanpa/tensorboardX))
- but some ramp-up time is needed to understand some of the concepts (session, graph, variable scope)
- transparent use of the GPU
- can be harder to debug

**PyTorch**

- younger, but also well documented and fast-growing community
- more pythonic and numpy-like approach, easier to get used to the dynamic-graph paradigm
- designed for faster prototyping and research
- transparent use of the GPU
- easy to debug and customize
