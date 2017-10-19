# Math-cv
An attempt to build a deep learning model that generates Latex markup from an image of a math equation. The project started after finding the request for research published by OpenAI (https://openai.com/requests-for-research/#im2latex). I'm using this project to learn the ins and outs of Tensorflow, and to reinforce the materials taught in Andrew Ng's deep learning course specialization on Coursera.

## Model
The model consists of 5 convolutional layers, an encoder, and a decoder. Each convolutional layer uses a relu activation function and dropout for regualrization. The encoder is a simple fully connected layer. The decoder is an LSTM layer followed by a fully connected layer.

#### Multi-GPU
Thanks to the resources provided by ComputeCanada and SHARCNET, I am able to train the model on much better setup than my macbook. I've been using 4 Tesla P100's for trainning. In a multi-GPU setup, there are 4 of the previously described model that share variables.
![alt text](https://github.com/Smurgs/math-cv/blob/master/model_preview.png "Single Device Model Preview")

## Results
I'll update when training is completed

## Usage
The top-level python file can be used to download the training set, preprocess the data, and train the model. Use the config file (mathch/config.py) to tweak hyper-parameters and other settings.
Python requirements:
* Tensorflow
* Numpy 
* Pillow

Other requirements:
* Node
```bash
Stefans-MacBook-Pro:math-cv Stefan$ python mathcv.py 
MathCv v0.1.0
Usage: mathcv command
Commands:
    download            download the im2latex dataset
    preprocess          preprocess the train/val/test dataset
    train               train the mathcv model
```
