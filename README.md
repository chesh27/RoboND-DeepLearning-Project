[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

## Deep Learning Project ##
This is the fourth project in the Udacity Robotics Software Engineer Nanodegree. 

In this project, I train a deep neural network to identify and track a target in simulation. So-called “follow me” applications like this are key to many fields of robotics and the very same techniques applied here could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.

I built and trained a fully convolutional neural network to allow a drone to identify and follow a specific target - the 'hero' - within an image in a simulated environment. I was able to achieve a final IoU of ~42% (0.421). 

[image_0]: ./docs/misc/sim_screenshot.png
![alt text][image_0] 

## Setup Instructions
**Clone the repository**
```
$ git clone https://github.com/udacity/RoboND-DeepLearning.git
```

**Download the data**

Save the following three files into the data folder of the cloned repository. 

[Training Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/train.zip) 

[Validation Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/validation.zip)

[Sample Evaluation Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Project/sample_evaluation_data.zip)

**Download the QuadSim binary**

To interface your neural net with the QuadSim simulator, you must use a version QuadSim that has been custom tailored for this project. The previous version that you might have used for the Controls lab will not work.

The simulator binary can be downloaded [here](https://github.com/udacity/RoboND-DeepLearning/releases/latest)

**Install Dependencies**

You'll need Python 3 and Jupyter Notebooks installed to do this project.  The best way to get setup with these if you are not already is to use Anaconda following along with the [RoboND-Python-Starterkit](https://github.com/udacity/RoboND-Python-StarterKit).

If for some reason you choose not to use Anaconda, you must install the following frameworks and packages on your system:
* Python 3.x
* Tensorflow 1.2.1
* NumPy 1.11
* SciPy 0.17.0
* eventlet 
* Flask
* h5py
* PIL
* python-socketio
* scikit-image
* transforms3d
* PyQt4/Pyqt5

## Implement the Segmentation Network
1. Download the training dataset from above and extract to the project `data` directory.
2. Implement your solution in model_training.ipynb
3. Train the network locally, or on [AWS](https://classroom.udacity.com/nanodegrees/nd209/parts/09664d24-bdec-4e64-897a-d0f55e177f09/modules/cac27683-d5f4-40b4-82ce-d708de8f5373/lessons/197a058e-44f6-47df-8229-0ce633e0a2d0/concepts/27c73209-5d7b-4284-8315-c0e07a7cd87f?contentVersion=1.0.0&contentLocale=en-us).
- I chose to incorporate 2 encoder layers and 2 decoder layers, connected by a 1x1 convolutional layer in my fully convolutional network. The first encoding layer takes the image as raw input and finds simple structures within it, such as edges, using semantic segmentation (pixel by pixel). The second layer then takes the output of the first and gradually identifies more complex patterns, such as faces. This output becomes the input to the 1x1 convolutional layer, which reduces dimensionality similar to a fully connected layer, but it preserves spatial information. Finally, there are 2 decoder layers which essentially reverse the process of the encoder layers, and upscale the image back to its original dimensions. This is followed by a softmax activation function that finally returns the output.   
a.	Create an encoder_block
b.	Create a decoder_block
c.	Build the FCN consisting of 2 encoder blocks, a 1x1 convolution, and 2 decoder blocks. This step requires experimentation with different numbers of layers and filter sizes to build your model.

[image_1]: ./docs/misc/network_architecture.png
![alt text][image_1] 

For this project, a fully connected neural network would not have worked, as spatial information would have been lost. Thus, we built a convolutional network, which preserves the spatial relationship between pixels by learning image features using small squares of input data. They are invariant to spatial translations, making our task of identifying targets within an image much more efficient. 

### Image Preprocessing ###
Before the network is trained, the images first need to be undergo a preprocessing step. The preprocessing step transforms the depth masks from the sim, into binary masks suitable for training a neural network. It also converts the images from .png to .jpeg to create a reduced sized dataset, suitable for uploading to AWS. 
To run preprocessing:
```
$ python preprocess_ims.py
```

## Training, Predicting and Scoring ##
With your training and validation data having been generated or downloaded from the above section of this repository, you are free to begin working with the neural net.

### Training your Model ###
**Prerequisites**
- Training data is in `data` directory
- Validation data is in the `data` directory
- The folders `data/train/images/`, `data/train/masks/`, `data/validation/images/`, and `data/validation/masks/` should exist and contain the appropriate data

To train complete the network definition in the `model_training.ipynb` notebook and then run the training cell with appropriate hyperparameters selected.

I initially tried training on my local system but it took too long to even run one epoch, so I switched to the AWS GPU instance. I used a very small learning rate of 0.001 and this seemed to work well for all my tests. For the batch size, I initially tried 256 but this ended up taking too long so I reduced it to 64. I initially trained 2 epochs, which brought my loss down to about 0.09. I thought this was already quite good, but I decided to run it again with 10 epochs. I noticed the loss went down drastically to about 0.03 by the 2nd epoch, so I stopped it and trained it one last time with just 1 epoch. The steps per epoch, validation steps and workers were left at their default values of 200, 50 and 2 respectively. I finally got a loss of 0.0349 and a validation loss of 0.0431. I saved this model, which you will find in my submission. 

[image_2]: ./docs/misc/parameters.png
![alt text][image_2] 

After the training run completed, the model was stored in the `data/weights` directory as an [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) file, and a configuration_weights file. 

## Scoring ##

To score the network on the Follow Me task, two types of error are measured. First the intersection over the union for the pixelwise classifications is computed for the target channel. 

In addition to this we determine whether the network detected the target person or not. If more then 3 pixels have probability greater then 0.5 of being the target person then this counts as the network guessing the target is in the image. 

We determine whether the target is actually in the image by whether there are more then 3 pixels containing the target in the label mask. 

Using the above the number of detection true_positives, false positives, false negatives are counted. 

**How the Final score is Calculated**

The final score is the pixelwise `average_IoU*(n_true_positive/(n_true_positive+n_false_positive+n_false_negative))` on data similar to that provided in sample_evaulation_data


## Experimentation: Testing in Simulation
1. Copy your saved model to the weights directory `data/weights`.
2. Launch the simulator, select "Spawn People", and then click the "Follow Me" button.
3. Run the realtime follower script
```
$ python follower.py my_amazing_model.h5
```
## Results

With my network I achieved a final IoU of 0.421 and a final grade score of 0.289. While the quad is following the target, I achieved a result of 0 false negatives and 0 false positives. However, I had several false positives while the quad is on patrol and the target is not visible as well as several false negatives when detecting the target from far away.

[image_3]: ./docs/misc/results1.png
![alt text][image_3] 

[image_4]: ./docs/misc/results2.png
![alt text][image_4] 

[image_5]: ./docs/misc/results3.png
![alt text][image_5] 

## Limitations / Future enhancements 

I am certain that if I collect my own data I will be able to achieve greater accuracy and efficiency; a much higher loss reduction and better overall score, as the volume of training data seems to be the predominant factor that determines the success of a neural network. In the same way that children learn to recognize objects by seeing them repeatedly, my network can become much more adept at identifying the hero if it is trained on more images of the hero in different possible scenarios. I could also continue to adjust the hyperparameters and/or add more layers to my network. 

[image_6]: ./docs/misc/limits.png
![alt text][image_6] 
