#**Behavioral Cloning** 
**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

###Files Submitted & Code Quality
####1. Submission includes all required files and can be used to run the simulator in autonomous mode
My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* process_image.py containing functions for image augmentation and pre-processing
* model.h5 containing a trained convolution neural network with all the learned weights
* model.json saved model
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.json
```
####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. model.py imports utility functions from within process_image.py. Methods in process_image.py are mostly for image augmentations (left/right shift, randomly introducing brightness and  image flipping, and cropping) to randomly introduce various driving scenes/scenarios. model.py shows the pipeline (get_model()) used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed
The model I used is an implementation of NVIDIA's CNN architecture as specified in [NVIDIA's End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316v1.pdf).
The first layer performs image normalization using a Keras lambda layer. The first convolutional layer uses 3 filters and 1x1 kernel to do initial feature extractions. The next three convolutional layers uses 24,36,and 48 filters all of which have 2x2 stride and 5x5 kernel. The last two convolutional layers both have 64 filters, 3x3 kernel and 1x1 stride (model.py get_model() method line 28). The model includes "ELU" layers to introduce nonlinearity, for fast learning, and better generalization [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](http://arxiv.org/abs/1511.07289). 


####2. Attempts to reduce overfitting in the model
The model contains dropout(0.10) layers in order to reduce overfitting (model.py get_model). 
The data sets were shuffled and splitted into test(80%) and validation(20%)  sets.
The model was trained on different data sets by using on-the-fly image augmentation to create more driving scenes/scenarios.
The augmentation routines are written and inspired by [Vivek Yadav's post] (https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.3iotk6hco). Validation data sets were augmented and pre-processed one time and used in the entire training activity. 
Model was tested in the simulator and vehicle was able to stay on the track [car driving autonomously](https://youtu.be/z3z2mb9RJAE)

####3. Model parameter tuning
The model used an adam optimizer. 

####4. Appropriate training data
The following data were collected for the training/validation sets 
* 3 laps of driving while staying in middle lane.
* 2 laps of driving while staying in middle lane in opposite direction.
  The data collected in track 1 is bias towards left turns to balance it we need to collect center lane driving in opposite      direction (counter-clockwise).

  
###Model Architecture and Training Strategy
####1. Solution Design Approach
Initially I trained the model using batch size of 256, epoch 15, and 20224 samples per epoch and noticed that the training and validation loss still has some room to converge to a minimum. The initial model had the car drive inside the lane for a few seconds then eventually send the car off-road. 
Initial training:

* 20224/20000  127s  loss: 0.1071 - val_loss: 0.0936
* 20224/20000  122s  loss: 0.0883 - val_loss: 0.0894
* 20224/20000  125s  loss: 0.0857 - val_loss: 0.0871
* 20224/20000  124s  loss: 0.0844 - val_loss: 0.0857
* 20224/20000  123s  loss: 0.0824 - val_loss: 0.0863
* 20224/20000  124s  loss: 0.0830 - val_loss: 0.0826
* 20224/20000  124s  loss: 0.0788 - val_loss: 0.0806
* 20224/20000  124s  loss: 0.0792 - val_loss: 0.0791
* 20224/20000  124s  loss: 0.0775 - val_loss: 0.0777
* 20224/20000  124s  loss: 0.0749 - val_loss: 0.0747
* 20224/20000  125s  loss: 0.0732 - val_loss: 0.0725
* 20224/20000  124s  loss: 0.0705 - val_loss: 0.0729
* 20224/20000  124s  loss: 0.0706 - val_loss: 0.0705
* 20224/20000  123s  loss: 0.0694 - val_loss: 0.0721
* 20224/20000  126s  loss: 0.0669 - val_loss: 0.0658

I tried to use initial model and fine tune it. The resulting training and validation loss improved and car was able to drive the track continously without going outside the road. [car driving autonomously](https://youtu.be/z3z2mb9RJAE)

Second training:
0224/20000 [===] - 124s - loss: 0.0988 - val_loss: 0.0637
Epoch 1/1
20224/20000 [===] - 123s - loss: 0.0845 - val_loss: 0.0624
Epoch 1/1
20224/20000 [===] - 123s - loss: 0.0820 - val_loss: 0.0643
Epoch 1/1
20224/20000 [===] - 122s - loss: 0.0817 - val_loss: 0.0700
Epoch 1/1
20224/20000 [===] - 124s - loss: 0.0808 - val_loss: 0.0619
Epoch 1/1
20224/20000 [===] - 123s - loss: 0.0778 - val_loss: 0.0572
Epoch 1/1
20224/20000 [===] - 124s - loss: 0.0766 - val_loss: 0.0595
Epoch 1/1
20224/20000 [===] - 123s - loss: 0.0779 - val_loss: 0.0545
Epoch 1/1
20224/20000 [===] - 122s - loss: 0.0742 - val_loss: 0.0537
Epoch 1/1
20224/20000 [===] - 123s - loss: 0.0716 - val_loss: 0.0512
Epoch 1/1
20224/20000 [===] - 123s - loss: 0.0693 - val_loss: 0.0504
Epoch 1/1
20224/20000 [===] - 121s - loss: 0.0706 - val_loss: 0.0497
Epoch 1/1
20224/20000 [===] - 124s - loss: 0.0669 - val_loss: 0.0517
Epoch 1/1
20224/20000 [===] - 120s - loss: 0.0664 - val_loss: 0.0483
Epoch 1/1
20224/20000 [===] - 123s - loss: 0.0651 - val_loss: 0.0480


I did another model training with batch size of 256, epoch 25, and 20224 samples per epoch and car was able to drive the lane successfully without goind outside the road and no additional fine tuning needed.

20224/20000 [===] - 128s - loss: 0.1335 - val_loss: 0.0841
Epoch 1/1
20224/20000 [===] - 122s - loss: 0.0994 - val_loss: 0.0773
Epoch 1/1
20224/20000 [===] - 121s - loss: 0.0921 - val_loss: 0.0744
Epoch 1/1
20224/20000 [===] - 121s - loss: 0.0944 - val_loss: 0.0774
Epoch 1/1
20224/20000 [===] - 119s - loss: 0.0896 - val_loss: 0.0725
Epoch 1/1
20224/20000 [===] - 121s - loss: 0.0913 - val_loss: 0.0683
Epoch 1/1
20224/20000 [===] - 121s - loss: 0.0848 - val_loss: 0.0667
Epoch 1/1
20224/20000 [===] - 123s - loss: 0.0861 - val_loss: 0.0672
Epoch 1/1
20224/20000 [===] - 121s - loss: 0.0861 - val_loss: 0.0642
Epoch 1/1
20224/20000 [===] - 121s - loss: 0.0831 - val_loss: 0.0635
Epoch 1/1
20224/20000 [===] - 122s - loss: 0.0805 - val_loss: 0.0634
Epoch 1/1
20224/20000 [===] - 123s - loss: 0.0822 - val_loss: 0.0629
Epoch 1/1
20224/20000 [===] - 120s - loss: 0.0798 - val_loss: 0.0674
Epoch 1/1
20224/20000 [===] - 121s - loss: 0.0789 - val_loss: 0.0609
Epoch 1/1
20224/20000 [===] - 122s - loss: 0.0761 - val_loss: 0.0625
Epoch 1/1
20224/20000 [===] - 122s - loss: 0.0751 - val_loss: 0.0590
Epoch 1/1
20224/20000 [===] - 121s - loss: 0.0742 - val_loss: 0.0606
Epoch 1/1
20224/20000 [===] - 121s - loss: 0.0755 - val_loss: 0.0569
Epoch 1/1
20224/20000 [===] - 120s - loss: 0.0709 - val_loss: 0.0600
Epoch 1/1
20224/20000 [===] - 122s - loss: 0.0703 - val_loss: 0.0550
Epoch 1/1
20224/20000 [===] - 121s - loss: 0.0705 - val_loss: 0.0556
Epoch 1/1
20224/20000 [===] - 120s - loss: 0.0664 - val_loss: 0.0557
Epoch 1/1
20224/20000 [===] - 122s - loss: 0.0653 - val_loss: 0.0527
Epoch 1/1
20224/20000 [===] - 120s - loss: 0.0640 - val_loss: 0.0525
Epoch 1/1
20224/20000 [===] - 121s - loss: 0.0641 - val_loss: 0.0531



####2. Final Model Architecture

The final model architecture (model.py get_model()) is the NVIDIA's CNN architecture as specified in [NVIDIA's End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316v1.pdf) with 0.10 dropout and ELU activation.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

#### ems show NVIDIAs image




####3. Creation of the Training Set & Training Process
The indexes to the image files were shuffled and split to test(80%) and validation(20%) sets. A one time data generation with augmentation on the validation set is done and used in the entire training activity. Batch training set is generated on-the-fly using the augmentation listed below via keras data generator. 

 * List of augmentation used
    - shifting image to the right and left to mimic the effect of car driving at different position on the road and adding 0.004 steering angle units per pixel shift to the right while subtracting 0.004 steering angle units per pixel shift to the left. Shifting vertically is also done to simulate car driving up or down the slope (process_image.py trans_image())
    - Brightness augmentation is done to simulate different time of day. Brightness level is chosen randomly (process_image.py augment_brightness_camera_images())
    - Image flipping is also done to for more driving data for better generalization of the model. Steering angle sign is reversed during image flipping.
    - Captured left and right camera images are also used. The steering angle recorded during data collection is based on the center camera position. In order to use the left and right camera image as if they came from the center camera, we add a 0.25 to recorded steering angle and map it to left camera image and subtract 0.25 for right camera image. By using images captured from right and left camera we can simulate the effect of driving on the side  of the road and recovering back to center (process_image.py preprocess_image_file_train()).
    - Cropping is also done to remove the horizon and car's hood which is not really needed by the model in predecting the corresponding steering angle (process_image.py preprocessImage(). After Cropping the image is resized to 66x200 for NVIDIA's expected image input shape.
 
 *Show the image here





