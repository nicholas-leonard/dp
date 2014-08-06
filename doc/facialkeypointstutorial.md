<a name="FacialKeypointsTutorial"/>
# Facial Keypoints Tutorial #
In this tutorial, we demonstrate how the __dp__ library can be used 
to build convolution neural networks and easily extended using Feedback 
objects and the Mediator. To make things more spicy, we consider 
a case study involving its practical application to a 
[Kaggle](https://www.kaggle.com) challenge provided by the University of Montreal: 
[Facial Keypoints Detection](https://www.kaggle.com/c/facial-keypoints-detection).
We will attempt to keep this tutorial as complete yet concise as possible.

## Planning ##
The first step is to determine how to approach the problem and outline the 
necessary components that will be needed to get the model working. 
It is in this step that one plans the final model(s) and components that
will be required to get your experiments running. In our case, we had 
already implemented a similar model in Pylearn2 such that we had a general 
idea what worked well for this particular problem. 

The problem has each 96x96 black-and-white images associated to 
15 keypoints, each identifies by an (x,y) coordinate. The problem is 
thus a regression where the target is a vector of 15x2=30 values 
bounded between 0 and 96, the size of the image. If you think like me, 
your initial reflex might be to use a simple multi-layer perception 
(or neural network) with a Linear output and a Mean Square Error 
Criterion. Or maybe we can bound the output by using a Sigmoid 
(which bound it between 0 and 1), and then scale the output by a 
constant greater than 96. 

However, these approaches don't work well in practice. An alternative 
solution is to model the output space as 30 vectors of size 97, and 
translate each target value to a small (standard deviation of about 1) 
gaussian blur centered at the keypoint coordinate. This increases the 
precision of the new targets as compared to just using a one-hot vector 
(a vector with one 1, the rest being zeros).

The use of a gaussian blur centered on the target, which amounts to 
predicting multinomial probabilities, can be combined with the 
DistKLDivCriterion.

## Building Components ##

 * FacialKeypoints : wrapper for the DataSource;
 * facialkeypointsdetector.lua : launch script with cmd-line options for specifying Model assembly and Experiment hyper-parameters; 
 * FKDKaggle : a Feedback for creating a Kaggle submission out of predictions;
 * FacialKeypoints : a Feedback for monitoring performance;
 * nn.MultiSoftMax : a nn.Module that will allow us to apply a softmax for each keypoint.
