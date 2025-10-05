---
layout: post
title: “Conveyor Flow” - Fine-tuning a YOLO Model Using Client Data
image: "/posts/dl-search-engine-title-img-flowers.png"
tags: [Deep Learning, Computer Vision, YOLO, Python]
---

Here at *We-Do-Data-Science-So-You-Don't-Have-To Limited* our work is never done! This morning a client that manufactures conveyor equipment (the kind of thing you see in large warehouses and distribution centres) got in touch to see if we could help them detect objects on their conveyor belts. We said we should be able to do that pretty quickly using a YOLO model from Ultralytics. The client has given us some sample data to work with and if we can show our skills using the sample data they'll ask us to take on a bigger project. The client is in a bit of a hurry and we need to act quickly to win the project, so let's get to it!

---

# Table of Contents
- [1. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth/Next Steps](#overview-growth)
- [2. Sample Data Overview](#sample-data-overview)
- [3. Fine-tuning Overview](#transfer-learning-overview)
- [4. Loading the Dataset](#loading-dataset)
- [5. Testing Baseline Performance](#baseline-performance)
- [6. Training the Model](#model-training)
- [7. Testing Custom Model Performance](#custom-performance)
- [8. Inspecting Training Progress](#training-progress)
- [9. Discussion, Growth & Next Steps](#growth-next-steps)

---

# Project Overview <a name="overview-main"></a>
### Context <a name="overview-context"></a>
About 15 years ago our client invested in some software that could detect different objects (such as boxes or trays) on a conveyor belt. They use this to detect issues (such as a product that hasn't been placed in a box) before the product reaches the machine that attaches the address label.  This helps prevent the label getting attached to the product itself, which can lead to the product getting thrown in the reject bin, wasting money. At the time it was state-of-the-art and helped them win lots of business kitting-out parcel distribution warehouses with conveyor systems. However, recently their own customers have been telling them that they've seen some really impressive object detection demonstrations from rival conveyor manufacturers, and our client is worried that they've fallen behind in that regard, so they're now shopping around to see if it's time to invest in a new generation of object dectection software. 

To start with the client want us to work with some convyor image data that they found online. If they're impressed by what we can achieve with that dataset they'll spend some time creating a set of images that is specific to their own products and the environments that their products are typically used in.

### Actions <a name="overview-actions"></a>
To show our client what can be achieved with state-of-the-art computer vision models we'll use the latest computer vision model from the folk over at [Ultralytics](https://www.ultralytics.com/). Their models are not only highly accurate, they're also fast enough to run in real-time and keep up with the flow of objects on a fast-moving conveyor system, so they're ideally suited to our client's use case.

We'll use one of the Ultralytics pretrained models and do a small amount of customizing (training) on the client's dataset, to show the client what's possible.

### Results <a name="overview-results"></a>
Here we can see that the model did really well at detecting cardboard boxes - it found all 167 of them. How awesome is that! After just 10 epochs of training the model was able to correctly detect 100% of the cardboard boxes in our validation set!

### Discussion, Growth & Next Steps <a name="overview-growth"></a>
Our work with this dataset serves as a nice proof-of-concept that we can show to our client. The next steps would be for the client to gather some even better training data for us - ideally we want a large set of images that represent the actual environments that their products get used in - then we could train the object detection using that dataset for a larger number of epochs to get even better results.

---

# Sample Data Overview <a name="sample-data-overview"></a>
Our client has asked us to use the *conveyor* dataset that they found in an online image repository called [Roboflow Universe](https://roboflow.com/universe). This dataset has a total of 1,135 images split into 837 training images, 211 validation images and 87 test images.

Each of the image files is accompanied by a text file that specifies the bounding boxes surrounding objects in the image, for objects in one of three categories: 'cardboard box', 'conveyor' and 'kartonbox'.

---

# Fine-tuning Overview <a name="transfer-learning-overview"></a>
### Overview
So we can get our results in front of our client before they go cool on the whole idea, and also to keep costs down, we'll start with a pre-existing model (rather than create our own model from scratch. We'll see how it performs on the sample data. Since we've done this kind of thing before our experience tells us that, to get even better results, it may be worth fine-tuning the model with some of the sample data provided by the customer. 


### YOLO
The YOLO algorithm has become really popular in the last few years, in large part due to its ability to combine high accuracy while also being fast enough to process video streams in real time. It achieves this by 'only looking once' - to detect objects in an image it only needs to make one forward propagation pass through its neural network, hence the name "You Only Look Once" (YOLO). This is in contrast to other object detection approaches which need to feed an image through their neural network mulitple times (often dozens or even hundreds of times) in order to detect all the objects in the image. 
[[link to paper(s) but also need to train a model with data!  that's where ultralytics comes in!]]

Of course, having a really great algorithm is only one side of the computer vision equation - you need to couple the algorithm with lots of good quality data. That's where Ultralytics comes in. They've [[]].

---

# Loading the Dataset <a name="loading-dataset"></a>
Before we do anything else we need to install some Python packages to our Colab runtime. We'll make use of the `ultralytics` Python package for training our model and making predictions, and the `roboflow` package for loading the sample dataset our client has asked us to use. 

In the code below, we install the `ultralytics` and `roboflow` packages to our Google Colab runtime.

```python
!pip install ultralytics
!pip install roboflow
```

Next, we need to grab our dataset from Roboflow. To do this we need the API key that Roboflow generated for us when we created our Roboflow account, and which we then stored in `Secrets` pane in Google Colab.

In the code below, we:
* Import the required Python packages
* Retrieve our Roboflow API key from the Colab userdata store
* Load the conveyor dataset from Roboflow into our Colab runtime

```python
from google.colab import userdata

from roboflow import Roboflow
rf = Roboflow(api_key=userdata.get("ROBOFLOW_API_KEY"))
project = rf.workspace("conveyor-550m0").project("conveyor-hhrzw")
version = project.version(3)
dataset = version.download("yolov11")
```

---

# Testing Baseline Performance <a name="baseline-performance"></a>
Our experience tells us that we may need fine-tune a pretrained model using our client's data, but before we spend time and money on model training, let's test that assumption. In other words, let's see how an off-the-shelf model does at detecting objects in the conveyor dataset. We'll use the nano version of the Ultralytics YOLO11 models. This is the smallest of their YOLO11 models but it will do nicely for our proof-of-concept testing.

[[@software{yolo11_ultralytics,
  author = {Glenn Jocher and Jing Qiu},
  title = {Ultralytics YOLO11},
  version = {11.0.0},
  year = {2024},
  url = {https://github.com/ultralytics/ultralytics},
  orcid = {0000-0001-5950-6979, 0000-0003-3783-7069},
  license = {AGPL-3.0}
}]]

### Creating Our Model Object
```python
from ultralytics import YOLO
# Create a model object using Ultralytic's YOLO11 nano object detection model
model = YOLO("yolo11n.pt")
```
### Inspecting Performance With a Sample Image
Next we'll ask the model to make a prediction (i.e. detect objects) using one of the images in the conveyor dataset's validation set.

```python
# Ask the YOLO11 nano model to detect objects in a sample image
results = model.predict("/content/conveyor-3/valid/images/box_0817_png.rf.016105c89863b0d33defe831ee0dea92.jpg")
# Display the image in our notebook, annotated with the objects the model detected
results[0].show()
```

When we run the code above the following image appears in our notebook:
![Object detection using baseline model with a sample image](/img/posts/baseline_object_detection_sample.png)

Well, clearly that's not quite what we're looking for! The model missed the cardboard box on the conveyor, and it detected what it thinks is a chair, but which is actually a conveyor belt support frame.

### Inspecting Performance With a Batch of Sample Images
To get a better picture of how our baseline model performes we need to do more than test just one sample image. Handily for us the `ultralytics` Python package gives us a very easy way to visualize the model's performance on our conveyor dataset.

In the code below we call the model's `val` method which will perform object detection on each image in the conveyor dataset's validation set and compare the results to the known object classes and locations in those images (which is provided with the dataset).
```python
results = model.val(data="/content/conveyor-3/data.yaml")
```

OK, let's take a look at the output.  Let's see what the model detected in a small batch of images from our validation set.

![Object detection using baseline model with sample validation batch](/img/posts/baseline_val_batch2_pred.jpg)

Well, there's certainly room for improvement! The model did correctly detect a person in one of the images, but in six of the images it didn't detect anything and in four of the images it was pretty determined it was going to tell us that there is a chair.

---

# Training the Model <a name="model-training"></a>]
We'll now go on to train the model using our dataset. Since much of the pretrained model's power comes from the large dataset it was trained on, we won't be wiping all that work and starting again. Instead, we'll be *fine-tuning* the model using our data, so we take advantage of the pretrained model's power, but then do some 'final adjustments' to make it perform really well on the particular dataset we want to use it on.

Using the `ultralytics` Python package we only need one line of code to get the fine-tuning up-and-running.

```python
# Fine-tune the Ultralytic's YOLO11 nano model using our conveyor dataset
train_results = model.train(data="/content/conveyor-3/data.yaml", epochs=10)
```

Since we're at the proof of concept stage and dont' want to spend too much money on training we specify just 10 epochs of training (that is, 10 full passes of all the data through the YOLO model). If the results are promising after 10 epochs we can always train with a larger number of epochs later.

---

# Testing Custom Model Performance <a name="custom-performance"></a>
Now let's see how I model does on the same small batch of images from our validation set.

![Object detection using custom model with sample validation batch](/img/posts/custom_val_batch2_pred.jpg)

Wow, how cool is that! The model correctly identified cardboard boxes in four of the images and drew very accurate bounding boxes around them. In the other 12 images it correctly located the conveyor belt. In three of those images it did detect two conveyors when actually there is only one, but since we've only done 10 epochs of training so far that's still pretty impressive.

Next, let's look at the 'confusion matrix', which we can find at `/content/runs/detect/train/confusion_matrix.png` in the Files pane of our Colab notebook. The confusion matrix is a chart that shows what the model got right and what it got wrong. If you've never seen a confusion matrix before, the gist of it is that, if everything is detected correctly, we should see a diagonal line of squares stretching from the top-left of the matrix to its bottom-right. Any squares that are off that diagonal are either false positives (where the model detected an object that isn't actually there) or false negatives (where the model missed an object that is actually present in the image). 

![Confusion matrix for custom model predictions on conveyor dataset](/img/posts/custom_object_detection_confusion_matrix.png)

Here we can see that the model did really well at detecting cardboard boxes - it found all 167 of them. How awesome is that! After just 10 epochs of training the model was able to correctly detect 100% of the cardboard boxes in our validation set!

With conveyors the model did pretty well, correctly detecting 42 out of a total of 46. 

[[Hang on, we shouldn't have any false negatives for background, as we don't have that in our dataset YAML]]

---

# Inspecting Training Progress <a name="training-progress"></a>
Since we've only trained for 10 epochs the custom model weights may not have had time to settle down to their 'final' values, in other words, to the values that they would have if we trained for a very large number of epochs. 

We can take a look at `/content/runs/detect/train/results.png` which `ultralytics` produces for us.

![Training results charts when training custom model on our dataset](/img/posts/results_train_10_epochs.png)

We can see that the loss function values (which are used in the training algorithm to 'push' the model weights in the right direction) and the output metrics (which measure model performance) have settled down reasonably well, though there is definitely scope for them to settle down further if we train for more than 10 epochs.

---

# Discussion, Growth & Next Steps <a name="growth-next-steps"></a>
Our work with this dataset serves as a nice proof-of-concept that we can show to our client. The next steps would be for the client to gather some even better training data for us - ideally we want a large set of images that represent the actual environments that their products get used in - then we could train the object detection using that dataset for a larger number of epochs to get even better results.



