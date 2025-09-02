---
layout: post
title: “Flower Power” - Creating An Image Search Engine Using Deep Learning
image: "/posts/dl-search-engine-title-img-flowers.png"
tags: [Deep Learning, CNN, Data Science, Computer Vision, Python]
---

Here at *We-Do-Data-Science-So-You-Don't-Have-To Limited* our work is never done! This morning a local florist got in touch to ask if we could help build a search engine for an app he's creating. The idea is that a customer would take a picture of a flower with their phone and the search engine would show them similar flowers from his product range. I told him we should be able to use transfer learning to build him a search engine quite quickly, so let's get to it!

---

# Table of Contents
- [1. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth/Next Steps](#overview-growth)
- [2. Sample Data Overview](#sample-data-overview)
- [3. Transfer Learning Overview](#transfer-learning-overview)
- [4. Adjusting the VGG16 Model](#vgg16-setup)
- [5. Loading the Dataset](#loading-dataset)
- [6. Image Preprocessing and Featurisation](#image-preprocessing)
- [7. Execute Search](#execute-search)
- [8. Discussion, Growth & Next Steps](#growth-next-steps)

---

# Project Overview <a name="overview-main"></a>
### Context <a name="overview-context"></a>
Our client finds that customers often come into his shop wanting to buy the kind of flowers that they've seen in a neighbour's garden or during a walk through the countryside. Sometimes it's hard to figure out what flower they're describing so usually he asks them to take a photo of the flower and show it to him.

This works quite well, but often he finds that customers want a type of flower that he doesn't have in stock in his own shop, but which he knows he can get from the wholesale market. Once they've made the effort of coming to his shop customers usually want to take flowers away there and then, so they either end up settling for something that isn't quite what they were looking for or, worse still, they walk out without buying anything.

So, he'd like us to build a search engine that let's customers identify which flower they're looking at and to then order through his app for next day delivery.  He'd like the search engine to include all the types of flowers that he can buy at the wholesale flower market. If a customer orders a flower he doesn't have in stock he can pop to the wholesale market the next morning and then ship it to them.

### Actions <a name="overview-actions"></a>
To implement our search engine we'll take inspiration from the vector embedding approach that is currently very popular for text-based document search. We'll use the pre-trained VGG16 network, but, instead of using it to make a class prediction, we'll modify it to produce a "feature vector" representation of each flower image. We'll then use these vectors to compare images and find images of flowers that are similar to the image provided by the end user (the florist's customer).

To get our vector representation of an image we'll replace the max pooling layer in the VGG16 architecture with an average pooling layer and take that layer's output as our vector representation of an image.

We'll pre-process the 6149 images in our dataset to make them all the same size, and then pass them through the modified VGG16 network to extract their feature vectors and store these as a database of image vectors.  When a search image is provided we'll apply the same pre-processing before passing the image through the network and then using cosine similarity to find images from the database that are good matches for the search image.

We'll then reduce the search hits down to a list of unique flower types and return one image of the five most similar types of flower to the user.

### Results <a name="overview-results"></a>
We tested two sample images and found that our search engine was able to return images of flowers that are pretty good matches for the sample images provied. You can see the sample images and the search hits in the dedicated section, below.

### Discussion, Growth & Next Steps <a name="overview-growth"></a>
We coded this project up as a proof of concept. For a production app we'd split the code that does the searching out into a separate script to allow a front-end engineer to use this code when developing the user app while the AI engineers continue to refine the vectorisation model.

For a production app we'd also want to query the florist's product database so our the app could provide users with current price and availability information for each of the search results.

The early results we got with a couple of sample images were quite encouraging. The next step would be to use a large batch of test images and test for accuracy by comparing the first search engine hit with the known flower category of each test image.  Here we only looked at cosine similarity, but it would be interesting to investigate other distance metrics, such as Euclidean distance, to see if search accuracy could be made even better.

The first match by cosine similarity is in effect a class prediction.  It would be interesting to compare accuracy of this against the baseline VGG prediction accuracy on the same dataset.

---

# Sample Data Overview <a name="sample-data-overview"></a>

By happy coincidence it turns out that the range of flowers sold at the wholesale market is exactly the same as the range of flower varieties in the *Oxford 102 Category Flower Dataset*, so we'll use that as our image dataset.

- [102 Category Flower Dataset](https://www.robots.ox.ac.uk/%7Evgg/data/flowers/102/): *Nilsback, M-E. and Zisserman, A.*, "Automated Flower Classification over a Large Number of Classes", *Proceedings of the Indian Conference on Computer Vision, Graphics and Image Processing*, 2008

This dataset contains images of 102 different types of flowers that are commonly found in the United Kingdom. One unusual aspect of this dataset is that its test set (6149 images) is much larger than its train set (1000 images). For our purposes it makes sense to use the 6149 images in the test set to create our vector database of flower images (we'll call this our "database set") and use a sample of images from the training set to test our search engine.

This is a sample of the images in the database set:

![Deep Learning Search Engine - Image Examples](/img/posts/search-engine-image-examples-flowers.png)

In image datasets it's not uncommon to encounter multiple different sizes of images and at this stage we notice that the dataset contains images of various sizes, ranging from 500x500 to 1168x500. For now we make a note of this fact, and later we'll add some code handle it.

---

# Transfer Learning Overview <a name="transfer-learning-overview"></a>
### Overview
The idea of transfer learning is that we take a model that was trained for some other purpose and, after making a few adjustments, apply it to some other task. In other words we  *transfer* the capabilities of the model from the original task it was designed for to our own, specific task.

We'll use one of the VGG family of models. These were developed by the Visual Geometry Group (VGG) at Oxford University. Specifically, we'll use VGG16 which consists of 13 convolutional layers and 3 fully-connected layers and was trained on the famous *ImageNet* database of millions of images. ImageNet contains more than 20,000 categories of images, so very many of them will *not* be of flowers. Our hope though, is that it will be sufficiently powerful that when we leverage its classification abilities on our task it'll enable us to produce a pretty good search engine. 

By default the output of VGG16 would be a prediction of which class an image belongs to - that is, a prediction of what is depicted in the picture. For our search engine we want to present the user with a selection of flowers that are the same as, or quite similar to, the search image, so we'll remove the final few layers of the VGG model and instead get it to output a numeric representation of an image in a format that we can use do compare images and see which ones are similar.

This *transfer learning* approach should save us a lot of time and effort and enable us to leverage the power of an existing (and very successful) model. To apply machine learning techniques to a business problem you don't always have to create your own model architecture (which can involve many iterations and much tweeking) and go to the expense (both computation and monetary) of training your own model from scratch.

![VGG16 Architecture](/img/posts/vgg16-architecture-flowers.png)

### Vector Representation of Images
The VGG16 model came second in the classification track of the ImageNet Challenge 2014. To adapt it to our task we'll first remove the final four layers (three fully-connected layers and the softmax prediction layer) then, since the output of the 7x7x512 max pooling layer would be multiple arrays, we'll replace it with an average pooling layer that will output a single numeric vector for each image we path through the network, which is what we want for our image simiilarity comparision.

In other words, our plan is to user the power of VGG16's 13 convolutional layers to produce a representation of an image that we can readily use in our image search task.

---

# Adjusting the VGG16 Model <a name="vgg16-setup"></a>
We make use of Keras for loading VGG16 and for tailoring it (removing the final few layers and using average pooling instead of max pooling). 

In the code below, we:
* Import the required Python packages
* Define variables containing the default image parameters for VGG16
* Create an instance of VGG16 without its 'top' (that is, its fully-connected and softmax layers) and with average pooling for the new final layer
* Save the model to h5 format so we don't have to recreate it every time we want to use our search engine

```python
# Import required packages
import tensorflow_datasets as tfds
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, smart_resize
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle

# Default image shape for VGG16 is 224x224 with 3 channels
img_width = 224
img_height = 224
num_channels = 3

# Load pre-trained VGG model excluding the 'top' and using average pooling
# so we get a vector (rather than array) representation of each image
vgg = VGG16(input_shape=(img_width, img_height, num_channels),
            include_top=False,
            pooling='avg')
print(vgg.summary())

# Create our keras model object using our VGG instance
model = Model(inputs=vgg.input, outputs=vgg.output)

# Save our model to file
model.save('vgg16-flower-search-engine.h5')
```

The `print(vgg.summary())` line gives us a summary of the model architecture:
```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 224, 224, 3)]     0         
_________________________________________________________________                                                                 
 block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792      
_________________________________________________________________                                                                 
 block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928     
_________________________________________________________________                                                                 
 block1_pool (MaxPooling2D)  (None, 112, 112, 64)      0         
_________________________________________________________________                                                                 
 block2_conv1 (Conv2D)       (None, 112, 112, 128)     73856     
_________________________________________________________________                                                                 
 block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584    
_________________________________________________________________                                                                 
 block2_pool (MaxPooling2D)  (None, 56, 56, 128)       0         
_________________________________________________________________                                                                 
 block3_conv1 (Conv2D)       (None, 56, 56, 256)       295168    
_________________________________________________________________                                                                 
 block3_conv2 (Conv2D)       (None, 56, 56, 256)       590080    
_________________________________________________________________                                                                 
 block3_conv3 (Conv2D)       (None, 56, 56, 256)       590080    
_________________________________________________________________                                                                 
 block3_pool (MaxPooling2D)  (None, 28, 28, 256)       0         
_________________________________________________________________                                                                 
 block4_conv1 (Conv2D)       (None, 28, 28, 512)       1180160   
_________________________________________________________________                                                                 
 block4_conv2 (Conv2D)       (None, 28, 28, 512)       2359808   
_________________________________________________________________                                                                 
 block4_conv3 (Conv2D)       (None, 28, 28, 512)       2359808   
_________________________________________________________________                                                                 
 block4_pool (MaxPooling2D)  (None, 14, 14, 512)       0         
_________________________________________________________________                                                                 
 block5_conv1 (Conv2D)       (None, 14, 14, 512)       2359808   
_________________________________________________________________
 block5_conv2 (Conv2D)       (None, 14, 14, 512)       2359808   
_________________________________________________________________
 block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808   
_________________________________________________________________
 block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0         
_________________________________________________________________                                                                 
 global_average_pooling2d (G  (None, 512)              0         
 lobalAveragePooling2D)                                          
=================================================================
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0
_________________________________________________________________

```

The output shape of the final layer `(None, 512)` indicates that the output we'll get for each image we run through the model will be a vector of length 512. That will do nicely for our image comparison. (The `None` in the layer shape is a placeholder for batch dimension, which is used when you push batches of images through a Keras model).

---

# Loading Our Dataset <a name="loading-dataset"></a>
We use TensorFlow Datasets to import the *Oxford 102 Category Flower Dataset*. We pass the `with_info` parameter so we can get the class name metadata (that is, the names of the different types of flowers in the dataset).

Note that we don't need to specify a location for the dataset. Instead, we just specify its name *oxford_flowers102* and TensorFlow Datasets will grab it from a repository for us.

```python
# Load the Oxford Flowers 102 dataset, swapping train for test
(train, test), dataset_info = tfds.load('oxford_flowers102', split=['test', 'train'], with_info=True)
# Get list of the types of flowers in the dataset
class_names = dataset_info.features['label'].names

```

---

# Image Preprocessing and Featurisation <a name="image-preprocessing"></a>
### Preprocessing
Next we'll create a helper function that we can call for each image before we put the image through the network. Since the images in the dataset are not all the same size the helper function will resize the supplied image to the size that the VGG16 is designed to take, which we specified above in our `img_width` and `img_height` variables.

```python
# Helper function to pre-process each image
def preprocess_image(image):
    image = smart_resize(image, (img_height, img_width))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(np.copy(image))
    
    return image
```
<br>
The `preprocess_image` function does the following:
* Resizes the image to the size that VGG16 is designed to take
* Turns the image into a numpy array
* Adds the batch dimension that Keras expects
* Uses the pre-processing method for VGG16 provided by Keras, which applies adjustments to the colour channels
* Returns the image as a numpy array

### Preprocess and Featurise Database Images
We'll now create a feature vector for each of the 6149 images in our database set.

```python
# A list to store the images in for easy access later
images = []
# A numpy array to store the image feature vectors in
image_vectors = np.empty((0,512))

for image in train:
    images.append(image)
    image = preprocess_image(image['image'])

    # Use our VGG16 model to get the image's feature vector
    image_vector = model.predict(image)
    image_vectors = np.append(image_vectors, image_vector, axis=0)

# Save the feature vectors to file for use when searching
pickle.dump(image_vectors, open('image_vectors.p', 'wb'))
```

---

# Execute Search <a name="execute-search"></a>
We'll now go on to search in our store of vectorized images to find images that are similar to some sample end user images.

### Setting Up Our Vector Search Model
Next we'll:
* Load our adjusted VGG16 model
* Load our feature vector store
* Create a KNN model that we'll use to assess vector similarity
* Create a vectorized representation of some sample end user images
* Search in our vector store for image vectors that are similar to the vectorized sample images

```python
# Load our model and vector database
model = load_model('vgg16-flower-search-engine.h5', compile=False)
image_vectors = pickle.load(open('image_vectors.p', 'rb'))
```

`image_vectors` is our vector database that we want to search in, but it's not yet in a format that we can readily search though. To make it searchable we'll make use of the *NearestNeighbors* model in *scikit-learn*. When we provide it with the vector representation of a sample image it will return for us the vectors that are the most similar. We'll specify that the *NearestNeighbors* model uses *cosine similarity* to determine vector similarity - this means the model will calculate the angle between the sample image vector and each vector in the store, and then rank the vectors in order of *lowest* angle. The idea behind this is that vectors are deemed to be similar if they point in the same direction. 

In the code below we:
* Create our *NearestNeighbors* model and specify that it should return 100 vectors that are similar to any sample vector we provide it with
* Fit the *NearestNeighbors* model using our `image_vectors` vector store

```python
# Create a KNN object for vector search
neighbors = NearestNeighbors(n_neighbors=100, metric='cosine')

# Fit the KNN model using our image vectors
neighbors.fit(image_vectors)
```

### Pre-processing and Featurising Sample Search Images
The first image we'll use to demonstrate the search engine functionality is from the test set we loaded earlier, from the *Oxford 102 Category Flower Dataset*:
![Sample search image 1](/img/posts/sample-search-image-1.png)

We preprocess and featurise our sample image in exactly the same was as we did for the image we added to our vector store.
```python
# Grab two sample images from our test set
sample_search_images = list(test.take(2).as_numpy_iterator())

# Pre-process first the sample image
sample_search_image = preprocess_image(sample_search_images[0]['image'])
# Use our VGG16 model to create feature vector for the first sample image
image_vector = model.predict(sample_search_image)
``` 

Next we pass our sample image vector to the *NearestNeighbors* model:

```python
# Get similar images from the KNN model
image_distances, image_indices = neighbors.kneighbors(image_vector)
# Convert the image indices and distances to lists, to make the data easier to plot later
image_indices = list(image_indices[0])
image_distances = list(image_distances[0])
```

### Plot Search Results
We asked for 100 results from the KNN model so that we can filter out image of flowers that are the same type and still have a decent number of flowers to return to the customer. We'll now plot the sample search image, together with the most similar image from each of the categories of flowers that made it into the top 100. We'll annotate the plot with the cosine similiarity distance for each search result (the smaller the better). Let's take a look!

```python
# Plot the search image
plt.imshow(sample_search_images[0]['image'])
plt.text(0, -5, f"Category: {class_names[sample_search_images[0]['label']]}")

# From our image store, grab the images whose vectors were found to match that of the sample image
search_result_images = [images[i] for i in image_indices]

# Plot the search result images
plt.figure(figsize=(12,9))
result_classes = []
for counter, result_image in enumerate(search_result_images):
    result_class = result_image['label'].numpy()
    # Filter the images so that, for each type of flower, we only have the image
    # that most closely matched the sample
    if result_class in result_classes:
        continue
    else:
        result_classes.append(result_class)

    # Plot up to 9 different flower images to send to the customer
    if len(result_classes) < 10:
        ax = plt.subplot(3, 3, len(result_classes))
        plt.imshow(result_image['image'])
        plt.text(0, -5, f"{round(image_distances[counter],3)} Category: {class_names[result_class]}")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
plt.show()
```

The search image and search results are below:

**Search Image**
<br>
![Sample search image 1](/img/posts/sample-search-image-1.png)
<br>
<br>
**Search Results**
![Sample search results 1](/img/posts/sample-search-results-1.png)
<br>
<br>
Pretty impressive results!  From the 102 categories of images in our database of flowers, these are the nine categories that are most similar to the sample image.

The results with the first sample are pretty nice, so let's try another example...

**Search Image**
<br>
![Sample search image 1](/img/posts/sample-search-image-2.png)
<br>
<br>
**Search Results**
![Sample search results 2](/img/posts/sample-search-results-2.png)

<br>
For both sample images our the first hit returned by our search engine was the same category as the sample image, which is really great! The other images the search engine returned will give the customer a nice selection of flowers to choose from. The app our client is developing would show the customer the prices of all the search hit flowers, so the customer would be able to pick one they like the look of at a price that suits their budget.

---

# Discussion, Growth & Next Steps <a name="growth-next-steps"></a>
We coded this project up as a proof of concept. For a production app we'd split the code that does the searching out into a separate script to allow a front-end engineer to use this code when developing the user app while the AI engineers continue to refine the vectorisation model.

For a production app we'd also want to query the florist's product database so our the app could provide users with current price and availability information for each of the search results.

The early results we got with a couple of sample images were quite encouraging. The next step would be to use a large batch of test images and see test for accuracy by comparing the first search engine hit with the known flower category of each test image.  Here we only looked at cosine similarity, but it would be interesting to investigate other distance metrics, such as Euclidean distance, to see if search accuracy could be made even better.

The first match by cosine similarity is in effect a class prediction.  It would be interesting to compare accuracy of this against the baseline VGG prediction accuracy on the same dataset.






