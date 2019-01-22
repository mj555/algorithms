# algorithms

# Table of Contents

  1.  A Brief Overview of the Different R-CNN Algorithms for Object Detection
  2.  Understanding the Problem Statement
  3.  Setting up the System
  4.  Data Exploration
  5.  Implementing Faster R-CNN

# A Brief Overview of the Different R-CNN Algorithms for Object Detection

Let’s quickly summarize the different algorithms in the R-CNN family (R-CNN, Fast R-CNN, and Faster R-CNN) that we saw in the first article. This will help lay the ground for our implementation part later when we will predict the bounding boxes present in previously unseen images (new data).

R-CNN extracts a bunch of regions from the given image using selective search, and then checks if any of these boxes contains an object. We first extract these regions, and for each region, CNN is used to extract specific features. Finally, these features are then used to detect objects. Unfortunately, R-CNN becomes rather slow due to these multiple steps involved in the process.

![](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2018/10/rcnn.png)

Fast R-CNN, on the other hand, passes the entire image to ConvNet which generates regions of interest (instead of passing the extracted regions from the image). Also, instead of using three different models (as we saw in R-CNN), it uses a single model which extracts features from the regions, classifies them into different classes, and returns the bounding boxes.

All these steps are done simultaneously, thus making it execute faster as compared to R-CNN. Fast R-CNN is, however, not fast enough when applied on a large dataset as it also uses selective search for extracting the regions.

![](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2018/10/Fast-rcnn.png)

Faster R-CNN fixes the problem of selective search by replacing it with Region Proposal Network (RPN). We first extract feature maps from the input image using ConvNet and then pass those maps through a RPN which returns object proposals. Finally, these maps are classified and the bounding boxes are predicted.

![](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2018/10/Faster-rcnn.png)

I have summarized below the steps followed by a Faster R-CNN algorithm to detect objects in an image:

* Take an input image and pass it to the ConvNet which returns feature maps for the image

* Apply Region Proposal Network (RPN) on these feature maps and get object proposals
* Apply ROI pooling layer to bring down all the proposals to the same size
* Finally, pass these proposals to a fully connected layer in order to classify any predict the bounding boxes for   the image

CNN: Divides the image into multiple regions and then classifies each region into various classes. Needs a lot of regions to predict accurately and hence high computation time.

R-CNN: Uses selective search to generate regions. Extracts around 2000 regions from each image. 40-50 seconds High computation time as each region is passed to the CNN separately. Also, it uses three different models for making predictions.

Fast R-CNN: Each image is passed only once to the CNN and feature maps are extracted. Selective search is used on these maps to generate predictions. Combines all the three models used in R-CNN together. 2 seconds Selective search is slow and hence computation time is still high.

Faster R-CNN: Replaces the selective search method with region proposal network (RPN) which makes the algorithm much faster. 0.2 seconds Object proposal takes time and as there are different systems working one after the other, the performance of systems depends on how the previous system has performed.

Now that we have a grasp on this topic, it’s time to jump from the theory into the practical part of our article. Let’s implement Faster R-CNN using a really cool (and rather useful) dataset with potential real-life applications!


# Understanding the Problem Statement

We will be working on a healthcare related dataset and the aim here is to solve a Blood Cell Detection problem. Our task is to detect all the Red Blood Cells (RBCs), White Blood Cells (WBCs), and Platelets in each image taken via microscopic image readings. Below is a sample of what our final predictions should look like:

![](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2018/10/example.jpg)

The reason for choosing this dataset is that the density of RBCs, WBCs and Platelets in our blood stream provides a lot of information about the immune system and hemoglobin. This can help us potentially identify whether a person is healthy or not, and if any discrepancy is found in their blood, actions can be taken quickly to diagnose that.

Manually looking at the sample via a microscope is a tedious process. And this is where Deep Learning models play such a vital role. They can classify and detect the blood cells from microscopic images with impressive precision.

The full blood cell detection dataset for our challenge can be downloaded [from here](https://github.com/Shenggan/BCCD_Dataset). I have modified the data a tiny bit for the scope of this article:

    The bounding boxes have been converted from the given .xml format to a .csv format
    I have also created the training and test set split on the entire dataset by randomly picking images for the split

Note that we will be using the popular Keras framework with a TensorFlow backend in Python to train and build our model.

 
# Setting up the System

Before we actually get into the model building phase, we need to ensure that the right libraries and frameworks have been installed. The below libraries are required to run this project:

* pandas
* matplotlib
* tensorflow
* keras==2.0.3
* numpy
* opencv-python
* sklearn
* h5py

Most of the above mentioned libraries will already be present on your machine if you have Anaconda and Jupyter Notebooks installed. Additionally, I recommend downloading the requirement.txt file from [this link](https://drive.google.com/file/d/1R4O0stMW9Wjksg-o7c54svntDiyask1B/view) and use that to install the remaining libraries. Type the following command in the terminal to do this:

pip install -r requirement.txt

Alright, our system is now set and we can move on to working with the data!


# Data Exploration

It’s always a good idea (and frankly, a mandatory step) to first explore the data we have. This helps us not only unearth hidden patterns, but gain a valuable overall insight into what we are working with. The three files I have created out of the entire dataset are:

 train_images: Images that we will be using to train the model. We have the classes and the actual bounding boxes for each class in this folder.
 test_images: Images in this folder will be used to make predictions using the trained model. This set is missing the classes and the bounding boxes for these classes.
 train.csv: Contains the name, class and bounding box coordinates for each image. There can be multiple rows for one image as a single image can have more than one object.

Let’s read the .csv file (you can create your own .csv file from the original dataset if you feel like experimenting) and print out the first few rows. We’ll need to first import the below libraries for this:

There are 6 columns in the train file. Let’s understand what each column represents:

    image_names: contains the name of the image
    cell_type: denotes the type of the cell
    xmin: x-coordinate of the bottom left part of the image
    xmax: x-coordinate of the top right part of the image
    ymin: y-coordinate of the bottom left part of the image
    ymax: y-coordinate of the top right part of the image

Let’s now print an image to visualize what we’re working with:

# reading single image using imread function of matplotlib
image = plt.imread('images/1.jpg')
plt.imshow(image)

![](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2018/10/Screenshot-from-2018-10-29-18-32-43.png)

This is what a blood cell image looks like. Here, the blue part represents the WBCs, and the slightly red parts represent the RBCs. Let’s look at how many images, and the different type of classes, there are in our training set.

Number of unique training images:

  train['image_names'].nunique()

So, we have 254 training images

Number of classes:
  train['cell_type'].value_counts()

We have three different classes of cells, i.e., RBC, WBC and Platelets. Finally, let’s look at how an image with detected objects will look like:

  fig = plt.figure()

add axes to the image:

  ax = fig.add_axes([0,0,1,1])

read and plot the image:

  image = plt.imread('images/1.jpg')
   plt.imshow(image)

iterating over the image for different objects:
  for _,row in train[train.image_names == "1.jpg"].iterrows():
    xmin = row.xmin
    xmax = row.xmax
    ymin = row.ymin
    ymax = row.ymax
    
    width = xmax - xmin
    height = ymax - ymin
    
    # assign different color to different classes of objects
    if row.cell_type == 'RBC':
        edgecolor = 'r'
        ax.annotate('RBC', xy=(xmax-40,ymin+20))
    elif row.cell_type == 'WBC':
        edgecolor = 'b'
        ax.annotate('WBC', xy=(xmax-40,ymin+20))
    elif row.cell_type == 'Platelets':
        edgecolor = 'g'
        ax.annotate('Platelets', xy=(xmax-40,ymin+20))
        
    # add bounding boxes to the image
    rect = patches.Rectangle((xmin,ymin), width, height, edgecolor = edgecolor, facecolor = 'none')
    
    ax.add_patch(rect)
    
![](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2018/10/Screenshot-from-2018-10-31-18-00-50.png)
    
This is what a training example looks like. We have the different classes and their corresponding bounding boxes. Let’s now train our model on these images. We will be using the keras_frcnn library to train our model as well as to get predictions on the test images.

#Implementing Faster R-CNN

tbr
