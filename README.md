# Hyperspectral CNN
A convolutional neural network in Tensorflow to classify pixels in a 1096 x 1096 hyperspectral satellite image.  

## Background
Hyperspectral imagery is used in a number of applications ranging from geospatial and climate modeling to land cover classification and predicting optimal mining and oil-drilling sites. Instead of acquiring images with shape *m* x *n* x 3, as in color photographs, hyperspectral sensors collect data of size *m* x *n* x &lambda;, where &lambda; is at least 100. A pixel's spectral signature, defined as all of the &lambda; values corresponding to that pixel, is often very rich in information, and it is often possible to distinguish between the type of material (e.g. water, concrete, vegetation, etc.) in the pixel using this information. Certain wavelengths present in the spectral signature can even distinguish between subtypes of each class. For example, information about the amount of near-infrared light reflected by a plant can help determine whether the plant is healthy or unhealthy even before observable changes to the plant's physical appearance begin. Here we use a convolutional neural network to classify every pixel in a hyperspectral satellite image, Pavia Centre. 
We use the ground truth to determine what class out of nine the pixel belongs to.  
  
  
<p align="center">
<b>Figure 1: The gray-scale satellite image.</b><br>
  <br><br>
  <img src="https://github.com/MichaelTeti/LandCoverClassification/blob/master/Pavia_60.png">
</p>  
  
  
<p align="center">
<b>Figure 2: The different classes in the image assigned arbitrary colors.</b><br>
  <br><br>
  <img src="https://github.com/MichaelTeti/LandCoverClassification/blob/master/Pavia_gt.png">
</p>  
  
  
<p align="center">
<b>Figure 3: The labels of each class.</b><br>
  <br><br>
  <img src="https://github.com/MichaelTeti/LandCoverClassification/blob/master/pavia.png">
</p>  
  
  
<p align="center">
<b>Figure 4: A hyperspectral image can be thought of as a 3D data cube. The spectral signature for a single pixel can be seen on the side of the image.</b><br>
  <br><br>
  <img src="https://github.com/MichaelTeti/LandCoverClassification/blob/master/3907041_orig.png">
</p>  
  
  
As you can see, there are many classes with the same exact crop. The network is able to distinguish between these and get a 99% validation accuracy. For more satellite image results, check out [my website](https://mtetiresearch.com/gallery/).
