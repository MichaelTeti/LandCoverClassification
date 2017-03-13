# Hyperspectral CNN
A convolutional neural network in Tensorflow to classify pixels in a 1096 x 1096 hyperspectral satellite image.  

## Background
Hyperspectral imagery is used in a number of applications ranging from geospatial and climate modeling to land cover classification and predicting optimal mining and oil-drilling sites. Instead of acquiring images with shape *m* x *n* x 3, as in color photographs, hyperspectral sensors collect data of size *m* x *n* x &lambda, where &lambda is at least 100. A pixel's spectral signature, defined as all of the $\lambda$ values corresponding to that pixel, is often very rich in information, and it is often possible to distinguish between the type of material (e.g. water, concrete, vegetation, etc.) in the pixel using this information \cite{baraniuk2011introduction}. Certain wavelengths present in the spectral signature can even distinguish between subtypes of each class. For example, information about the amount of near-infrared light reflected by a plant can help determine whether the plant is healthy or unhealthy even before observable changes to the plant's physical appearance begin \cite{beeri2007estimating}. 

![alt tag](https://github.com/MichaelTeti/LandCoverClassification/blob/master/Pavia_60.png)
![alt tag](https://github.com/MichaelTeti/LandCoverClassification/blob/master/Pavia_gt.png)
