# Orchids-Color-Detection

This repository is an implementation of a method described in the paper "Automated color detection in orchids using color labels and deep learning" (https://doi.org/10.1371/journal.pone.0259036).
The method was proposed to detect Color of Flower (CF) and Color of Labellum (CL) in Orchids flower using deep learning and color label.

The repositoy includes:
- Source code for building color classifiers using Deep Learning
	Basically, source code for each architecture used in the paper is the same, only need to change the type of pre-trained model and use different color schemes. 
	In this repository, the attached code is for Xception.
- Pre-trained weights for each classifiers.
- Source code for determining the color (output color label).

To use our code (redo our experiments):
1. Please setup the environment based on requirements.txt.
2. The images for training, validation and testing should be put in different folders. 
3. The dataset (all of the images and the labels for training, validation and testing) can be downloaded from https://doi.org/10.7910/DVN/0HNECY. 
4. For using the images used in the method, please find the folder Color_Classifier -- Multiple_Color -- CF1, CF2, CL1 and CL2.
5. The weight of the deep learning model is saved in the folder "Model".

# Citation

Use this bibtex to cite the work describe in this repository:

- @article{10.1371/journal.pone.0259036,
    doi = {10.1371/journal.pone.0259036},
    author = {Apriyanti, Diah Harnoni AND Spreeuwers, Luuk J. AND Lucas, Peter J. F. AND Veldhuis, Raymond N. J.},
    journal = {PLOS ONE},
    publisher = {Public Library of Science},
    title = {Automated color detection in orchids using color labels and deep learning},
    year = {2021},
    month = {10},
    volume = {16},
    url = {https://doi.org/10.1371/journal.pone.0259036},
    pages = {1-27},
    abstract = {The color of particular parts of a flower is often employed as one of the features to differentiate between flower types. Thus, color is also used in flower-image classification. Color labels, such as ‘green’, ‘red’, and ‘yellow’, are used by taxonomists and lay people alike to describe the color of plants. Flower image datasets usually only consist of images and do not contain flower descriptions. In this research, we have built a flower-image dataset, especially regarding orchid species, which consists of human-friendly textual descriptions of features of specific flowers, on the one hand, and digital photographs indicating how a flower looks like, on the other hand. Using this dataset, a new automated color detection model was developed. It is the first research of its kind using color labels and deep learning for color detection in flower recognition. As deep learning often excels in pattern recognition in digital images, we applied transfer learning with various amounts of unfreezing of layers with five different neural network architectures (VGG16, Inception, Resnet50, Xception, Nasnet) to determine which architecture and which scheme of transfer learning performs best. In addition, various color scheme scenarios were tested, including the use of primary and secondary color together, and, in addition, the effectiveness of dealing with multi-class classification using multi-class, combined binary, and, finally, ensemble classifiers were studied. The best overall performance was achieved by the ensemble classifier. The results show that the proposed method can detect the color of flower and labellum very well without having to perform image segmentation. The result of this study can act as a foundation for the development of an image-based plant recognition system that is able to offer an explanation of a provided classification.},
    number = {10},
}

and 

- @data{DVN/0HNECY_2020,
author = {Apriyanti, D.H. and Spreeuwers, L.J. and Lucas, P.J.F. and Veldhuis, R.N.J.},
publisher = {Harvard Dataverse},
title = {{Orchid Flowers Dataset}},
year = {2020},
version = {V1},
doi = {10.7910/DVN/0HNECY},
url = {https://doi.org/10.7910/DVN/0HNECY}
}
