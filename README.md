# Predict 19 Cephlometric Landmarks from a Cephalogram
Simple PyTorch solution to find 19 cephalometric landmarks from a lateral cephalogram for CLI/in a project.

Model used to predict landmarks is trained from the source code of [Cephalometric Landmark Detection by Attentive Feature Pyramid Fusion and Regression-Voting](https://arxiv.org/pdf/1908.08841.pdf).

## How to use py-ceph?
py-ceph accepts CLI usage for batch process or single file interactive view.
____

```commandline
py-ceph --image_src my-single-image-path
```
For single image processing, it will output:
- the image interactively with a grid
- the labeled landmarks to the console
____
```commandline
py-ceph --image_folder my-image-folder-path
```
For batch processing, it will create a folder in the images folder path named `predicted_images` and output:
- each image overlayed with predicted ceph landmarks
- csv of the coordinates for each labeled landmark

## To-do
- Non-CLI usage

## Dependencies
- torch
- numpy
- torch
- matplotlib
- scikit-image