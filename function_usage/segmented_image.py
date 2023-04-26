import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from torchvision import models
from PIL import Image
import PIL
import matplotlib.pyplot as plt
import torch
import numpy as np

# https://pytorch.org/vision/main/models/generated/torchvision.models.segmentation.fcn_resnet101.html
# https://pytorch.org/vision/0.12/generated/torchvision.models.segmentation.fcn_resnet101.html
# Constructs a Fully-Convolutional Network model with a ResNet-101 backbone.
# pretrained (bool) – If True, returns a model pre-trained on COCO train2017 which contains the same classes as Pascal VOC
fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()

# https://www.geeksforgeeks.org/python-pil-image-open-method/
# PIL is the Python Imaging Library which provides the python interpreter with image editing capabilities.
# The Image module provides a class with the same name which is used to represent a PIL image.
# PIL.Image.open() Opens and identifies the given image file.
img = Image.open('C:/Users/karen/Downloads/MixedTern1.jpg')
# print(type(img))
# <class 'PIL.JpegImagePlugin.JpegImageFile'>
plt.imshow(img)
plt.show()
# This method will show image in any image viewer
# just like what we get when open in the folder
# img.show()

# Apply the transformations needed
import torchvision.transforms as T
# https://pytorch.org/vision/0.9/transforms.html
# Transforms are common image transformations. They can be chained together using Compose
# All transformations accept PIL Image, Tensor Image or batch of Tensor Images as input.
# Tensor Image is a tensor with (C, H, W) shape, where C is a number of channels, H and W are image height and width
# Batch of Tensor Images is a tensor of (B, C, H, W) shape, where B is a number of images in the batch.
trf = T.Compose([T.Resize(256),
                 T.CenterCrop(224),
                 T.ToTensor(),
                 T.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])])
# T.Compose:
# Composes several transforms together. This transform does not support torchscript.
###### torchvision.transforms.Resize(size, interpolation=<InterpolationMode.BILINEAR: 'bilinear'>)
# Resize the input image to the given size
# Parameters:	size (sequence or int) – Desired output size.
###### torchvision.transforms.CenterCrop(size)
# Crops the given image at the center. parameter: size - Desired output size of the crop
# T.CenterCrop(224)
##### torchvision.transforms.ToTensor
# Convert a PIL Image or numpy.ndarray to tensor.
##### torchvision.transforms.functional.normalize(tensor: torch.Tensor,
#               mean: List[float], std: List[float], inplace: bool = False)
# Normalize a tensor image with mean and standard deviation.
# 1. resize 2. crop 3. change to tensor 4. normalize tensor

# torch.unsqueeze(input, dim) → Tensor
# Returns a new tensor with a dimension of size one inserted at the specified position.
inp = trf(img).unsqueeze(0)


# Pass the input through the net
out = fcn(inp)['out']
print(out.shape)

##### torch.argmax(input) → LongTensor
# Returns the indices of the maximum value of all elements in the input tensor.
##### Tensor.detach()
# Returns a new Tensor, detached from the current graph.
om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
print(om.shape)
print(np.unique(om))


# Define the helper function
def decode_segmap(image, nc=21):
    label_colors = np.array([(0, 0, 0),  # 0=background
                             # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                             (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                             # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                             (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                             # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                             (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                             # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                             (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb

rgb = decode_segmap(om)
plt.imshow(rgb)
plt.show()


import cv2
import numpy as np

# a video capture object
cam = cv2.VideoCapture(0)

while(True):
    # read a video frame by frame
    ret, frame = cam.read()

    img = PIL.Image.fromarray(frame)
    inp = trf(img).unsqueeze(0)
    out = fcn(inp)['out']
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    rgb = decode_segmap(om)

    cv2.imshow('frame', frame)
    cv2.imshow('segment_img', rgb)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# release the capture object
cam.release()
# destroy all windows
cv2.destroyAllWindows()
