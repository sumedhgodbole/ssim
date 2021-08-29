import pytorch_msssim
import torch
import numpy as np
import cv2
from PIL import Image
from torch.autograd import Variable

img_path = "images/"

npImg1 = cv2.imread(img_path + "car_image1.jpg")
npImg2 = cv2.imread(img_path + "car_image1.jpg")

print("original ",npImg1.shape)
img_resize = np.rollaxis(npImg1, 2)
print("rolled ", img_resize.shape)



img1 = torch.from_numpy(np.rollaxis(npImg1, 2)).float().unsqueeze(0)
img2 = torch.from_numpy(np.rollaxis(npImg2, 2)).float().unsqueeze(0)

# img1 = Variable( img1, requires_grad = False)
# img2 = Variable( img2, requires_grad = False)

print(torch.cuda.is_available())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

m = pytorch_msssim.MSSSIM()

#img1 = torch.rand(1, 1, 256, 256)
#img2 = torch.rand(1, 1, 256, 256)

print(pytorch_msssim.msssim(img1, img2).item())
print(m(img1, img2))