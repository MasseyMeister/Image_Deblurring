import torch
import cv2
import deblur
import numpy as np
import glob as glob
import os
from torchvision.utils import save_image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# set computation device

model = deblur.DeblurCNN().to(device)
model.load_state_dict(torch.load('../outputs/model.pth'))

image_path = glob.glob('../input/bird.jpeg')
# we get all the image paths using the glob module.

image = cv2.imread(image_path, cv2.IMREAD_COLOR)
# Then we use a for loop to test on each of the images in directory
test_image_name = image_path.split(os.path.sep)[-1].split('.')[0]
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = image.reshape(image.shape[0], image.shape[1], 1)
cv2.imwrite(f"../outputs/test_{test_image_name}.png", image)
# get just the image name so that we can easily save the original
# and test image and the output image to the disk.
image = image / 255.  # normalize the pixel values between 0 and 1
cv2.imshow('Greyscale image', image)
# We convert the image to greyscale format at line 6 and make
# it channels-last so as to visualize it using OpenCV. Then we
# show the original test image.
cv2.waitKey(0)
model.eval()
with torch.no_grad():
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = torch.tensor(image, dtype=torch.float).to(device)
    image = image.unsqueeze(0)
    # we switch to eval mode and make the image.
    # channels then convert it into a torch tensor.
    # Then we unsqueeze it to add an extra batch dimension.
    outputs = model(image)
    # feeds the image to the model and we get the output.
outputs = outputs.cpu()
save_image(outputs, f"../outputs/output_{test_image_name}.png")
outputs = outputs.detach().numpy()
outputs = outputs.reshape(outputs.shape[2],
                            outputs.shape[3],
                            outputs.shape[1])
print(outputs.shape)
cv2.imshow('Output', outputs)
# save the output image to disk, 
# visualize it using OpenCV.
cv2.waitKey(0)


