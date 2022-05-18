import os
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split

# Constructing the argument parser
# The default number of epochs is 40.
# --epochs is the only command line argument
# that we will give while executing the program.
# we define the save_decoded_image() function. We will call
# this image while validating the model to save the deblurred images.

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=40,
                    help='number of epochs to train the model for')
args = vars(parser.parse_args())


def save_decoded_image(img, name):
    img = img.view(img.size(0), 3, 224, 224)
    save_image(img, name)


# In the below code block, we are also selecting the computation device.
# It is better if you have a GPU in your system for running the program.
# We are defining a batch size of 2.

image_dir = '../outputs/saved_images'
os.makedirs(image_dir, exist_ok=True)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
batch_size = 2

# gauss_blur and sharp are two lists containing the image file paths to the
# blurred and sharp images respectively. We are also sorting the list so that
# the corresponding blurred and sharp image file paths align with each other.

gauss_blur = os.listdir('../input/gaussian_blurred')
gauss_blur.sort()
sharp = os.listdir('../input/sharp')
sharp.sort()

# Then we define two more lists, x_blur and y_sharp which contain the Gaussian
# blurred and sharp image paths respectively. x_blur will act as the training
# data and y_blur will act as the label. So, while training we compare the
# blurred image outputs with the sharp image. We will try to make the blurred
# images as similar to the sharp images as possible.

x_blur = []

for i in range(len(gauss_blur)):
    x_blur.append(gauss_blur[i])

y_sharp = []

for i in range(len(sharp)):
    y_sharp.append(sharp[i])

(x_train, x_val, y_train, y_val) = train_test_split(x_blur,
                                                    y_sharp,
                                                    test_size=0.25)
print(f"Train data instances: {len(x_train)}")
print(f"Validation data instances: {len(x_val)}")

# define transforms
# First, we are converting the image to PIL image format.
# Then we are resizing the image to 224×224 dimensions.
# Finally, we are converting the image to torch tensors.

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


class DeblurDataset(Dataset):
    '''
    In the __init__() function, we initialize the blurred images paths, sharp
    images paths, and the image transforms. After, we have the __getitem__()
    function this reads the blurred image from the blurred image paths.
    Then we apply the transforms to the images. At line 17, we read the sharp
    images, which will act as the labels while training and validating.
    We are applying the same transform to the sharp images as well.
    '''
    def __init__(self, blur_paths, sharp_paths=None, transforms=None):
        self.X = blur_paths
        self.y = sharp_paths
        self.transforms = transforms

    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        blur_image = cv2.imread(f"../input/gaussian_blurred/{self.X[i]}")

        if self.transforms:
            blur_image = self.transforms(blur_image)

        if self.y is not None:
            sharp_image = cv2.imread(f"../input/sharp/{self.y[i]}")
            sharp_image = self.transforms(sharp_image)
            return (blur_image, sharp_image)
        else:
            return blur_image

# Next, we will define the train data, validation data, train loader, and
# validation loader. We will use a batch size of 2 as defined above. Also
# we will shuffle the training data only and not the validation data.


train_data = DeblurDataset(x_train, y_train, transform)
val_data = DeblurDataset(x_val, y_val, transform)
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)


class DeblurCNN(nn.Module):
    def __init__(self):
        super(DeblurCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=2)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


model = DeblurCNN().to(device)
print(model)

# We are using the MSE (Mean Square Error Loss) as we will be comparing the
# pixels of blurred images and sharp images. The optimizer is Adam with a
# learning rate of 0.001. We are also defining a ReduceLROnPlateau() learning
# rate scheduler. The patience is 5 and factor is 5. So, if the loss value
# does not improve for 5 epochs, the new learning rate will be
# old_learning_rate * 0.5.

# the loss function
criterion = nn.MSELoss()
# the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=5,
        factor=0.5,
        verbose=True
    )


# We are only keeping track of the loss values while training. running_loss
# keeps track of the batch-wise loss.
# Below we are calculating the loss for the outputs and the sharp_image. This
# is because we want to make the blurry images as similar to the sharp images
# as possible. We are calculating the loss for each epoch
# (train_loss). We are returning the train_loss value.

def fit(model, dataloader, epoch):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader),
                        total=int(len(train_data)/dataloader.batch_size)):
        blur_image = data[0]
        sharp_image = data[1]
        blur_image = blur_image.to(device)
        sharp_image = sharp_image.to(device)
        optimizer.zero_grad()
        outputs = model(blur_image)
        loss = criterion(outputs, sharp_image)
        # backpropagation
        loss.backward()
        # update the parameters
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss/len(dataloader.dataset)
    print(f"Train Loss: {train_loss:.5f}")

    return train_loss

# Below we are keeping track of the validation loss values. We do no need to
# backpropagate the gradients or update the parameters.
# We are saving one instance of the sharp image and the blurred image from the
# last batch. This will help compare the blurred, sharp and deblurred images.
# Then we are saving the last deblurred image from the last
# batch. This takes place every epoch. So, by the end of the training we will
# have 40 deblurred images (one image per epoch) in the saved_images folder.


def validate(model, dataloader, epoch):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader),
                            total=int(len(val_data)/dataloader.batch_size)):
            blur_image = data[0]
            sharp_image = data[1]
            blur_image = blur_image.to(device)
            sharp_image = sharp_image.to(device)
            outputs = model(blur_image)
            loss = criterion(outputs, sharp_image)
            running_loss += loss.item()

            if epoch == 0 and i == \
                    int((len(val_data)/dataloader.batch_size)-1):

                save_decoded_image(
                    sharp_image.cpu().data,
                    name=f"../outputs/saved_images/sharp{epoch}.jpg")
                save_decoded_image(
                    blur_image.cpu().data,
                    name=f"../outputs/saved_images/blur{epoch}.jpg")
            if i == int((len(val_data)/dataloader.batch_size)-1):
                save_decoded_image(
                    outputs.cpu().data,
                    name=f"../outputs/saved_images/val_deblurred{epoch}.jpg")

        val_loss = running_loss/len(dataloader.dataset)
        print(f"Val Loss: {val_loss:.5f}")

        return val_loss

# We are storing the train and validation loss values of each epoch in the
# train_loss and val_loss lists respectively. we are then applying the
# learning rate scheduler as well.

train_loss = []
val_loss = []
start = time.time()
for epoch in range(args['epochs']):
    print(f"Epoch {epoch+1} of {args['epochs']}")
    train_epoch_loss = fit(model, trainloader, epoch)
    val_epoch_loss = validate(model, valloader, epoch)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    scheduler.step(val_epoch_loss)
end = time.time()
print(f"Took {((end-start)/60):.3f} minutes to train")

# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('../outputs/loss.png')
plt.show()
# save the model to disk
print('Saving model...')
torch.save(model.state_dict(), '../outputs/model.pth')
