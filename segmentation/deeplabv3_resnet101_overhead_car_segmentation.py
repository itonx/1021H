from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights

# Load segmentation model
model = models.segmentation.deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
model.eval()

# Load image
img = Image.open('./cars/car3.jpg').convert('RGB')
img_original = np.array(img)

# Configure the model
preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
input_batch = preprocess(img).unsqueeze(0)

if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)['out']

# Convert to probabilities (value between 0 and 1) for cars
car_index = 7
car_prob = output[0][car_index].sigmoid()

# Select pixels where probability of being a car is > 0.9
mask_1d = (car_prob > 0.9).cpu().numpy().astype(np.uint8)

# Turn 1d array into a 3d array
mask_2d = mask_1d.reshape(mask_1d.shape)
mask_3d = np.zeros((mask_1d.shape[0],mask_1d.shape[1], 3), dtype=np.uint8)
mask_3d[mask_2d == 1, 1] = 255  # Set green channel for car pixels

# Add mask to the original image
mask_nonzero = np.any(mask_3d != [0, 0, 0], axis=-1)  # shape: (H, W)
img_result = img_original.copy()
img_result[mask_nonzero] = mask_3d[mask_nonzero]

# Print results
plt.figure(figsize=(10, 5), num='Semantic Segmentation')
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Car Segmentation")
plt.imshow(img_result)
plt.axis('off')

plt.show()