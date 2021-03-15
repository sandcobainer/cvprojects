import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from proj2_code.dl_utils import predict_labels
from proj2_code.image_loader import ImageLoader


def visualize(model: torch.nn.Module,
              split: str,
              data_transforms,
              data_base_path: str = '../data') -> None:
  loader = ImageLoader(data_base_path, split=split, transform=data_transforms)
  class_labels = loader.class_dict
  class_labels = {ele.lower(): class_labels[ele] for ele in class_labels}
  labels = {class_labels[ele]: ele for ele in class_labels}
  paths_and_labels = loader.load_imagepaths_with_labels(class_labels)
  selected = random.choices(paths_and_labels, k=4)
  fig, axs = plt.subplots(2, 2)
  for i in range(4):
    img = loader.load_img_from_path(selected[i][0])
    with torch.no_grad():
      outputs = model(data_transforms(img).unsqueeze(
          0).to(next(model.parameters()).device))
      predicted = predict_labels(outputs).item()
    axs[i//2, i % 2].imshow(img, cmap='gray')
    axs[i // 2, i % 2].set_title('Predicted:{}|Correct:{}'.format(
        labels[predicted], labels[selected[i][1]]))
    axs[i // 2, i % 2].axis('off')
  fig.tight_layout()
  plt.subplots_adjust(wspace=0.5)
  plt.show()
