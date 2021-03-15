import torch
import torch.nn as nn
from torchvision.models import alexnet


class MyAlexNet(nn.Module):
  def __init__(self):
    '''
    Init function to define the layers and loss function

    Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention to understand what it means

    Note: Do not forget to freeze the layers of alexnet except the last one. Otherwise the training will take a long time. To freeze a layer, set the
    weights and biases of a layer to not require gradients.

    Note: Map elements of alexnet to self.cnn_layers and self.fc_layers.

    Note: Remove the last linear layer in Alexnet and add your own layer to 
    perform 15 class classification.

    Note: Download pretrained alexnet using pytorch's API (Hint: see the import statements)
    '''
    super().__init__()
    alexmodel = alexnet(True, True)

    # freeze alexmodel
    for param in alexmodel.parameters():
      param.requires_grad = False

    # map cnn_layers and flatten
    self.cnn_layers = nn.Sequential( 
      alexmodel.features,
      nn.Flatten()
    )

    # map fc_layers and replace final layer
    self.fc_layers = alexmodel.classifier
    self.fc_layers[-1] = nn.Linear(4096,15)
    self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')

  def forward(self, x: torch.tensor) -> torch.tensor:
    '''
    Perform the forward pass with the net

    Note: do not perform soft-max or convert to probabilities in this function

    Args:
    -   x: the input image [Dim: (N,C,H,W)]
    Returns:
    -   y: the output (raw scores) of the net [Dim: (N,15)]
    '''
    model_output = None
    x = x.repeat(1, 3, 1, 1)  # as AlexNet accepts color images
    
    cnn_output = self.cnn_layers(x)
    model_output = self.fc_layers(cnn_output)

    return model_output
