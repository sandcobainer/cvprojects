import torch
import torch.nn as nn


class SimpleNetDropout(nn.Module):
  def __init__(self):
    '''
    Init function to define the layers and loss function

    Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention to understand what it means
    '''
    super().__init__()

    self.cnn_layers = nn.Sequential()
    self.fc_layers = nn.Sequential()
    self.loss_criterion = None

    self.cnn_layers = nn.Sequential(
      nn.Conv2d(1, 10, 5),
      nn.MaxPool2d(3, stride=3),
      nn.ReLU(),
      
      nn.Conv2d(10, 20, 5),
      nn.MaxPool2d(3, stride=3),
      nn.ReLU()
      )  # conv2d and supporting layers here

    self.fc_layers = nn.Sequential(
      nn.Flatten(),
      nn.Linear(500,100),
      nn.Dropout(),
      nn.Linear(100,15)
    )  # linear and supporting layers here
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
    cnn_output = self.cnn_layers(x)
    model_output = self.fc_layers(cnn_output)
    return model_output
