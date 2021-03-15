import torch
from torch.quantization import DeQuantStub, QuantStub

from proj2_code.my_alexnet import MyAlexNet


class MyAlexNetQuantized(MyAlexNet):
  def __init__(self):
    '''
    Init function to define the layers and loss function.
    '''
    super().__init__()

    self.quant = QuantStub()
    self.dequant = DeQuantStub()

  def forward(self, x: torch.tensor) -> torch.tensor:
    '''
    Perform the forward pass with the net.

    Hints:
    1. Use the self.quant() and self.dequant() layer on input/output.

    Args:
    -   x: the input image [Dim: (N,C,H,W)]
    Returns:
    -   y: the output (raw scores) of the net [Dim: (N,15)]
    '''
    model_output = None

    x = x.repeat(1, 3, 1, 1)  # as AlexNet accepts color images
    ############################################################################
    # Student code begin
    ############################################################################

    raise NotImplementedError(
        'forward() for MyAlexNetQuantized is not implemented')

    ############################################################################
    # Student code end
    ############################################################################

    return model_output
