import copy

import torch

from proj2_code.image_loader import ImageLoader
from proj2_code.my_alexnet import MyAlexNet
from proj2_code.my_alexnet_quantized import MyAlexNetQuantized


def quantize_model(float_model: MyAlexNet,
                   train_loader: ImageLoader) -> MyAlexNetQuantized:
  '''
  Quantize the input model to int8 weights.

  Args:
  -   float_model: model with fp32 weights.
  -   train_loader: training dataset.
  Returns:
  -   quantized_model: equivalent model with int8 weights.
  '''

  # copy the weights from original model (still floats)
  quantized_model = MyAlexNetQuantized()
  quantized_model.cnn_layers = copy.deepcopy(float_model.cnn_layers)
  quantized_model.fc_layers = copy.deepcopy(float_model.fc_layers)

  quantized_model = quantized_model.to('cpu')

  quantized_model.eval()

  ##############################################################################
  # Student code begin
  ##############################################################################

  raise NotImplementedError('quantize_model() not implemented')

  ##############################################################################
  # Student code end
  ##############################################################################

  quantized_model.eval()

  return quantized_model
