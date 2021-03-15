'''
Utilities to be used along with the deep model
'''

import torch


def predict_labels(model_output: torch.tensor) -> torch.tensor:
  '''
  Predicts the labels from the output of the model.

  Args:
  -   model_output: the model output [Dim: (N, 15)]
  Returns:
  -   predicted_labels: the output labels [Dim: (N,)]
  '''
  predicted_labels = torch.zeros(model_output.shape[0])
  for i in range(model_output.shape[0]):
    predicted_labels[i] = torch.argmax(model_output[i,:])

  return predicted_labels


def compute_loss(model: torch.nn.Module,
                 model_output: torch.tensor,
                 target_labels: torch.tensor,
                 is_normalize: bool = True) -> torch.tensor:
  '''
  Computes the loss between the model output and the target labels

  Note: we have initialized the loss_criterion in the model with the sum
  reduction.

  Args:
  -   model: model (which inherits from nn.Module), and contains loss_criterion
  -   model_output: the raw scores output by the net [Dim: (N, 15)]
  -   target_labels: the ground truth class labels [Dim: (N, )]
  -   is_normalize: bool flag indicating that loss should be divided by the
                    batch size
  Returns:
  -   the loss value
  '''
  loss = None
  if(model.loss_criterion.__class__.__name__ == 'CrossEntropyLoss'):
    loss = model.loss_criterion(model_output, target_labels)

  elif (model.loss_criterion.__class__.__name__ == 'KLDivLoss'):
    model_log = torch.log(torch.nn.functional.softmax(model_output, dim=0))
    target_log = torch.zeros(model_log.shape).cuda()

    i=0
    for l in target_labels:
      target_log[i, l] = 1.
      i+=1
    loss = model.loss_criterion(model_log, target_log.float())
  
  return loss
