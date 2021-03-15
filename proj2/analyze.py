'''
Contains helper function which will help evaluate the models
'''
from typing import List, Callable, Dict, Union, Tuple

import torch
import numpy as np
from proj2_code.image_loader import ImageLoader
from pycm import ConfusionMatrix


def evaluate(model: torch.nn.Module, image_loader: ImageLoader, is_cuda: bool,
             metrics_fn: List[Callable[..., np.array]],
             metrics_args: List[Dict] = None, model_dir: str = '') -> Tuple[Union[ConfusionMatrix, np.array], List[np.array]]:
  '''
    Returns the list of metrics scores that evaluate the models prediction

    Args:
    - model: the model to be evaluated
    - image_loader: the image dataset containing the images that are used to evaluate the model
    - is_cuda: boolean flag to suggest whether or not to use cuda.
    - metrics_fn: list of functions from sklearn.metrics
    - metrics_args: list of dictionary containing additional arguments to pass to each metrics functions
    - model_dir: (Optional) directory to the model checkpoint to be loaded
    Returns:
    - confusion_matrix: the confusion matrix. You don't need to compute it manually, instead, 
                        you can use the function from sklearn.metrics or pycm.
    - scores: list of metrics scores, which are the outputs of the functions given in metrics_fn

    HINT:
    e.g. if you want to use the metrics of accuracy score and balanced accuracy score to evaluate
    the SimpleNet model, you can use this function like this:

    from sklearn import accuracy_score, balanced_accuracy_score
    acc_score, balanced_acc_score = evaluate(simple_net, image_loader,
                                             [accuracy_score, balanced_accuracy_score],
                                             [{}, {'adjusted': True}],
                                             model_dir)
    Note that this is just an example usage, and please use other metrics instead of accuracy because it's
    already reported previously when we train and test the models.

    Note that the input data and the model should be on the same device. Therefore, if the model is on GPU,
    you need to put the input data to GPU as well. You can take advantage of the is_cuda argument.
  '''
  confusion_matrix = None
  scores = []

  y_true = []  # ground truth labels
  y_pred = []  # model predicted labels
  y_prob = []  # model output probabilities

  if not metrics_args:
    metrics_args = [{}] * len(metrics_fn)

  image_loader = torch.utils.data.DataLoader(image_loader, batch_size=1)
  if model_dir:
    checkpoint = torch.load(model_dir)
    model.load_state_dict(checkpoint['model_state_dict'])

  if is_cuda:
    model = model.to('cuda')

  model.eval()
  ############################################################################
  # Student code begin
  ############################################################################
  raise NotImplementedError('evaluate not implemented')

  ############################################################################
  # Student code end
  ############################################################################

  # you can replace y_pred with y_prob for metrics that takes in probabilities
  # instead of predicted labels
  for metrics, args in zip(metrics_fn, metrics_args):
    scores.append(metrics(y_true, y_pred, **args))

  return confusion_matrix, scores
