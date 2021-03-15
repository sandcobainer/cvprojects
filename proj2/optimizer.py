'''
This class contains helper functions which will help get the optimizer
'''

import torch


def get_optimizer(model: torch.nn.Module,
                  config: dict) -> torch.optim.Optimizer:
  '''
  Returns the optimizer initializer according to the config on the model.

  Note: config has a minimum of three entries. Feel free to add more entries if you want.
  But do not change the name of the three existing entries

  Args:
  - model: the model to optimize for
  - config: a dictionary containing parameters for the config
  Returns:
  - optimizer: the optimizer
  '''

  optimizer = None

  optimizer_type = config["optimizer_type"]
  learning_rate = config["lr"]
  weight_decay = config["weight_decay"]

  if (optimizer_type == 'sgd'):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate 
                ,weight_decay=weight_decay)

  return optimizer
