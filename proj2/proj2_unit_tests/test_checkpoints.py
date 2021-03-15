'''
Test the presence of checkpoint files
'''
import os


def test_simple_net_checkpoint():
  assert os.path.exists(os.path.join('model_checkpoints', 'simple_net', 'checkpoint.pt')) or os.path.exists(os.path.join('..', 'model_checkpoints', 'simple_net', 'checkpoint.pt'))

def test_simple_net_dropout_checkpoint():
  assert os.path.exists(os.path.join('model_checkpoints', 'simple_net_dropout', 'checkpoint.pt')) or os.path.exists(os.path.join('..', 'model_checkpoints', 'simple_net_dropout', 'checkpoint.pt'))  

def test_alexnet_checkpoint():
  assert os.path.exists(os.path.join('model_checkpoints', 'alexnet', 'checkpoint.pt')) or os.path.exists(os.path.join('..', 'model_checkpoints', 'alexnet', 'checkpoint.pt'))
