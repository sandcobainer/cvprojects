from proj2_code.proj2_unit_tests.test_models import *
from proj2_code.simple_net import SimpleNet


def test_simple_net():
  '''
  Tests the SimpleNet contains desired number of corresponding layers
  '''
  this_simple_net = SimpleNet()

  _, output_dim, counter, *_ = extract_model_layers(this_simple_net)

  assert counter['Conv2d'] >= 2
  assert counter['Linear'] >= 2
  assert counter['ReLU'] >= 2
  assert output_dim == 15
