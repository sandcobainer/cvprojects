import numpy as np
import torch

from proj2_code.data_transforms import get_fundamental_transforms
from proj2_code.image_loader import ImageLoader


def test_dataset_length():
  try:
    train_image_loader = ImageLoader(root_dir='proj2_code/proj2_unit_tests/small_data/', split='train', transform=get_fundamental_transforms(
        inp_size=(64, 64), pixel_mean=np.array([0.01]), pixel_std=np.array([1.001])
    ))

    test_image_loader = ImageLoader(root_dir='proj2_code/proj2_unit_tests/small_data/', split='test', transform=get_fundamental_transforms(
        inp_size=(64, 64), pixel_mean=np.array([0.01]), pixel_std=np.array([1.001])
    ))
  except:
    train_image_loader = ImageLoader(root_dir='../proj2_code/proj2_unit_tests/small_data/', split='train', transform=get_fundamental_transforms(
        inp_size=(64, 64), pixel_mean=np.array([0.01]), pixel_std=np.array([1.001])
    ))

    test_image_loader = ImageLoader(root_dir='../proj2_code/proj2_unit_tests/small_data/', split='test', transform=get_fundamental_transforms(
        inp_size=(64, 64), pixel_mean=np.array([0.01]), pixel_std=np.array([1.001])
    ))
    
  
  assert len(train_image_loader) == 30
  assert len(test_image_loader) == 39


def test_unique_vals():
  try:
    train_image_loader = ImageLoader(root_dir='proj2_code/proj2_unit_tests/small_data/', split='train', transform=get_fundamental_transforms(
        inp_size=(64, 64), pixel_mean=np.array([0.01]), pixel_std=np.array([1.001])
    ))
  except:
    train_image_loader = ImageLoader(root_dir='../proj2_code/proj2_unit_tests/small_data/', split='train', transform=get_fundamental_transforms(
        inp_size=(64, 64), pixel_mean=np.array([0.01]), pixel_std=np.array([1.001])
    ))

  item1 = train_image_loader.__getitem__(2)
  item2 = train_image_loader.__getitem__(5)

  assert not torch.allclose(item1[0], item2[0])


def test_class_values():
  try:
    test_image_loader = ImageLoader(root_dir='proj2_code/proj2_unit_tests/small_data/', split='test', transform=get_fundamental_transforms(
        inp_size=(64, 64), pixel_mean=np.array([0.01]), pixel_std=np.array([1.001])
    ))
  except:
    test_image_loader = ImageLoader(root_dir='../proj2_code/proj2_unit_tests/small_data/', split='test', transform=get_fundamental_transforms(
        inp_size=(64, 64), pixel_mean=np.array([0.01]), pixel_std=np.array([1.001])
    ))

  class_labels = test_image_loader.class_dict
  class_labels = {ele.lower():class_labels[ele] for ele in class_labels}

  expected_vals = {
      'opencountry': 0,
      'industrial': 1,
      'office': 2,
      'insidecity': 3,
      'kitchen': 4,
      'tallbuilding': 5,
      'mountain': 6,
      'forest': 7,
      'store': 8,
      'livingroom': 9,
      'street': 10,
      'bedroom': 11,
      'coast': 12,
      'suburb': 13,
      'highway': 14
  }

  assert len(class_labels) == 15
  assert set(class_labels.keys()) == set(expected_vals.keys())
  assert set(class_labels.values()) == set(expected_vals.values())


def test_load_img_from_path():
  
  try:
    test_image_loader = ImageLoader(root_dir='proj2_code/proj2_unit_tests/small_data', split='train', transform=get_fundamental_transforms(
        inp_size=(64, 64), pixel_mean=np.array([0.01]), pixel_std=np.array([1.001])
    ))
    im_path = 'proj2_code/proj2_unit_tests/small_data/train/bedroom/image_0003.jpg'
  except:
    test_image_loader = ImageLoader(root_dir='../proj2_code/proj2_unit_tests/small_data', split='train', transform=get_fundamental_transforms(
        inp_size=(64, 64), pixel_mean=np.array([0.01]), pixel_std=np.array([1.001])
    ))
    im_path = '../proj2_code/proj2_unit_tests/small_data/train/bedroom/image_0003.jpg'


  im_np = np.asarray(test_image_loader.load_img_from_path(im_path))

  try:
    expected_data = np.loadtxt('proj2_code/proj2_unit_tests/test_data/sample_inp.txt')
  except:
    expected_data = np.loadtxt('../proj2_code/proj2_unit_tests/test_data/sample_inp.txt')

  assert np.allclose(expected_data, im_np)


if __name__ == '__main__':
  test_load_img_from_path()
