import cv2
import numpy as np
import PIL
import torch
import torchvision.transforms as transforms


def im2single(im: np.ndarray) -> np.ndarray:
    """
      Args:
      - img: uint8 array of shape (m,n,c) or (m,n) and in range [0,255]
      Returns:
      - im: float or double array of identical shape and in range [0,1]
    """
    im = im.astype(np.float32) / 255
    return im


def single2im(im: np.ndarray) -> np.ndarray:
    im *= 255
    im = im.astype(np.uint8)
    return im


# def load_image(path):
#     return im2single(cv2.imread(path))[:, :, ::-1]


def load_image(path: str) -> torch.Tensor:
    """
      Args:
      - path: string representing a file path to an image
      Returns:
      - float or double array of shape (m,n,c) or (m,n) and in range [0,1],
        representing an RGB image
    """
    img = PIL.Image.open(path)
    img = np.asarray(img)
    float_img_rgb = im2single(img)
    torch_img_rgb = torch.from_numpy(float_img_rgb)
    return torch_img_rgb


def load_image_gray(path: str) -> np.ndarray:
    img = load_image(path)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def rgb2gray(img: np.ndarray) -> np.ndarray:
    """ Use the coefficients used in OpenCV, found here:
            https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html

        Args:
        -   Numpy array of shape (M,N,3) representing RGB image

        Returns:
        -   Numpy array of shape (M,N) representing grayscale image
    """
    # Grayscale coefficients
    c = [0.299, 0.587, 0.114]
    return img[:, :, 0] * c[0] + img[:, :, 1] * c[1] + img[:, :, 2] * c[2]


def load_image_gray_tensor(path: str) -> torch.FloatTensor:
    """
    Args:
    - path: string representing a file path to an image

    Returns:
    - float tensor of shape (m,n) and in range [0,1],
      representing a image in gray scale
    """

    gray_img = load_image_gray(path)
    tensor_type = torch.FloatTensor
    torch.set_default_tensor_type(tensor_type)
    to_tensor = transforms.ToTensor()
    gray_img_tensor = to_tensor(gray_img).unsqueeze(0)

    return gray_img_tensor


def arrayToTensor(img: np.ndarray) -> torch.FloatTensor:
    tensor_type = torch.FloatTensor
    torch.set_default_tensor_type(tensor_type)
    to_tensor = transforms.ToTensor()
    gray_img_tensor = to_tensor(img).unsqueeze(0)

    return gray_img_tensor
