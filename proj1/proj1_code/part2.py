import torch


def my_imfilter(image, filter):
    """
    Apply a filter to an image. Return the filtered image.
    Args
    - image: Torch tensor of shape (m, n, c)
    - filter: Torch tensor of shape (k, j)
    Returns
    - filtered_image: Torch tensor of shape (m, n, c)
    HINTS:
    - You may not use any libraries that do the work for you. Using torch to work
     with matrices is fine and encouraged. Using OpenCV or similar to do the
     filtering for you is not allowed.
    - I encourage you to try implementing this naively first, just be aware that
     it may take a long time to run. You will need to get a function
     that takes a reasonable amount of time to run so that the TAs can verify
     your code works.
    - Useful functions: torch.nn.functional.pad
    """
    filtered_image = torch.zeros(image.shape).float()
    padded_image = torch.FloatTensor()
    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1
    kx = filter.shape[0]
    ky = filter.shape[1]
    pad_x = int (kx / 2)
    pad_y = int (ky / 2)
    padded_image = torch.nn.functional.pad(image, (0,0,pad_y,pad_y,pad_x,pad_x))
    filter = filter.float().flatten()
    
    for i in range(pad_x, padded_image.shape[0] - pad_x):
      for j in range(pad_y, padded_image.shape[1] - pad_y):
        r_win = padded_image[i-pad_x : i-pad_x + kx, j-pad_y : j-pad_y + ky, 0]
        g_win = padded_image[i-pad_x : i-pad_x + kx, j-pad_y : j-pad_y + ky, 1]
        b_win = padded_image[i-pad_x : i-pad_x + kx, j-pad_y : j-pad_y + ky, 2]

        filtered_image[i-pad_x,j - pad_y,0] = torch.dot(filter, r_win.flatten())
        filtered_image[i-pad_x,j - pad_y,1] = torch.dot(filter, g_win.flatten())
        filtered_image[i-pad_x,j - pad_y,2] = torch.dot(filter, b_win.flatten())

    return filtered_image

def create_hybrid_image(image1, image2, filter):
    """
    Takes two images and a low-pass filter and creates a hybrid image. Returns
    the low frequency content of image1, the high frequency content of image 2,
    and the hybrid image.

    Args
    - image1: Torch tensor of dim (m, n, c)
    - image2: Torch tensor of dim (m, n, c)
    - filter: Torch tensor of dim (x, y)
    Returns
    - low_frequencies: Torch tensor of shape (m, n, c)
    - high_frequencies: Torch tensor of shape (m, n, c)
    - hybrid_image: Torch tensor of shape (m, n, c)

    HINTS:
    - You will use your my_imfilter function in this function.
    - You can get just the high frequency content of an image by removing its low
      frequency content. Think about how to do this in mathematical terms.
    - Don't forget to make sure the pixel values of the hybrid image are between
      0 and 1. This is known as 'clipping' ('clamping' in torch).
    - If you want to use images with different dimensions, you should resize them
      in the notebook code.
    """

    hybrid_image = torch.Tensor()
    low_frequencies = torch.Tensor()
    high_frequencies = torch.Tensor()

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]
    assert filter.shape[0] <= image1.shape[0]
    assert filter.shape[1] <= image1.shape[1]
    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

    low_frequencies = my_imfilter(image1, filter)
    
    lf_image2 = my_imfilter(image2, filter) 
    high_frequencies = image2 - lf_image2

    hybrid_image = low_frequencies + high_frequencies
    hybrid_image = torch.clamp(hybrid_image,0.0, 1.0)
    return low_frequencies, high_frequencies, hybrid_image
