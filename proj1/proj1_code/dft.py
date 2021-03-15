import torch


def DFT_matrix(N):
    """
    Takes the square matrix dimension as input, generate the DFT matrix correspondingly
    Args
    - N: the DFT matrix dimension
    Returns
    - U: the generated DFT matrix (torch.Tensor) of size (N,N,2);
    the real part is represented by U[:,:,0], and the complex part is represented by U[:,:,1]
    """
    U = torch.Tensor()

    torch.pi = torch.acos(torch.zeros(1)).item() * \
        2  # which is 3.1415927410125732
        
    #############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################

    raise NotImplementedError('`DFT_matrix` function in '
                              + '`dft.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE
    ############################################################################

    return U


def Compl_mul_Real(m1, m2):
    """
    Takes the one complex tensor matrix and a real matrix, and do matrix multiplication
    Args
    - m1: the Tensor matrix (m,n,2) which represents complex number;
    E.g., the real part is t1[:,:,0], the imaginary part is t1[:,:,1]
    - m2: the real matrix (m,n)
    Returns
    - U: matrix multiplication result in the same form as input m1
    """
    real1 = m1[:, :, 0]
    imag1 = m1[:, :, 1]
    real2 = m2
    imag2 = torch.zeros(real2.shape)
    return torch.stack([torch.matmul(real1, real2) - torch.matmul(imag1, imag2),
                        torch.matmul(real1, imag2) + torch.matmul(imag1, real2)], dim=-1)


def Compl_mul_Compl(m1, m2):
    """
    Takes the two complex tensor matrix and do matrix multiplication
    Args
    - t1, t2: the Tensor matrix (m,n,2) which represents complex number;
    E.g., the real part is t1[:,:,0], the imaginary part is t1[:,:,1]
    Returns
    - U: matrix multiplication result in the same form as input
    """
    real1 = m1[:, :, 0]
    imag1 = m1[:, :, 1]
    real2 = m2[:, :, 0]
    imag2 = m2[:, :, 1]
    return torch.stack([torch.matmul(real1, real2) - torch.matmul(imag1, imag2),
                        torch.matmul(real1, imag2) + torch.matmul(imag1, real2)], dim=-1)


def my_dft(img):
    """
    Takes a square image as input, performs Discrete Fourier Transform for the image matrix
    This function is expected to behave similar as torch.rfft(x,2,onesided=False)
    Args
    - img: a 2D grayscale image (torch.Tensor) whose width equals height, size: (N,N)
    Returns
    - dft: the DFT results of img; the size should be (N,N,2),
    where the real part is dft[:,:,0], while the imag part is dft[:,:,1]
    - hints: we provide two function to do complex matrix multiplication:
    Compl_mul_Real and Compl_mul_Compl
    """
    dft = torch.Tensor()

    assert img.shape[0] == img.shape[1], "Input image should be a square matrix"

    #############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################


    raise NotImplementedError('`my_dft` function in '
                              + '`dft.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE
    ############################################################################

    return dft


def dft_filter(img):
    """
    Takes an square image as input, performs a low-pass filter and return the filtered image
    Args
    - img: a 2D grayscale image whose width equals height, size: (N,N)
    Returns
    - img_back: the filtered image whose size is also (N,N)
    Hints
    - You will need your implemented DFT filter for this function
    - We don't care how much frequency you want to retain, if only it returns reasonable results
    - Since you already implemented DFT part, you're allowed to use the torch.ifft in this part for convenience, though not necessary
    """

    img_back = torch.Tensor()

    #############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################

    raise NotImplementedError('`dft_filter` function in '
                              + '`dft.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE
    ############################################################################

    return img_back
