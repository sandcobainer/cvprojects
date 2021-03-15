import torch


def my_1dfilter(signal: torch.FloatTensor,
                kernel: torch.FloatTensor) -> torch.FloatTensor:
    """Filters the signal by the kernel.

    output = signal * kernel where * denotes the cross-correlation function.
    Cross correlation is similar to the convolution operation with difference
    being that in cross-correlation we do not flip the sign of the kernel.

    Reference: 
    - https://mathworld.wolfram.com/Cross-Correlation.html
    - https://mathworld.wolfram.com/Convolution.html

    Note:
    1. The shape of the output should be the same as signal.
    2. You may use zero padding as required. Please do not use any other 
       padding scheme for this function.
    3. Take special care that your function performs the cross-correlation 
       operation as defined even on inputs which are asymmetric.

    Args:
        signal (torch.FloatTensor): input signal. Shape=(N,)
        kernel (torch.FloatTensor): kernel to filter with. Shape=(K,)

    Returns:
        torch.FloatTensor: filtered signal. Shape=(N,)
    """
    
    
    padding_size = int (kernel.shape[0]/2)
    padding = torch.zeros(padding_size)
    padded_signal = torch.cat([padding, signal, padding])
    N = signal.shape[0]
    K = kernel.shape[0]
    filtered_signal = torch.ones([N]).float()
    
    for i in range(padding_size, padded_signal.shape[0] - padding_size):
        filtered_signal[i-padding_size] = torch.dot(padded_signal[i-padding_size:i-padding_size+K],kernel)

    return filtered_signal
