import torch

from proj1_code.dft import DFT_matrix, my_dft

def test_dft_matrix():
    """
    Test a 4*4 dft matrix
    """

    dft_m_student = DFT_matrix(4)

    if(dft_m_student.shape[0]!=4):
        print("The dimension of input matrix for this file is 4")
        return False
    
    correct_real = torch.Tensor([[1,1,1,1],[1,0,-1,0],[1,-1,1,-1],[1,0,-1,0]])/4
    correct_imag = torch.Tensor([[0,0,0,0],[0,-1,0,1],[0,0,0,0],[0,1,0,-1]])/4
    
    if(torch.mean(dft_m_student[:,:,0]-correct_real)<0.001 and torch.mean(dft_m_student[:,:,1]-correct_imag)<0.001):
        print('Success! The DFT matrix for dimension 4 is correct!')
        return True
    else:
        print('DFT Matrix is not correct, please double check your implementation')


def test_dft():
    """
    Test DFT for A matrix
    """
    
    A = torch.Tensor([[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0]])
    dft_student = my_dft(A)
    
    if(dft_student.shape[0]!=4):
        print("The dimension of input matrix for this file is 4")
        return False
    
    correct_real = torch.Tensor([[1,-1,1,-1],[0,0,0,0],[0,0,0,0],[0,0,0,0]])/4
    correct_imag = torch.zeros((4,4))
    
    if(torch.mean(dft_student[:,:,0]-correct_real)<0.001 and torch.mean(dft_student[:,:,1]-correct_imag)<0.001):
        print('Success! The DFT matrix for A is correct!')
        return True
    else:
        print('DFT Matrix is not correct, please double check your implementation')