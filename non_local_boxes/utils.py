#import numpy as np
import torch

#
#   WIRINGS
#

def W_BS09(n):  # n is the number of columns
    W = torch.Tensor([0, 0, 1, 1,              # f_1(x, a_2) = x
            0, 0, 1, 1,              # g_1(y, b_2) = y
            0, 0, 0, 1,              # f_2(x, a_1) = a_1*x
            0, 0, 0, 1,              # g_2(y, b_1) = b_1*y
            0, 1, 1, 0, 0, 1, 1, 0,  # f_3(x, a_1, a_2) = a_1 + a_2 mod 2
            0, 1, 1, 0, 0, 1, 1, 0   # g_3(y, b_1, b_2) = b_1 + b_2 mod 2
            ])
    return torch.reshape(torch.kron(torch.ones(n), W), (32, -1))


def random_wiring(n):  # n is the number of columns
    return torch.rand((32, n))

def random_extremal_wiring(n):  # n is the number of columns
    return torch.randint(2, (32,n))




#
#   Link between MATRICES and TENSORS
#

def matrix_to_tensor(Matrix):
    T = torch.reshape(Matrix, (2,2,2,2))
    T = torch.transpose(T, 0, 2)
    T = torch.transpose(T, 1, 3)
    return T

def tensor_to_matrix(Tensor):
    M = torch.transpose(Tensor, 0, 2)
    M = torch.transpose(M, 1, 3)
    return torch.reshape(M, (4,4))




#
#   BOXES
#

def P_L(mu, nu, sigma, tau):
    new_box = torch.zeros((4,4))
    for a in range(2):
        for b in range(2):
            for x in range(2):
                for y in range(2):
                    if a==(mu*x+nu)%2 and b==(sigma*y+tau)%2:
                        new_box[2*x+y, 2*a+b] = 1         
    return new_box


def P_NL(mu, nu, sigma):
    new_box = torch.zeros((4,4))
    for a in range(2):
        for b in range(2):
            for x in range(2):
                for y in range(2):
                    if (a+b)%2==(x*y + mu*x + nu*y + sigma)%2:
                        new_box[2*x+y, 2*a+b] = 0.5               
    return new_box



P_0 = P_L(0,0,0,0)
P_1 = P_L(0,1,0,1)
SR = (P_0 + P_1)/2
PR = P_NL(0,0,0)
PRbar = P_NL(0,0,1)
I = 0.25*torch.ones((4,4))
SRbar = 2*I-SR


def corNLB(p):
    return p*PR + (1-p)*SR

                





#
#   CHSH
#

CHSH = torch.zeros((4,4))

for a in range(2):
    for b in range(2):
        for x in range(2):
            for y in range(2):
                if (a+b)%2 == x*y:
                    CHSH[2*x+y, 2*a+b]=0.25

CHSH_flat = matrix_to_tensor( CHSH ).flatten()