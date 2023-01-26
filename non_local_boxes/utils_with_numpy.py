import numpy as np

#
#   WIRINGS
#

def W_BS09(n):  # n is the number of columns
    W = [0, 0, 1, 1,              # f_1(x, a_2) = x
            0, 0, 1, 1,              # g_1(y, b_2) = y
            0, 0, 0, 1,              # f_2(x, a_1) = a_1*x
            0, 0, 0, 1,              # g_2(y, b_1) = b_1*y
            0, 1, 1, 0, 0, 1, 1, 0,  # f_3(x, a_1, a_2) = a_1 + a_2 mod 2
            0, 1, 1, 0, 0, 1, 1, 0   # g_3(y, b_1, b_2) = b_1 + b_2 mod 2
            ]
    return np.tensordot(np.ones(n), W, axes=0).T


def random_wiring(n):  # n is the number of columns
    return np.random.rand(32, n)

def random_extremal_wiring(n):  # n is the number of columns
    return np.random.randint(2, size=(32,n))




#
#   Link between MATRICES and TENSORS
#

def matrix_to_tensor(Matrix):
    return np.transpose(np.reshape(Matrix, (2,2,2,2)), (2,3,0,1))

def tensor_to_matrix(Tensor):
    return np.reshape(np.transpose(Tensor, (2,3,0,1)), (4,4))




#
#   BOXES
#

def P_L(mu, nu, sigma, tau):
    new_box = np.zeros((4,4))
    
    for a in range(2):
        for b in range(2):
            for x in range(2):
                for y in range(2):
                    if a==(mu*x+nu)%2 and b==(sigma*y+tau)%2:
                        new_box[2*x+y, 2*a+b] = 1
                        
    return new_box


def P_NL(mu, nu, sigma):
    new_box = np.zeros((4,4))
    
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
I = 0.25*np.ones((4,4))
SRbar = 2*I-SR


def corNLB(p):
    return p*PR + (1-p)*SR

                





#
#   CHSH
#

CHSH = np.zeros((4,4))

for a in range(2):
    for b in range(2):
        for x in range(2):
            for y in range(2):
                if (a+b)%2 == x*y:
                    CHSH[2*x+y, 2*a+b]=0.25

CHSH_flat = matrix_to_tensor( CHSH ).flatten()