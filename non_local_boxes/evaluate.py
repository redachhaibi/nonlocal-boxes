#import numpy as np
import torch
import non_local_boxes.utils


nb_columns = int(1e0)





#
#   A(W)
#

# A1 is a 2x4x4x32-tensor
A1 = torch.zeros( (2, 4, 4, 32) )
for x in range(2):
    for j in range(4):
        
        sign = 1
        if j >= 2:
            sign=-1
            
        A1[x, 0, j, 0] = sign*(x-1)
        A1[x, 1, j, 0] = sign*(x-1)
        A1[x, 2, j, 1] = sign*(x-1)
        A1[x, 3, j, 1] = sign*(x-1)
        
        A1[x, 0, j, 2] = sign*(-x)
        A1[x, 1, j, 2] = sign*(-x)
        A1[x, 2, j, 3] = sign*(-x)
        A1[x, 3, j, 3] = sign*(-x)


# A2 is a 2x4x4xn tensor
A2 = torch.zeros( (2, 4, 4, nb_columns) )
for x in range(2):
    for i in range(4):
        for k in range(4):
            for alpha in range(nb_columns):
                if k<=1:
                    A2[x, i, k]=1


# A3 is a 2x4x4x32-tensor
A3 = torch.zeros( (2, 4, 4, 32) )
for y in range(2):
    for j in range(4):
        
        sign = 1
        if j==1 or j==3:
            sign=-1
            
        A3[y, 0, j, 0 +4] = sign*(y-1)
        A3[y, 2, j, 0 +4] = sign*(y-1)
        A3[y, 1, j, 1 +4] = sign*(y-1)
        A3[y, 3, j, 1 +4] = sign*(y-1)
        
        A3[y, 0, j, 2 +4] = sign*(-y)
        A3[y, 2, j, 2 +4] = sign*(-y)
        A3[y, 1, j, 3 +4] = sign*(-y)
        A3[y, 3, j, 3 +4] = sign*(-y)


# A4 is a 2x4x4xn-tensor
A4 = torch.zeros( (2, 4, 4, nb_columns) )
for y in range(2):
    for i in range(4):
        for k in range(4):
            for alpha in range(nb_columns):
                if k==0 or k==2:
                    A4[y, i, k]=1


def A(W):    # W is a 32xn matrix
    T1 = torch.tensordot(A1, W, dims=1) + A2
    #T2 = torch.reshape(torch.kron(torch.ones((2)), T1), (2,2,4,4,-1))
    T2 = T1.repeat(2, 1, 1, 1, 1)
    T3 = torch.transpose(T2, 0, 1)
    S1 = torch.tensordot(A3, W, dims=1) + A4
    #S2 = torch.reshape(torch.kron(torch.ones((2)), S1), (2,2,4,4,-1))
    S2 = S1.repeat(2, 1, 1, 1, 1)
    R = torch.transpose(T3 * S2, 4, 2)
    return torch.transpose(R, 4, 3)
    # the output is a 2x2xnx4x4 tensor





#
#   B(W)
#

# B1 is a 2x4x4x32-tensor
B1 = torch.zeros( (2, 4, 4, 32) )
for x in range(2):
    for l in range(4):
        
        sign = 1
        if l>=2:
            sign=-1
            
        B1[x, 0, l, 0 +8] = sign * (x-1)
        B1[x, 1, l, 0 +8] = sign * (x-1)
        B1[x, 2, l, 1 +8] = sign * (x-1)
        B1[x, 3, l, 1 +8] = sign * (x-1)
        
        B1[x, 0, l, 2 +8] = sign * (-x)
        B1[x, 1, l, 2 +8] = sign * (-x)
        B1[x, 2, l, 3 +8] = sign * (-x)
        B1[x, 3, l, 3 +8] = sign * (-x)


# B2 is equal to A2
B2 = A2


# B3 is a 2x4x4x32-tensor
B3 = torch.zeros( (2, 4, 4, 32) )
for y in range(2):
    for l in range(4):
        
        sign=1
        if l==1 or l==3:
            sign=-1
        
        B3[y, 0, l, 0 +12] = sign * (y-1)
        B3[y, 2, l, 0 +12] = sign * (y-1)
        B3[y, 1, l, 1 +12] = sign * (y-1)
        B3[y, 3, l, 1 +12] = sign * (y-1)
        
        B3[y, 0, l, 2 +12] = sign * (-y)
        B3[y, 2, l, 2 +12] = sign * (-y)
        B3[y, 1, l, 3 +12] = sign * (-y)
        B3[y, 3, l, 3 +12] = sign * (-y)

# B4 is equal to A4
B4 = A4

def B(W):    # W is a 32xn matrix
    T1 = torch.tensordot(B1, W, dims=1) + B2
    #T2 = torch.reshape(torch.kron(torch.ones((2)), T1), (2,2,4,4,-1))
    T2 = T1.repeat(2, 1, 1, 1, 1)
    T3 = torch.transpose(T2, 0, 1)
    S1 = torch.tensordot(B3, W, dims=1) + B4
    #S2 = torch.reshape(torch.kron(torch.ones((2)), S1), (2,2,4,4,-1))
    S2 = S1.repeat(2, 1, 1, 1, 1)
    R = torch.transpose(T3 * S2, 4, 2)
    return torch.transpose(R, 4, 3)
    # the output is a 2x2xnx4x4 tensor





#
#   C(W)
#

# C1 is a 2x2x4x4x32-tensor
C1 = torch.zeros( (2, 2, 4, 4, 32) )
for a in range(2):
    for x in range(2):
        for j in range(4):
            if j<=1:
                C1[a, x, 0, j, 0 +16] = -(1-x) * (-1)**a
                C1[a, x, 1, j, 0 +16] = -(1-x) * (-1)**a
                C1[a, x, 2, j, 1 +16] = -(1-x) * (-1)**a
                C1[a, x, 3, j, 1 +16] = -(1-x) * (-1)**a
                
                C1[a, x, 0, j, 4 +16] = -(x) * (-1)**a
                C1[a, x, 1, j, 4 +16] = -(x) * (-1)**a
                C1[a, x, 2, j, 5 +16] = -(x) * (-1)**a
                C1[a, x, 3, j, 5 +16] = -(x) * (-1)**a
            
            if j>=2:
                C1[a, x, 0, j, 0 +18] = -(1-x) * (-1)**a
                C1[a, x, 1, j, 0 +18] = -(1-x) * (-1)**a
                C1[a, x, 2, j, 1 +18] = -(1-x) * (-1)**a
                C1[a, x, 3, j, 1 +18] = -(1-x) * (-1)**a
                
                C1[a, x, 0, j, 4 +18] = -(x) * (-1)**a
                C1[a, x, 1, j, 4 +18] = -(x) * (-1)**a
                C1[a, x, 2, j, 5 +18] = -(x) * (-1)**a
                C1[a, x, 3, j, 5 +18] = -(x) * (-1)**a


# C2 is a 2x2x4x4xnb_columns-tensor
C2 = torch.zeros( (2, 2, 4, 4, nb_columns) )
for x in range(2):
    for i in range(4):
        for j in range(4):
            for alpha in range(nb_columns):
                C2[0, x, i, j]=1


def C(W):    # W is a 32xn matrix
    T1 = torch.tensordot(C1, W, dims=1) + C2
    # T2 = torch.reshape(torch.kron( torch.ones((2,2)), T1), (2,2,2,2,4,4,-1))
    T2 = T1.repeat(2, 2, 1, 1, 1, 1, 1)
    R = torch.transpose(T2, 0, 1)
    R = torch.transpose(R, 0, 3)
    R = torch.transpose(R, 0, 2)
    R = torch.transpose(R, 6, 4)
    R = torch.transpose(R, 6, 5)
    return R
    # the output is a 2x2x2x2xnx4x4 tensor





#
#   D(W)
#

# D1 is a 2x2x4x4x32-tensor
D1 = torch.zeros( (2, 2, 4, 4, 32) )
for b in range(2):
    for y in range(2):
        for j in range(4):
            if j==0 or j==2:
                D1[b, y, 0, j, 0 + 24] = -(1-y) * (-1)**b
                D1[b, y, 2, j, 0 + 24] = -(1-y) * (-1)**b
                D1[b, y, 1, j, 1 + 24] = -(1-y) * (-1)**b
                D1[b, y, 3, j, 1 + 24] = -(1-y) * (-1)**b
                
                D1[b, y, 0, j, 4 + 24] = -(y) * (-1)**b
                D1[b, y, 2, j, 4 + 24] = -(y) * (-1)**b
                D1[b, y, 1, j, 5 + 24] = -(y) * (-1)**b
                D1[b, y, 3, j, 5 + 24] = -(y) * (-1)**b
            
            if j==1 or j==3:
                D1[b, y, 0, j, 0 + 26] = -(1-y) * (-1)**b
                D1[b, y, 2, j, 0 + 26] = -(1-y) * (-1)**b
                D1[b, y, 1, j, 1 + 26] = -(1-y) * (-1)**b
                D1[b, y, 3, j, 1 + 26] = -(1-y) * (-1)**b
                
                D1[b, y, 0, j, 4 + 26] = -(y) * (-1)**b
                D1[b, y, 2, j, 4 + 26] = -(y) * (-1)**b
                D1[b, y, 1, j, 5 + 26] = -(y) * (-1)**b
                D1[b, y, 3, j, 5 + 26] = -(y) * (-1)**b

# D2 is a 2x2x4x4xnb_columns-tensor
D2 = torch.zeros( (2, 2, 4, 4, nb_columns) )
for y in range(2):
    for i in range(4):
        for j in range(4):
            for alpha in range(nb_columns):
                D2[0, y, i, j] = 1

def D(W):    # W is a 32xn matrix
    T1 = torch.tensordot(D1, W, dims=1) + D2
    # T2 = torch.reshape(torch.kron( torch.ones((2,2)), T1), (2,2,2,2,4,4,-1))
    T2 = T1.repeat(2, 2, 1, 1, 1, 1, 1)
    R = torch.transpose(T2, 1, 2)
    R = torch.transpose(R, 6, 4)
    R = torch.transpose(R, 6, 5)
    return R
    # the output is a 2x2x2x2xnx4x4 tensor





#
#   R(W, P, Q) := P x_W Q
#

def R(W, P, Q):     # W is a 32xn matrix, P and Q are 4x4 matrices
    T1 = torch.tensordot(A(W), P, dims=1)  # green term
    T2 = torch.transpose(torch.tensordot(B(W), Q, dims=1), 4, 3)  # blue term
    #T3 = torch.reshape(torch.kron(torch.ones((2,2)), T1*T2), (2,2,2,2,-1,4,4))
    T3 = (T1*T2).repeat(2,2,1,1,1,1,1)
    T4 = T3 * C(W) * D(W)  # the big bracket
    return torch.tensordot(T4, torch.ones((4, 4)), dims=2)
    #T5 = torch.tensordot(T4, torch.ones((4)), dims=1)
    #return torch.tensordot(T5, torch.ones((4)), dims=1)
    # the output is a 2x2x2x2xn tensor

def R_tensor(W, P, Q):     # W is a 32xn matrix, P and Q are 2x2x2x2 tensors
    P = non_local_boxes.utils.tensor_to_matrix(P)
    Q = non_local_boxes.utils.tensor_to_matrix(Q)
    T1 = torch.tensordot(A(W), P, dims=1)  # green term
    T2 = torch.transpose(torch.tensordot(B(W), Q, dims=1), 4, 3)  # blue term
    #T3 = torch.reshape(torch.kron(torch.ones((2,2)), T1*T2), (2,2,2,2,-1,4,4))
    T3 = (T1*T2).repeat(2,2,1,1,1,1,1)
    T4 = T3 * C(W) * D(W)  # the big bracket
    return torch.tensordot(T4, torch.ones((4, 4)), dims=2)
    #T5 = torch.tensordot(T4, torch.ones((4)), dims=1)
    #return torch.tensordot(T5, torch.ones((4)), dims=1)
    # the output is a 2x2x2x2xn tensor



#
#   h
#

def h_flat(R):  # R is a 2x2x2x2xn tensor
    R = torch.reshape(R, (16, -1))
    return torch.tensordot( non_local_boxes.utils.CHSH_flat, R, dims=1 )  # scalar product of CHSH and each column of R


def h_prime_flat(R):  # R is a 2x2x2x2xn tensor
    R = torch.reshape(R, (16, -1))
    return torch.tensordot( non_local_boxes.utils.CHSH_prime_flat, R, dims=1 )  # scalar product of CHSH and each column of R




#
#   FUNCTION TO MAXIMIZE in W
#

def phi_flat(W, P, Q):
    # W is a 32xn matrix
    # P is a box: a 4x4 matrix
    # Q is a box: a 4x4 matrix
    
    return h_flat( R(W,P,Q) )
    # the output is a list of numbers between 0 and 1 (n terms)



def phi_power(W, P, N):
    # W is a 32xn matrix
    # P is a box: a 4x4 matrix
    # N is the power of P

    # # # Q1=torch.clone(P)
    # # # Q1=non_local_boxes.utils.matrix_to_tensor(Q1)  
    # # # # Q1 is a 2x2x2x2 tensor
    
    # # # Q2 = R(W, non_local_boxes.utils.tensor_to_matrix(Q1), P)
    # # # # Q2 is a 2x2x2x2xn tensor

    # # # Q3 = torch.zeros(2, 2, 2, 2, nb_columns)
    # # # for alpha in range(nb_columns):
    # # #     Q3[:,:,:,:,alpha] = R(W, non_local_boxes.utils.tensor_to_matrix(Q2[:,:,:,:,alpha]), P)[:,:,:,:,alpha]

    # Q = torch.clone(P)
    # Q = non_local_boxes.utils.matrix_to_tensor(Q) # Q is a 2x2x2x2 tensor
    # Q = R(W, P, P) 
    # for k in range(N-2):
    #     for alpha in range(nb_columns):
    #         Q[:,:,:,:,alpha] = R(W, non_local_boxes.utils.tensor_to_matrix(Q[:,:,:,:,alpha]), P)[:,:,:,:,alpha]
    # return h_flat( Q )

    Q = torch.zeros(N+1,2,2,2,2,nb_columns)
    Q[2,:,:,:,:,:] = R(W, P, P) 
    for k in range(N-2):
        for alpha in range(nb_columns):
            Q[k+3,:,:,:,:,alpha] = R(W, non_local_boxes.utils.tensor_to_matrix(Q[k+2,:,:,:,:,alpha]), P)[:,:,:,:,alpha]

    
    return h_flat( Q[N,:,:,:,:,:] )
    # the output is a list of numbers between 0 and 1 (n terms)



def box_power_recursive(W, P, N):
    if N==1:
        return P
    if N==2:
        return R(W, P, P) 
    Q = box_power_recursive(W,P,N-1)
    Q_prime = torch.zeros_like(Q)
    for alpha in range(nb_columns):
        Q_prime[:,:,:,:,alpha] = R(W, non_local_boxes.utils.tensor_to_matrix(Q[:,:,:,:,alpha]), P)[:,:,:,:,alpha]
    return Q_prime


def phi_power_recursive(W, P, N):
    return h_flat(box_power_recursive(W, P, N))