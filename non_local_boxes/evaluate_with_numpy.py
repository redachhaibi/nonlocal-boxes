import numpy as np
import non_local_boxes.utils


nb_columns = 100000





#
#   A(W)
#

# A1 is a 2x4x4x32-tensor
A1 = np.zeros( (2, 4, 4, 32) )
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


# A2 is a 2x4x4xnb_columns-tensor
A2 = np.zeros( (2, 4, 4, nb_columns) )
for x in range(2):
    for i in range(4):
        for k in range(4):
            for alpha in range(nb_columns):
                if k<=1:
                    A2[x, i, k]=1


# A3 is a 2x4x4x32-tensor
A3 = np.zeros( (2, 4, 4, 32) )
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


# A4 is a 2x4x4xnb_columns-tensor
A4 = np.zeros( (2, 4, 4, nb_columns) )
for y in range(2):
    for i in range(4):
        for k in range(4):
            for alpha in range(nb_columns):
                if k==0 or k==2:
                    A4[y, i, k]=1


def A(W):    # W is a 32xn matrix
    T1 = np.tensordot(A1, W, axes=([3, 0])) + A2
    T2 = np.tensordot(np.ones((2)), T1, axes=0)
    T3 = np.transpose(T2, (1,0,2,3,4))
    S1 = np.tensordot(A3, W, axes=([3, 0])) + A4
    S2 = np.tensordot(np.ones((2)), S1, axes=0)
    return np.transpose(T3 * S2, (0,1,4,2,3))
    # the output is a 2x2xnx4x4 tensor





#
#   B(W)
#

# B1 is a 2x4x4x32-tensor
B1 = np.zeros( (2, 4, 4, 32) )
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
B3 = np.zeros( (2, 4, 4, 32) )
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
    T1 = np.tensordot(B1, W, axes=([3, 0])) + B2
    T2 = np.tensordot(np.ones((2)), T1, axes=0)
    T3 = np.transpose(T2, (1,0,2,3,4))
    S1 = np.tensordot(B3, W, axes=([3, 0])) + B4
    S2 = np.tensordot(np.ones((2)), S1, axes=0)
    return np.transpose(T3 * S2, (0,1,4,2,3))
    # the output is a 2x2xnx4x4 tensor





#
#   C(W)
#

# C1 is a 2x2x4x4x32-tensor
C1 = np.zeros( (2, 2, 4, 4, 32) )
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
C2 = np.zeros( (2, 2, 4, 4, nb_columns) )
for x in range(2):
    for i in range(4):
        for j in range(4):
            for alpha in range(nb_columns):
                C2[0, x, i, j]=1


def C(W):    # W is a 32xn matrix
    T1 = np.tensordot(C1, W, axes=([4, 0])) + C2
    T2 = np.tensordot( np.ones((2,2)), T1, axes=0)  # Kronecker product
    return np.transpose(T2, (2, 0, 3, 1, 6, 4, 5))
    # the output is a 2x2x2x2xnx4x4 tensor





#
#   D(W)
#

# D1 is a 2x2x4x4x32-tensor
D1 = np.zeros( (2, 2, 4, 4, 32) )
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
D2 = np.zeros( (2, 2, 4, 4, nb_columns) )
for y in range(2):
    for i in range(4):
        for j in range(4):
            for alpha in range(nb_columns):
                D2[0, y, i, j] = 1

def D(W):    # W is a 32xn matrix
    T1 = np.tensordot(D1, W, axes=([4, 0])) + D2
    T2 = np.tensordot( np.ones((2,2)), T1, axes=0)  # Kronecker product
    return np.transpose(T2, (0, 2, 1, 3, 6, 4, 5))
    # the output is a 2x2x2x2xnx4x4 tensor





#
#   R(W, P, Q) := P x_W Q
#

def R(W, P, Q):     # W is a 32xn matrix, P and Q are 4x4 matrices
    T1 = np.tensordot(A(W), P, axes=([4, 0]))  # green term
    T2 = np.transpose(np.tensordot(B(W), Q, axes=([4, 0])), (0,1,2,4,3))  # blue term
    T3 = np.tensordot(np.ones((2,2)), T1*T2, axes = 0)   # Kronecker product
    T4 = T3 * C(W) * D(W)  # the big bracket
    T5 = np.tensordot(T4, np.ones((4)), axes=([6, 0]))
    return np.tensordot(T5, np.ones((4)), axes = ([5,0]))
    # the output is a 2x2x2x2xn tensor





#
#   h
#

def h_flat(R):  # R is a 2x2x2x2xn tensor
    R = np.reshape(R, (16, -1))
    return np.dot( non_local_boxes.utils.CHSH_flat, R )  # scalar product of CHSH and each column of R





#
#   FUNCTION TO MAXIMIZE
#

def phi_flat(W, P, Q):
    # W is a 32xn matrix
    # P is a box: a 4x4 matrix
    # Q is a box: a 4x4 matrix
    
    return h_flat( R(W,P,Q) )
    # the output is a number between 0 and 1