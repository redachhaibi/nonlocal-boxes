import numpy as np

W_BS09 = [0, 0, 1, 1,              # f_1(x, a_2) = x
          0, 0, 1, 1,              # g_1(y, b_2) = y
          0, 0, 0, 1,              # f_2(x, a_1) = a_1*x
          0, 0, 0, 1,              # g_2(y, b_1) = b_1*y
          0, 1, 1, 0, 0, 1, 1, 0,  # f_3(x, a_1, a_2) = a_1 + a_2 mod 2
          0, 1, 1, 0, 0, 1, 1, 0   # g_3(y, b_1, b_2) = b_1 + b_2 mod 2
         ]

def random_wiring(n):
    #return np.array([[random.random() for _ in range(32)] for _ in range(nb_columns)]).T
    return np.random.rand(32, n)
