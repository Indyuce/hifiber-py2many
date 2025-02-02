#!/usr/bin/python
import os
import random
from fibertree import Tensor

if __name__ == "__main__":
    K = 10
    M = 10
    N = 10
    A_MK = Tensor(rank_ids=["K", "M", "N"], shape=[K, M])
    B_NK = Tensor(rank_ids=["K", "M", "N"], shape=[K, N])

    Z_MN = Tensor(rank_ids=["M", "N"], shape=[M,N])
    z_m = Z_MN.getRoot()
    a_m = A_MK.getRoot()
    b_n = B_NK.getRoot()
    for (m, (z_n, a_k)) in z_m << a_m:
        for (n, (z_ref, b_k)) in z_n << b_n:
            for (k, (a_val, b_val)) in a_k & b_k:
                z_ref += a_val * b_val
