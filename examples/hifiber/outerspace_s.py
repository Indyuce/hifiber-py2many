#!/usr/bin/python
import os
import random
from fibertree import Tensor

if __name__ == "__main__":
    K = 10
    M = 10
    N = 10
    A_KM = Tensor(rank_ids=["K", "M", "N"], shape=[K, M])
    B_KN = Tensor(rank_ids=["K", "M", "N"], shape=[K, N])

    T_KMN = Tensor(rank_ids=["K", "M", "N"], shape=[K,M,N])
    t_k = T_KMN.getRoot()
    a_k = A_KM.getRoot()
    b_k = B_KN.getRoot()
    for (k, (t_m, (a_m, b_n))) in t_k << (a_k & b_k):
        for (m, (t_n, a_val)) in t_m << a_m:
            for (n, (t_ref, b_val)) in t_n << b_n:
                t_ref += a_val * b_val
    Z_MN = Tensor(rank_ids=["M", "N"], shape=[M,N])
    z_m = Z_MN.getRoot()
    t_k = T_KMN.getRoot()
    for (k, t_m) in t_k:
        for (m, (z_n, t_n)) in z_m << t_m:
            for (n, (z_ref, t_val)) in z_n << t_n:
                z_ref += t_val