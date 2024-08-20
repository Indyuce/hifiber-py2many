#!/usr/bin/python
import os
import random
from fibertree import Tensor

if __name__ == "__main__":
    K = 10
    M = 10
    N = 10

    A_KM = Tensor(rank_ids=["K", "M"], shape=[K, M])
    B_KN = Tensor(rank_ids=["K", "N"], shape=[K, N])

    T0_KMN = Tensor(rank_ids=["K", "M", "N"], shape=[K, M, N])
    t0_k = T0_KMN.getRoot()
    a_k = A_KM.getRoot()
    b_k = B_KN.getRoot()
    for k, (t0_m, (a_m, b_n)) in t0_k << (a_k & b_k):
        for m, (t0_n, a_val) in t0_m << a_m:
            for n, (t0_ref, b_val) in t0_n << b_n:
                t0_ref += a_val * b_val
    tmp0 = T0_KMN
    tmp1 = tmp0.swizzleRanks(rank_ids=["M", "K", "N"])
    T0_MKN = tmp1



    T1_MKN = Tensor(rank_ids=["M", "K", "N"], shape=[M, K, N])
    t1_m = T1_MKN.getRoot()
    t0_m = T0_MKN.getRoot()
    for m, (t1_k, t0_k) in t1_m << t0_m:
        for k, (t1_n, t0_n) in t1_k << t0_k:
            for n, (t1_ref, t0_val) in t1_n << t0_n:
                t1_ref += t0_val




    Z_MN = Tensor(rank_ids=["M", "N"], shape=[M, N])
    z_m = Z_MN.getRoot()
    T1_MNK = T1_MKN.swizzleRanks(rank_ids=["M", "N", "K"])
    t1_m = T1_MNK.getRoot()
    for m, (z_n, t1_n) in z_m << t1_m:
        for n, (z_ref, t1_k) in z_n << t1_n:
            for k, t1_val in t1_k:
                z_ref += t1_val
