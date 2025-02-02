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

    Z_NM = Tensor(rank_ids=["N", "M"], shape=[N, M])
    tmp0 = A_KM
    tmp1 = tmp0.splitUniform(128, depth=0)
    A_K1K0M = tmp1
    A_K1K0M.setRankIds(rank_ids=["K1", "K0", "M"])
    tmp2 = B_KN
    tmp3 = tmp2.splitUniform(128, depth=0)
    B_K1K0N = tmp3
    B_K1K0N.setRankIds(rank_ids=["K1", "K0", "N"])
    z_n = Z_NM.getRoot()
    A_K1MK0 = A_K1K0M.swizzleRanks(rank_ids=["K1", "M", "K0"])
    B_K1NK0 = B_K1K0N.swizzleRanks(rank_ids=["K1", "N", "K0"])
    tmp4 = A_K1MK0
    tmp5 = tmp4.flattenRanks(depth=1, levels=1, coord_style="tuple")
    A_K1MK0_flat = tmp5
    A_K1MK0_flat.setRankIds(rank_ids=["K1", "MK0"])
    b_k1 = B_K1NK0.getRoot()
    a_k1 = A_K1MK0_flat.getRoot()
    #canvas = createCanvas(A_K1MK0_flat, B_K1NK0, Z_NM)
    for k1_pos, (k1, (a_mk0, b_n)) in enumerate(a_k1 & b_k1):
        A_MK0 = Tensor.fromFiber(rank_ids=["MK0"], fiber=a_mk0)
        tmp6 = A_MK0
        tmp7 = tmp6.splitEqual(16384)
        A_MK01MK00 = tmp7
        A_MK01MK00.setRankIds(rank_ids=["MK01", "MK00"])
        a_mk01 = A_MK01MK00.getRoot()
        for mk01_pos, (mk01, a_mk00) in enumerate(a_mk01):
            for n_pos, (n, (z_m, b_k0)) in enumerate(z_n << b_n):
                for mk00_pos, ((m, k0), a_val) in enumerate(a_mk00):
                    z_ref = z_m.getPayloadRef(m)
                    b_val = b_k0.getPayload(k0)
                    z_ref += a_val * b_val
                    #  canvas.addActivity((k1, (m, k0)), (k1, n, k0), (n, m), spacetime=((mk00_pos,), (k1_pos, mk01_pos, n_pos)))
    tmp8 = Z_NM
    tmp9 = tmp8.swizzleRanks(rank_ids=["M", "N"])
    Z_MN = tmp9
    #displayCanvas(canvas)
