#!/usr/bin/python
import os
import random
from fibertree import Metrics, Tensor
from fibertree.model import Compute, Format, Traffic
from teaal.parse import Einsum as EinsumParser
from teaal.parse import Mapping as MappingParser
from teaal.parse import Architecture as ArchitectureParser
from teaal.parse import Bindings as BindingsParser
from teaal.parse import Format as FormatParser
from teaal.trans.hifiber import HiFiber

if __name__ == "__main__":
    K = 70
    M = 70
    N = 70

    K1 = M1 = N1 = 16
    K0 = M0 = N0 = 4

    A_KM = Tensor(rank_ids=["K", "M"], shape=[K, M])
    B_KN = Tensor(rank_ids=["K", "N"], shape=[K, N])

    Z_N2M2M1N1M0N0 = Tensor(rank_ids=["N2", "M2", "M1", "N1", "M0", "N0"])
    tmp0 = A_KM
    tmp1 = tmp0.splitUniform(M1, depth=1)
    tmp2 = tmp1.splitUniform(M0, depth=2)
    A_KM2M1M0 = tmp2
    A_KM2M1M0.setRankIds(rank_ids=["K", "M2", "M1", "M0"])
    tmp3 = A_KM2M1M0
    tmp4 = tmp3.splitUniform(K1, depth=0)
    tmp5 = tmp4.splitUniform(K0, depth=1)
    A_K2K1K0M2M1M0 = tmp5
    A_K2K1K0M2M1M0.setRankIds(rank_ids=["K2", "K1", "K0", "M2", "M1", "M0"])
    tmp6 = B_KN
    tmp7 = tmp6.splitUniform(N1, depth=1)
    tmp8 = tmp7.splitUniform(N0, depth=2)
    B_KN2N1N0 = tmp8
    B_KN2N1N0.setRankIds(rank_ids=["K", "N2", "N1", "N0"])
    tmp9 = B_KN2N1N0
    tmp10 = tmp9.splitUniform(K1, depth=0)
    tmp11 = tmp10.splitUniform(K0, depth=1)
    B_K2K1K0N2N1N0 = tmp11
    B_K2K1K0N2N1N0.setRankIds(rank_ids=["K2", "K1", "K0", "N2", "N1", "N0"])
    z_n2 = Z_N2M2M1N1M0N0.getRoot()
    A_K2M2M1K1M0K0 = A_K2K1K0M2M1M0.swizzleRanks(rank_ids=["K2", "M2", "M1", "K1", "M0", "K0"])
    B_N2K2N1K1N0K0 = B_K2K1K0N2N1N0.swizzleRanks(rank_ids=["N2", "K2", "N1", "K1", "N0", "K0"])
    a_k2 = A_K2M2M1K1M0K0.getRoot()
    b_n2 = B_N2K2N1K1N0K0.getRoot()
    #canvas = createCanvas(A_K2M2M1K1M0K0, B_N2K2N1K1N0K0, Z_N2M2M1N1M0N0)
    for n2_pos, (n2, (z_m2, b_k2)) in enumerate(z_n2 << b_n2):
        for k2_pos, (k2, (a_m2, b_n1)) in enumerate(a_k2 & b_k2):
            for m2_pos, (m2, (z_m1, a_m1)) in enumerate(z_m2 << a_m2):
                for m1_pos, (m1, (z_n1, a_k1)) in enumerate(z_m1 << a_m1):
                    for n1_pos, (n1, (z_m0, b_k1)) in enumerate(z_n1 << b_n1):
                        for k1_pos, (k1, (a_m0, b_n0)) in enumerate(a_k1 & b_k1):
                            for m0_pos, (m0, (z_n0, a_k0)) in enumerate(z_m0 << a_m0):
                                for n0_pos, (n0, (z_ref, b_k0)) in enumerate(z_n0 << b_n0):
                                    for k0_pos, (k0, (a_val, b_val)) in enumerate(a_k0 & b_k0):
                                        z_ref += a_val * b_val
                                        #canvas.addActivity((k2, m2, m1, k1, m0, k0), (n2, k2, n1, k1, n0, k0), (n2, m2, m1, n1, m0, n0), spacetime=((k1_pos,), (n2_pos, k2_pos, m2_pos, m1_pos, n1_pos, m0_pos, n0_pos, k0_pos)))
    tmp12 = Z_N2M2M1N1M0N0
    tmp13 = tmp12.swizzleRanks(rank_ids=["M2", "M1", "M0", "N2", "N1", "N0"])
    tmp14 = tmp13.mergeRanks(depth=3, levels=2, coord_style="absolute")
    tmp15 = tmp14.mergeRanks(depth=0, levels=2, coord_style="absolute")
    tmp15.setRankIds(rank_ids=["M", "N"])
    Z_MN = tmp15
    #displayCanvas(canvas)
