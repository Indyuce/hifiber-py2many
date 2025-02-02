//! ```cargo
//! [package]
//! edition = "2021"
//! [dependencies]
//! fibertree = "*"
//! hifiber = "*"
//! random = "*"
//! ```

#![allow(clippy::assertions_on_constants)]
#![allow(clippy::bool_comparison)]
#![allow(clippy::collapsible_else_if)]
#![allow(clippy::comparison_to_empty)]
#![allow(clippy::double_parens)] // https://github.com/adsharma/py2many/issues/17
#![allow(clippy::eq_op)]
#![allow(clippy::let_with_type_underscore)]
#![allow(clippy::map_identity)]
#![allow(clippy::needless_return)]
#![allow(clippy::nonminimal_bool)]
#![allow(clippy::partialeq_to_none)]
#![allow(clippy::print_literal)]
#![allow(clippy::ptr_arg)]
#![allow(clippy::redundant_static_lifetimes)] // https://github.com/adsharma/py2many/issues/266
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::useless_vec)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(unused_imports)]
#![allow(unused_mut)]
#![allow(unused_parens)]

use hifiber::core::eager::EagerFiber;
use hifiber::core::tensor::Tensor;
use std::collections;
use std::io::Result;

use hifiber::core::tensor::Tensor;
pub fn main() -> Result<()> {
    let K = 10;
    let M = 10;
    let N = 10;
let A_KM = Tensor::new_empty(vec!["K", "M", "N"], vec![K, M]);
let B_KN = Tensor::new_empty(vec!["K", "M", "N"], vec![K, N]);
let T0_KMN = Tensor::new_empty(vec!["K", "M", "N"], vec![K, M, N]);
let t0_k = T0_KMN.get_root_mut();
let a_k = A_KM.get_root_mut();
let b_k = B_KN.get_root_mut();
for (k, (t0_m, (a_m, b_n))) in (t0_k << (a_k & b_k)) {
    for (m, (t0_n, a_val)) in (t0_m << a_m) {
        for (n, (t0_ref, b_val)) in (t0_n << b_n) {
            t0_ref += (a_val * b_val);
        }
    }
}
let tmp0 = T0_KMN;
let tmp1: Tensor = tmp0.swizzle_ranks(vec!["M", "K", "N"]);
let T0_MKN: Tensor = tmp1;
let T1_MKN = Tensor::new_empty(vec!["M", "K", "N"], vec![M, K, N]);
let mut t1_m = T1_MKN.get_root_mut();
let t0_m = T0_MKN.get_root_mut();
for (m_pos, (m, (t1_k, t0_k))) in (t1_m << t0_m) {
    for (k_pos, (k, (t1_n, t0_n))) in (t1_k << t0_k) {
        for (n_pos, (n, (t1_ref, t0_val))) in (t1_n << t0_n) {
            t1_ref += t0_val;
        }
    }
}
    let Z_MN = Tensor::new_empty(vec!["M", "N"], vec![M, N]);
    let T1_MNK: Tensor = T1_MKN.swizzle_ranks(vec!["M", "N", "K"]);
    let z_m = Z_MN.get_root_mut();
    t1_m = T1_MNK.get_root_mut();
    for (m, (z_n, t1_n)) in (z_m << t1_m) {
        for (n, (z_ref, t1_k)) in (z_n << t1_n) {
            for (k, t1_val) in t1_k {
                z_ref += t1_val;
            }
        }
    }
    Ok(())
}
