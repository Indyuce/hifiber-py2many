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
use std::collections;
use std::io::Result;

use hifiber::core::tensor::Tensor;
pub fn main() -> Result<()> {
    let K = 10;
    let M = 10;
    let N = 10;
    let A_MK = Tensor::new_empty(vec!["K", "M", "N"], vec![K as usize, M as usize]);
    let B_NK = Tensor::new_empty(vec!["K", "M", "N"], vec![K as usize, N as usize]);
    let Z_MN = Tensor::new_empty(vec!["M", "N"], vec![M as usize, N as usize]);
    let z_m: &mut EagerFiber = Z_MN.get_root_mut();
    let a_m: &mut EagerFiber = A_MK.get_root_mut();
    let b_n: &mut EagerFiber = B_NK.get_root_mut();
    for (m, (z_n, a_k)) in (z_m << a_m) {
        for (n, (z_ref, b_k)) in (z_n << b_n) {
            for (k, (a_val, b_val)) in (a_k & b_k) {
                z_ref += (a_val * b_val);
            }
        }
    }
    Ok(())
}
