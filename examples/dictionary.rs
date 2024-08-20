//! ```cargo
//! [package]
//! edition = "2021"
//! [dependencies]
//!
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

use std::collections::HashMap;
use std::io::Result;

pub fn main() -> Result<()> {
    let prices: &HashMap<&str, i32> = &[("apple", 5), ("banana", 10)]
        .iter()
        .cloned()
        .collect::<HashMap<_, _>>();
    let my_purchase: &HashMap<&str, i32> = &[("apple", 1), ("banana", 6)]
        .iter()
        .cloned()
        .collect::<HashMap<_, _>>();
    let grocery_bill: None = my_purchase
        .keys()
        .map(|fruit| (prices[fruit] * my_purchase[fruit]))
        .collect::<Vec<_>>()
        .iter()
        .sum();
    println!("{} {}", "I owe the grocer ", grocery_bill);
    Ok(())
}
