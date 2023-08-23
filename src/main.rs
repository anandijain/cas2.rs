use std::rc::Rc;

use ordered_float::NotNan;
use BinOpType::*;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Ex {
    Const(NotNan<f64>),
    Var(String),
    Par(String),
    BinOp(BinOpType, Rc<Ex>, Rc<Ex>),
    UnOp(UnOpType, Rc<Ex>),
    Der(Rc<Ex>, usize),
    // ... other variants
}

// Define BinOpType, UnOpType, etc.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum BinOpType {
    Add,
    Sub,
    Mul,
    Div,
    // ... other operators
}

// macro_rules! define_op_overloads {
//     ($($trait:ident, $method:ident, $op:path);* $(;)?) => {
//         $(
//             impl std::ops::$trait for Rc<Ex> {
//                 type Output = Rc<Ex>;

//                 fn $method(self, rhs: Self) -> Self::Output {
//                     Rc::new(binop($op, &self, &rhs))
//                 }
//             }
//         )*
//     }
// }

// // std::ops::
// define_op_overloads! {
//     Add, add, BinOpType::Add;
//     Sub, sub, BinOpType::Sub;
//     Mul, mul, BinOpType::Mul;
// };

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum UnOpType {
    Neg,
    // ... other unary operations
}

pub fn c(val: f64) -> Rc<Ex> {
    Rc::new(Ex::Const(NotNan::new(val).unwrap()))
}

pub fn var(name: &str) -> Rc<Ex> {
    Rc::new(Ex::Var(name.to_string()))
}

pub fn par(name: &str) -> Rc<Ex> {
    Rc::new(Ex::Par(name.to_string()))
}

pub fn binop(op: BinOpType, lhs: &Rc<Ex>, rhs: &Rc<Ex>) -> Rc<Ex> {
    Rc::new(Ex::BinOp(op, lhs.clone(), rhs.clone()))
}

pub fn unop(op: UnOpType, operand: &Rc<Ex>) -> Rc<Ex> {
    Rc::new(Ex::UnOp(op, operand.clone()))
}

pub fn der(expr: &Rc<Ex>, n: usize) -> Rc<Ex> {
    Rc::new(Ex::Der(expr.clone(), n))
}
macro_rules! vars {
    ($($name:ident),* $(,)?) => {
        $(
            let $name = var(stringify!($name));
        )*
    };
}

macro_rules! pars {
    ($($name:ident),* $(,)?) => {
        $(
            let $name = par(stringify!($name));
        )*
    };
}
// Using the above definitions:

fn main() {
    vars!(x, y, T, x_t, y_t);
    pars!(r, g);

    let mut eqs = vec![
        binop(Sub, &der(&x, 1), &x_t),
        binop(Sub, &der(&y, 1), &y_t),
        binop(Sub, &der(&x_t, 1), &binop(Mul, &T, &x)),
        binop(Sub, &der(&y_t, 1), &binop(Sub, &binop(Mul, &T, &y), &g)),
        binop(
            Sub,
            &binop(Add, &binop(Mul, &x, &x), &binop(Mul, &y, &y)),
            &binop(Mul, &r, &r),
        ),
    ];
    eqs.sort();
    for eq in &eqs {
        println!("{:?}", eq);

    }
}
