#![feature(is_sorted)]
#![allow(warnings)]
extern crate colored;
use colored::*;

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::Ex::*;
use ordered_float::NotNan;
use BinOpType::*;

#[macro_export]
macro_rules! vars {
    ($($name:ident),* $(,)?) => {
        $(
            let $name = var(stringify!($name));
        )*
    };
}

#[macro_export]
macro_rules! pars {
    ($($name:ident),* $(,)?) => {
        $(
            let $name = par(stringify!($name));
        )*
    };
}

#[derive(Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Ex {
    Const(NotNan<f64>),
    Var(String), // Real valued. implictly depends on time
    Par(String), // Real
    BinOp(BinOpType, Rc<Ex>, Rc<Ex>),
    UnOp(UnOpType, Rc<Ex>),
    Der(Rc<Ex>, usize),
    // Int
    // Real
    // Complex
    // Rat
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum BinOpType {
    Add,
    Sub,
    Mul,
    Div,
    // Pow,
    // Eq,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum UnOpType {
    Neg,
    Sin,
    Cos,
    Exp,
    Log,
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

pub fn binop(op: BinOpType, lhs: Rc<Ex>, rhs: Rc<Ex>) -> Rc<Ex> {
    Rc::new(Ex::BinOp(op, lhs, rhs))
}

pub fn unop(op: UnOpType, operand: Rc<Ex>) -> Rc<Ex> {
    Rc::new(Ex::UnOp(op, operand))
}

pub fn der(expr: Rc<Ex>, n: usize) -> Rc<Ex> {
    Rc::new(Ex::Der(expr, n))
}

macro_rules! gen_binop_funcs {
    ($($func_name:ident : $bin_op:ident),*) => {
        $(
            pub fn $func_name(lhs: Rc<Ex>, rhs: Rc<Ex>) -> Rc<Ex> {
                binop(BinOpType::$bin_op, lhs, rhs)
            }
        )*
    };
}

// Generate functions add, sub, mul, div
gen_binop_funcs! {
    add: Add,
    sub: Sub,
    mul: Mul,
    div: Div
}

pub fn get_der(expr: Rc<Ex>) -> Rc<Ex> {
    match &*expr {
        Const(_) => Rc::new(Const(NotNan::new(0.0).unwrap())), // d/dt(c) = 0
        Var(name) => Rc::new(Der(Rc::new(Var(name.clone())), 1)), // d/dt(x) = Der(x, 1)
        Par(_) => Rc::new(Const(NotNan::new(0.0).unwrap())),   // d/dt(parameter) = 0
        BinOp(BinOpType::Add, left, right) => {
            // (u + v)' = u' + v'
            let left_der = get_der(left.clone());
            let right_der = get_der(right.clone());
            binop(Add, left_der, right_der)
        }
        BinOp(BinOpType::Sub, left, right) => {
            // (u - v)' = u' - v'
            let left_der = get_der(left.clone());
            let right_der = get_der(right.clone());
            binop(Sub, left_der, right_der)
        }
        BinOp(BinOpType::Mul, left, right) => {
            // (u * v)' = u' * v + u * v'
            let left_der = get_der(left.clone());
            let right_der = get_der(right.clone());
            binop(
                Add,
                binop(Mul, left_der, right.clone()),
                binop(Mul, left.clone(), right_der),
            )
        }
        BinOp(BinOpType::Div, left, right) => {
            // (u / v)' = (u' * v - u * v') / v^2
            let left_der = get_der(left.clone());
            let right_der = get_der(right.clone());
            let numerator = binop(
                Sub,
                binop(Mul, left_der, right.clone()),
                binop(Mul, left.clone(), right_der),
            );
            let denominator = binop(Mul, right.clone(), right.clone());
            binop(Div, numerator, denominator)
        }
        UnOp(UnOpType::Neg, operand) => {
            // (-u)' = -u'
            let operand_der = get_der(operand.clone());
            unop(UnOpType::Neg, operand_der)
        }
        // Add more cases for other unary operations (e.g., Sin, Cos) as needed.
        Der(inner, n) => der(inner.clone(), n + 1),
        _ => panic!("get_der: unsupported expression: {:?}", expr),
    }
}

pub fn replace_all(expr: Rc<Ex>, target: &Rc<Ex>, replacement: Rc<Ex>) -> Rc<Ex> {
    if &*expr == &**target {
        return replacement.clone();
    }

    match &*expr {
        Ex::Const(_) | Ex::Var(_) | Ex::Par(_) => expr.clone(),
        Ex::BinOp(op, lhs, rhs) => {
            let new_lhs = replace_all(lhs.clone(), target, replacement.clone());
            let new_rhs = replace_all(rhs.clone(), target, replacement.clone());
            binop(op.clone(), new_lhs, new_rhs)
        }
        Ex::UnOp(op, operand) => {
            let new_operand = replace_all(operand.clone(), target, replacement.clone());
            unop(op.clone(), new_operand)
        }
        Ex::Der(operand, n) => {
            let new_operand = replace_all(operand.clone(), target, replacement.clone());
            der(new_operand, *n)
        }
    }
}

pub fn simplify(expr: Rc<Ex>) -> Rc<Ex> {
    match &*expr {
        Ex::Const(_) => expr.clone(),
        Ex::Var(_) => expr.clone(),
        Ex::Par(_) => expr.clone(),
        Ex::BinOp(op, lhs, rhs) => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());

            // Constant Folding
            if let (Ex::Const(left_val), Ex::Const(right_val)) =
                (&*lhs_simplified, &*rhs_simplified)
            {
                match op {
                    Add => return c((**left_val + **right_val)),
                    Sub => return c((**left_val - **right_val)),
                    Mul => return c((**left_val * **right_val)),
                    Div => {
                        if *right_val == NotNan::new(0.0).unwrap() {
                            panic!("Division by zero");
                        }
                        return c((**left_val / **right_val));
                    }
                }
            }

            // Zero Laws
            if let Ex::Const(val) = &*rhs_simplified {
                if *val == NotNan::new(0.0).unwrap() {
                    match op {
                        Add | Sub => return lhs_simplified.clone(),
                        Mul => return c(0.0),
                        Div => panic!("Division by zero"), // x / 0 is undefined
                    }
                } else if *val == NotNan::new(1.0).unwrap() {
                    if matches!(op, Div) {
                        return lhs_simplified.clone(); // x / 1 = x
                    }
                }
            }

            if let Ex::Const(val) = &*lhs_simplified {
                if *val == NotNan::new(0.0).unwrap() {
                    match op {
                        Add => return rhs_simplified.clone(),
                        Sub => return unop(UnOpType::Neg, rhs_simplified.clone()),
                        Mul => return c(0.0),
                        Div => return c(0.0), // 0 / x = 0, x != 0
                    }
                }
            }

            // Other Simplification Rules (Identity laws, etc.)
            // ...
            if lhs_simplified == rhs_simplified {
                match op {
                    Add => return mul(lhs_simplified.clone(), c(2.0)),
                    Sub => return c(0.0),
                    Mul => return mul(lhs_simplified.clone(), lhs_simplified.clone()), // or pow(lhs_simplified, c(2.0)) if you later add a Pow operation
                    Div => return c(1.0),
                }
            }

            binop(op.clone(), lhs_simplified, rhs_simplified)
        }
        Ex::UnOp(op, operand) => {
            let operand_simplified = simplify(operand.clone());
            unop(op.clone(), operand_simplified)
        }
        Ex::Der(operand, n) => {
            let operand_simplified = simplify(operand.clone());
            // If the operand is a binomial, apply the product rule or chain rule
            if let Ex::BinOp(op, lhs, rhs) = &*operand_simplified {
                match op {
                    // If it's addition or subtraction, the derivative distributes across the terms.
                    Add | Sub => {
                        let lhs_der = der(simplify(lhs.clone()), *n);
                        let rhs_der = der(simplify(rhs.clone()), *n);
                        return binop(op.clone(), lhs_der, rhs_der);
                    }
                    // If it's multiplication or division, you'd apply the product rule or quotient rule,
                    // but let's not get into that right now.
                    _ => {}
                }
            }
            // If it's not a binomial or other special case, just return the derivative of the simplified operand.
            der(operand_simplified, *n)
        }
    }
}

pub fn full_simplify(mut expr: Rc<Ex>) -> Rc<Ex> {
    let mut i = 0;
    loop {
        let simplified = simplify(expr.clone());
        if *simplified == *expr {
            return simplified;
        }
        expr = simplified;
        i += 1;
        if i > 100 {
            panic!("full_simplify: too many iterations");
        }
    }
}

impl Ex {
    pub fn to_pretty_string(&self) -> String {
        match self {
            Ex::Const(value) => format!("{}", value.to_string().green()),
            Ex::Var(name) => format!("{}[t]", name.blue()),
            Ex::Par(name) => format!("{}", name.red()),
            Ex::BinOp(op, lhs, rhs) => {
                let op_str = match op {
                    BinOpType::Add => "+",
                    BinOpType::Sub => "-",
                    BinOpType::Mul => "*",
                    BinOpType::Div => "/",
                };
                format!(
                    "({} {} {})",
                    lhs.to_pretty_string(),
                    op_str,
                    rhs.to_pretty_string()
                )
            }
            Ex::UnOp(op, operand) => {
                let op_str = match op {
                    UnOpType::Neg => "-", // this is broken
                    UnOpType::Sin => "Sin",
                    UnOpType::Cos => "Cos",
                    UnOpType::Exp => "Exp",
                    UnOpType::Log => "Log",
                };
                format!("{}[{}]", op_str, operand.to_pretty_string())
            }
            Ex::Der(operand, n) => format!("D[{},{{{},{}}}]", operand.to_pretty_string(), "t", n),
        }
    }
}

fn extract_vars_and_ders(expr: Rc<Ex>) -> HashSet<Rc<Ex>> {
    let mut result = HashSet::new();

    match &*expr {
        Ex::Var(_) | Ex::Der(_, _) => {
            result.insert(expr.clone());
        }
        Ex::Par(_) => {}   // Skip parameters
        Ex::Const(_) => {} // Skip constants
        Ex::BinOp(_, left, right) => {
            result.extend(extract_vars_and_ders(left.clone()));
            result.extend(extract_vars_and_ders(right.clone()));
        }
        Ex::UnOp(_, operand) => {
            result.extend(extract_vars_and_ders(operand.clone()));
        }
    }

    result
}

pub fn vars_from_eqs(eqs: &[Rc<Ex>]) -> Vec<Rc<Ex>> {
    let mut v_nodes: Vec<_> = eqs
        .iter()
        .map(|eq| extract_vars_and_ders(eq.clone()))
        .fold(HashSet::new(), |mut acc, set| {
            acc.extend(set);
            acc
        })
        .into_iter()
        .collect();
    v_nodes.sort();
    v_nodes
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct BipartiteGraph {
    // pub ne: usize,
    pub fadjlist: Vec<HashSet<usize>>, // each element is a set of v_nodes (indices) that occur in the equation. len == neqs
    pub badjlist: Vec<HashSet<usize>>, // this is which equations the v_nodes occur in. len == nvars
}

impl BipartiteGraph {
    pub fn from_equations(eqs: &[Rc<Ex>]) -> Self {
        let vars = vars_from_eqs(eqs);
        Self::from_eqs_and_vars(eqs, &vars)
    }

    pub fn from_eqs_and_vars(eqs: &[Rc<Ex>], vars: &[Rc<Ex>]) -> Self {
        // assert!(vars.is_sorted());
        // assert!(eqs.is_sorted());

        // Initialize the adjacency lists
        let mut fadjlist = vec![HashSet::new(); eqs.len()];
        let mut badjlist = vec![HashSet::new(); vars.len()];

        // Iterate through the equations
        for (i, eq) in eqs.iter().enumerate() {
            let eq_ = eq.clone();
            let _vars = extract_vars_and_ders(eq_);
            // let mut extracted_vars_ders = vars.iter().collect::<Vec<_>>();
            // extracted_vars_ders.sort();
            for var_or_der in _vars {
                if let Some(index) = vars.iter().position(|node| *node == var_or_der) {
                    badjlist[index].insert(i);
                    fadjlist[i].insert(index);
                }
            }
        }

        BipartiteGraph { fadjlist, badjlist }
    }

    pub fn nv(&self) -> usize {
        self.fadjlist.len() + self.badjlist.len()
    }
    pub fn ne(&self) -> usize {
        self.fadjlist.iter().map(|inner_vec| inner_vec.len()).sum()
    }
}

#[derive(Default, Debug)]
pub struct DiffGraph {
    pub to: Vec<Option<usize>>,
    pub from: Vec<Option<usize>>,
}

#[derive(Debug)]
pub struct Structure {
    pub var_diff: DiffGraph, // v_node -> v_node (A in paper)
    pub eq_diff: DiffGraph,  // e_node -> e_node (B in paper)
    pub g: BipartiteGraph,   // e_node (srcs) <-> v_node (dsts)
    pub solveable_graph: Option<BipartiteGraph>,
}

#[derive(Debug)]
pub struct Eq {
    lhs: Rc<Ex>,
    rhs: Rc<Ex>,
}

#[derive(Debug)]
pub struct State {
    pub eqs: Vec<Rc<Ex>>,      // eqs
    pub fullvars: Vec<Rc<Ex>>, // holds vars and ders.
    pub structure: Structure,
    pub extra_eqs: Vec<Rc<Ex>>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Matching {
    pub m: Vec<Option<usize>>, // v_node -> e_node (Assign in paper)
}

impl Matching {
    pub fn new(size: usize) -> Self {
        Matching {
            m: vec![None; size],
        }
    }
}

/// (3b-l) Delete all V-nodes with A(. )!=0 and all their incident edges from the graph
/// so we have an order where der(x, 1) is before x and is "higher".
/// but for pendulum, T is the highest as der(T, n) is not in eqs
pub fn computed_highest_diff_variables(structure: &Structure) -> Vec<bool> {
    let nvars = structure.var_diff.to.len();
    let mut varwhitelist = vec![false; nvars];

    for var in 0..nvars {
        let mut current_var = var;
        if structure.var_diff.to[current_var].is_none() && !varwhitelist[current_var] {
            while structure.g.badjlist[current_var].is_empty() {
                match structure.var_diff.from[current_var] {
                    Some(next_var) => current_var = next_var,
                    None => break,
                }
            }
            varwhitelist[current_var] = true;
        }
    }

    for var in 0..nvars {
        if !varwhitelist[var] {
            continue;
        }

        let mut current_var = var;
        while let Some(next_var) = structure.var_diff.to[current_var] {
            if varwhitelist[next_var] {
                varwhitelist[var] = false;
                break;
            }
            current_var = next_var;
        }
    }

    varwhitelist
}

pub fn fullvars_to_diffgraph(fullvars: &[Rc<Ex>]) -> DiffGraph {
    let mut to: Vec<Option<usize>> = vec![None; fullvars.len()];
    let mut from: Vec<Option<usize>> = vec![None; fullvars.len()];

    for (idx, expr) in fullvars.iter().enumerate() {
        match &**expr {
            Ex::Var(_) => {
                let der_expr = der(expr.clone(), 1);
                if let Some(der_idx) = fullvars.iter().position(|x| *x == der_expr) {
                    to[idx] = Some(der_idx);
                }
            }
            Ex::Der(inner_expr, 1) => {
                if let Some(original_idx) = fullvars.iter().position(|x| *x == *inner_expr) {
                    from[idx] = Some(original_idx);
                }
            }
            _ => {}
        }
    }

    DiffGraph { to, from }
}

pub fn state_from_eqs(eqs: &[Rc<Ex>], vars: &[Rc<Ex>]) -> State {
    let mut eq_to_diff = DiffGraph::default(); // this stays empty so not mut

    let mut var_to_diff = fullvars_to_diffgraph(&vars);
    let mut g = BipartiteGraph::from_eqs_and_vars(eqs, vars);

    for (i, eq) in eqs.iter().enumerate() {
        eq_to_diff.to.push(None);
        eq_to_diff.from.push(None);
    }

    let structure = Structure {
        var_diff: var_to_diff,
        eq_diff: eq_to_diff,
        g,
        solveable_graph: None,
    };

    let state = State {
        eqs: eqs.to_vec(),
        fullvars: vars.to_vec(),
        structure,
        extra_eqs: Vec::new(),
    };
    state
}

// assumes v is an e_node
pub fn unmatched_neighbors(m: &Matching, g: &BipartiteGraph, v: usize) -> Vec<usize> {
    g.fadjlist[v]
        .iter()
        .filter(|&j| m.m[*j].is_none()) // j is v_node here
        .map(|&j| j)
        .collect()
}
// pub fn uncolored_neighbors(g, v) -> Vec<usize> {}

/// the difference between the base case and the recursive case is that
/// base case we find an unmatched highest_diff_var and match it immediately
/// recursive case we go to an unvisted, but matched highest_diff_var and try to find an augmenting path
/// if you record the path it is clear that only e_node indices are passed to augmenting_path
pub fn augmenting_path(
    m: &mut Matching,
    g: &BipartiteGraph,
    v: usize, // e_node
    colored_eqs: &mut Vec<bool>,
    colored_vars: &mut Vec<bool>,
    is_highest_diff_var: &Vec<bool>, //impl Fn(usize) -> bool,
) -> bool {
    // println!("augmenting_path: v = {}", v);
    // println!("colored_eqs: {:?}", colored_eqs);
    // println!("colored_vars: {:?}", colored_vars);
    colored_eqs[v] = true;

    // variables depending on equation v
    for j in g.fadjlist[v].iter() {
        // if we find a highest_diff_var that is unmatched, we can match it
        if m.m[*j].is_none() && is_highest_diff_var[*j] {
            m.m[*j] = Some(v);
            return true;
        }
    }

    // variables depending on equation v
    for j in g.fadjlist[v].iter() {
        // uncolored highest_diff_var can be traversed
        if !colored_vars[*j] && is_highest_diff_var[*j] {
            colored_vars[*j] = true;
            let k = m.m[*j].unwrap(); // the equation variable[j] is matched to
            if augmenting_path(m, g, k, colored_eqs, colored_vars, is_highest_diff_var) {
                m.m[*j] = Some(v);
                return true;
            }
        }
    }
    false
}

pub fn pants(s: &mut State) -> Matching {
    // println!("{:?}", s.fullvars);
    // println!("{:?}", s.structure.var_diff);
    let mut highest_diff_vars = computed_highest_diff_variables(&s.structure);

    let mut structure = &mut s.structure;

    // let mut vd = &mut structure.var_diff;
    // let mut ed = &mut structure.eq_diff;
    let mut g = &mut structure.g;
    let mut m = Matching::new(g.badjlist.len());

    let mut neqs = g.fadjlist.len();
    let mut nvars = g.badjlist.len();

    let mut colored_eqs = vec![false; neqs];
    let mut colored_vars = vec![false; nvars];

    for keq in 0..neqs {
        // println!("STARTING EQ {} {:?}", keq, s.eqs[keq]);
        let mut eq = keq;
        let mut path_found = false;

        loop {
            colored_vars.as_mut_slice().fill(false);
            colored_eqs.as_mut_slice().fill(false);

            path_found = augmenting_path(
                &mut m,
                g,
                eq,
                &mut colored_eqs,
                &mut colored_vars,
                &highest_diff_vars,
            );
            if path_found {
                // println!("FOUND A PATH for eq# {} {:?}, {:?}", eq, s.eqs[eq], m);
                break;
            }
            // println!("colored_eqs: {:?}", colored_eqs);
            for vidx in 0..colored_vars.len() {
                if !colored_vars[vidx] {
                    continue;
                }
                nvars += 1;

                colored_vars.push(false);

                g.badjlist.push(HashSet::new()); // g new vertex

                // this adds the new vertex and edge to the var_to_diff graph
                structure.var_diff.to[vidx] = Some(nvars - 1); // is -1 correct?
                structure.var_diff.to.push(None); // the new vertex has no outgoing edges
                structure.var_diff.from.push(Some(vidx)); // but there is a new incoming edge

                // the new variable is highest diffed since it is the diff of a previously highest diffed variable
                highest_diff_vars[vidx] = false;
                highest_diff_vars.push(false);
                highest_diff_vars[structure.var_diff.to[vidx].unwrap()] = true;

                // add the new variable to fullvars
                let nv_ex = s.fullvars[vidx].clone();
                let new_var = get_der(nv_ex);
                // println!("TOOKA  NEWVAR {:?}", new_var);
                s.fullvars.push(new_var);

                // m is a matching from v
                m.m.push(None);
                assert_eq!(g.badjlist.len(), nvars);
                assert_eq!(structure.var_diff.to.len(), nvars);
                assert_eq!(m.m.len(), nvars);
            }

            for eidx in 0..colored_eqs.len() {
                if colored_eqs[eidx] {
                    neqs += 1;
                    colored_eqs.push(false);

                    // g new equation
                    g.fadjlist.push(HashSet::new());

                    structure.eq_diff.to[eidx] = Some(neqs - 1);
                    structure.eq_diff.to.push(None); // the new equation has no outgoing edges
                    structure.eq_diff.from.push(Some(eidx)); // but there is a new incoming edge
                                                             // let new_eq = get_der(s.eqs[eidx].clone()); // eq to diff
                    let eq_to_get_der = s.eqs[eidx].clone();

                    // let new_eq = Eq{ lhs: get_der(eq_to_get_der.lhs), rhs: get_der(eq_to_get_der.rhs) };
                    let new_eq = get_der(eq_to_get_der); // eq to diff
                    s.eqs.push(new_eq.clone());
                    println!("TOOKA  DER {:?}", new_eq);

                    // vars connected to eq[eidx]
                    // probably can get around the clone here
                    for e_to_j in g.fadjlist[eidx].clone().iter() {
                        g.fadjlist[neqs - 1].insert(*e_to_j);
                        g.badjlist[*e_to_j].insert(neqs - 1);

                        g.fadjlist[neqs - 1].insert(structure.var_diff.to[*e_to_j].unwrap());
                        g.badjlist[structure.var_diff.to[*e_to_j].unwrap()].insert(neqs - 1);
                    }
                    // println!("{:?}", g);
                }
            }
            for vidx in 0..colored_vars.len() {
                // println!("in the place we dont go");
                if !colored_vars[vidx] {
                    continue;
                }
                // var_eq_matching[var_to_diff[var]] = eq_to_diff[var_eq_matching[var]]
                m.m[structure.var_diff.to[vidx].unwrap()] = structure.eq_diff.to[m.m[vidx].unwrap()]
            }
            // println!("{:?}", );
            eq = structure.eq_diff.to[eq].unwrap();
        }
    }
    m
}

pub fn map_vars_to_eqs(s: &State, m: &Matching) -> HashMap<Rc<Ex>, Rc<Ex>> {
    let mut map = HashMap::new();

    // Iterate through the Matching vector
    for (var_index, eq_index_option) in m.m.iter().enumerate() {
        // If there's a matching equation for this variable, add to map
        if let Some(eq_index) = eq_index_option {
            let var = s.fullvars[var_index].clone();
            let eq = s.eqs[*eq_index].clone();

            map.insert(var, eq);
        }
    }

    map
}

pub fn example2() -> (Vec<Rc<Ex>>, Vec<Rc<Ex>>) {
    vars!(x, y);
    let dx = der(x.clone(), 1);
    let dy = der(y.clone(), 1);

    let eq1 = binop(Add, dx.clone(), dy.clone());
    // x + y^2 == 0
    let eq2 = binop(Add, x.clone(), binop(Mul, y.clone(), y.clone()));

    let eqs = vec![eq1, eq2];
    (eqs, vec![dx.clone(), dy.clone(), x.clone(), y.clone()])
}

/// example 3 in the paper
pub fn pend_sys() -> Vec<Rc<Ex>> {
    vars!(x, y, T, x_t, y_t);
    pars!(r, g);

    let mut vars = vec![x.clone(), y.clone(), T.clone(), x_t.clone(), y_t.clone()];
    vars.sort();

    let mut eqs = vec![
        binop(Sub, der(x.clone(), 1), x_t.clone()),
        binop(Sub, der(y.clone(), 1), y_t.clone()),
        binop(Sub, der(x_t.clone(), 1), binop(Mul, T.clone(), x.clone())),
        binop(
            Sub,
            der(y_t.clone(), 1),
            binop(Sub, binop(Mul, T.clone(), y.clone()), g.clone()),
        ),
        binop(
            Sub,
            binop(
                Add,
                binop(Mul, x.clone(), x.clone()),
                binop(Mul, y.clone(), y.clone()),
            ),
            binop(Mul, r.clone(), r.clone()),
        ),
    ];
    // eqs.sort();
    eqs
}

// =K,(Co-C)-R,
// (33a) #=KI(To-T)+K:zR K3(T-Tc),
// 0--R-g exp(-g4/T)C.
// pub fn example_4() -> Vec<Rc<Ex>>{
//     vars!();
//     pars!(K1, K2, K3, K4);

// }
