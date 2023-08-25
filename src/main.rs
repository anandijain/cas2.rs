#![feature(is_sorted)]
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use ordered_float::NotNan;
use petgraph::dot::{self, Dot};
// use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::Graph;
use petgraph::Undirected;

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
    eqs.sort();
    eqs
}

use BinOpType::*;

#[derive(Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Ex {
    Const(NotNan<f64>),
    Var(String), // Real implictly depends on time
    Par(String), // Real
    BinOp(BinOpType, Rc<Ex>, Rc<Ex>),
    UnOp(UnOpType, Rc<Ex>),
    Der(Rc<Ex>, usize),
}

#[derive(Clone, Hash, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum BinOpType {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum UnOpType {
    Neg,
    Sin,
    Cos,
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

#[derive(Debug, Default)]
pub struct BipartiteGraph {
    ne: usize,
    fadjlist: Vec<Vec<usize>>,
    badjlist: Vec<Vec<usize>>,
}

impl BipartiteGraph {
    pub fn from_equations(eqs: &[Rc<Ex>]) -> Self {
        let v_nodes = vars_from_eqs(eqs);

        // Initialize the adjacency lists
        let mut fadjlist = vec![Vec::new(); v_nodes.len()];
        let mut badjlist = vec![Vec::new(); eqs.len()];

        // Iterate through the equations
        for (i, eq) in eqs.iter().enumerate() {
            let extracted_vars_ders = extract_vars_and_ders(eq.clone());
            for var_or_der in extracted_vars_ders {
                if let Some(index) = v_nodes.iter().position(|node| *node == var_or_der) {
                    fadjlist[index].push(i);
                    badjlist[i].push(index);
                }
            }
        }

        BipartiteGraph {
            ne: fadjlist.iter().map(|inner_vec| inner_vec.len()).sum(),
            fadjlist,
            badjlist,
        }
    }

    pub fn nv(&self) -> usize {
        self.fadjlist.len() + self.badjlist.len()
    }
}

#[derive(Default, Debug)]
pub struct DiffGraph {
    to: Vec<Option<usize>>,
    from: Vec<Option<usize>>,
}

#[derive(Debug)]
pub struct Structure {
    var_diff: DiffGraph, // v_node -> v_node (A in paper)
    eq_diff: DiffGraph,  // e_node -> e_node (B in paper)
    g: BipartiteGraph,   // e_node (srcs) <-> v_node (dsts)
    solveable_graph: Option<BipartiteGraph>,
}

#[derive(Debug)]
pub struct State {
    eqs: Vec<Rc<Ex>>,      // eqs
    fullvars: Vec<Rc<Ex>>, // holds vars and ders.
    structure: Structure,
    extra_eqs: Vec<Rc<Ex>>,
}

#[derive(Debug)]
pub struct Matching {
    m: Vec<Option<usize>>, // e_node -> v_node (Assign in paper)
}

impl Matching {
    pub fn new(size: usize) -> Self {
        Matching {
            m: vec![None; size],
        }
    }
}

pub fn computed_highest_diff_variables(structure: Structure) -> Vec<bool> {
    let nvars = structure.var_diff.to.len();
    let mut varwhitelist = vec![false; nvars];

    for var in 0..nvars {
        let mut current_var = var;
        if structure.var_diff.to[current_var].is_none() && !varwhitelist[current_var] {
            while structure.g.fadjlist[current_var].is_empty() {
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
            Ex::Var(_) | Ex::Par(_) => {
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

pub fn state_from_eqs(eqs: &[Rc<Ex>]) -> State {
    let mut eq_to_diff = DiffGraph::default(); // this stays empty so not mut

    // almost certainly slow
    let vars = vars_from_eqs(eqs);
    let mut var_to_diff = fullvars_to_diffgraph(&vars);
    // its possible that idxs get messed up between g and vars
    let mut g = BipartiteGraph::from_equations(eqs);

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
        fullvars: vars,
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

pub fn aug(
    m: &mut Matching,
    g: &BipartiteGraph,
    v: usize, // e_node
    colored_eqs: &mut Vec<usize>,
    colored_vars: &mut Vec<usize>,
) -> bool {
    // colored_eqs[v] = true;
    // for j in unmatched_neighbors(&m, &g, v) {}
    for j in g.fadjlist[v].iter() {
        if m.m[*j].is_none() {
            m.m[*j] = Some(v);
            return true;
        }
    }

    // for j in uncolored_neighbors(g, v) {}
    for j in g.fadjlist[v].iter() {
        if !colored_eqs.contains(j) {
            colored_eqs.push(*j);
            let k = m.m[*j].unwrap(); // the equation variable[j] is matched to
            if aug(m, g, *j, colored_eqs, colored_vars) {
                m.m[*j] = Some(v);
                return true;
            }
        }
    }
    false
}

fn main() {
    let eqs = pend_sys();
    let g2 = BipartiteGraph::from_equations(&eqs);
    println!("{:?} {}", g2, g2.nv());
    let fullvars = vars_from_eqs(&eqs);
    assert_eq!(fullvars.len(), 9);
    assert!(fullvars.is_sorted());
    println!("{:?}", fullvars);

    let state = state_from_eqs(&eqs);
    println!("{:?}", state);


    // println!("{:?}", );
    let X = (0..2).collect::<Vec<_>>();
    let Y = (0..2).collect::<Vec<_>>();
    let E = vec![(0, 0), (0, 1), (1, 0)];

    let mut bg = BipartiteGraph::default();

    bg.ne = E.len();
    bg.fadjlist = vec![vec![0, 1], vec![0]];
    bg.badjlist = vec![vec![0, 1], vec![0]];

    let mut m = Matching::new(bg.badjlist.len());
    m.m[0] = Some(0);

    let mut colored_eqs = Vec::new();
    let mut colored_vars = Vec::new();
    let pf = aug(&mut m, &bg, 1, &mut colored_eqs, &mut colored_vars);
    println!("{:?}", m);
    println!("{:?}", pf);
}
