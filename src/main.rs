#![feature(is_sorted)]
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use ordered_float::NotNan;
use petgraph::dot::{self, Dot};
// use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::Graph;
use petgraph::Undirected;

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

#[derive(Debug)]
pub struct BipartiteGraph {
    pub left_set: HashSet<usize>,
    pub right_set: HashSet<usize>,
    pub edges: Vec<(usize, usize)>,
}

impl BipartiteGraph {
    pub fn new() -> Self {
        BipartiteGraph {
            left_set: HashSet::new(),
            right_set: HashSet::new(),
            edges: Vec::new(),
        }
    }

    pub fn add_vertex(&mut self, vertex: usize, left: bool) {
        if left {
            self.left_set.insert(vertex);
        } else {
            self.right_set.insert(vertex);
        }
    }

    pub fn add_edge(
        &mut self,
        left_vertex: usize,
        right_vertex: usize,
    ) -> Result<(), &'static str> {
        if self.left_set.contains(&left_vertex) && self.right_set.contains(&right_vertex) {
            self.edges.push((left_vertex, right_vertex));
            Ok(())
        } else {
            Err("Vertices for the edge are not in the respective sets.")
        }
    }

    // More methods depending on what you need
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

fn create_bipartite_from_equations(eqs: &[Rc<Ex>]) -> BipartiteGraph {
    let mut graph = BipartiteGraph::new();

    // Extract unique variables/derivatives from all equations and sort them
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

    // Add nodes for each unique variable/derivative to the graph
    for i in 0..v_nodes.len() {
        graph.add_vertex(i, false);
    }

    // Add nodes for each equation to the graph and
    // link them to the variables/derivatives they contain
    for (i, eq) in eqs.iter().enumerate() {
        graph.add_vertex(i, true);

        let extracted_vars_ders = extract_vars_and_ders(eq.clone());
        for var_or_der in extracted_vars_ders {
            if let Some(index) = v_nodes.iter().position(|node| *node == var_or_der) {
                graph.add_edge(i, index).unwrap();
            }
        }
    }

    graph
}

#[derive(Debug)]
pub struct BipartiteGraph2 {
    ne: usize,
    fadjlist: Vec<Vec<usize>>,
    badjlist: Vec<Vec<usize>>,
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

impl BipartiteGraph2 {
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

        BipartiteGraph2 {
            ne: fadjlist.iter().map(|inner_vec| inner_vec.len()).sum(),
            fadjlist,
            badjlist,
        }
    }

    pub fn nv(&self) -> usize {
        self.fadjlist.len() + self.badjlist.len()
    }
}
// impl From<BipartiteGraph2> for Graph<Rc<Ex>, (), Undirected> {
//     fn from(graph: BipartiteGraph2) -> Self {
//         let mut g = Graph::<(), (), Undirected>::new_undirected();

//         let n_fadj = graph.fadjlist.len();
//         let n_badj = graph.badjlist.len();

//         // Create nodes for fadjlist and badjlist
//         let mut node_indices = Vec::with_capacity(n_fadj + n_badj);
//         for _ in 0..(n_fadj + n_badj) {
//             node_indices.push(g.add_node(()));
//         }

//         // Create edges based on adjacency relations
//         for (f_idx, adjacencies) in graph.fadjlist.iter().enumerate() {
//             for &adj in adjacencies {
//                 g.add_edge(node_indices[f_idx], node_indices[n_fadj + adj], ());
//             }
//         }

//         g
//     }
// }

pub fn build_bipartite_graph2_electric_boogaloo(eqs: &[Rc<Ex>]) -> Graph<Rc<Ex>, (), Undirected> {
    let mut g: Graph<Rc<Ex>, (), Undirected> = Graph::default();

    let v_nodes = vars_from_eqs(eqs);

    // dvars_list.sort(); // vnodes
    // let eqs = &system.equations; // e nodes
    assert!(eqs.is_sorted());

    for (i, eq) in eqs.iter().enumerate() {
        let eq_node_idx = g.add_node(eq.clone());
        // e_nodes.insert(eq_node_idx);
    }

    for (i, var) in v_nodes.iter().enumerate() {
        let var_node_idx = g.add_node(var.clone());
        // v_nodes.insert(var_node_idx);
    }

    for (i, eq) in eqs.iter().enumerate() {
        // let mut variables = HashSet::new();
        // extract_diff_variables(&eq, &mut variables);

        for var in extract_vars_and_ders(eq.clone()) {
            let var_node_idx = g.node_indices().find(|&n| g[n] == var.clone()).unwrap();
            g.add_edge((i as u32).into(), var_node_idx, ());
        }
    }

    g
}
pub enum Node {
    Src(usize),
    Dst(usize),
}

// #[derive(Debug)]
// pub struct System {
//     equations: Vec<Rc<Ex>>,
//     variables: Vec<Rc<Ex>>, // doesn't hold ders
//     parameters: Vec<Rc<Ex>>,
// }


#[derive(Default, Debug)]
pub struct DiffGraph {
    to: Vec<Option<usize>>,
    from: Vec<Option<usize>>,
}

#[derive(Debug)]
pub struct Structure {
    var_diff: DiffGraph, // v_node -> v_node 
    eq_diff: DiffGraph, // e_node -> e_node
    g: BipartiteGraph2, // e_node (srcs) <-> v_node (dsts)
    solveable_graph: Option<BipartiteGraph2>, 
}

#[derive(Debug)]
pub struct State {
    eqs: Vec<Rc<Ex>>,      // eqs
    fullvars: Vec<Rc<Ex>>, // holds vars and ders.
    structure: Structure,
    extra_eqs: Vec<Rc<Ex>>,
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
    let mut g = BipartiteGraph2::from_equations(eqs);
    
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

// construct_augmenting_path!(m::Matching, g::BipartiteGraph, vsrc, dstfilter, vcolor=falses(ndsts(g)), ecolor=nothing) -> path_found::Bool

fn main() {
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

    let der_x = der(x.clone(), 1);
    assert_eq!(der_x, der(x.clone(), 1));
    let g = create_bipartite_from_equations(&eqs);
    println!("{:?}", g);

    let g2 = BipartiteGraph2::from_equations(&eqs);
    println!("{:?} {}", g2, g2.nv());
    let fullvars = vars_from_eqs(&eqs);
    assert_eq!(fullvars.len(), 9);
    assert!(fullvars.is_sorted());
    println!("{:?}", fullvars);

    
    let state  = state_from_eqs(&eqs);
    println!("{:?}", state);

}
