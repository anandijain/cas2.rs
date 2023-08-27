#![feature(is_sorted)]
#![allow(warnings)]
use std::rc::Rc;

use cas2::BinOpType::*;
use cas2::*;

// The type signature specifies that it takes a function `f` which takes Rc<Ex> and returns Rc<Ex>,
// an initial expression `init`, and the number of times `n` the function should be applied.
pub fn nest(f: impl Fn(Rc<Ex>) -> Rc<Ex>, init: Rc<Ex>, n: usize) -> Rc<Ex> {
    let mut result = init;
    for _ in 0..n {
        result = f(result.clone());
    }
    result
}

// Similar to `nest`, but returns a Vec<Rc<Ex>> containing all intermediate results.
pub fn nest_list(f: impl Fn(Rc<Ex>) -> Rc<Ex>, init: Rc<Ex>, n: usize) -> Vec<Rc<Ex>> {
    let mut result = vec![init.clone()];
    let mut current = init;

    for _ in 0..n {
        current = f(current.clone());
        result.push(current.clone());
    }

    result
}

pub fn nest_and_replace() {
    vars!(x);
    // Define a simple function to square an expression
    let square = |expr: Rc<Ex>| binop(Mul, expr.clone(), expr.clone());

    // Nest: Square x, 3 times
    let nested_expr = nest(square, x.clone(), 3);
    // nested_expr will be (((x*x) * (x*x)) * ((x*x) * (x*x)))

    // NestList: Square x, 3 times, collecting all intermediate results
    let nested_expr_list = nest_list(square, x.clone(), 3);
    // nested_expr_list will contain [x, x*x, (x*x)*(x*x), ((x*x)*(x*x))*((x*x)*(x*x))]
    println!("{:?}", nested_expr_list);
    // let highest_diffed_eqs =
    // let vars = vars_from_eqs(&feqs);
    // let mut s2 = state_from_eqs(&feqs, vars.as_slice());
    // .map(|(i, v)| (s.fullvars[i], s.fullvars[v]));
}

fn main() {
    let sys = pend_sys();
    let vars = vars_from_eqs(&sys);
    // println!("VARS: {:?}", vars);
    let mut s = state_from_eqs(&sys, vars.as_slice());
    let m = pants(&mut s);
    // let m = pants(&mut s); // never terminates upon second call (need maxiters)
    // println!("{:?}", s);
    // println!("{:?}", m);

    let var_eq_matching = map_vars_to_eqs(&s, &m);
    // println!("{:?}", var_eq_matching);

    let seq8 = full_simplify(s.eqs[8].clone());
    println!("{}", seq8.to_pretty_string());

    // let foo = s.structure.var_diff.to.iter().enumerate();
    // let mut tups = vec![];
    // for (i, v) in s.structure.var_diff.to.iter().enumerate() {
    //     if v.is_some() {
    //         tups.push((s.fullvars[i].clone(), s.fullvars[v.unwrap()].clone()));
    //     }
    // }
    // let fvs = s.fullvars.clone();
    // let feqs = s.eqs.clone();
    // let vec = m.m;
    // let result: Vec<(usize, usize)> = vec
    //     .iter()
    //     .enumerate()
    //     .filter_map(|(idx, &opt)| opt.map(|val| (idx, val)))
    //     .collect();

    // //hardcoded rn
    // // the constrait diff chain is eq idxs 4 5 and 8
    // let highest_diffd_eqs: Vec<_> = vec![2, 3, 6, 7, 8]
    //     .iter()
    //     .map(|&i| s.eqs[i].clone())
    //     .collect();

    // println!("HIGHEST DIFFD EQS{:#?}", highest_diffd_eqs);

    // let vars2 = vars_from_eqs(&highest_diffd_eqs);
    // println!("VARS2: {:?}", vars2);
    // let mut state2 = state_from_eqs(&highest_diffd_eqs, vars2.as_slice());
    // let m2 = pants(&mut state2);
    // // println!("{:?}", state2);
    // println!("{:?}", m2);


    // [Some(0), None,     None,     None,             None,             Some(2),            Some(4),          Some(3),          Some(1)]
    // [Var("T"),Var("x"), Var("y"), Der(Var("x"), 1), Der(Var("x"), 2), Der(Var("x_t"), 1), Der(Var("y"), 1), Der(Var("y"), 2), Der(Var("y_t"), 1)]
    // if we rerun pants on the highest diffed eqs, we only find 9 vars
}

#[cfg(test)]
mod test {
    use ordered_float::NotNan;

    use super::*;
    #[test]
    fn augmenting_path_test() {
        let eqs = pend_sys();
        let g2 = BipartiteGraph::from_equations(&eqs);
        let fullvars = vars_from_eqs(&eqs);
        assert_eq!(fullvars.len(), 9);
        assert!(fullvars.is_sorted());

        // let state = state_from_eqs(&eqs);

        let X = (0..2).collect::<Vec<_>>();
        let Y = (0..2).collect::<Vec<_>>();
        let E = vec![(0, 0), (0, 1), (1, 0)];

        let mut bg = BipartiteGraph::default();

        bg.fadjlist = vec![
            [0, 1].iter().cloned().collect(),
            [0].iter().cloned().collect(),
        ];
        bg.badjlist = vec![
            [0, 1].iter().cloned().collect(),
            [0].iter().cloned().collect(),
        ];

        let mut m = Matching::new(bg.badjlist.len());
        m.m[0] = Some(0);

        // println!("{:?}", bg);
        // an example eqs for the above graph
        vars!(x, y, z);
        // x + y, x
        // [1, 2], [1] == fadjlist
        let mut eqs = vec![binop(Add, x.clone(), y.clone()), x.clone()];
        // eqs.sort();
        let vars = vec![x.clone(), y.clone()];

        let g3 = BipartiteGraph::from_eqs_and_vars(&eqs, &vars);

        assert_eq!(bg, g3);

        let mut colored_eqs = vec![false; bg.fadjlist.len()];
        let mut colored_vars = vec![false; bg.badjlist.len()];

        let state = state_from_eqs(&eqs, &vars);
        let mut highest_diff_vars = computed_highest_diff_variables(&state.structure);

        // println!("{:?}", highest_diff_vars);
        let pf = augmenting_path(
            &mut m,
            &bg,
            1,
            &mut colored_eqs,
            &mut colored_vars,
            &highest_diff_vars,
        );
        // println!("{:?}", m);
        // println!("{:?}", pf);

        let bg2 = BipartiteGraph::from_eqs_and_vars(
            &vec![x.clone(), y.clone()],
            &vec![x.clone(), y.clone(), z.clone()],
        );
        // println!("{:?}", bg2);

        let (example_eqs, example_vars) = example2();
        let mut s = state_from_eqs(&example_eqs, &example_vars);
        let matching = pants(&mut s);
        println!("{:?}", matching);

        // STOCHASTIC FAILURE STILL
        assert_eq!(
            matching,
            Matching {
                m: vec![Some(0), Some(2), None, None]
            }
        );
    }

    #[test]
    pub fn symbolic_differentiation() {
        vars!(x, y);
        let dx = der(x.clone(), 1);
        let dy = der(y.clone(), 1);

        // dx + dy == 0
        let eq1 = binop(Add, dx.clone(), dy.clone());
        // x + y^2 == 0
        let eq2 = binop(Add, x.clone(), binop(Mul, y.clone(), y.clone()));

        let eqs = vec![eq1.clone(), eq2.clone()];

        let deq2 = get_der(eq2.clone());
        // dx + 2y*dy
        let ydy = binop(Mul, y.clone(), dy.clone());
        let expected_deq2 = binop(
            Add,
            dx.clone(),
            binop(
                Add,
                binop(Mul, dy.clone(), y.clone()),
                binop(Mul, y.clone(), dy.clone()),
            ),
        );
        assert_eq!(deq2, expected_deq2);
    }

    #[test]
    fn test_constant_folding() {
        let expr = binop(Add, c(1.0), binop(Mul, c(2.0), c(3.0)));
        let simplified = simplify(expr);
        assert_eq!(*simplified, Ex::Const(NotNan::new(7.0).unwrap()));
    }

    #[test]
    fn test_zero_laws_addition() {
        let expr = binop(Add, c(0.0), var("x"));
        let simplified = simplify(expr);
        assert_eq!(*simplified, Ex::Var("x".to_string()));
    }

    #[test]
    fn test_zero_laws_multiplication() {
        let expr = binop(Mul, c(0.0), var("x"));
        let simplified = simplify(expr);
        assert_eq!(*simplified, Ex::Const(NotNan::new(0.0).unwrap()));
    }

    #[test]
    fn test_zero_laws_subtraction() {
        let expr = binop(Sub, var("x"), c(0.0));
        let simplified = simplify(expr);
        assert_eq!(*simplified, Ex::Var("x".to_string()));
    }

    #[test]
    fn test_zero_laws_division_by_zero() {
        let expr = binop(Div, var("x"), c(0.0));
        let should_panic = std::panic::catch_unwind(|| {
            simplify(expr);
        });
        assert!(should_panic.is_err());
    }

    #[test]
    fn test_zero_laws_division_by_one() {
        let expr = binop(Div, var("x"), c(1.0));
        let simplified = simplify(expr);
        assert_eq!(*simplified, Ex::Var("x".to_string()));
    }

    #[test]
    fn test_zero_laws_division_of_zero() {
        let expr = binop(Div, c(0.0), var("x"));
        let simplified = simplify(expr);
        assert_eq!(*simplified, Ex::Const(NotNan::new(0.0).unwrap()));
    }

    #[test]
    fn test_multiple_simplification_needed() {
        vars!(var_a, var_b, var_c);

        let initial_expr = binop(
            Add,
            binop(Add, var_a.clone(), binop(Add, var_b.clone(), c(0.0))),
            binop(Add, c(0.0), var_c.clone()),
        );

        let once_simplified = simplify(initial_expr.clone());
        assert_eq!(
            *once_simplified,
            *binop(Add, binop(Add, var_a.clone(), var_b.clone()), var_c.clone())
        );

        let twice_simplified = simplify(once_simplified);
        assert_eq!(
            *twice_simplified,
            *binop(Add, binop(Add, var_a.clone(), var_b.clone()), var_c.clone())
        ); // This will be different if you decide to implement further flattening or similar rules
    }

    #[test]
    fn test_full_simplify() {
        vars!(var_a, var_b, var_c);

        let initial_expr = binop(
            Add,
            binop(Add, var_a.clone(), binop(Add, var_b.clone(), c(0.0))),
            binop(Add, c(0.0), var_c.clone()),
        );

        // Use full_simplify, which should reach a fixed point
        let fully_simplified = full_simplify(initial_expr.clone());

        // Assuming further flattening or similar rules aren't added to the simplify function,
        // the fully_simplified expression should match this
        assert_eq!(
            *fully_simplified,
            *binop(Add, binop(Add, var_a.clone(), var_b.clone()), var_c.clone())
        );
    }
}
