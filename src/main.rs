#![feature(is_sorted)]
#![allow(warnings)]
use cas2::*;
use cas2::BinOpType::*;

fn main() {
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

    bg.ne = E.len();
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
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn augmenting_path_test() {
        todo!()
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
}
