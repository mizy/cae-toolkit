use structural_core::{solve, LinearSystem};
fn system() -> LinearSystem {
    LinearSystem {
        row_ptr: vec![0, 2, 5, 7],
        col_idx: vec![0, 1, 0, 1, 2, 1, 2],
        values: vec![4., 1., 1., 3., 1., 1., 2.],
        rhs: vec![6., 10., 8.],
    }
}
#[test]
fn cpu_known_solution_and_iteration_limit() {
    let a = system();
    let r = pollster::block_on(solve(&a, "cpu", 1e-8, 100)).unwrap();
    assert!(r.converged);
    for (u, want) in r.solution.iter().zip([1., 2., 3.]) {
        assert!((u - want).abs() < 1e-7);
    }
    assert!(
        !pollster::block_on(solve(&a, "cpu", 1e-8, 1))
            .unwrap()
            .converged
    );
}
#[test]
fn reject_invalid_and_nonsymmetric_matrix() {
    let mut a = system();
    a.col_idx[0] = 99;
    assert!(a.validate().is_err());
    let mut a = system();
    a.values[1] = 2.;
    assert!(a.validate().is_err());
}
#[test]
#[ignore = "requires a real GPU adapter"]
fn gpu_known_solution_and_true_residual() {
    let a = system();
    let r = pollster::block_on(solve(&a, "gpu", 1e-5, 2000)).unwrap();
    assert!(r.converged);
    assert!(r.relative_residual < 1e-5);
    for (u, want) in r.solution.iter().zip([1., 2., 3.]) {
        assert!((u - want).abs() < 1e-4);
    }
}
