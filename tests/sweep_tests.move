#[test_only]
module sweep::sweep_tests;

use sweep::sat;

#[test]
fun test_literal_operations() {
    use grid::point;
    use sui::vec_map;

    let p1 = point::new(0, 0);
    let lit_pos = sat::literal_new(p1, true);
    let lit_neg = sat::literal_negate(&lit_pos);

    let mut assignment = vec_map::empty();
    assignment.insert(p1, true);

    assert!(sat::literal_is_satisfied(&lit_pos, &assignment), 0);
    assert!(sat::literal_is_falsified(&lit_neg, &assignment), 1);

    assignment.remove(&p1);
    assignment.insert(p1, false);
    assert!(sat::literal_is_satisfied(&lit_neg, &assignment), 2);
    assert!(sat::literal_is_falsified(&lit_pos, &assignment), 3);
}

#[test]
fun test_clause_operations() {
    use grid::point;
    use sui::vec_map;

    let p1 = point::new(0, 0);
    let p2 = point::new(0, 1);

    let lit1 = sat::literal_new(p1, true);
    let lit2 = sat::literal_new(p2, false);
    let clause = sat::clause_new(vector[lit1, lit2]);

    let mut assignment = vec_map::empty();

    // Empty assignment - clause not satisfied or falsified
    assert!(!sat::clause_is_satisfied(&clause, &assignment), 0);
    assert!(!sat::clause_is_falsified(&clause, &assignment), 1);

    // First literal satisfied
    assignment.insert(p1, true);
    assert!(sat::clause_is_satisfied(&clause, &assignment), 2);

    // Both literals falsified
    assignment.remove(&p1);
    assignment.insert(p1, false);
    assignment.insert(p2, true);
    assert!(sat::clause_is_falsified(&clause, &assignment), 3);
}
