#[test_only]
module sweep::sweep_tests;

use grid::point;
use sui::vec_map;
use sui_system::test_runner;
use sweep::{game, sat};

#[test]
fun test_literal_operations() {
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

#[test]
fun test_game_initialization() {
    let mut runner = test_runner::new().build();
    let game = game::new(3, 2, runner.ctx());

    assert!(game.turn() == 0, 0);
    assert!(game.mines() == 2, 1);
    assert!(game.revealed() == 0, 2);

    let grid = game.grid();

    // Check all tiles are initially hidden
    let (width, height) = (grid.rows(), grid.cols());
    width.do!(|x| {
        height.do!(|y| {
            assert!(game.get_square(x, y).is_none(), 3);
        });
    });

    game.destroy();
    runner.finish();
}

#[test]
fun test_reveal_tile() {
    let mut runner = test_runner::new().build();
    let mut rng = sui::random::new_generator_from_seed_for_testing(b"test_seed");
    let mut game = game::new(3, 2, runner.ctx());

    game.reveal(1, 1, &mut rng); // Reveal center tile

    assert!(game.turn() == 1, 0);
    assert!(game.revealed() == 1, 1);

    // Check tile is revealed
    assert!(game.get_square(1, 1).is_some(), 2);

    game.destroy();
    runner.finish();
}

#[test]
fun test_get_neighbors_corner() {
    let mut runner = test_runner::new().build();
    let game = game::new(3, 2, runner.ctx());

    let corner_pos = grid::point::new(0, 0);
    let neighbors = game::get_neighbors(&game, corner_pos);

    // Corner should have 3 neighbors
    assert!(neighbors.length() == 3, 0);

    game.destroy();
    runner.finish();
}

#[test]
fun test_get_neighbors_center() {
    let mut runner = test_runner::new().build();
    let game = game::new(3, 2, runner.ctx());

    let center_pos = grid::point::new(1, 1);
    let neighbors = game::get_neighbors(&game, center_pos);

    // Center should have 8 neighbors
    assert!(neighbors.length() == 8, 0);

    game.destroy();
    runner.finish();
}

#[test]
fun test_get_neighbors_edge() {
    let mut runner = test_runner::new().build();
    let game = game::new(3, 2, runner.ctx());

    let edge_pos = grid::point::new(1, 0);
    let neighbors = game::get_neighbors(&game, edge_pos);

    // Edge should have 5 neighbors
    assert!(neighbors.length() == 5, 0);

    game.destroy();
    runner.finish();
}

#[test]
#[expected_failure]
fun test_reveal_already_revealed() {
    let mut runner = test_runner::new().build();
    let mut rng = sui::random::new_generator_from_seed_for_testing(b"test_seed");
    let mut game = game::new(3, 2, runner.ctx());

    game.reveal(1, 1, &mut rng);
    game.reveal(1, 1, &mut rng); // Should abort

    game.destroy();
    runner.finish();
}

#[test]
fun test_first_click_safety() {
    let mut rng = sui::random::new_generator_from_seed_for_testing(b"test_seed");
    let mut runner = test_runner::new().build();
    let mut game = game::new(3, 2, runner.ctx());

    // First click should never be a mine
    let result = game.reveal(1, 1, &mut rng);
    assert!(result == true, 0); // Game should continue

    // Should be revealed
    assert!(game.get_square(1, 1).is_some(), 1);

    game.destroy();
    runner.finish();
}
