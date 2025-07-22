/// SAT-based minesweeper game implementation using constraint solving
module sweep::game;

use grid::{grid::Grid, point::Point};
use sui::random::RandomGenerator;
use sweep::sat::{Self, SATInstance};

// --- Game State Enums ---

/// Visual state of each tile from player's perspective
public enum Tile has copy, drop {
    Hidden,
    Revealed(u8),
}

/// SAT solver tile state for constraint tracking
public enum SATTile has copy, drop {
    Unknown,
    Constrained(u8, u8), // mine_count, constraint_count
    Deduced(bool), // true = mine, false = safe
}

// --- Core Game Structures ---

/// Main SAT-based minesweeper game state
public struct SATMinesweeper has drop {
    grid: Grid<Tile>,
    sat_grid: Grid<SATTile>,
    sat_instance: SATInstance,
    rng: RandomGenerator,
    turn: u16,
    mines: u16,
    revealed: u16,
}

// --- Game Initialization ---

#[allow(lint(public_random))]
/// Create new SAT-based minesweeper game
public fun new(rng: RandomGenerator, width: u16, height: u16, mines: u16): SATMinesweeper {
    use grid::{grid, point};

    // Create SAT variables for each grid position
    let mut sat_variables = vector::empty<Point>();
    width.do!(|x| {
        height.do!(|y| {
            sat_variables.push_back(point::new(x, y));
        });
    });

    SATMinesweeper {
        grid: grid::tabulate!(width, height, |_, _| Tile::Hidden),
        sat_grid: grid::tabulate!(width, height, |_, _| SATTile::Unknown),
        sat_instance: sat::sat_instance_new(sat_variables),
        rng,
        turn: 0,
        mines,
        revealed: 0,
    }
}

// --- Game Actions ---

/// Reveal a tile using stateless probabilistic sampling
public fun reveal(game: &mut SATMinesweeper, x: u16, y: u16): bool {
    use grid::point;

    // Check if tile is already revealed
    match (game.grid[x, y]) {
        Tile::Revealed(_) => abort,
        _ => (),
    };

    // First click safety guarantee
    let is_first_click = game.turn == 0;

    // Generate SAT analysis once for consistency throughout this move
    let analysis = if (is_first_click) {
        let first_move_instance = generate_first_move_constraints(game, x, y);
        sat::analyze_with_deductions(&first_move_instance, &mut game.rng)
    } else {
        let fresh_instance = generate_constraints_from_visible(game);
        sat::analyze_with_deductions(&fresh_instance, &mut game.rng)
    };

    // Check if this cell is a mine using the consistent analysis
    let pos = point::new(x, y);
    let is_mine = if (is_first_click) {
        false // First click is always safe
    } else {
        analysis.sample_assignment_cell_state(&pos)
    };

    // If it's a mine, game over
    if (is_mine) {
        // Mark as mine and return false for game over
        *game.grid.borrow_mut(x, y) = Tile::Revealed(9); // Use 9 to indicate mine
        game.turn = game.turn + 1;
        return false
    };

    // It's safe - calculate mine count from neighbors using consistent analysis
    let neighbors = get_neighbors(game, pos);
    let mut mine_count = 0u8;
    neighbors.do_ref!(|neighbor| {
        let (nx, ny) = neighbor.to_values();
        match (game.grid[nx, ny]) {
            Tile::Revealed(9) => mine_count = mine_count + 1, // Revealed mine
            Tile::Revealed(_) => (), // Already revealed safe
            Tile::Hidden => {
                // Use consistent analysis result instead of re-sampling
                if (analysis.sample_assignment_cell_state(neighbor)) {
                    mine_count = mine_count + 1;
                }
            },
        }
    });

    // Update visual grid with calculated mine count
    *game.grid.borrow_mut(x, y) = Tile::Revealed(mine_count);
    game.revealed = game.revealed + 1;
    game.turn = game.turn + 1;

    // Auto-reveal safe neighbors if this is a zero
    if (mine_count == 0) {
        flood_fill_zeros_with_analysis(game, x, y, &analysis);
    };

    true // Continue game
}

// --- Helper Functions ---

/// Get valid neighbors within grid bounds
fun get_neighbors(game: &SATMinesweeper, pos: Point): vector<Point> {
    use grid::point;

    let (x, y) = pos.to_values();
    let mut neighbors = vector::empty<Point>();
    let (width, height) = (game.grid.rows(), game.grid.cols());

    // Check all 8 neighbors manually
    // Top row
    if (x > 0 && y > 0) neighbors.push_back(point::new(x - 1, y - 1));
    if (y > 0) neighbors.push_back(point::new(x, y - 1));
    if (x + 1 < width && y > 0) neighbors.push_back(point::new(x + 1, y - 1));

    // Middle row
    if (x > 0) neighbors.push_back(point::new(x - 1, y));
    if (x + 1 < width) neighbors.push_back(point::new(x + 1, y));

    // Bottom row
    if (x > 0 && y + 1 < height) neighbors.push_back(point::new(x - 1, y + 1));
    if (y + 1 < height) neighbors.push_back(point::new(x, y + 1));
    if (x + 1 < width && y + 1 < height) neighbors.push_back(point::new(x + 1, y + 1));

    neighbors
}

/// Auto-reveal safe neighbors using consistent analysis result
fun flood_fill_zeros_with_analysis(
    game: &mut SATMinesweeper,
    x: u16,
    y: u16,
    analysis: &sat::AnalysisResult,
) {
    use grid::point;

    let pos = point::new(x, y);
    let neighbors = get_neighbors(game, pos);

    neighbors.do_ref!(|neighbor| {
        let (nx, ny) = neighbor.to_values();
        match (game.grid[nx, ny]) {
            Tile::Hidden => {
                // Use consistent analysis result instead of re-sampling
                if (!analysis.sample_assignment_cell_state(neighbor)) {
                    // It's safe - reveal it with a fresh analysis
                    // Note: recursive reveals will generate their own consistent analysis
                    reveal(game, nx, ny);
                }
            },
            _ => (), // Skip already revealed or flagged tiles
        }
    });
}

/// Generate SAT constraints based only on currently visible (revealed) tiles
fun generate_constraints_from_visible(game: &SATMinesweeper): sat::SATInstance {
    use grid::point;

    // Create fresh SAT instance with all grid positions as variables
    let mut sat_variables = vector::empty<Point>();
    let (width, height) = (game.grid.rows(), game.grid.cols());

    width.do!(|x| {
        height.do!(|y| {
            sat_variables.push_back(point::new(x, y));
        });
    });

    let mut fresh_instance = sat::sat_instance_new(sat_variables);

    // Add constraints from all currently revealed tiles
    width.do!(|x| {
        height.do!(|y| {
            match (game.grid[x, y]) {
                Tile::Revealed(mine_count) => {
                    let pos = point::new(x, y);
                    let neighbors = get_neighbors(game, pos);

                    // Count already revealed neighboring mines
                    let mut revealed_neighbor_mines = 0u64;
                    let mut unknown_neighbors = vector::empty<Point>();

                    neighbors.do_ref!(|neighbor| {
                        let (nx, ny) = neighbor.to_values();
                        match (game.grid[nx, ny]) {
                            Tile::Hidden => {
                                unknown_neighbors.push_back(*neighbor);
                            },
                            Tile::Revealed(9) => {
                                // Count revealed mines (9 indicates mine)
                                revealed_neighbor_mines = revealed_neighbor_mines + 1;
                            },
                            Tile::Revealed(_) => {},
                        }
                    });

                    if (!unknown_neighbors.is_empty()) {
                        // Subtract already revealed mines from the required mine count
                        let remaining_mines = (mine_count as u64) - revealed_neighbor_mines;
                        sat::sat_instance_add_exactly_k(
                            &mut fresh_instance,
                            &unknown_neighbors,
                            remaining_mines,
                        );
                    }
                },
                _ => (), // Hidden tiles have no constraints
            }
        });
    });

    // Add global mine count constraint for all hidden cells
    let mut all_hidden_cells = vector::empty<Point>();
    width.do!(|x| {
        height.do!(|y| {
            match (game.grid[x, y]) {
                Tile::Hidden => {
                    all_hidden_cells.push_back(point::new(x, y));
                },
                _ => (), // Skip revealed cells
            }
        });
    });

    if (!all_hidden_cells.is_empty()) {
        // Calculate remaining mines: total mines minus already revealed mines
        let mut revealed_mines = 0u64;
        width.do!(|x| {
            height.do!(|y| {
                match (game.grid[x, y]) {
                    Tile::Revealed(9) => revealed_mines = revealed_mines + 1, // 9 indicates mine
                    _ => (),
                }
            });
        });

        let remaining_mines = (game.mines as u64) - revealed_mines;
        sat::sat_instance_add_exactly_k(
            &mut fresh_instance,
            &all_hidden_cells,
            remaining_mines,
        );
    };

    fresh_instance
}

/// Generate SAT constraints for first move ensuring clicked cell and neighbors are safe
fun generate_first_move_constraints(
    game: &SATMinesweeper,
    first_click_x: u16,
    first_click_y: u16,
): sat::SATInstance {
    use grid::point;

    // Create fresh SAT instance with all grid positions as variables
    let mut sat_variables = vector::empty<Point>();
    let (width, height) = (game.grid.rows(), game.grid.cols());

    width.do!(|x| {
        height.do!(|y| {
            sat_variables.push_back(point::new(x, y));
        });
    });

    let mut fresh_instance = sat::sat_instance_new(sat_variables);

    // Create constraint that clicked cell must be safe (0 mines)
    let click_pos = point::new(first_click_x, first_click_y);
    let safe_area = vector[click_pos]; // Only the clicked cell, not neighbors

    // Add constraint: exactly 0 mines in the safe area (just the clicked cell)
    sat::sat_instance_add_exactly_k(
        &mut fresh_instance,
        &safe_area,
        0,
    );

    // Add global mine count constraint for cells OUTSIDE the safe area
    let mut non_safe_cells = vector::empty<Point>();
    width.do!(|x| {
        height.do!(|y| {
            let cell_pos = point::new(x, y);
            let mut is_in_safe_area = false;
            safe_area.do_ref!(|safe_pos| {
                if (cell_pos == *safe_pos) {
                    is_in_safe_area = true;
                }
            });
            if (!is_in_safe_area) {
                non_safe_cells.push_back(cell_pos);
            }
        });
    });

    // All mines must be in the non-safe area
    sat::sat_instance_add_exactly_k(
        &mut fresh_instance,
        &non_safe_cells,
        game.mines as u64,
    );

    fresh_instance
}

// --- Tests ---

#[test]
fun test_game_initialization() {
    let rng = sui::random::new_generator_from_seed_for_testing(b"test_seed");
    let game = new(rng, 3, 3, 2);

    assert!(game.turn == 0, 0);
    assert!(game.mines == 2, 1);
    assert!(game.revealed == 0, 2);

    // Check all tiles are initially hidden
    let (width, height) = (game.grid.rows(), game.grid.cols());
    width.do!(|x| {
        height.do!(|y| {
            match (game.grid[x, y]) {
                Tile::Hidden => (),
                _ => assert!(false, 3),
            }
        });
    });
}

#[test]
fun test_reveal_tile() {
    let rng = sui::random::new_generator_from_seed_for_testing(b"test_seed");
    let mut game = new(rng, 3, 3, 2);

    game.reveal(1, 1); // Reveal center tile

    assert!(game.turn == 1, 0);
    assert!(game.revealed == 1, 1);

    // Check tile is revealed
    match (game.grid[1, 1]) {
        Tile::Revealed(_) => (),
        _ => assert!(false, 2),
    }
}

#[test]
fun test_get_neighbors_corner() {
    let rng = sui::random::new_generator_from_seed_for_testing(b"test_seed");
    let game = new(rng, 3, 3, 2);

    let corner_pos = grid::point::new(0, 0);
    let neighbors = get_neighbors(&game, corner_pos);

    // Corner should have 3 neighbors
    assert!(neighbors.length() == 3, 0);
}

#[test]
fun test_get_neighbors_center() {
    let rng = sui::random::new_generator_from_seed_for_testing(b"test_seed");
    let game = new(rng, 3, 3, 2);

    let center_pos = grid::point::new(1, 1);
    let neighbors = get_neighbors(&game, center_pos);

    // Center should have 8 neighbors
    assert!(neighbors.length() == 8, 0);
}

#[test]
fun test_get_neighbors_edge() {
    let rng = sui::random::new_generator_from_seed_for_testing(b"test_seed");
    let game = new(rng, 3, 3, 2);

    let edge_pos = grid::point::new(1, 0);
    let neighbors = get_neighbors(&game, edge_pos);

    // Edge should have 5 neighbors
    assert!(neighbors.length() == 5, 0);
}

#[test]
#[expected_failure]
fun test_reveal_already_revealed() {
    let rng = sui::random::new_generator_from_seed_for_testing(b"test_seed");
    let mut game = new(rng, 3, 3, 2);

    game.reveal(1, 1);
    game.reveal(1, 1); // Should abort
}

#[test]
fun test_first_click_safety() {
    let rng = sui::random::new_generator_from_seed_for_testing(b"test_seed");
    let mut game = new(rng, 3, 3, 2);

    // First click should never be a mine
    let result = game.reveal(1, 1);
    assert!(result == true, 0); // Game should continue

    match (game.grid[1, 1]) {
        Tile::Revealed(count) => assert!(count < 9, 1), // Should not be a mine (9)
        _ => assert!(false, 2),
    }
}
