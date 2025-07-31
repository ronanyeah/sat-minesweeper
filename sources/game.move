/// SAT-based minesweeper game implementation using constraint solving
module sweep::game;

use grid::{grid::{Self, Grid}, point::{Self, Point}};
use sui::random::RandomGenerator;
use sweep::sat;

// --- Game State Enums ---

/// Visual state of each tile from player's perspective
public enum Tile has copy, drop, store {
    Hidden,
    Revealed(u8),
}

// --- Core Game Structures ---

/// Main SAT-based minesweeper game state
public struct SATMinesweeper has key, store {
    id: UID,
    grid: Grid<Tile>,
    turn: u16,
    mines: u16,
    revealed: u16,
}

public fun to_string(tile: &Tile): std::string::String {
    match (tile) {
        Tile::Revealed(n) => (*n).to_string(),
        Tile::Hidden => b" ".to_string(),
    }
}

// --- Game Initialization ---

/// Create new SAT-based minesweeper game
public fun new(size: u16, mines: u16, ctx: &mut TxContext): SATMinesweeper {
    let id = object::new(ctx);
    SATMinesweeper {
        id,
        grid: grid::tabulate!(size, size, |_, _| Tile::Hidden),
        turn: 0,
        mines,
        revealed: 0,
    }
}

public fun destroy(game: SATMinesweeper) {
    let SATMinesweeper {
        id,
        grid: _,
        turn: _,
        mines: _,
        revealed: _,
    } = game;
    object::delete(id);
}

public fun get_square(game: &SATMinesweeper, x: u16, y: u16): Option<u8> {
    match (game.grid[x, y]) {
        Tile::Revealed(n) => option::some(n),
        Tile::Hidden => option::none(),
    }
}

public fun grid(game: &SATMinesweeper): Grid<Tile> {
    game.grid
}

public fun turn(game: &SATMinesweeper): u16 {
    game.turn
}

public fun revealed(game: &SATMinesweeper): u16 {
    game.revealed
}

public fun mines(game: &SATMinesweeper): u16 {
    game.mines
}

// --- Game Actions ---

#[allow(lint(public_random))]
/// Reveal a tile using stateless probabilistic sampling
public fun reveal(game: &mut SATMinesweeper, x: u16, y: u16, rng: &mut RandomGenerator): bool {
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
        sat::analyze_with_deductions(&first_move_instance, rng)
    } else {
        let fresh_instance = generate_constraints_from_visible(game);
        sat::analyze_with_deductions(&fresh_instance, rng)
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
        flood_fill_zeros_with_analysis(game, x, y, &analysis, rng);
    };

    true // Continue game
}

public fun render(game: &SATMinesweeper): std::string::String {
    game.grid.to_string!()
}

// --- Helper Functions ---

/// Get valid neighbors within grid bounds
public fun get_neighbors(game: &SATMinesweeper, pos: Point): vector<Point> {
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
    rng: &mut RandomGenerator,
) {
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
                    reveal(game, nx, ny, rng);
                }
            },
            _ => (), // Skip already revealed or flagged tiles
        }
    });
}

/// Generate SAT constraints based only on currently visible (revealed) tiles
fun generate_constraints_from_visible(game: &SATMinesweeper): sat::SATInstance {
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
