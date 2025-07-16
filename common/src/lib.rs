use itertools::Itertools;
use std::collections::{HashMap, HashSet, VecDeque};
use varisat::{CnfFormula, ExtendFormula, Lit, Solver, Var};

/// Represents a 2D coordinate on the minesweeper board.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Point {
    pub x: usize,
    pub y: usize,
}

/// The visible state of a single cell on the board.
/// This is the only state that is "remembered" between moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Cell {
    Hidden,
    Revealed(u8), // The u8 is the number of adjacent mines.
}

/// The main game struct, holding the visible board state and game rules.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Game {
    pub width: usize,
    pub height: usize,
    /// The visible state of the board. This is the "ground truth" for the solver.
    pub board: Vec<Vec<Cell>>,
    /// The total number of mines the board contains. This acts as a global constraint.
    pub total_mines: usize,
    /// Tracks the current status of the game (playing, won, lost).
    pub game_state: GameState,
}

/// Represents a single constraint for the SAT solver.
/// For example, a revealed '1' creates a constraint that exactly 1 mine must be among its hidden neighbors.
#[derive(Debug, Clone)]
pub struct Constraint {
    /// The set of hidden cells this constraint applies to.
    pub variables: Vec<Point>,
    /// The exact number of mines that must be present among the constraint's variables.
    pub required_mines: usize,
}

/// Collection of all constraints used by the SAT solver.
pub struct Constraints {
    /// Set of all hidden cells (variables) in the current game state.
    pub variables: HashSet<Point>,
    /// Constraints from revealed numbered cells.
    pub local_constraints: Vec<Constraint>,
    /// Global constraint ensuring the total mine count is correct.
    pub global_constraint: Constraint,
}

/// The possible outcomes of a solver's analysis for a single hidden cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeducedState {
    ForcedMine,   // All valid solutions require this cell to be a mine.
    ForcedSafe,   // All valid solutions require this cell to be safe.
    Undetermined, // Valid solutions exist for this cell being either a mine or safe.
}

/// The result returned by the core constraint solver.
pub type SolverResult = HashMap<Point, DeducedState>;

/// The result of a full board analysis, containing both logical deductions and a sample solution.
pub struct AnalysisResult {
    /// The deduced state for every hidden cell (variable).
    pub deductions: SolverResult,
    /// A single, valid, concrete assignment of mines to all hidden cells that satisfies all constraints.
    pub sample_assignment: HashMap<Point, bool>,
}

/// Represents the current state of the game.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum GameState {
    Playing,
    Won,
    Lost,
}

// --- Game Implementation (orchestrating the logic) ---

impl Game {
    pub fn new(width: usize, height: usize, total_mines: usize) -> Self {
        if total_mines >= width * height {
            panic!("Total mines must be less than the number of cells on the board.");
        }
        Game {
            width,
            height,
            board: vec![vec![Cell::Hidden; width]; height],
            total_mines,
            game_state: GameState::Playing,
        }
    }

    /// Deserializes a game state from bytes.
    pub fn deserialize(bts: &Vec<u8>) -> Self {
        bcs::from_bytes(bts).unwrap()
    }

    /// Serializes the game state to bytes.
    pub fn serialize(&self) -> Vec<u8> {
        bcs::to_bytes(self).unwrap()
    }

    /// The primary function called when a player clicks to reveal a cell.
    ///
    /// This function orchestrates the entire stateless process for a single move:
    /// 1. Handles pre-checks and the special case for the first move.
    /// 2. Builds the current set of constraints from the visible board.
    /// 3. Calls the solver to deduce, solve, and sample a valid board state.
    /// 4. Updates the board with the revealed cell and its number.
    /// 5. Triggers a cascading flood fill if a '0' is revealed.
    /// 6. Checks for a win condition.
    pub fn reveal_cell(&mut self, at: Point) -> anyhow::Result<bool> {
        // --- 1. Pre-checks and Special Cases ---
        if !matches!(self.board[at.y][at.x], Cell::Hidden) {
            return Ok(true);
        }
        if self.game_state != GameState::Playing {
            anyhow::bail!("game_ended");
        }

        // --- 2. Build Constraints ---
        let constraints = if self.is_first_move() {
            self.build_constraints_first_move(at)
        } else {
            self.build_constraints()
        };

        // --- 3. Analyze Board State and Process Move ---
        self.process_move_with_constraints(at, constraints)
    }

    /// Helper to determine if this is the very first move of the game.
    fn is_first_move(&self) -> bool {
        self.board
            .iter()
            .all(|row| row.iter().all(|cell| matches!(cell, Cell::Hidden)))
    }

    /// Processes a move with the given constraints, handling the core game logic.
    fn process_move_with_constraints(
        &mut self,
        at: Point,
        constraints: Constraints,
    ) -> anyhow::Result<bool> {
        // --- Analyze Board State (Solve, Deduce, and Sample) ---
        let analysis = analyze_board_state(&constraints)?;

        // --- Use the sampled assignment for this move's outcome ---
        let assignment = analysis.sample_assignment;

        if *assignment.get(&at).unwrap_or(&false) {
            self.game_state = GameState::Lost;
            return Ok(false);
        }

        // --- Reveal and Cascade ---
        let mine_count = self.count_adjacent_mines(at, &assignment);

        if mine_count == 0 {
            self.flood_fill_reveal(at, &assignment);
        } else {
            self.board[at.y][at.x] = Cell::Revealed(mine_count);
        };

        // --- Check for Win Condition ---
        if self.check_win_condition() {
            self.game_state = GameState::Won;
        }

        Ok(true)
    }

    /// Counts mines around a point using a definitive `assignment` map.
    fn count_adjacent_mines(&self, point: Point, assignment: &HashMap<Point, bool>) -> u8 {
        let mut count = 0;
        for neighbor in self.get_neighbors(point) {
            // A neighbor is a mine if it's true in the assignment map.
            if *assignment.get(&neighbor).unwrap_or(&false) {
                count += 1;
            }
        }
        count
    }

    /// Checks if all non-mine cells have been revealed, resulting in a win.
    /// This is done by building the current constraints and checking if the solver
    /// determines that all remaining hidden cells are *forced* to be mines. This is
    /// the only way to be 100% certain of a win state.
    pub fn check_win_condition(&self) -> bool {
        // Build the constraints based on the current visible board.
        let constraints = self.build_constraints();

        // The game is won if the number of hidden cells matches the total mines.
        if constraints.variables.is_empty() {
            let remaining = self
                .board
                .iter()
                .flatten()
                .filter(|c| matches!(c, Cell::Hidden))
                .count();
            return remaining == self.total_mines;
        }

        // Analyze the board state to get deductions.
        let analysis = analyze_board_state(&constraints).expect("analysis crash");

        // The game is won if and only if every single remaining hidden cell
        // is a forced mine. If even one is undetermined or safe, the game is not over.
        analysis
            .deductions
            .values()
            .all(|&state| state == DeducedState::ForcedMine)
    }

    /// A helper function to translate the current visible `board` into a formal
    /// set of constraints that the solver can understand.
    ///
    /// It identifies all hidden cells and generates a `Constraint` for each
    /// revealed number, as well as a single global constraint for the total mine count.
    pub fn build_constraints(&self) -> Constraints {
        let mut variables = HashSet::new();
        let mut local_constraints = Vec::new();

        // First pass: Identify all variables (hidden cells) and generate local constraints.
        for y in 0..self.height {
            for x in 0..self.width {
                let current_point = Point { x, y };
                let cell = self.board[y][x];

                match cell {
                    Cell::Hidden => {
                        variables.insert(current_point);
                    }
                    Cell::Revealed(number) => {
                        let mut constraint_variables = Vec::new();

                        // Analyze neighbors to build the constraint
                        for neighbor_point in self.get_neighbors(current_point) {
                            match self.board[neighbor_point.y][neighbor_point.x] {
                                // A hidden neighbor is an unknown variable for this constraint.
                                Cell::Hidden => {
                                    constraint_variables.push(neighbor_point);
                                }
                                // A revealed neighbor provides no information for this constraint.
                                Cell::Revealed(_) => {}
                            }
                        }

                        // Only create a constraint if there are still unknown variables.
                        if !constraint_variables.is_empty() {
                            local_constraints.push(Constraint {
                                variables: constraint_variables,
                                required_mines: number as usize,
                            });
                        }
                    }
                }
            }
        }

        // Second pass: Create the global constraint.
        // This constraint applies to all hidden cells on the board.

        let global_vars: Vec<Point> = variables.iter().cloned().collect();

        let global_constraint = Constraint {
            variables: global_vars,
            required_mines: self.total_mines,
        };

        Constraints {
            variables,
            local_constraints,
            global_constraint,
        }
    }

    /// Builds constraints for the first move, ensuring the clicked cell and its neighbors are safe.
    pub fn build_constraints_first_move(&self, first_click: Point) -> Constraints {
        let mut variables = HashSet::new();
        let mut local_constraints = Vec::new();

        // All cells are variables on the first move
        for y in 0..self.height {
            for x in 0..self.width {
                variables.insert(Point { x, y });
            }
        }

        // Create a constraint that the clicked cell and its neighbors must be safe (0 mines)
        let mut safe_area = vec![first_click];
        for neighbor in self.get_neighbors(first_click) {
            safe_area.push(neighbor);
        }

        // This constraint ensures the safe area has 0 mines
        local_constraints.push(Constraint {
            variables: safe_area,
            required_mines: 0,
        });

        // Global constraint: total mines on the board
        let global_vars: Vec<Point> = variables.iter().cloned().collect();
        let global_constraint = Constraint {
            variables: global_vars,
            required_mines: self.total_mines,
        };

        Constraints {
            variables,
            local_constraints,
            global_constraint,
        }
    }

    /// Performs flood fill reveal for cells with 0 adjacent mines using a sampled assignment.
    fn flood_fill_reveal(
        &mut self,
        start_point: Point,
        assignment: &HashMap<Point, bool>,
    ) -> Vec<Point> {
        let mut revealed_points = Vec::new();
        let mut queue = VecDeque::from([start_point]);
        let mut visited = HashSet::from([start_point]);

        while let Some(point) = queue.pop_front() {
            // Only process hidden cells
            if !matches!(self.board[point.y][point.x], Cell::Hidden) {
                continue;
            }

            // Calculate the mine count using the assignment
            let mine_count = self.count_adjacent_mines(point, assignment);
            self.board[point.y][point.x] = Cell::Revealed(mine_count);
            revealed_points.push(point);

            // If it's a 0, add its neighbors to the queue
            if mine_count == 0 {
                for neighbor in self.get_neighbors(point) {
                    if !visited.contains(&neighbor)
                        && matches!(self.board[neighbor.y][neighbor.x], Cell::Hidden)
                    {
                        queue.push_back(neighbor);
                        visited.insert(neighbor);
                    }
                }
            }
        }

        revealed_points
    }

    /// A helper function to get all valid neighbor coordinates for a given point.
    /// It correctly handles board edges and corners.
    fn get_neighbors(&self, point: Point) -> impl Iterator<Item = Point> {
        let width = self.width;
        let height = self.height;

        // Define potential neighbor offsets (from -1 to 1 in both x and y)
        (-1..=1).flat_map(move |dy| {
            (-1..=1).filter_map(move |dx| {
                // Skip the center point itself (dx=0, dy=0)
                if dx == 0 && dy == 0 {
                    return None;
                }

                // Calculate potential neighbor coordinates
                let nx = point.x as isize + dx;
                let ny = point.y as isize + dy;

                // Check if the neighbor is within board bounds
                if nx >= 0 && nx < width as isize && ny >= 0 && ny < height as isize {
                    Some(Point {
                        x: nx as usize,
                        y: ny as usize,
                    })
                } else {
                    None
                }
            })
        })
    }
}

// --- The Main `analyze_board_state` Function ---

/// The core analytical function of the game. It takes the current board state
/// (represented by variables and constraints) and performs a full SAT analysis.
///
/// It produces two key pieces of information:
/// 1. `deductions`: A map indicating whether each hidden cell is `ForcedMine`,
///    `ForcedSafe`, or `Undetermined`. This is found by testing each possibility.
/// 2. `sample_assignment`: A single, complete, valid assignment of mines and safe
///    cells that satisfies all constraints. This is generated efficiently from the
///    solver's internal model.
///
/// This function replaces the need for a separate sampling step (like backtracking)
/// by leveraging the SAT solver for both deduction and sampling, which is significantly
/// more performant and simpler.
pub fn analyze_board_state(constraints: &Constraints) -> anyhow::Result<AnalysisResult> {
    let Constraints {
        variables,
        local_constraints,
        global_constraint,
    } = constraints;

    let mut solver = Solver::new();
    let mut var_map: HashMap<Point, Var> = HashMap::new();

    // 1. Allocate SAT variables for each Point
    for &point in variables {
        var_map.insert(point, solver.new_var());
    }

    // 2. Encode all constraints as CNF using batch operations
    let mut formula = CnfFormula::new();
    for constraint in local_constraints
        .iter()
        .chain(std::iter::once(global_constraint))
    {
        let lits: Vec<Lit> = constraint
            .variables
            .iter()
            .filter_map(|p| var_map.get(p).copied().map(|v| Lit::from_var(v, true)))
            .collect();
        encode_exactly_k_to_formula(&mut formula, &mut solver, &lits, constraint.required_mines);
    }

    // Add all clauses to solver at once
    solver.add_formula(&formula);

    // 3. Check if the problem is solvable.
    if !solver.solve()? {
        anyhow::bail!("solve_fail");
    };

    // 4. Generate a single valid assignment (a "model") from the solver.
    let model = solver.model().ok_or(anyhow::anyhow!("solver_model_fail"))?;
    let mut sample_assignment = HashMap::new();
    for (&point, &var) in &var_map {
        // The model contains the literals that are true.
        // Check if the literal for our variable being a mine is in the model.
        let is_mine = model.contains(&Lit::from_var(var, true));
        sample_assignment.insert(point, is_mine);
    }

    // 5. Deduce the state for each variable by testing both assignments efficiently.
    let mut deductions = SolverResult::new();
    for (&point, &var) in &var_map {
        let lit_mine = Lit::from_var(var, true);
        let lit_safe = Lit::from_var(var, false);

        // Use incremental solving with assumptions.
        // Test if the cell can be a mine.
        let mine_possible = {
            solver.assume(&[lit_mine]);
            let result = solver.solve().unwrap_or(false);
            // Clear assumptions for next test.
            solver.assume(&[]);
            result
        };

        // Test if the cell can be safe.
        let safe_possible = {
            solver.assume(&[lit_safe]);
            let result = solver.solve().unwrap_or(false);
            // Clear assumptions for next test.
            solver.assume(&[]);
            result
        };

        let state = match (mine_possible, safe_possible) {
            (true, true) => DeducedState::Undetermined,
            (true, false) => DeducedState::ForcedMine,
            (false, true) => DeducedState::ForcedSafe,
            (false, false) => anyhow::bail!("state_collision"),
        };
        deductions.insert(point, state);
    }

    Ok(AnalysisResult {
        deductions,
        sample_assignment,
    })
}

/// Encodes an "exactly k" constraint into the CNF formula.
fn encode_exactly_k_to_formula(
    formula: &mut CnfFormula,
    solver: &mut Solver,
    vars: &[Lit],
    k: usize,
) {
    encode_at_most_k_to_formula(formula, solver, vars, k);
    encode_at_least_k_to_formula(formula, solver, vars, k);
}

/// Encodes an "at most k" constraint into the CNF formula.
fn encode_at_most_k_to_formula(
    formula: &mut CnfFormula,
    solver: &mut Solver,
    vars: &[Lit],
    k: usize,
) {
    if k >= vars.len() {
        return; // Always satisfiable.
    }
    if k == 0 {
        // All variables must be false.
        for &lit in vars {
            formula.add_clause(&[!lit]);
        }
        return;
    }

    // Use sequential counter encoding for efficiency
    if vars.len() <= 10 {
        // For small constraints, use naive encoding. to avoid overhead.
        for combo in vars.iter().copied().combinations(k + 1) {
            let clause: Vec<Lit> = combo.iter().map(|&lit| !lit).collect();
            formula.add_clause(&clause);
        }
    } else {
        // Use sequential counter for larger constraints.
        encode_sequential_counter_at_most_k_to_formula(formula, solver, vars, k);
    }
}

/// Encodes an "at least k" constraint into the CNF formula.
fn encode_at_least_k_to_formula(
    formula: &mut CnfFormula,
    solver: &mut Solver,
    vars: &[Lit],
    k: usize,
) {
    if k == 0 {
        return; // Always satisfied.
    }
    if k > vars.len() {
        // Unsatisfiable - add empty clause.
        formula.add_clause(&[]);
        return;
    }

    if vars.len() <= 10 {
        // For small constraints, use naive encoding.
        for combo in vars.iter().copied().combinations(vars.len() - k + 1) {
            formula.add_clause(&combo);
        }
    } else {
        // Use sequential counter for larger constraints.
        encode_sequential_counter_at_least_k_to_formula(formula, solver, vars, k);
    }
}

/// Efficient sequential counter encoding for "at most k" constraints.
fn encode_sequential_counter_at_most_k_to_formula(
    formula: &mut CnfFormula,
    solver: &mut Solver,
    vars: &[Lit],
    k: usize,
) {
    let n = vars.len();
    if n == 0 || k >= n {
        return;
    }
    if k == 0 {
        // All variables must be false.
        for &lit in vars {
            formula.add_clause(&[!lit]);
        }
        return;
    }

    // Create auxiliary variables: s[i][j] for i in 0..n, j in 0..k
    let mut aux_vars = Vec::new();
    for _ in 0..n {
        let mut row = Vec::new();
        for _ in 0..=k {
            row.push(solver.new_var());
        }
        aux_vars.push(row);
    }

    // Base case: s[0][0] iff not x[0], s[0][1] iff x[0]
    formula.add_clause(&[!vars[0], Lit::from_var(aux_vars[0][1], true)]);
    formula.add_clause(&[vars[0], !Lit::from_var(aux_vars[0][1], true)]);
    formula.add_clause(&[Lit::from_var(aux_vars[0][0], true)]);
    if k > 1 {
        for j in 2..=k {
            formula.add_clause(&[!Lit::from_var(aux_vars[0][j], true)]);
        }
    }

    // Recursive case: s[i][j] = s[i-1][j] OR (x[i] AND s[i-1][j-1])
    for i in 1..n {
        for j in 0..=k {
            let s_i_j = Lit::from_var(aux_vars[i][j], true);
            let s_prev_j = Lit::from_var(aux_vars[i - 1][j], true);

            if j == 0 {
                // s[i][0] = s[i-1][0] AND not x[i]
                formula.add_clause(&[!s_i_j, s_prev_j]);
                formula.add_clause(&[!s_i_j, !vars[i]]);
                formula.add_clause(&[!s_prev_j, vars[i], s_i_j]);
            } else {
                let s_prev_j_minus_1 = Lit::from_var(aux_vars[i - 1][j - 1], true);
                // s[i][j] = s[i-1][j] OR (x[i] AND s[i-1][j-1])
                formula.add_clause(&[!s_i_j, s_prev_j, vars[i]]);
                formula.add_clause(&[!s_i_j, s_prev_j, s_prev_j_minus_1]);
                formula.add_clause(&[!s_prev_j, s_i_j]);
                formula.add_clause(&[!vars[i], !s_prev_j_minus_1, s_i_j]);
            }
        }
    }

    // Final constraint: s[n-1][k] must be true
    formula.add_clause(&[Lit::from_var(aux_vars[n - 1][k], true)]);
}

/// Efficient sequential counter encoding for "at least k" constraints.
fn encode_sequential_counter_at_least_k_to_formula(
    formula: &mut CnfFormula,
    solver: &mut Solver,
    vars: &[Lit],
    k: usize,
) {
    // "at least k" is equivalent to "at most (n-k)" negated variables.
    let negated_vars: Vec<Lit> = vars.iter().map(|&lit| !lit).collect();
    encode_sequential_counter_at_most_k_to_formula(formula, solver, &negated_vars, vars.len() - k);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_game_initialization() {
        // Test that a new game is properly initialized with correct dimensions and state
        let game = Game::new(5, 5, 3);
        assert_eq!(game.width, 5);
        assert_eq!(game.height, 5);
        assert_eq!(game.total_mines, 3);
        assert_eq!(game.game_state, GameState::Playing);

        // Verify all cells start as hidden
        for row in &game.board {
            for cell in row {
                assert_eq!(*cell, Cell::Hidden);
            }
        }
    }

    #[test]
    #[should_panic(expected = "Total mines must be less than the number of cells on the board.")]
    fn test_game_initialization_too_many_mines() {
        // Test that creating a game with mines >= total cells panics
        Game::new(3, 3, 9);
    }

    #[test]
    fn test_first_move_always_safe() {
        // Test that the first move is always safe and reveals cells
        let mut game = Game::new(5, 5, 10);
        let result = game.reveal_cell(Point { x: 2, y: 2 }).unwrap();

        // First move should always succeed (not hit a mine)
        assert!(result);
        assert_eq!(game.game_state, GameState::Playing);

        // The clicked cell should be revealed with a number
        assert!(matches!(game.board[2][2], Cell::Revealed(_)));
    }

    #[test]
    fn test_constraint_building() {
        // Test that constraints are built correctly from a partially revealed board
        let mut game = Game::new(3, 3, 2);

        // Set up a board state with one revealed cell
        game.board[1][1] = Cell::Revealed(1);

        let Constraints {
            variables,
            local_constraints,
            global_constraint,
        } = game.build_constraints();

        // Should have 8 variables (9 total cells - 1 revealed)
        assert_eq!(variables.len(), 8);

        // Should have 1 local constraint from the revealed '1' cell
        assert_eq!(local_constraints.len(), 1);

        // Global constraint should match the total mine count
        assert_eq!(global_constraint.required_mines, 2);
    }

    #[test]
    fn test_get_neighbors() {
        // Test that neighbor calculation works correctly for different board positions
        let game = Game::new(3, 3, 1);

        // Corner cell (0,0) should have 3 neighbors
        let corner_neighbors: Vec<Point> = game.get_neighbors(Point { x: 0, y: 0 }).collect();
        assert_eq!(corner_neighbors.len(), 3);

        // Center cell (1,1) should have 8 neighbors
        let center_neighbors: Vec<Point> = game.get_neighbors(Point { x: 1, y: 1 }).collect();
        assert_eq!(center_neighbors.len(), 8);

        // Edge cell (1,0) should have 5 neighbors
        let edge_neighbors: Vec<Point> = game.get_neighbors(Point { x: 1, y: 0 }).collect();
        assert_eq!(edge_neighbors.len(), 5);
    }

    #[test]
    fn test_simple_analysis() {
        // Test SAT solver analysis with a simple symmetric constraint scenario
        let variables = HashSet::from([Point { x: 0, y: 0 }, Point { x: 0, y: 1 }]);
        let local_constraints = vec![Constraint {
            variables: vec![Point { x: 0, y: 0 }, Point { x: 0, y: 1 }],
            required_mines: 1,
        }];
        let global_constraint = Constraint {
            variables: vec![Point { x: 0, y: 0 }, Point { x: 0, y: 1 }],
            required_mines: 1,
        };

        let constraints = Constraints {
            variables,
            local_constraints,
            global_constraint,
        };

        let result = analyze_board_state(&constraints);
        assert!(result.is_ok());

        let analysis = result.unwrap();
        let deductions = analysis.deductions;
        let sample = analysis.sample_assignment;

        // With symmetric constraints (exactly 1 mine between 2 cells), both should be undetermined
        assert_eq!(
            deductions.get(&Point { x: 0, y: 0 }),
            Some(&DeducedState::Undetermined)
        );
        assert_eq!(
            deductions.get(&Point { x: 0, y: 1 }),
            Some(&DeducedState::Undetermined)
        );

        // The sample assignment must satisfy the constraint (exactly 1 mine)
        let mine_count = sample.values().filter(|&&is_mine| is_mine).count();
        assert_eq!(mine_count, 1);
    }

    #[test]
    fn test_first_move_constraint_building() {
        // Test that first move constraints ensure a safe opening
        let game = Game::new(5, 5, 10);
        let click_point = Point { x: 2, y: 2 };

        let Constraints {
            variables,
            local_constraints,
            global_constraint,
        } = game.build_constraints_first_move(click_point);

        // All cells should be variables on the first move
        assert_eq!(variables.len(), 25);

        // Should have 1 local constraint ensuring the safe area has 0 mines
        assert_eq!(local_constraints.len(), 1);

        // The safe area constraint should guarantee 0 mines in clicked cell and neighbors
        let safe_constraint = &local_constraints[0];
        assert_eq!(safe_constraint.required_mines, 0);
        assert!(safe_constraint.variables.contains(&click_point));

        // Global constraint should distribute all mines across the board
        assert_eq!(global_constraint.required_mines, 10);
        assert_eq!(global_constraint.variables.len(), 25);
    }

    #[test]
    fn test_hitting_mine() {
        // Create a small board with high mine density to increase chance of hitting a mine
        let mut game = Game::new(3, 3, 5);

        // Try multiple clicks until we hit a mine (statistically should happen quickly)
        let mut mine_hit = false;
        for y in 0..3 {
            for x in 0..3 {
                if game.game_state == GameState::Playing {
                    let result = game.reveal_cell(Point { x, y }).unwrap();
                    if !result {
                        // We hit a mine
                        mine_hit = true;
                        assert_eq!(game.game_state, GameState::Lost);
                        break;
                    }
                }
            }
            if mine_hit {
                break;
            }
        }
    }

    #[test]
    fn test_optimized_solver_performance() {
        // Test the optimized solver on a larger board to verify it works correctly
        let mut game = Game::new(10, 10, 20);

        // Simulate a partially revealed board
        let result = game.reveal_cell(Point { x: 5, y: 5 }).unwrap();
        assert!(result);

        // Make a few more moves to create constraints
        let _ = game.reveal_cell(Point { x: 3, y: 3 });
        let _ = game.reveal_cell(Point { x: 7, y: 7 });

        // Build constraints and analyze - this exercises the optimized code paths
        let constraints = game.build_constraints();

        // Verify the analysis works with the optimizations
        let analysis = analyze_board_state(&constraints);
        assert!(analysis.is_ok());

        let result = analysis.unwrap();

        // Verify we get consistent results
        let total_mines_in_sample = result
            .sample_assignment
            .values()
            .filter(|&&is_mine| is_mine)
            .count();
        assert!(total_mines_in_sample <= 20);

        // Verify deductions are consistent with sample
        for (point, &deduced_state) in &result.deductions {
            let sample_is_mine = *result.sample_assignment.get(point).unwrap_or(&false);
            match deduced_state {
                DeducedState::ForcedMine => assert!(sample_is_mine),
                DeducedState::ForcedSafe => assert!(!sample_is_mine),
                DeducedState::Undetermined => {} // Can be either
            }
        }
    }
}
