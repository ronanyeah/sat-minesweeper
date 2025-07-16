use minesweeper::*;
use rand::prelude::IndexedRandom;
use std::thread;
use std::time::Duration;

fn main() {
    // --- 1. Initialization ---
    let mut game = Game::new(10, 10, 15);
    let mut rng = rand::rng();

    println!("--- Autonomous Minesweeper Bot ---");
    println!("Strategy: Prioritize logically safe moves, guess randomly otherwise.");
    println!("Initial Board:");
    print_board(&game);
    thread::sleep(Duration::from_secs(2));

    // --- 2. Game Loop ---
    let mut move_count = 0;
    while game.game_state == GameState::Playing {
        move_count += 1;
        println!("\n--- Move #{} ---", move_count);

        // --- 3. Bot's Decision Logic ---

        // First, analyze the board to find logically forced moves.
        let constraints = game.build_constraints();
        let analysis_result = analyze_board_state(&constraints);

        let point_to_reveal: Option<Point>;

        if let Ok(analysis) = analysis_result {
            // Strategy 1: Find a cell that is guaranteed to be safe.
            let safe_move = analysis
                .deductions
                .iter()
                .find(|&(_, &state)| state == DeducedState::ForcedSafe)
                .map(|(&point, _)| point);

            if safe_move.is_some() {
                println!("Logic found a guaranteed safe cell.");
                point_to_reveal = safe_move;
            } else {
                // Strategy 2: No safe move found, so make a random guess.
                println!("No logically safe move found. Making a random guess...");
                let hidden_cells: Vec<Point> = constraints
                    .variables
                    .iter()
                    .filter(|&&p| matches!(game.board[p.y][p.x], Cell::Hidden))
                    .cloned()
                    .collect();

                point_to_reveal = hidden_cells.choose(&mut rng).cloned();
            }
        } else {
            // If the solver returns an error, the game state is inconsistent.
            // This is a loss condition handled inside reveal_cell, but we can stop early.
            println!("Solver found an inconsistency! Game should be over.");
            game.game_state = GameState::Lost;
            break;
        }

        // --- 4. Execute the Chosen Move ---
        if let Some(point) = point_to_reveal {
            println!("Bot reveals ({}, {})...", point.x, point.y);

            game.reveal_cell(point).unwrap();

            print_board(&game);
        } else {
            // This happens if there are no hidden cells left to click,
            // which usually means the game has been won or is in a strange state.
            println!("No valid moves left for the bot to make.");
            break;
        }

        // Add a delay to make the game watchable
        thread::sleep(Duration::from_millis(500));
    }

    // --- 5. Final Result ---
    println!("\n--- Game Over ---");

    match game.game_state {
        GameState::Won => println!("Result: The bot won!"),
        GameState::Lost => println!("Result: The bot hit a mine and lost."),
        GameState::Playing => println!("Result: The game ended unexpectedly."),
    }
}

fn print_board(game: &Game) {
    // Print header
    print!("   ");
    for x in 0..game.width {
        print!("{:^3}", x);
    }
    println!("\n  +{}", "---".repeat(game.width));

    // Print rows
    for (y, row) in game.board.iter().enumerate() {
        print!("{:^2}|", y);
        for cell in row {
            let display = match cell {
                Cell::Hidden => " ■ ".to_string(),
                Cell::Revealed(0) => " 0 ".to_string(),
                Cell::Revealed(n) => format!(" {} ", n),
            };
            print!("{}", display);
        }
        println!();
    }
    println!();
}
