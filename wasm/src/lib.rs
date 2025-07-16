use minesweeper as ms;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn create_game(size: u8, mines: u8) -> Vec<u8> {
    console_error_panic_hook::set_once();

    let game = ms::Game::new(size as usize, size as usize, mines as usize);
    let bts = game.serialize();
    bts
}

#[wasm_bindgen]
pub fn validate(bts: Vec<u8>) -> bool {
    console_error_panic_hook::set_once();

    let game = ms::Game::deserialize(&bts);
    game.check_win_condition()
}

#[wasm_bindgen]
pub fn choose_cell(bts: Vec<u8>, x: usize, y: usize) -> Result<Vec<u8>, String> {
    console_error_panic_hook::set_once();

    let mut game = ms::Game::deserialize(&bts);
    let point = ms::Point { x, y };
    let res = game.reveal_cell(point).map_err(|e| e.to_string())?;
    let mut xs = game.serialize();
    xs.push(if res { 0 } else { 1 });
    Ok(xs)
}

#[wasm_bindgen]
pub fn get_cells(bts: Vec<u8>) -> Vec<i8> {
    console_error_panic_hook::set_once();

    let game = ms::Game::deserialize(&bts);
    game.board
        .into_iter()
        .map(|row| {
            row.into_iter().map(|cell| match cell {
                ms::Cell::Hidden => -1,
                ms::Cell::Revealed(n) => n as i8,
            })
        })
        .flatten()
        .collect()
}
