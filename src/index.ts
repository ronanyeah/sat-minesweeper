import "./index.css";
const { Elm } = require("./Main.elm");
import wasmInit, * as wasm from "../public/wasm/wasm.js";
import { ElmApp, Flags } from "./ports";

let game = new Uint8Array();

(async () => {
  const flags: Flags = {};
  const app: ElmApp = Elm.Main.init({
    node: document.getElementById("app"),
    flags,
  });

  app.ports.log.subscribe((txt) => console.log(txt));

  app.ports.alert.subscribe((txt) => alert(txt));

  app.ports.validateBoard.subscribe(() =>
    (async () => {
      const win = wasm.validate(game);
      app.ports.gameResult.send(win);
    })().catch(console.error)
  );

  app.ports.createGame.subscribe(({ size, mines }) =>
    (async () => {
      game = wasm.create_game(size, mines);
      const board = await renderBoard();
      app.ports.updateBoard.send([0, board]);
    })().catch(console.error)
  );

  app.ports.selectSquare.subscribe(([x, y]) =>
    (async () => {
      let res = wasm.choose_cell(game, x, y);
      const outcome = res[res.length - 1];
      game = res.slice(0, -1);
      const board = await renderBoard();
      if (outcome == 1) {
        app.ports.hitMine.send([x, y]);
      } else {
        app.ports.updateBoard.send([outcome, board]);
      }
    })().catch(console.error)
  );

  await wasmInit();
})().catch((e) => {
  console.error(e);
});

async function renderBoard(): Promise<number[][]> {
  const data = wasm.get_cells(game);
  return buildBoard(data);
}

function buildBoard(data: Int8Array): number[][] {
  const flat = Array.from(data);
  const size = Math.sqrt(flat.length);
  const result2D = [];
  for (let i = 0; i < size; i++) {
    result2D.push(Array.from(flat.slice(i * size, (i + 1) * size)));
  }
  return result2D;
}
