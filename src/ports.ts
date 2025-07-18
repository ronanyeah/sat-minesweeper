/* This file was generated by github.com/ronanyeah/elm-port-gen */

interface ElmApp {
  ports: Ports;
}

interface Ports {
  alert: PortOut<string>;
  log: PortOut<string>;
  validateBoard: PortOut<null>;
  createGame: PortOut<{
    size: number;
    mines: number;
  }>;
  selectSquare: PortOut<[number, number]>;
  gameResult: PortIn<boolean>;
  hitMine: PortIn<[number, number]>;
  updateBoard: PortIn<[number, number[][]]>;
}

interface PortOut<T> {
  subscribe: (_: (_: T) => void) => void;
}

interface PortIn<T> {
  send: (_: T) => void;
}

type PortResult<E, T> =
    | { err: E; data: null }
    | { err: null; data: T };

interface Flags {
  
}

function portOk<E, T>(data: T): PortResult<E, T> {
  return { data, err: null };
}

function portErr<E, T>(err: E): PortResult<E, T> {
  return { data: null, err };
}

export { ElmApp, PortResult, portOk, portErr, Flags };