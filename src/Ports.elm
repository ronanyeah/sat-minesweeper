port module Ports exposing (..)


type alias Flags =
    {}



-- OUT


port alert : String -> Cmd msg


port log : String -> Cmd msg


port validateBoard : () -> Cmd msg


port createGame : { size : Int, mines : Int } -> Cmd msg


port selectSquare : ( Int, Int ) -> Cmd msg



-- IN


port gameResult : (Bool -> msg) -> Sub msg


port hitMine : (( Int, Int ) -> msg) -> Sub msg


port updateBoard : (( Int, List (List Int) ) -> msg) -> Sub msg
