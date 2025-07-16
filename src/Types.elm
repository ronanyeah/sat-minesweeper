module Types exposing (..)

import Set exposing (Set)


type alias Model =
    { squares : List (List Square)
    , status : GameStatus
    , boardLoading : Bool
    , flags : Set ( Int, Int )
    , mine : Maybe ( Int, Int )
    , gameSettings : GameSettings
    , touchHoldTimer : Maybe ( Int, Int )
    }


type Msg
    = BoardCb ( Int, List (List Int) )
    | SelectSquare ( Int, Int )
    | FlagToggle ( Int, Int )
    | HitMine ( Int, Int )
    | GameResult Bool
    | UpdateSize Int
    | UpdateMines Int
    | StartGame
    | RestartGame
    | TouchStart ( Int, Int )
    | TouchEnd ( Int, Int )
    | TouchHoldComplete ( Int, Int )


type GameStatus
    = Standby
    | InProgress
        -- 'did win' outcome
        (Maybe Bool)


type alias GameSettings =
    { size : Int
    , mines : Int
    }


type alias Square =
    { x : Int
    , y : Int
    , mines : Maybe Int
    }
