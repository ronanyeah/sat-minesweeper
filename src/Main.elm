module Main exposing (main)

import Browser
import Ports
import Set
import Types exposing (Model, Msg)
import Update exposing (update)
import View exposing (view)


main : Program Ports.Flags Model Msg
main =
    Browser.element
        { init = init
        , view = view
        , update = update
        , subscriptions = subscriptions
        }


init : Ports.Flags -> ( Model, Cmd Msg )
init _ =
    ( { squares = []
      , flags = Set.empty
      , mine = Nothing
      , status = Types.Standby
      , boardLoading = False
      , gameSettings = { size = 8, mines = 8 }
      , touchHoldTimer = Nothing
      }
    , Ports.log "app start"
    )


subscriptions : Model -> Sub Msg
subscriptions _ =
    Sub.batch
        [ Ports.updateBoard Types.BoardCb
        , Ports.hitMine Types.HitMine
        , Ports.gameResult Types.GameResult
        ]
