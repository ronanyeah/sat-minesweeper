module View exposing (view)

import Element exposing (..)
import Element.Background as Background
import Element.Border as Border
import Element.Font as Font
import Element.Input as Input
import Helpers.View exposing (style, when, whenAttr)
import Html exposing (Html)
import Html.Events
import Json.Decode as Decode
import Material.Icons as Icons
import Material.Icons.Types exposing (Icon)
import Maybe.Extra exposing (unwrap)
import Set
import Types exposing (..)


view : Model -> Html Msg
view model =
    case model.status of
        Standby ->
            initScreen model

        InProgress outcome ->
            gameView model outcome


initScreen : Model -> Html Msg
initScreen model =
    [ paragraph [ Font.size 24, Font.bold, centerX ] [ text "Stateless Probabilistic Minesweeper" ]
    , [ newTabLink [ Font.underline, hover ]
            { url = "https://x.com/kostascrypto/status/1940444390255665509"
            , label = text "Project objectives"
            }
      , text "•"
      , newTabLink [ Font.underline, hover ]
            { url = "https://github.com/ronanyeah/sat-minesweeper"
            , label = text "Code"
            }
      ]
        |> row [ spacing 20, centerX ]
    , column [ spacing 20, centerX, padding 20 ]
        [ row [ spacing 20, centerX ]
            [ el [ width (px 100) ] (text "Size:")
            , btn (Just (UpdateSize -1)) [ Background.color (rgb255 200 200 200), padding 10 ] (text "▼")
            , el [ centerX ] (text (String.fromInt model.gameSettings.size))
            , btn (Just (UpdateSize 1)) [ Background.color (rgb255 200 200 200), padding 10 ] (text "▲")
            ]
        , row [ spacing 20, centerX ]
            [ el [ width (px 100) ] (text "Mines:")
            , btn (Just (UpdateMines -1)) [ Background.color (rgb255 200 200 200), padding 10 ] (text "▼")
            , el [ centerX ] (text (String.fromInt model.gameSettings.mines))
            , btn (Just (UpdateMines 1)) [ Background.color (rgb255 200 200 200), padding 10 ] (text "▲")
            ]
        , btn (Just StartGame)
            [ Background.color (rgb255 100 200 100)
            , padding 15
            , centerX
            , Font.bold
            ]
            (text "Start Game")
        ]
    ]
        |> column [ centerX, spacing 20, padding 20 ]
        |> Element.layoutWith
            { options =
                [ Element.focusStyle
                    { borderColor = Nothing
                    , backgroundColor = Nothing
                    , shadow = Nothing
                    }
                ]
            }
            [ width fill
            , height fill
            , Background.color (rgb255 192 192 192)
            ]


gameView : Model -> Maybe Bool -> Html Msg
gameView model outcome =
    [ [ [ "Click to reveal • Ctrl+Click to flag (hold on mobile)"
            |> text
        ]
            |> paragraph [ Font.size 14, Font.italic, Font.color (rgb255 100 100 100) ]
      , [ "Reveal all safe squares to win!"
            |> text
        ]
            |> paragraph [ Font.size 14, Font.italic, Font.center, Font.color (rgb255 100 100 100) ]
      ]
        |> column [ centerX, spacing 5 ]
    , minesweeperGrid model outcome
    , [ "Flags: "
            ++ String.fromInt (Set.size model.flags)
            |> text
      , "Mines: "
            ++ String.fromInt model.gameSettings.mines
            |> text
      ]
        |> row [ spaceEvenly, width fill ]
    , case outcome of
        Just True ->
            column [ spacing 15, centerX ]
                [ el [ Font.size 24, Font.bold, centerX, Font.color (rgb255 0 128 0) ] (text "🎉 You Won!")
                , btn (Just RestartGame)
                    [ Background.color (rgb255 100 200 100)
                    , padding 10
                    , centerX
                    , Font.bold
                    ]
                    (text "Play Again")
                ]

        Just False ->
            column [ spacing 15, centerX ]
                [ el [ Font.size 24, Font.bold, centerX, Font.color (rgb255 255 0 0) ] (text "💥 Game Over!")
                , btn (Just RestartGame)
                    [ Background.color (rgb255 100 200 100)
                    , padding 10
                    , centerX
                    , Font.bold
                    ]
                    (text "Try Again")
                ]

        Nothing ->
            none
    ]
        |> column [ centerX, padding 20, spacing 10 ]
        |> Element.layoutWith
            { options =
                [ Element.focusStyle
                    { borderColor = Nothing
                    , backgroundColor = Nothing
                    , shadow = Nothing
                    }
                ]
            }
            [ width fill
            , height fill
            , Background.color (rgb255 192 192 192)
            ]


minesweeperGrid : Model -> Maybe Bool -> Element Msg
minesweeperGrid model outcome =
    let
        gridRows =
            model.squares
                |> List.map
                    (\row ->
                        row
                            |> List.map (\sq -> mineCell model sq outcome)
                            |> Element.row [ spacing 0 ]
                    )
    in
    gridRows
        |> Element.column
            [ spacing 0
            , Background.color (rgb255 128 128 128)
            , Border.width 2
            , Border.color (rgb255 64 64 64)
            ]


mineCell : Model -> Types.Square -> Maybe Bool -> Element Msg
mineCell model sq outcome =
    let
        mine =
            model.mine

        flags =
            model.flags

        isMine =
            mine
                |> unwrap False (\( x, y ) -> sq.x == x && sq.y == y)

        isZero =
            if isMine then
                False

            else
                sq.mines
                    |> unwrap False ((==) 0)

        isRevealed =
            sq.mines /= Nothing

        gameEnded =
            outcome /= Nothing

        size =
            45
    in
    el
        ([ width (px size)
         , height (px size)
         , Background.color
            (if isZero then
                rgb255 175 175 175

             else
                rgb255 192 192 192
            )
         , (if isMine then
                text "💥"

            else
                sq.mines
                    |> unwrap
                        (icon Icons.flag 35
                            |> el [ Font.color (rgb255 255 0 0), centerX, centerY ]
                            |> when (Set.member ( sq.x, sq.y ) flags)
                        )
                        (\n ->
                            String.fromInt n
                                |> text
                                |> when (n /= 0)
                        )
           )
            |> el [ Font.size 35, centerX, centerY ]
            |> inFront
         ]
            ++ (if isZero then
                    []

                else
                    [ Border.widthEach { top = 2, right = 1, bottom = 1, left = 2 }
                    , Border.color (rgb255 255 255 255)
                    , Border.innerGlow (rgb255 128 128 128) 1
                    , Border.glow (rgb255 64 64 64) 1
                    ]
               )
            ++ (if not isRevealed && not gameEnded && not model.boardLoading then
                    [ hover
                    , pointer
                    , Html.Events.on "click" (ctrlClickDecoder sq.x sq.y)
                        |> htmlAttribute
                    , Html.Events.on "touchstart" (touchStartDecoder sq.x sq.y)
                        |> htmlAttribute
                    , Html.Events.on "touchend" (touchEndDecoder sq.x sq.y)
                        |> htmlAttribute
                    , Html.Events.on "touchcancel" (touchCancelDecoder sq.x sq.y)
                        |> htmlAttribute
                    ]

                else
                    []
               )
            ++ (if model.boardLoading then
                    [ style "cursor" "wait"
                    ]

                else
                    []
               )
        )
        none


ctrlClickDecoder : Int -> Int -> Decode.Decoder Msg
ctrlClickDecoder x y =
    Decode.field "ctrlKey" Decode.bool
        |> Decode.andThen
            (\ctrlPressed ->
                if ctrlPressed then
                    Decode.succeed (FlagToggle ( x, y ))

                else
                    Decode.succeed (SelectSquare ( x, y ))
            )


touchStartDecoder : Int -> Int -> Decode.Decoder Msg
touchStartDecoder x y =
    Decode.succeed (TouchStart ( x, y ))


touchEndDecoder : Int -> Int -> Decode.Decoder Msg
touchEndDecoder x y =
    Decode.succeed (TouchEnd ( x, y ))


touchCancelDecoder : Int -> Int -> Decode.Decoder Msg
touchCancelDecoder x y =
    Decode.succeed (TouchCancel ( x, y ))


btn : Maybe msg -> List (Attribute msg) -> Element msg -> Element msg
btn msg attrs elem =
    Input.button
        ((hover |> whenAttr (msg /= Nothing)) :: attrs)
        { onPress = msg
        , label = elem
        }


hover : Attribute msg
hover =
    Element.mouseOver [ fade ]


fade : Element.Attr a b
fade =
    Element.alpha 0.7


icon : Icon msg -> Int -> Element msg
icon ic n =
    ic n Material.Icons.Types.Inherit
        |> Element.html
        |> el []
