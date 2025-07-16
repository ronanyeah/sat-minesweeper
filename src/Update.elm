module Update exposing (update)

import Ports
import Process
import Set
import Task
import Types exposing (..)


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        BoardCb ( _, v ) ->
            let
                sqs =
                    v
                        |> List.indexedMap
                            (\y row ->
                                row
                                    |> List.indexedMap
                                        (\x val ->
                                            { x = x
                                            , y = y
                                            , mines =
                                                if val == -1 then
                                                    Nothing

                                                else
                                                    Just val
                                            }
                                        )
                            )
            in
            { model
                | squares = sqs
                , boardLoading = False
            }
                |> handleBoard

        SelectSquare x ->
            if isEnded model.status then
                ( model, Cmd.none )

            else
                ( { model | boardLoading = True }, Ports.selectSquare x )

        HitMine x ->
            ( { model
                | mine = Just x
                , status = Types.InProgress (Just False)
                , boardLoading = False
              }
            , Cmd.none
            )

        FlagToggle point ->
            if isEnded model.status then
                ( model, Cmd.none )

            else
                let
                    flags =
                        if Set.member point model.flags then
                            Set.remove point model.flags

                        else
                            Set.insert point model.flags
                in
                { model
                    | flags = flags
                }
                    |> handleBoard

        UpdateSize delta ->
            let
                newSize =
                    max 5 (min 15 (model.gameSettings.size + delta))

                settings =
                    model.gameSettings

                newSettings =
                    { settings | size = newSize }
            in
            ( { model | gameSettings = newSettings }, Cmd.none )

        UpdateMines delta ->
            let
                maxMines =
                    model.gameSettings.size * model.gameSettings.size - 1

                newMines =
                    max 1 (min maxMines (model.gameSettings.mines + delta))

                settings =
                    model.gameSettings

                newSettings =
                    { settings | mines = newMines }
            in
            ( { model | gameSettings = newSettings }, Cmd.none )

        StartGame ->
            ( { model | status = Types.InProgress Nothing }
            , Ports.createGame model.gameSettings
            )

        GameResult ok ->
            ( { model
                | status =
                    if ok then
                        Types.InProgress (Just True)

                    else
                        model.status
                , boardLoading = False
              }
            , if ok then
                Ports.alert "You won!"

              else
                Ports.alert "Incorrect solution!"
            )

        RestartGame ->
            ( { model
                | status = Types.Standby
                , squares = []
                , flags = Set.empty
                , mine = Nothing
                , touchHoldTimer = Nothing
              }
            , Ports.createGame model.gameSettings
            )

        TouchStart point ->
            if isEnded model.status then
                ( model, Cmd.none )

            else
                ( { model | touchHoldTimer = Just point }
                , Task.perform (\_ -> TouchHoldComplete point) (Process.sleep 350)
                )

        TouchEnd point ->
            case model.touchHoldTimer of
                Just holdPoint ->
                    if point == holdPoint then
                        -- Short tap - select square
                        ( { model | touchHoldTimer = Nothing, boardLoading = True }
                        , Ports.selectSquare point
                        )

                    else
                        ( { model | touchHoldTimer = Nothing }, Cmd.none )

                Nothing ->
                    ( model, Cmd.none )

        TouchCancel _ ->
            ( { model | touchHoldTimer = Nothing }, Cmd.none )

        TouchHoldComplete point ->
            case model.touchHoldTimer of
                Just holdPoint ->
                    if point == holdPoint then
                        -- Long hold - toggle flag
                        let
                            flags =
                                if Set.member point model.flags then
                                    Set.remove point model.flags

                                else
                                    Set.insert point model.flags
                        in
                        { model
                            | flags = flags
                            , touchHoldTimer = Nothing
                        }
                            |> handleBoard

                    else
                        ( { model | touchHoldTimer = Nothing }, Cmd.none )

                Nothing ->
                    ( model, Cmd.none )


isEnded : GameStatus -> Bool
isEnded status =
    case status of
        InProgress Nothing ->
            False

        _ ->
            True


{-| clear invalid flags and trigger win condition
-}
handleBoard : Model -> ( Model, Cmd Msg )
handleBoard model =
    let
        revealed =
            model.squares
                |> List.concat
                |> List.filter (.mines >> (/=) Nothing)
                |> List.length

        hiddenSquares =
            model.squares
                |> List.concatMap
                    (List.filterMap
                        (\sq ->
                            if sq.mines == Nothing then
                                Just ( sq.x, sq.y )

                            else
                                Nothing
                        )
                    )

        flags =
            Set.filter (\val -> List.member val hiddenSquares)
                model.flags

        allFilled =
            revealed
                + Set.size flags
                == model.gameSettings.size
                ^ 2
                || List.length hiddenSquares
                == model.gameSettings.mines
    in
    if allFilled then
        ( { model
            | flags = flags
            , boardLoading = True
          }
        , Ports.validateBoard ()
        )

    else
        ( { model
            | flags = flags
          }
        , Cmd.none
        )
