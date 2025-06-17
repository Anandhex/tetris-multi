using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System;
using UnityEngine.InputSystem;

public class SocketTetrisAgent : MonoBehaviour, IPlayerInputController
{
    private Board board;
    private Piece currentPiece;
    private bool isExecutingAction = false;
    private bool waitingForNewPiece = false;
    private bool pythonConnected = false;

    [Header("Curriculum Parameters")]
    public int curriculumBoardHeight = 20;
    public int curriculumBoardPreset = 0;
    public int allowedTetrominoTypes = 7;

    // Timing
    private float lastStateTime = 0f;
    private float stateUpdateInterval = 0.1f;

    public float lastReward = 0f;
    public bool gameOver = false;

    // Action execution
    private int targetColumn = -1;
    private int targetRotation = -1;
    private bool actionCompleted = false;

    void Start()
    {
        // Subscribe to socket events
        if (SocketManager.Instance != null)
        {
            SocketManager.Instance.OnCommandReceived += HandleCommand;
            SocketManager.Instance.OnPythonConnected += OnPythonConnected;
            SocketManager.Instance.OnPythonDisconnected += OnPythonDisconnected;
        }
        else
        {
            Debug.LogError("SocketManager not found! Make sure it's in the scene.");
        }
    }



    void HandleCommand(GameCommand command)
    {
        switch (command.type)
        {
            case "action":
                if (command.action != null && !isExecutingAction)
                {
                    if (command.action.col != -1 || command.action.rot != -1)
                    {
                        ExecuteAction(command.action.col, command.action.rot);
                    }
                    else
                    {
                        board.DumpTilemap(board.Bounds);
                        Debug.Log("Triggered" + command.action.col + ":" + command.action.rot);
                        TriggerGameOver();
                    }
                }
                break;


            case "reset":
                // Reset the board + agent flags
                ResetGame();
                // Immediately send the new initial state back to Python
                break;

            case "request_states":
                RequestStates();

                break;


        }
    }

    void RequestStates()
    {
        // Prepare the piece for spawn but don't actually spawn it yet
        // Clear any existing piece from the board to ensure clean state calculation
        if (currentPiece != null)
        {
            board.Clear(currentPiece);
        }

        // Set the current piece reference
        SetCurrentPiece(board.activePiece);

        // Calculate all possible moves without placing the piece on the board
        var metrics = GetMoveMetricsForCurrentPiece();

        // Convert to the format expected by Python
        var dict = metrics.ToDictionary(
            kv => $"{kv.Key.Item1}:{kv.Key.Item2}",
            kv => kv.Value
        );

        // Debug.Log($"Sending {dict.Count} possible states to Python");
        SocketManager.Instance.SendEvent("possible_states", dict);
    }

    void ExecuteAction(int colIdx, int rotation)
    {
        if (currentPiece == null || board == null)
        {
            Debug.LogError("Cannot execute action: piece or board is null");
            return;
        }
        board.Clear(currentPiece);
        isExecutingAction = true;
        // Validate the action before executing
        ExecuteDirectPlacementSync(colIdx, rotation);

    }



    public void ExecuteDirectPlacementSync(int col, int rot)
    {
        // Debug.Log($"Executing: col={col} rot={rot}");

        if (currentPiece == null)
        {
            Debug.LogError("No current piece to execute action on");
            isExecutingAction = false;
            return;
        }


        // 1) Rotate to target
        int curRot = currentPiece.rotationIndex;
        int needed = (rot - curRot + 4) % 4;

        for (int i = 0; i < needed; i++)
        {
            currentPiece.Rotate(1);

        }

        // 2) Move horizontally
        int leftX = currentPiece.cells.Min(c => currentPiece.position.x + c.x);
        int halfW = board.Bounds.width / 2;
        int targetX = col - halfW;
        int offset = targetX - leftX;
        var dir = offset > 0 ? Vector3Int.right : Vector3Int.left;

        for (int i = 0; i < Mathf.Abs(offset); i++)
        {
            var newPos = currentPiece.position + dir;

            currentPiece.position = newPos;
        }

        // 3) Hard drop
        while (true)
        {
            var downPos = currentPiece.position + Vector3Int.down;
            if (!board.IsValidPosition2(currentPiece, downPos))
            {
                break; // Hit bottom or piece
            }
            currentPiece.position = downPos;
        }

        // 4) Final placement
        FinalizePlacement();
    }

    public void TriggerGameOver()
    {
        gameOver = true;
        SendGameState();
        gameOver = false;
        isExecutingAction = false;

    }
    void FinalizePlacement()
    {

        board.Set(currentPiece);


        // Clear any completed lines
        SendGameState();

        // Send game state after placement

        // Reset flags
        actionCompleted = true;
        isExecutingAction = false;
        waitingForNewPiece = false;

        // Clear current piece reference
        board.SpawnPiece();


    }





    void ResetGame()
    {
        gameOver = false;
        lastReward = 0f;
        isExecutingAction = false;
        waitingForNewPiece = false;
        actionCompleted = false;

        if (board != null)
        {

            board.playerScore = 0;
            board.ClearBoard();
            StateReset();
            board.SpawnPiece();

        }
    }

    void OnPythonConnected()
    {
        pythonConnected = true;
        Debug.Log("Python AI connected - Ready for 40-action Tetris (10 columns × 4 rotations)!");
        gameOver = false;
        lastStateTime = Time.time;
    }

    void OnPythonDisconnected()
    {
        Debug.Log("Python AI disconnected");
    }
    void StateReset()
    {
        GameState state = new GameState();
        SocketManager.Instance.SendGameState(state);
    }
    void SendGameState()
    {
        if (board == null || SocketManager.Instance == null)
            return;

        // 1) Compute lines cleared this step by replaying a clear on a copy of the board
        //    (or you can cache the last lines cleared in your OnLinesCleared callback)
        // int linesCleared = board.ClearLinesCount();
        int linesCleared = board.ClearLines();

        // 2) Shaped reward: +1 per placement, +lines²×width, −2 if game over
        float reward = 1f + (linesCleared * linesCleared) * board.boardSize.x;
        if (gameOver)
            reward -= 2f;

        // 3) Build minimal payload
        var payload = new GameState();
        payload.reward = reward;
        payload.gameOver = gameOver;
        // Debug.Log("somethinggg");

        // 4) Send it over the socket (using our SendEvent helper)
        SocketManager.Instance.SendGameState(payload);

        // 5) Reset lastReward if needed
        lastReward = 0f;
    }







    // IPlayerInputController implementation - Updated method names
    public bool GetLeft()
    {
        return false; // Not used in direct placement mode
    }

    public bool GetRight()
    {
        return false; // Not used in direct placement mode
    }

    public bool GetDown()
    {
        return false; // Not used in direct placement mode
    }

    public bool GetRotateLeft()
    {
        return false; // Not used in direct placement mode
    }

    public bool GetRotateRight()
    {
        return false; // Not used in direct placement mode
    }

    public bool GetHardDrop()
    {
        return false; // Not used in direct placement mode
    }

    public void SetCurrentPiece(Piece piece)
    {
        currentPiece = piece;

    }

    public void SetBoard(Board gameBoard)
    {
        board = gameBoard;
        board.inputController = this;
        lastStateTime = Time.time;   // reset your timer so you don’t immediately resend
    }

    public void OnGameOver()
    {
        gameOver = true;
        lastReward = -10f; // Penalty for game over
        isExecutingAction = false;
        // Send game over state BEFORE resetting
        SendGameState();

    }


    public void OnLinesCleared(int lines)
    {
        lastReward += lines * lines * 25f; // 25, 100, 225, 400 for 1, 2, 3, 4 lines

        if (lines == 4) // Tetris bonus
        {
            lastReward += 100f;
        }

    }

    void OnDestroy()
    {
        if (SocketManager.Instance != null)
        {
            SocketManager.Instance.OnCommandReceived -= HandleCommand;
            SocketManager.Instance.OnPythonConnected -= OnPythonConnected;
            SocketManager.Instance.OnPythonDisconnected -= OnPythonDisconnected;
        }
    }


    /// <summary>
    /// Enumerate all valid (column, rotation) placements for the current piece,
    /// simulate each drop on a cloned board, and return metrics.
    /// </summary>
    private Dictionary<(int col, int rot), float[]> GetMoveMetricsForCurrentPiece()
    {
        var results = new Dictionary<(int, int), float[]>();

        if (currentPiece == null || board == null)
        {
            Debug.LogWarning("Cannot calculate move metrics: piece or board is null");
            return results;
        }
        // Create snapshots for simulation
        var origBoard = new BoardData(board);
        var origPiece = new PieceState(currentPiece);
        if (origBoard.IsGameOverCondition())
        {
            Debug.Log("Game over condition detected - center columns in top row are blocked");
            return results; // Return empty dictionary
        }

        int w = board.boardSize.x;
        int halfW = board.Bounds.width / 2;


        for (int rot = 0; rot < currentPiece.data.RotationCount; rot++)
        {
            for (int colIdx = 0; colIdx < w; colIdx++)
            {
                try
                {

                    // Create fresh clones for simulation
                    var simBoard = origBoard.Clone();
                    var simPiece = origPiece.Clone();

                    int rotCount = simPiece.data.RotationCount;
                    int steps = (rot - simPiece.rotationIndex + rotCount) % rotCount;
                    for (int i = 0; i < steps; i++)
                    {
                        simPiece.RotateCW();
                    }


                    // Move horizontally
                    int leftX = simPiece.Cells.Min(c => simPiece.position.x + c.x);
                    int targetX = colIdx - halfW;
                    int offset = targetX - leftX;
                    var dir = offset > 0 ? Vector2Int.right : Vector2Int.left;
                    bool couldMoveAllTheWay = true;
                    for (int s = 0; s < Mathf.Abs(offset); s++)
                    {
                        var np = simPiece.position + dir;
                        if (!simBoard.IsValidPosition(simPiece, np))
                        {
                            couldMoveAllTheWay = false;
                            break;
                        }
                        simPiece.position = np;
                    }
                    if (!couldMoveAllTheWay)
                        continue;

                    // Hard drop
                    while (simBoard.IsValidPosition(simPiece, simPiece.position + Vector2Int.down))
                    {
                        simPiece.position += Vector2Int.down;
                    }


                    //     Debug.Log($"Simulated placement @ col={colIdx}, rot={rot}:\n" +
                    //   simBoard.DumpToString());
                    // Place piece and calculate metrics
                    int lines = simBoard.PlaceAndClear(simPiece);
                    float holes = simBoard.CountHoles();
                    float bumpiness = simBoard.GetBumpinessScore();
                    float height = simBoard.CalculateStackHeight();
                    results[(colIdx, rot)] = new float[] { lines, holes, bumpiness, height
};
                }
                catch (System.Exception e)
                {
                    Debug.LogError($"Error calculating metrics for col={colIdx}, rot={rot}: {e.Message}");
                    continue;
                }
            }
        }


        // now print out every move and its metrics:
        foreach (var kvp in results)
        {
            // deconstruct the tuple key into column index and rotation
            var (colIdx, rot) = kvp.Key;
            float[] metrics = kvp.Value;

            // metrics[0] = lines cleared
            // metrics[1] = holes created
            // metrics[2] = bumpiness
            // metrics[3] = resulting height
            Debug.Log($"Move → Column: {colIdx}, Rotation: {rot} | " +
                      $"Lines: {metrics[0]}, Holes: {metrics[1]}, " +
                      $"Bumpiness: {metrics[2]}, Height: {metrics[3]}");
        }
        Debug.Log($"Calculated metrics for {results.Count} valid moves");
        return results;
    }
    // Inside the BoardData class



}


