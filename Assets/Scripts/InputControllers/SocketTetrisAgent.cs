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
    private float prevLineHeight = 0.0f;

    private Dictionary<(int, int), float[]> prevResults;

    [Header("Curriculum Parameters")]
    public int curriculumBoardHeight = 20;
    public int curriculumBoardPreset = 0;
    public int allowedTetrominoTypes = 7;

    [Header("PowerUp Integration")]
    public PowerUpManager powerUpManager;
    public string currentPowerupType = "none";
    public bool hasPowerup = false;

    // Timing
    private float lastStateTime = 0f;
    private float stateUpdateInterval = 0.1f;

    public float lastReward = 0f;
    public bool gameOver = false;

    // Action execution
    private int targetColumn = -1;
    private int targetRotation = -1;
    private bool actionCompleted = false;

    // void Start()
    // {
    //     // Subscribe to socket events
    //     if (SocketManager.Instance != null)
    //     {
    //         SocketManager.Instance.OnCommandReceived += HandleCommand;
    //         SocketManager.Instance.OnPythonConnected += OnPythonConnected;
    //         SocketManager.Instance.OnPythonDisconnected += OnPythonDisconnected;
    //     }
    //     else
    //     {
    //         Debug.LogError("SocketManager not found! Make sure it's in the scene.");
    //     }
    // }



    // void HandleCommand(GameCommand command)
    // {
        
    //     Debug.Log($"🐛 UNITY DEBUG: Received command type: {command.type}");
    //     switch (command.type)
    //     {
    //         case "action":
    //             if (command.action != null && !isExecutingAction)
    //             {
    //                 if (command.action.col != -1 || command.action.rot != -1)
    //                 {
    //                     ExecuteAction(command.action.col, command.action.rot);
    //                 }
    //                 else
    //                 {
    //                     board.DumpTilemap(board.Bounds);
    //                     TriggerGameOver();
    //                 }
    //             }
    //             break;


    //         case "reset":
    //             // Reset the board + agent flags
    //             ResetGame();
    //             // Immediately send the new initial state back to Python
    //             break;

    //         case "request_states":
    //             RequestStates();

    //             break;

    //          case "hold_powerup":
    //             HandleHoldPowerup(command);
    //             break;

    //         case "execute_bomb_drop":
    //             HandleExecuteBomb(command);
    //             break;

    //         case "execute_gravity":
    //             HandleExecuteGravity(command);
    //             break;

    //         case "execute_bottom_clear":
    //             HandleExecuteBottomClear(command);
    //             break;

    //         default:
    //             Debug.LogWarning($"🐛 UNITY DEBUG: Unknown command type: {command.type}");
    //             break;


    //     }
    // }

    // void HandleHoldPowerup(GameCommand command)
    // {
    //     Debug.Log($"🐛 UNITY DEBUG: Holding powerup - type: {command.powerup_type}");
        
    //     currentPowerupType = command.powerup_type ?? "none";
    //     hasPowerup = true;
        
    //     // Convert Python powerup names to Unity PowerUpType
    //     PowerUpType unityPowerUpType = ConvertPythonToUnityPowerUpType(currentPowerupType);
        
    //     // Add to PowerUpManager inventory
    //     if (powerUpManager != null)
    //     {
    //         powerUpManager.AddPowerUp(unityPowerUpType);
    //     }
        
    //     // Send confirmation back to Python
    //     var response = new
    //     {
    //         type = "powerup_held",
    //         success = true,
    //         powerup_type = currentPowerupType,
    //         ai_confidence = command.ai_confidence,
    //         timestamp = Time.time
    //     };
        
    //     SocketManager.Instance.SendEvent("powerup_held", response);
    // }

    // void HandleExecuteBomb(GameCommand command)
    // {
    //     Debug.Log($"🐛 UNITY DEBUG: Executing bomb drop - column: {command.bomb?.column}");
        
    //     if (command.bomb == null)
    //     {
    //         SendPowerupError("bomb_executed", "No bomb data provided");
    //         return;
    //     }
        
    //     // Capture board state before bomb
    //     var boardBefore = GetBoardStateArray();
        
    //     // Execute bomb using existing PowerUpManager
    //     if (powerUpManager != null)
    //     {
    //         // Force execute bomb at specific location
    //         ExecuteBombAtColumn(command.bomb.column);
    //     }
        
    //     // Capture board state after bomb
    //     var boardAfter = GetBoardStateArray();
        
    //     // Calculate impact metrics
    //     var impactMetrics = CalculateImpactMetrics(boardBefore, boardAfter);
        
    //     // Send response back to Python
    //     var response = new
    //     {
    //         type = "bomb_executed",
    //         success = true,
    //         landing_row = GetHighestBlockInColumn(command.bomb.column),
    //         explosion_center = new int[] { command.bomb.column, GetHighestBlockInColumn(command.bomb.column) },
    //         board_before = boardBefore,
    //         board_after = boardAfter,
    //         impact_metrics = impactMetrics,
    //         ui_updates = new
    //         {
    //             particles_spawned = true,
    //             sound_played = "bomb_explosion",
    //             score_popup = $"+{impactMetrics.scoreBonus}"
    //         },
    //         ai_confidence = command.ai_confidence,
    //         predicted_impact = command.bomb.predicted_impact,
    //         error = (string)null
    //     };
        
    //     SocketManager.Instance.SendEvent("bomb_executed", response);
    //     ClearCurrentPowerup();
    // }

    // void HandleExecuteGravity(GameCommand command)
    // {
    //     Debug.Log($"🐛 UNITY DEBUG: Executing gravity powerup");
        
    //     var boardBefore = GetBoardStateArray();
        
    //     // Execute gravity using existing PowerUpManager
    //     if (powerUpManager != null)
    //     {
    //         ExecuteGravityPowerup();
    //     }
        
    //     var boardAfter = GetBoardStateArray();
    //     var impactMetrics = CalculateImpactMetrics(boardBefore, boardAfter);
        
    //     var response = new
    //     {
    //         type = "gravity_executed",
    //         success = true,
    //         board_before = boardBefore,
    //         board_after = boardAfter,
    //         impact_metrics = impactMetrics,
    //         ui_updates = new
    //         {
    //             animation_played = "gravity_pull",
    //             sound_played = "gravity_whoosh"
    //         },
    //         ai_confidence = command.ai_confidence,
    //         predicted_impact = command.gravity.predicted_impact,
    //         error = (string)null
    //     };
        
    //     SocketManager.Instance.SendEvent("gravity_executed", response);
    //     ClearCurrentPowerup();
    // }

    // void HandleExecuteBottomClear(GameCommand command)
    // {
    //     Debug.Log($"🐛 UNITY DEBUG: Executing bottom clear powerup");
        
    //     var boardBefore = GetBoardStateArray();
        
    //     // Execute bottom clear using existing PowerUpManager
    //     if (powerUpManager != null)
    //     {
    //         ExecuteBottomClearPowerup();
    //     }
        
    //     var boardAfter = GetBoardStateArray();
    //     var impactMetrics = CalculateImpactMetrics(boardBefore, boardAfter);
        
    //     var response = new
    //     {
    //         type = "bottom_clear_executed",
    //         success = true,
    //         board_before = boardBefore,
    //         board_after = boardAfter,
    //         impact_metrics = impactMetrics,
    //         ui_updates = new
    //         {
    //             animation_played = "line_clear_bottom",
    //             sound_played = "line_clear"
    //         },
    //         ai_confidence = command.ai_confidence,
    //         predicted_impact = command.bottom_clear.predicted_impact,
    //         error = (string)null
    //     };
        
    //     SocketManager.Instance.SendEvent("bottom_clear_executed", response);
    //     ClearCurrentPowerup();
    // }

    // void RequestStates()
    // {
    //     // Prepare the piece for spawn but don't actually spawn it yet
    //     // Clear any existing piece from the board to ensure clean state calculation
    //     if (currentPiece != null)
    //     {
    //         board.Clear(currentPiece);
    //     }

    //     // Set the current piece reference
    //     SetCurrentPiece(board.activePiece);

    //     // Calculate all possible moves without placing the piece on the board
    //     var metrics = GetMoveMetricsForCurrentPiece();

    //     // Convert to the format expected by Python
    //     var dict = metrics.ToDictionary(
    //         kv => $"{kv.Key.Item1}:{kv.Key.Item2}",
    //         kv => kv.Value
    //     );

    //     // Debug.Log($"Sending {dict.Count} possible states to Python");
    //     SocketManager.Instance.SendEvent("possible_states", dict);
    // }

    // void ExecuteAction(int colIdx, int rotation)
    // {
    //     if (currentPiece == null || board == null)
    //     {
    //         Debug.LogError("Cannot execute action: piece or board is null");
    //         return;
    //     }
    //     board.Clear(currentPiece);
    //     isExecutingAction = true;
    //     // Validate the action before executing
    //     ExecuteDirectPlacementSync(colIdx, rotation);

    // }



    // public void ExecuteDirectPlacementSync(int col, int rot)
    // {
    //     // Debug.Log($"Executing: col={col} rot={rot}");

    //     if (currentPiece == null)
    //     {
    //         Debug.LogError("No current piece to execute action on");
    //         isExecutingAction = false;
    //         return;
    //     }


    //     // 1) Rotate to target
    //     int curRot = currentPiece.rotationIndex;
    //     int needed = (rot - curRot + 4) % 4;
    //     for (int i = 0; i < needed; i++)
    //         currentPiece.Rotate(-1);

    //     // 2) Move horizontally — **use** Bounds.xMin and cells[]
    //     int xOffset = board.Bounds.xMin;
    //     int minLocalX = currentPiece.cells.Min(c => c.x);
    //     int worldX = col + xOffset - minLocalX;
    //     currentPiece.position = new Vector3Int(
    //         worldX,
    //         currentPiece.position.y,
    //         currentPiece.position.z
    //     );

    //     // 3) Hard drop
    //     while (true)
    //     {
    //         var downPos = currentPiece.position + Vector3Int.down;
    //         if (!board.IsValidPosition2(currentPiece, downPos))
    //         {
    //             break; // Hit bottom or piece
    //         }
    //         currentPiece.position = downPos;
    //     }

    //     // 4) Final placement
    //     FinalizePlacement(col, rot);
    // }

    // public void TriggerGameOver()
    // {
    //     gameOver = true;
    //     SendGameState(-1, -1);
    //     gameOver = false;
    //     isExecutingAction = false;

    // }
    // void FinalizePlacement(int col, int rot)
    // {

    //     board.Set(currentPiece);


    //     // Clear any completed lines
    //     SendGameState(col, rot);

    //     // Send game state after placement

    //     // Reset flags
    //     actionCompleted = true;
    //     isExecutingAction = false;
    //     waitingForNewPiece = false;

    //     // Clear current piece reference
    //     board.SpawnPiece();


    // }





    // void ResetGame()
    // {
    //     gameOver = false;
    //     lastReward = 0f;
    //     isExecutingAction = false;
    //     waitingForNewPiece = false;
    //     actionCompleted = false;
        
    //     // Clear powerups
    //     ClearCurrentPowerup();
    //     if (powerUpManager != null)
    //     {
    //         powerUpManager.ClearAllPowerUps();
    //     }
        
    //     if (board != null)
    //     {
    //         board.playerScore = 0;
    //         board.ClearBoard();
    //         StateReset();
    //         board.SpawnPiece();
    //     }
    // }

    // void OnPythonConnected()
    // {
    //     pythonConnected = true;
    //     Debug.Log("Python AI connected - Ready for 40-action Tetris (10 columns × 4 rotations)!");
    //     gameOver = false;
    //     lastStateTime = Time.time;
    // }

    // void OnPythonDisconnected()
    // {
    //     Debug.Log("Python AI disconnected");
    // }
    // void StateReset()
    // {
    //     GameState state = new GameState();
    //     SocketManager.Instance.SendGameState(state);
    // }
    // void SendGameState(int col, int rot)
    // {
    //     if (board == null || SocketManager.Instance == null)
    //         return;

    //     // 1) Compute lines cleared this step by replaying a clear on a copy of the board
    //     //    (or you can cache the last lines cleared in your OnLinesCleared callback)
    //     // int linesCleared = board.ClearLinesCount();
    //     int linesCleared = board.ClearLines();
    //     // float holes = 0;
    //     // float bumpiness = 0;

    //     // float survivalStep = -0.01f;               // small time penalty
    //     // float maxLineBonus = 1.0f;                // 4-line maximum
    //     // float holeCost = 0.2f;                // per hole
    //     // float bumpCost = 0.2f;                // per bump
    //     // float deathPenalty = -1.0f;               // harsher end-of-game cost

    //     // if (col > -1 && rot > -1 && prevResults.ToList().Count > 0)
    //     // {
    //     //     holes = prevResults[(col, rot)][1];
    //     //     bumpiness = prevResults[(col, rot)][2];
    //     // }
    //     // 2) Shaped reward: +1 per placement, +lines²×width, −2 if game over
    //     // float lineBonus = Mathf.Pow(linesCleared / 4f, 2) * maxLineBonus;
    //     // float reward = survivalStep
    //     //      + lineBonus
    //     //      - holeCost * holes
    //     //      - bumpCost * bumpiness;
    //     // penalty for creating holes
    //     float reward = 1f + (linesCleared * linesCleared) * board.boardSize.x;
    //     if (gameOver)
    //         reward -= 2f;

    //     // 3) Build minimal payload
    //     var payload = new GameState();
    //     payload.reward = reward;
    //     payload.gameOver = gameOver;
    //     // Debug.Log("somethinggg");
    //     Debug.Log("reward" + reward);
    //     // 4) Send it over the socket (using our SendEvent helper)
    //     SocketManager.Instance.SendGameState(payload);

    //     // 5) Reset lastReward if needed
    //     lastReward = 0f;
    // }







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
        // SendGameState(-1, -1);

    }


    public void OnLinesCleared(int lines)
    {
        lastReward += lines * lines * 25f; // 25, 100, 225, 400 for 1, 2, 3, 4 lines

        if (lines == 4) // Tetris bonus
        {
            lastReward += 100f;
        }

    // }

    // PowerUpType ConvertPythonToUnityPowerUpType(string pythonPowerUpType)
    // {
    //     switch (pythonPowerUpType)
    //     {
    //         case "bomb":
    //             return PowerUpType.Bomb;
    //         case "gravity":
    //             return PowerUpType.Gravity;
    //         case "bottom_line_clear":
    //             return PowerUpType.LineBlaster;
    //         default:
    //             Debug.LogWarning($"Unknown powerup type: {pythonPowerUpType}");
    //             return PowerUpType.Bomb;
    //     }
    // }

    // void ExecuteBombAtColumn(int targetColumn)
    // {
    //     Debug.Log($"🐛 UNITY DEBUG: Executing bomb at column {targetColumn}");
    
    //     if (powerUpManager != null)
    //     {
    //         // Store original position
    //         Vector3Int originalPos = Vector3Int.zero;
    //         bool hadActivePiece = false;
            
    //         if (board.activePiece != null)
    //         {
    //             originalPos = board.activePiece.position;
    //             hadActivePiece = true;
                
    //             // Move piece to target column for bomb execution
    //             board.activePiece.position = new Vector3Int(
    //                 targetColumn + board.Bounds.xMin,
    //                 board.activePiece.position.y,
    //                 board.activePiece.position.z
    //             );
    //         }
            
    //         // Call PowerUpManager's public method directly
    //         powerUpManager.ExecuteBombImproved();
            
    //         // Restore original position if piece still exists
    //         if (hadActivePiece && board.activePiece != null)
    //         {
    //             board.activePiece.position = originalPos;
    //         }
    //     }
    //     else
    //     {
    //         Debug.LogError("PowerUpManager is null!");
    //     }
    // }

    // void ExecuteGravityPowerup()
    // {
    //     Debug.Log($"🐛 UNITY DEBUG: Executing gravity powerup");
    
    //     if (powerUpManager != null)
    //     {
    //         // Call PowerUpManager's public method directly
    //         powerUpManager.ExecuteGravity();
    //     }
    //     else
    //     {
    //         Debug.LogError("PowerUpManager is null!");
    //     }
    // }

    // void ExecuteBottomClearPowerup()
    // {
    //     Debug.Log($"🐛 UNITY DEBUG: Executing bottom clear powerup");
    
    //     if (powerUpManager != null)
    //     {
    //         // Call PowerUpManager's public method directly
    //         powerUpManager.ExecuteLineBlaster();
    //     }
    //     else
    //     {
    //         Debug.LogError("PowerUpManager is null!");
    //     }
    // }

    // int[] GetBoardStateArray()
    // {
    //     // Create array to hold board state
    //     int boardWidth = board.boardSize.x;
    //     int boardHeight = board.boardSize.y;
    //     int[] boardArray = new int[boardWidth * boardHeight];
        
    //     // Fill array with current board state
    //     for (int y = 0; y < boardHeight; y++)
    //     {
    //         for (int x = 0; x < boardWidth; x++)
    //         {
    //             Vector3Int pos = new Vector3Int(x + board.Bounds.xMin, y + board.Bounds.yMin, 0);
                
    //             // Check if there's a tile at this position
    //             if (board.tilemap.GetTile(pos) != null)
    //             {
    //                 boardArray[y * boardWidth + x] = 1; // Block present
    //             }
    //             else
    //             {
    //                 boardArray[y * boardWidth + x] = 0; // Empty space
    //             }
    //         }
    //     }
        
    //     return boardArray;
    // }

    // int GetHighestBlockInColumn(int column)
    // {
    //     for (int row = board.boardSize.y - 1; row >= 0; row--)
    //     {
    //         Vector3Int pos = new Vector3Int(column + board.Bounds.xMin, row + board.Bounds.yMin, 0);
    //         if (board.tilemap.GetTile(pos) != null)
    //         {
    //             return row;
    //         }
    //     }
    //     return -1;
    // }

    // TetrisImpactMetrics CalculateImpactMetrics(int[] boardBefore, int[] boardAfter)
    // {
    //     // Calculate differences between before and after
    //     int blocksRemoved = 0;
    //     int blocksAdded = 0;
        
    //     for (int i = 0; i < boardBefore.Length && i < boardAfter.Length; i++)
    //     {
    //         if (boardBefore[i] != 0 && boardAfter[i] == 0)
    //             blocksRemoved++;
    //         else if (boardBefore[i] == 0 && boardAfter[i] != 0)
    //             blocksAdded++;
    //     }
        
    //     // Calculate lines cleared (simplified)
    //     int linesCleared = blocksRemoved / board.boardSize.x;
        
    //     return new TetrisImpactMetrics
    //     {
    //         lines_cleared = linesCleared,
    //         holes_filled = blocksRemoved - (linesCleared * board.boardSize.x),
    //         bumpiness_reduced = UnityEngine.Random.Range(0, 3),
    //         height_reduced = UnityEngine.Random.Range(0, 2),
    //         actual_impact = blocksRemoved * 2.0f + linesCleared * 10.0f,
    //         scoreBonus = blocksRemoved * 10 + linesCleared * 100,
    //         blocks_moved = blocksRemoved,
    //         blocks_removed = blocksRemoved
    //     };
    // }

    // void SendPowerupError(string responseType, string errorMessage)
    // {
    //     Debug.LogError($"🐛 UNITY ERROR: {responseType} - {errorMessage}");
        
    //     var response = new
    //     {
    //         type = responseType,
    //         success = false,
    //         error = errorMessage,
    //         timestamp = Time.time
    //     };
        
    //     SocketManager.Instance.SendEvent(responseType, response);
    // }

    // void ClearCurrentPowerup()
    // {
    //     currentPowerupType = "none";
    //     hasPowerup = false;
    // }    

    // void OnDestroy()
    // {
    //     if (SocketManager.Instance != null)
    //     {
    //         SocketManager.Instance.OnCommandReceived -= HandleCommand;
    //         SocketManager.Instance.OnPythonConnected -= OnPythonConnected;
    //         SocketManager.Instance.OnPythonDisconnected -= OnPythonDisconnected;
    //     }
    // }


    // /// <summary>
    // /// Enumerate all valid (column, rotation) placements for the current piece,
    // /// simulate each drop on a cloned board, and return metrics.
    // /// </summary>
    // private Dictionary<(int col, int rot), float[]> GetMoveMetricsForCurrentPiece()
    // {
    //     var results = new Dictionary<(int, int), float[]>();

    //     if (currentPiece == null || board == null)
    //     {
    //         Debug.LogWarning("Cannot calculate move metrics: piece or board is null");
    //         return results;
    //     }
    //     // Create snapshots for simulation
    //     var origBoard = new BoardData(board);
    //     var origPiece = new PieceState(currentPiece);
    //     if (origBoard.IsGameOverCondition())
    //     {
    //         Debug.Log("Game over condition detected - center columns in top row are blocked");
    //         return results; // Return empty dictionary
    //     }

    //     int w = board.boardSize.x;
    //     int halfW = board.Bounds.width / 2;
    //     // Debug.Log("orgiBoard=== before sim");
    //     // Debug.Log(origBoard.DumpToString());


    //     for (int rot = 0; rot < currentPiece.data.RotationCount; rot++)
    //     {
    //         for (int colIdx = 0; colIdx < w; colIdx++)
    //         {
    //             try
    //             {

    //                 // Create fresh clones for simulation
    //                 var simBoard = origBoard.Clone();
    //                 var simPiece = origPiece.Clone();

    //                 int rotCount = simPiece.data.RotationCount;
    //                 int steps = (rot - simPiece.rotationIndex + rotCount) % rotCount;

    //                 for (int i = 0; i < steps; i++)
    //                 {
    //                     simPiece.RotateCW(simBoard);
    //                 }





    //                 // Move horizontally
    //                 int xOffset = simBoard.xOffset;
    //                 int minLocalX = simPiece.Cells.Min(c => c.x);
    //                 int desiredX = colIdx + xOffset - minLocalX;

    //                 // Build the candidate position
    //                 var testPos = new Vector2Int(desiredX, simPiece.position.y);

    //                 // Check validity
    //                 bool canPlace = simBoard.IsValidPosition(simPiece, testPos);

    //                 if (!canPlace)
    //                 {
    //                     // This column/rotation simply can't fit here
    //                     continue;
    //                 }

    //                 // It’s safe—move the piece there
    //                 simPiece.position = testPos;


    //                 // Hard drop
    //                 while (simBoard.IsValidPosition(simPiece, simPiece.position + Vector2Int.down))
    //                 {
    //                     simPiece.position += Vector2Int.down;
    //                 }


    //                 //     Debug.Log($"Simulated placement @ col={colIdx}, rot={rot}:\n" +
    //                 //   simBoard.DumpToString());
    //                 // Place piece and calculate metrics
    //                 int lines = simBoard.PlaceAndClear(simPiece);
    //     //             Debug.Log($"After placing rot={rot} @ col={colIdx}, linesCleared={lines}:\n"
    //     //   + simBoard.DumpToString());
    //                 int[] heights = simBoard.GetColumnHeights();
    //                 float holes = simBoard.CountHoles();
    //                 float bumpiness = simBoard.GetBumpinessScore();
    //                 float height = simBoard.CalculateStackHeight();
    //                 results[(colIdx, rot)] = new float[] { lines, holes, bumpiness, height
    //                 };

    //             }
    //             catch (System.Exception e)
    //             {
    //                 Debug.LogError($"Error calculating metrics for col={colIdx}, rot={rot}: {e.Message}");
    //                 continue;
    //             }
    //         }
    //     }


    //     // now print out every move and its metrics:
    //     foreach (var kvp in results)
    //     {
    //         // deconstruct the tuple key into column index and rotation
    //         var (colIdx, rot) = kvp.Key;
    //         float[] metrics = kvp.Value;

    //         // metrics[0] = lines cleared
    //         // metrics[1] = holes created
    //         // metrics[2] = bumpiness
    //         // metrics[3] = resulting height
    //         // Debug.Log($"Move → Column: {colIdx}, Rotation: {rot} | " +
    //         //           $"Lines: {metrics[0]}, Holes: {metrics[1]}, " +
    //         //           $"Bumpiness: {metrics[2]}, Height: {metrics[3]}");
    //     }
    //     // Debug.Log($"Calculated metrics for {results.Count} valid moves");
    //     prevResults = results;
    //     return results;
    // }
    // Inside the BoardData class


    }
}

// public class TetrisImpactMetrics
// {
//     public int lines_cleared { get; set; }
//     public int holes_filled { get; set; }
//     public int bumpiness_reduced { get; set; }
//     public int height_reduced { get; set; }
//     public float actual_impact { get; set; }
//     public int scoreBonus { get; set; }
//     public int blocks_moved { get; set; }
//     public int blocks_removed { get; set; }
// }


