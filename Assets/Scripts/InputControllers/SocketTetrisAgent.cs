using UnityEngine;
using System.Collections;

public class SocketTetrisAgent : MonoBehaviour, IPlayerInputController
{
    private Board board;
    private Piece currentPiece;
    private bool isExecutingAction = false;
    private bool waitingForNewPiece = false;
    
    [Header("Curriculum Parameters")]
    public float curriculumBoardHeight = 20f;
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
    
    void Update()
    {
        // Send game state periodically, but only when not executing an action
        if (Time.time - lastStateTime > stateUpdateInterval && !isExecutingAction)
        {
            SendGameState();
            lastStateTime = Time.time;
        }
    }
    
  void HandleCommand(GameCommand command)
{
    switch (command.type)
    {
        case "action":
            if (command.action != null && !isExecutingAction)
            {
                ExecuteAction(command.action.actionIndex);
            }
            break;
            
        case "curriculum_change":
            if (command.curriculum != null)
            {
                ApplyCurriculumChange(command.curriculum);
                SendCurriculumConfirmation();
            }
            break;
            
        case "curriculum_status_request":
            SendGameState(); // Will include current curriculum info
            break;
            
        case "reset":
            if (command.reset != null && command.reset.resetBoard)
            {
                ResetGame();
            }
            break;
    }
}

void SendCurriculumConfirmation()
{
    // Send confirmation that curriculum was applied
    if (SocketManager.Instance != null)
    {
        GameState confirmationState = new GameState();
        confirmationState.curriculumBoardHeight = curriculumBoardHeight;
        confirmationState.curriculumBoardPreset = curriculumBoardPreset;
        confirmationState.allowedTetrominoTypes = allowedTetrominoTypes;
        confirmationState.curriculumConfirmed = true;
        
        SocketManager.Instance.SendGameState(confirmationState);
    }
}  
    void ExecuteAction(int actionIndex)
    {
        if (actionIndex < 0 || actionIndex >= 40)
        {
            Debug.LogWarning($"Invalid action index: {actionIndex}. Must be 0-39.");
            return;
        }
        
        if (currentPiece == null || isExecutingAction)
        {
            Debug.LogWarning("Cannot execute action: no current piece or already executing action");
            return;
        }
        
        // Decode action: action = column * 4 + rotation
        targetColumn = actionIndex / 4;  // 0-9
        targetRotation = actionIndex % 4; // 0-3
        
        Debug.Log($"Executing action {actionIndex}: Column {targetColumn}, Rotation {targetRotation}");
        
        isExecutingAction = true;
        actionCompleted = false;
        waitingForNewPiece = false;
        
        StartCoroutine(ExecuteDirectPlacement());
    }
    
    IEnumerator ExecuteDirectPlacement()
    {
        if (currentPiece == null)
        {
            isExecutingAction = false;
            yield break;
        }
        
        // First, apply the target rotation
        int currentRotation = currentPiece.rotationIndex;
        int rotationsNeeded = (targetRotation - currentRotation + 4) % 4;
        
        for (int i = 0; i < rotationsNeeded; i++)
        {
            // Clear current position
            board.Clear(currentPiece);
            
            // Try to rotate (using rotate right for simplicity)
            currentPiece.Rotate(1);
            
            // Check if rotation is valid, if not try wall kicks or revert
            if (!board.IsValidPosition(currentPiece, currentPiece.position))
            {
                // Simple wall kick attempts
                Vector3Int[] kickOffsets = { 
                    Vector3Int.left, Vector3Int.right, 
                    Vector3Int.left * 2, Vector3Int.right * 2,
                    Vector3Int.up, Vector3Int.down 
                };
                
                bool kickSuccessful = false;
                foreach (var offset in kickOffsets)
                {
                    if (board.IsValidPosition(currentPiece, currentPiece.position + offset))
                    {
                        currentPiece.position += offset;
                        kickSuccessful = true;
                        break;
                    }
                }
                
                if (!kickSuccessful)
                {
                    // Revert rotation if no valid position found
                    currentPiece.Rotate(-1);
                    Debug.LogWarning($"Could not rotate to target rotation {targetRotation}");
                    break;
                }
            }
            
            // Set piece in new position
            board.Set(currentPiece);
            yield return new WaitForSeconds(0.05f); // Small delay to show rotation
        }
        
        // Now move to target column
        int currentColumn = currentPiece.position.x;
        int columnsToMove = targetColumn - currentColumn;
        
        // Move horizontally
        for (int i = 0; i < Mathf.Abs(columnsToMove); i++)
        {
            board.Clear(currentPiece);
            
            Vector3Int moveDirection = columnsToMove > 0 ? Vector3Int.right : Vector3Int.left;
            Vector3Int newPosition = currentPiece.position + moveDirection;
            
            if (board.IsValidPosition(currentPiece, newPosition))
            {
                currentPiece.position = newPosition;
            }
            else
            {
                Debug.LogWarning($"Could not move to target column {targetColumn}");
                break;
            }
            
            board.Set(currentPiece);
            yield return new WaitForSeconds(0.05f); // Small delay to show movement
        }
        
        // Finally, drop the piece
        while (true)
        {
            board.Clear(currentPiece);
            Vector3Int newPosition = currentPiece.position + Vector3Int.down;
            
            if (board.IsValidPosition(currentPiece, newPosition))
            {
                currentPiece.position = newPosition;
                board.Set(currentPiece);
                yield return new WaitForSeconds(0.02f); // Fast drop
            }
            else
            {
                // Piece has landed
                board.Set(currentPiece);
                break;
            }
        }
        
        // Lock the piece
        board.Set(currentPiece);
        board.ClearLines();
        
        // Calculate reward for this placement
        CalculatePlacementReward();
        
        // Mark action as completed
        actionCompleted = true;
        isExecutingAction = false;
        waitingForNewPiece = true;
        
        // Spawn new piece
        board.SpawnPiece();
        
        // Send final state for this action
        yield return new WaitForSeconds(0.1f);
        SendGameState();
        waitingForNewPiece = false;
    }
    
    void CalculatePlacementReward()
    {
        // Reset reward
        lastReward = 0f;
        
        // Small reward for successful placement
        lastReward += 1f;
        
        // Penalty for creating holes
        int holes = board.CountHoles();
        lastReward -= holes * 2f;
        
        // Penalty for high stacks
        float stackHeight = board.CalculateStackHeight();
        if (stackHeight > 15)
        {
            lastReward -= (stackHeight - 15) * 1f;
        }
        
        // Bonus for keeping stack low
        if (stackHeight < 10)
        {
            lastReward += (10 - stackHeight) * 0.5f;
        }
        
        // Big bonus for perfect clear
        if (board.IsPerfectClear())
        {
            lastReward += 50f;
        }
        
        // Line clear rewards are handled separately in OnLinesCleared
    }
    
    void ApplyCurriculumChange(CurriculumData curriculum)
    {
        curriculumBoardHeight = curriculum.boardHeight;
        curriculumBoardPreset = curriculum.boardPreset;
        allowedTetrominoTypes = curriculum.allowedTetrominoTypes;
        
        Debug.Log($"Curriculum changed: Height={curriculumBoardHeight}, Preset={curriculumBoardPreset}, Types={allowedTetrominoTypes}");
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
            board.ApplyCurriculumBoardPreset();
        }
    }
    
    void OnPythonConnected()
    {
        Debug.Log("Python AI connected - Ready for 40-action Tetris (10 columns × 4 rotations)!");
        SendGameState();
    }
    
    void OnPythonDisconnected()
    {
        Debug.Log("Python AI disconnected");
    }
    
    void SendGameState()
    {
        if (board == null || SocketManager.Instance == null)
            return;
            
        GameState state = new GameState();
        
        // Get board state
        state.board = GetBoardState();
        state.currentPiece = GetCurrentPieceState();
        state.nextPiece = GetNextPieceState();
        state.piecePosition = currentPiece != null ? (Vector2Int)currentPiece.position : Vector2Int.zero;
        state.score = board.playerScore;
        state.gameOver = gameOver;
        state.reward = lastReward;
        state.episodeEnd = gameOver;
        
        // Action space information - Now these properties exist in GameState
        state.actionSpaceSize = 40;
        state.actionSpaceType = "column_rotation";
        state.isExecutingAction = isExecutingAction;
        state.waitingForAction = !isExecutingAction && !waitingForNewPiece && currentPiece != null;
           // Curriculum information
    state.curriculumBoardHeight = curriculumBoardHeight;
    state.curriculumBoardPreset = curriculumBoardPreset;
    state.allowedTetrominoTypes = allowedTetrominoTypes;
        // Additional metrics
        state.holesCount = board.CountHoles();
        state.stackHeight = board.CalculateStackHeight();
        state.perfectClear = board.IsPerfectClear();
        state.linesCleared = 0; // This should be set when lines are actually cleared
        
        SocketManager.Instance.SendGameState(state);
        
        // Reset reward after sending
        if (!gameOver)
    {
        lastReward = 0f;
    }
    }
    
    float[] GetBoardState()
    {
        var bounds = board.Bounds;
        float[] boardState = new float[bounds.width * bounds.height];
        
        int index = 0;
        // Send board from top to bottom for easier visualization
        for (int y = bounds.yMax - 1; y >= bounds.yMin; y--)
        {
            for (int x = bounds.xMin; x < bounds.xMax; x++)
            {
                Vector3Int pos = new Vector3Int(x, y, 0);
                boardState[index++] = board.tilemap.HasTile(pos) ? 1f : 0f;
            }
        }
        
        return boardState;
    }
    
    int[] GetCurrentPieceState()
    {
        if (currentPiece == null)
            return new int[] { 0, 0, 0, 0 }; // type, rotation, x, y
            
        return new int[] 
        {
            GetPieceTypeIndex(currentPiece.data),
            currentPiece.rotationIndex,
            currentPiece.position.x,
            currentPiece.position.y
        };
    }
    
    int[] GetNextPieceState()
    {
        if (board.nextPieceData.Equals(default))
            return new int[] { 0 };
            
        return new int[] { GetPieceTypeIndex(board.nextPieceData) };
    }
    
    int GetPieceTypeIndex(TetrominoData data)
    {
        for (int i = 0; i < board.tetrominoes.Length; i++)
        {
            if (board.tetrominoes[i].Equals(data))
                return i;
        }
        return 0;
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
        // Send state when new piece spawns
        if (!isExecutingAction)
        {
            SendGameState();
        }
    }
    
    public void SetBoard(Board gameBoard)
    {
        board = gameBoard;
    }

    public void OnGameOver()
    {
        gameOver = true;
        lastReward = -10f; // Penalty for game over
        isExecutingAction = false;

        // Send game over state BEFORE resetting
        SendGameState();

        // Wait a moment to ensure the message is sent
        StartCoroutine(DelayedReset());
    }
private IEnumerator DelayedReset()
{
    // Give time for the game over state to be sent and processed
    yield return new WaitForSeconds(0.5f);
    
    // Now reset the game
    gameOver = false;
    lastReward = 0f;
    isExecutingAction = false;
    waitingForNewPiece = false;
    actionCompleted = false;
    
    if (board != null)
    {
        board.ApplyCurriculumBoardPreset();
    }
}
    
   public void OnLinesCleared(int lines)
{
    // Reward for line clears (exponential for multi-line clears)
    lastReward += lines * lines * 25f; // 25, 100, 225, 400 for 1, 2, 3, 4 lines
    
    if (lines == 4) // Tetris bonus
    {
        lastReward += 100f;
    }
    
    // Send updated state immediately after line clear
    SendGameState();
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
}