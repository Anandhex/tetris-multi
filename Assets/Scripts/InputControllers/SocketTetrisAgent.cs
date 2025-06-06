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
    public int curriculumBoardPreset = 1;
    public int allowedTetrominoTypes = 1;

    private int currentLinesCleared = 0;

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
        if (!IsActionValid(actionIndex))
        {
            Debug.LogWarning($"Invalid action {actionIndex} rejected");
            lastReward = -50f; // Penalty for invalid action
            SendGameState();
            return;
        }

        // Simple mapping: 40 actions = 10 columns × 4 rotations
        int targetColumnIndex = actionIndex / 4;  // 0-9
        targetRotation = actionIndex % 4;         // 0-3

        // Map column index to actual board position
        targetColumn = GetBoardColumnFromIndex(targetColumnIndex);

        Debug.Log($"Action {actionIndex}: Column Index {targetColumnIndex} -> Board Column {targetColumn}, Rotation {targetRotation}");

        isExecutingAction = true;
        actionCompleted = false;
        waitingForNewPiece = false;

        StartCoroutine(ExecuteDirectPlacement());
    }
    int GetBoardColumnFromIndex(int columnIndex)
    {
        // Map action column index (0-9) to board coordinates
        var bounds = board.Bounds;

        // For a 10-wide board, distribute columns evenly across the board width
        if (bounds.width != 10)
        {
            Debug.LogWarning($"Board width is {bounds.width}, not 10. Mapping may be incorrect.");
        }

        // Simple mapping: column index directly maps to board position
        int boardColumn = bounds.xMin + columnIndex;

        // Clamp to board bounds just in case
        boardColumn = Mathf.Clamp(boardColumn, bounds.xMin, bounds.xMax - 1);

        return boardColumn;
    }
    IEnumerator ExecuteDirectPlacement()
    {
        if (currentPiece == null)
        {
            isExecutingAction = false;
            yield break;
        }

        // Store original state for debugging
        Vector3Int originalPosition = currentPiece.position;
        int originalRotation = currentPiece.rotationIndex;

        Debug.Log($"Starting placement: From {originalPosition} (rot {originalRotation}) to column {targetColumn} (rot {targetRotation})");

        // Step 1: Apply target rotation
        yield return StartCoroutine(RotatePiece());

        // Step 2: Move to target column  
        yield return StartCoroutine(MovePieceToColumn());

        // Step 3: Drop piece
        yield return StartCoroutine(DropPiece());

        // Step 4: Finalize placement
        FinalizePlacement();
    }
    IEnumerator RotatePiece()
    {
        int currentRotation = currentPiece.rotationIndex;
        int rotationsNeeded = (targetRotation - currentRotation + 4) % 4;

        for (int i = 0; i < rotationsNeeded; i++)
        {
            board.Clear(currentPiece);

            // Store current state for potential revert
            Vector3Int positionBeforeRotation = currentPiece.position;
            int rotationBeforeAttempt = currentPiece.rotationIndex;

            currentPiece.Rotate(1);

            // Try current position first
            if (board.IsValidPosition(currentPiece, currentPiece.position))
            {
                board.Set(currentPiece);
                yield return new WaitForSeconds(0.03f);
                continue;
            }

            // Try wall kicks
            Vector3Int[] kickOffsets = {
            Vector3Int.left, Vector3Int.right, Vector3Int.up,
            Vector3Int.left * 2, Vector3Int.right * 2,
            new Vector3Int(-1, 1, 0), new Vector3Int(1, 1, 0) // diagonal kicks
        };

            bool kickSuccessful = false;
            foreach (var offset in kickOffsets)
            {
                Vector3Int testPosition = positionBeforeRotation + offset;
                if (board.IsValidPosition(currentPiece, testPosition))
                {
                    currentPiece.position = testPosition;
                    kickSuccessful = true;
                    break;
                }
            }

            if (!kickSuccessful)
            {
                // Revert rotation
                currentPiece.Rotate(-1);
                currentPiece.position = positionBeforeRotation;
                Debug.LogWarning($"Could not rotate to target rotation {targetRotation}");
                break;
            }

            board.Set(currentPiece);
            yield return new WaitForSeconds(0.03f);
        }
    }

    IEnumerator MovePieceToColumn()
    {
        int currentColumn = currentPiece.position.x;
        int columnsToMove = targetColumn - currentColumn;

        // Move step by step
        while (columnsToMove != 0)
        {
            board.Clear(currentPiece);

            Vector3Int moveDirection = columnsToMove > 0 ? Vector3Int.right : Vector3Int.left;
            Vector3Int newPosition = currentPiece.position + moveDirection;

            if (board.IsValidPosition(currentPiece, newPosition))
            {
                currentPiece.position = newPosition;
                columnsToMove += (columnsToMove > 0) ? -1 : 1;
            }
            else
            {
                Debug.LogWarning($"Cannot move further. Target: {targetColumn}, Current: {currentPiece.position.x}");
                break;
            }

            board.Set(currentPiece);
            yield return new WaitForSeconds(0.03f);
        }
    }

    IEnumerator DropPiece()
    {
        int dropSteps = 0;
        while (true)
        {
            board.Clear(currentPiece);
            Vector3Int newPosition = currentPiece.position + Vector3Int.down;

            if (board.IsValidPosition(currentPiece, newPosition))
            {
                currentPiece.position = newPosition;
                board.Set(currentPiece);
                dropSteps++;
                yield return new WaitForSeconds(0.01f); // Fast drop
            }
            else
            {
                board.Set(currentPiece);
                break;
            }
        }

        Debug.Log($"Piece dropped {dropSteps} steps and landed at {currentPiece.position}");
    }

    void FinalizePlacement()
    {
        // Lock the piece in place
        board.Set(currentPiece);

        // Clear any completed lines
        board.ClearLines();

        // Calculate reward
        CalculatePlacementReward();

        // Mark action as completed
        actionCompleted = true;
        isExecutingAction = false;
        waitingForNewPiece = true;

        // Spawn new piece
        board.SpawnPiece();

        // Send final state
        StartCoroutine(SendStateAfterDelay());
    }

    IEnumerator SendStateAfterDelay()
    {
        yield return new WaitForSeconds(0.1f);
        SendGameState();
        waitingForNewPiece = false;
    }

    void CalculatePlacementReward()
    {
        // Reset reward
        lastReward = 0f;

        // Base placement reward
        lastReward += 1f;

        // Get current curriculum stage for adaptive scaling
        float maxHeight = curriculumBoardHeight;

        // === HOLE PENALTIES (Enhanced) ===
        int holes = board.CountHoles();
        float avgHoleDepth = board.CalculateAverageHoleDepth();
        lastReward -= holes * 25f * (1f + avgHoleDepth * 0.5f); // Deeper holes worse

        // === BUMPINESS PENALTY ===
        float bumpiness = board.CalculateBumpiness();
        lastReward -= bumpiness * 3f;

        // === WELL PENALTY ===
        int wells = board.CountWells();
        lastReward -= wells * 10f; // Wells are very bad

        // === HEIGHT MANAGEMENT (Curriculum-aware) ===
        float stackHeight = board.CalculateStackHeight();
        float heightRatio = stackHeight / maxHeight;

        // Progressive height penalty
        if (heightRatio <= 0.3f)
        {
            lastReward += 5f; // Good zone
        }
        else if (heightRatio <= 0.5f)
        {
            lastReward += 2f; // OK zone
        }
        else if (heightRatio <= 0.7f)
        {
            lastReward += 0f; // Neutral
        }
        else if (heightRatio <= 0.85f)
        {
            lastReward -= 10f; // Danger zone
        }
        else
        {
            lastReward -= 25f; // Critical zone
        }

        // === POTENTIAL LINE CLEAR BONUS ===
        int potentialLines = board.CountPotentialLineClears(2);
        lastReward += potentialLines * 8f; // Reward setting up line clears

        // === BOARD DENSITY MANAGEMENT ===
        float density = board.CalculateBoardDensity();
        if (density > 0.8f)
        {
            lastReward -= 15f; // Too dense is bad
        }
        else if (density < 0.3f && stackHeight > 5)
        {
            lastReward += 3f; // Good density with some height
        }

        // === T-SPIN OPPORTUNITY BONUS ===
        if (board.HasTSpinOpportunity())
        {
            lastReward += 20f; // Reward creating T-spin setups
        }

        // === PERFECT CLEAR BONUS ===
        if (board.IsPerfectClear())
        {
            lastReward += 500f; // Massive bonus for perfect clear
        }

        // === EFFICIENCY BONUS ===
        // Reward keeping the board clean and efficient
        float efficiency = 1f - (holes * 0.1f + bumpiness * 0.05f + wells * 0.15f);
        efficiency = Mathf.Clamp01(efficiency);
        lastReward += efficiency * 10f;
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

        // Action space information
        state.actionSpaceSize = 40;
        state.actionSpaceType = "column_rotation";
        state.isExecutingAction = isExecutingAction;
        state.waitingForAction = !isExecutingAction && !waitingForNewPiece && currentPiece != null;

        // Curriculum information - FIXED PROPERTY NAMES
        state.curriculumBoardHeight = curriculumBoardHeight;
        state.curriculumBoardPreset = curriculumBoardPreset;
        state.allowedTetrominoTypes = allowedTetrominoTypes;

        // Additional metrics
        state.holesCount = board.CountHoles();
        state.stackHeight = board.CalculateStackHeight();
        state.perfectClear = board.IsPerfectClear();
        state.linesCleared = board.GetTotalLinesCleared(); // This should be set when lines are actually cleared

        if (currentPiece != null)
        {
            state.currentPieceType = GetPieceTypeIndex(currentPiece.data);
            state.currentPieceX = currentPiece.position.x;
            state.currentPieceY = currentPiece.position.y;
            state.currentPieceRotation = currentPiece.rotationIndex;
        }
        else
        {
            state.currentPieceType = 0;
            state.currentPieceX = 0;
            state.currentPieceY = 0;
            state.currentPieceRotation = 0;
        }

        // Next piece type (single value, not array)
        if (!board.nextPieceData.Equals(default))
        {
            state.nextPieceType = GetPieceTypeIndex(board.nextPieceData);
        }
        else
        {
            state.nextPieceType = 0;
        }
        state.bumpiness = board.CalculateBumpiness();
        state.wells = board.CountWells();
        state.averageHoleDepth = board.CalculateAverageHoleDepth();
        state.potentialLineClears = board.CountPotentialLineClears(2);
        state.boardDensity = board.CalculateBoardDensity();
        state.tSpinOpportunity = board.HasTSpinOpportunity();

        // Efficiency score
        state.efficiencyScore = CalculateEfficiencyScore();
        SocketManager.Instance.SendGameState(state);

        // Reset reward after sending
        if (!gameOver)
        {
            currentLinesCleared = 0;
            lastReward = 0f;
        }
    }
    private bool IsActionValid(int actionIndex)
    {
        if (currentPiece == null || actionIndex < 0 || actionIndex >= 40)
            return false;

        int targetColumnIndex = actionIndex / 4;
        int targetRotation = actionIndex % 4;
        int targetColumn = GetBoardColumnFromIndex(targetColumnIndex);

        // Test if this action would be valid
        var bounds = board.Bounds;

        // Quick boundary check
        if (targetColumn < bounds.xMin || targetColumn >= bounds.xMax)
            return false;

        // Create a temporary piece to test the action
        // (You might need to implement a more sophisticated check)
        return true; // Simplified - implement full collision detection if needed
    }
    private float CalculateEfficiencyScore()
    {
        float holes = board.CountHoles();
        float bumpiness = board.CalculateBumpiness();
        float wells = board.CountWells();

        // Calculate efficiency as inverse of problems (0-1 scale)
        float problems = holes * 0.1f + bumpiness * 0.05f + wells * 0.15f;
        return Mathf.Clamp01(1f - problems / 10f); // Normalize to 0-1
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
        board.playerScore = 0;
        board.ResetTotalLinesCleared();

        if (board != null)
        {
            board.ApplyCurriculumBoardPreset();
        }
    }

    public void OnLinesCleared(int lines)
    {
        currentLinesCleared += lines;
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