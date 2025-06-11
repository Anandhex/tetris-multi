using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System;

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

    void Update()
    {
        // Send game state periodically, but only when not executing an action
        // if (!isExecutingAction && Time.time - lastStateTime > stateUpdateInterval)
        // {
        //     SendGameState();   // fires every 0.1 s even while waiting for next piece
        //     lastStateTime = Time.time;
        // }
        // if (pythonConnected && !isExecutingAction && Time.time - lastStateTime > stateUpdateInterval)
        // {
        //     SendGameState();
        //     lastStateTime = Time.time;
        // }
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
            return;
        }
        if (currentPiece == null || isExecutingAction)
        {
            return;
        }

        // 1) total rotations for this piece
        int rotationCount = currentPiece.data.RotationCount;  // always 4 :contentReference[oaicite:0]{index=0}

        // 2) decode flat action → columnIdx (0–9) and rotation (0–3)
        int columnIdx = actionIndex / rotationCount;  // e.g. 0–9
        int rotation = actionIndex % rotationCount;  // e.g. 0–3

        // 3) clamp into valid ranges
        columnIdx = Mathf.Clamp(columnIdx, 0, board.boardSize.x - 1);
        rotation = Mathf.Clamp(rotation, 0, rotationCount - 1);

        // 4) map columnIdx (0..9) into your board’s [-5..+4] x-coordinate
        int halfWidth = board.Bounds.width / 2;                         // 10/2=5 :contentReference[oaicite:1]{index=1}
        int boardColumn = columnIdx - halfWidth;

        // store for the placement coroutine
        targetColumn = boardColumn;
        targetRotation = rotation;

        Debug.Log($"Action {actionIndex}: idx={columnIdx} → x={boardColumn}, rot={rotation}");

        isExecutingAction = true;
        actionCompleted = false;
        waitingForNewPiece = false;

        StartCoroutine(ExecuteDirectPlacement());
    }
    int GetBoardColumnFromIndex(int columnIndex)
    {
        Vector3Int spawnCell = board.spawnPosition;
        int halfWidth = board.Bounds.width / 2;
        int leftmostX = spawnCell.x - halfWidth;
        int boardColumn = leftmostX + columnIndex;
        return Mathf.Clamp(boardColumn, board.Bounds.xMin, board.Bounds.xMax - 1);
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
                break;
            }

            board.Set(currentPiece);
            yield return new WaitForSeconds(0.03f);
        }
    }

    IEnumerator MovePieceToColumn()
    {
        // compute based on leftmost block
        int minCellX = currentPiece.cells.Min(c => c.x);
        int desiredPivotX = targetColumn - minCellX;


        while (currentPiece.position.x != desiredPivotX)
        {
            var dir = (desiredPivotX > currentPiece.position.x)
                      ? Vector3Int.right
                      : Vector3Int.left;
            var newPos = currentPiece.position + dir;

            board.Clear(currentPiece);
            if (!board.IsValidPosition(currentPiece, newPos))
            {

                yield break;
            }

            currentPiece.position = newPos;
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

    }

    void FinalizePlacement()
    {
        // Lock the piece in place
        board.Set(currentPiece);

        // Clear any completed lines
        board.ClearLines();

        // Calculate reward
        CalculatePlacementReward();
        SendGameState();

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
        pythonConnected = true;
        Debug.Log("Python AI connected - Ready for 40-action Tetris (10 columns × 4 rotations)!");
        gameOver = false;
        SendGameState();
        lastStateTime = Time.time;
    }

    void OnPythonDisconnected()
    {
        Debug.Log("Python AI disconnected");
    }

    void SendGameState()
    {
        if (board == null || SocketManager.Instance == null)
            return;

        if (currentPiece == null || currentPiece.cells == null)
        {
            Debug.LogWarning("SendGameState(): currentPiece is null — skipping state send.");
            return;
        }

        // Get ground state and metrics without modifying the actual board
        var groundData = GetGroundStateAndMetrics();

        GameState state = new GameState();

        // Use the calculated ground state data
        state.board = groundData.board;
        state.heights = groundData.heights;
        state.holesCount = groundData.holesCount;
        state.covered = groundData.covered;
        state.bumpiness = groundData.bumpiness;
        state.stackHeight = groundData.stackHeight;
        state.perfectClear = groundData.perfectClear;

        // Get piece information
        state.currentPiece = GetCurrentPieceState();
        state.nextPiece = GetNextPieceState();
        state.piecePosition = (Vector2Int)currentPiece.position;

        // Game metrics
        state.score = board.playerScore;
        state.gameOver = gameOver;
        state.reward = lastReward;
        state.episodeEnd = gameOver;

        // Action space information
        state.actionSpaceSize = 40;
        state.actionSpaceType = "column_rotation";
        state.isExecutingAction = isExecutingAction;
        state.waitingForAction = !isExecutingAction && !waitingForNewPiece && currentPiece != null;

        // Curriculum information
        state.curriculumBoardHeight = curriculumBoardHeight;
        state.curriculumBoardPreset = curriculumBoardPreset;
        state.allowedTetrominoTypes = allowedTetrominoTypes;

        // Other metrics
        state.linesCleared = board.playerScore / 100;
        state.validActions = currentPiece.validActions ?? new List<int>();

        // Send the game state
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
        if (board == null)
        {
            return;
        }
        if (currentPiece != null)
        {
            currentPiece.ComputeAndStoreValidMoves(board);
        }
        // Send state when new piece spawns

    }

    public void SetBoard(Board gameBoard)
    {
        board = gameBoard;
        board.inputController = this;
        SendGameState();
        lastStateTime = Time.time;   // reset your timer so you don’t immediately resend
    }
    public static List<int> GenerateValidActionIndices(Board board, TetrominoData data)
    {
        var validActions = new List<int>();
        var bounds = board.Bounds;
        var kicks = Data.WallKicks[data.tetromino];
        const int FULL_ROTATIONS = 4;

        // Starting orientation (usually 0)
        int fromRot = 0;


        for (int rot = 0; rot < data.RotationCount; rot++)
        {
            var cells = Data.GetCells(data.tetromino, rot);

            // Compute piece footprint
            int minX = cells.Min(c => c.x);
            int maxX = cells.Max(c => c.x);
            int maxYOffset = cells.Max(c => c.y);


            // Determine horizontal range in world coordinates
            int colLo = bounds.xMin - minX;
            int colHi = bounds.xMax - maxX;

            // Map rotation transition to SRS kick row (0–7)
            int kickRow;
            switch ((fromRot, rot))
            {
                case (0, 1): kickRow = 0; break;
                case (1, 2): kickRow = 1; break;
                case (2, 3): kickRow = 2; break;
                case (3, 0): kickRow = 3; break;

                case (1, 0): kickRow = 4; break;
                case (2, 1): kickRow = 5; break;
                case (3, 2): kickRow = 6; break;
                case (0, 3): kickRow = 7; break;

                default:
                    // Fallback to safe index
                    kickRow = 0;
                    break;
            }

            for (int worldX = colLo; worldX <= colHi; worldX++)
            {

                // Calculate initial spawn position in world coords
                Vector3Int spawnCell = board.spawnPosition;
                var spawnPos = new Vector3Int(
                    worldX,
                    spawnCell.y - maxYOffset,
                    0
                );


                // Quick check at spawn
                if (!IsValidPlacement(board, cells, spawnPos, bounds))
                {
                    continue;
                }

                // Test wall kicks
                Vector3Int kickPos = default;
                bool kicked = false;
                for (int k = 0; k < kicks.GetLength(1); k++)
                {
                    var off = kicks[kickRow, k];
                    var p = spawnPos + new Vector3Int(off.x, off.y, 0);

                    if (IsValidPlacement(board, cells, p, bounds))
                    {
                        kickPos = p;
                        kicked = true;
                        break;
                    }
                    else
                    {
                    }
                }

                if (!kicked)
                {
                    continue;
                }

                // Test sliding
                int targetCol = kickPos.x - bounds.xMin;
                if (!CanSlideTo(board, cells, spawnPos, targetCol, bounds))
                {
                    continue;
                }

                // Test drop
                var dropPos = kickPos;
                int dropSteps = 0;
                while (dropPos.y > bounds.yMin &&
                       IsValidPlacement(board, cells, dropPos + Vector3Int.down, bounds))
                {
                    dropPos += Vector3Int.down;
                    dropSteps++;
                }

                if (IsValidPlacement(board, cells, dropPos, bounds))
                {
                    int actionIndex = (kickPos.x - bounds.xMin) * data.RotationCount + rot;
                    validActions.Add(actionIndex);
                }
                else
                {
                }
            }

            // Update fromRot for next transition
            fromRot = rot;
        }

        return validActions;
    }

    private static bool CanSlideTo(Board board, Vector2Int[] cells, Vector3Int startPos, int targetCol, RectInt bounds)
    {
        int currentCol = startPos.x - bounds.xMin;
        int dx = targetCol - currentCol;
        int step = Math.Sign(dx);

        var pos = startPos;
        for (int i = 0; i < Math.Abs(dx); i++)
        {
            pos += new Vector3Int(step, 0, 0);
            if (!IsValidPlacement(board, cells, pos, bounds))
                return false;
        }
        return true;
    }



    private static bool IsValidPlacement(Board board, Vector2Int[] cells, Vector3Int position, RectInt bounds)
    {
        foreach (Vector2Int cell in cells)
        {
            // Calculate absolute position of this cell
            Vector3Int tilePosition = new Vector3Int(position.x + cell.x, position.y + cell.y, 0);

            // Check if position is within bounds
            Vector2Int tilePos2D = new Vector2Int(tilePosition.x, tilePosition.y);
            if (!bounds.Contains(tilePos2D))
            {
                return false;
            }

            // Check if position is already occupied
            if (board.tilemap.HasTile(tilePosition))
            {
                return false;
            }
        }

        return true;
    }


    public void ResetAgent()
    {
        gameOver = false;
        lastReward = 0f;
        isExecutingAction = false;
        waitingForNewPiece = false;
        actionCompleted = false;
        // ...any other state fields you need to clear...
    }
    private void DebugValidMoves()
    {
        List<int> validActions = GenerateValidActionIndices(board, currentPiece.data);

        Debug.Log("=== VALID ACTIONS FOR CURRENT PIECE ===");
        foreach (int action in validActions)
        {
            int column = action / 4;
            int rotation = action % 4;
            Debug.Log($"ActionIndex: {action} → Column: {column}, Rotation: {rotation}");
        }

        Debug.Log("Press Enter to continue...");
        StartCoroutine(WaitForEnterKey());
    }

    private IEnumerator WaitForEnterKey()
    {
        Time.timeScale = 0f; // Pause the game

        while (!Input.GetKeyDown(KeyCode.Return))
        {
            yield return null;
        }

        Time.timeScale = 1f; // Resume
        Debug.Log("Resuming...");
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

    // Add this method to your SocketTetrisAgent class
    private GameState GetGroundStateAndMetrics()
    {
        var bounds = board.Bounds;
        var stateData = new GameState();

        // Create a copy of the board state WITHOUT the active piece
        bool[,] groundBoard = new bool[bounds.width, bounds.height];
        float[] boardArray = new float[bounds.width * bounds.height];
        int[] heights = new int[bounds.width];
        print(currentPiece);
        // Copy only the locked tiles (not the active piece)
        int arrayIndex = 0;
        for (int y = bounds.yMax - 1; y >= bounds.yMin; y--)
        {
            for (int x = bounds.xMin; x < bounds.xMax; x++)
            {
                Vector3Int pos = new Vector3Int(x, y, 0);
                bool hasTile = board.tilemap.HasTile(pos);

                // Check if this position is occupied by the current piece
                bool isActivePiece = false;
                if (currentPiece != null)
                {
                    foreach (Vector2Int cell in currentPiece.cells)
                    {
                        Vector3Int piecePos = new Vector3Int(
                            currentPiece.position.x + cell.x,
                            currentPiece.position.y + cell.y,
                            0
                        );
                        if (piecePos.x == x && piecePos.y == y)
                        {
                            isActivePiece = true;
                            break;
                        }
                    }
                }

                // Only count as filled if it's a tile AND not part of active piece
                bool isGroundTile = hasTile && !isActivePiece;

                int boardX = x - bounds.xMin;
                int boardY = y - bounds.yMin;
                groundBoard[boardX, boardY] = isGroundTile;
                boardArray[arrayIndex++] = isGroundTile ? 1f : 0f;
            }
        }

        // Calculate heights
        for (int x = 0; x < bounds.width; x++)
        {
            heights[x] = 0;
            for (int y = bounds.height - 1; y >= 0; y--)
            {
                if (groundBoard[x, y])
                {
                    heights[x] = y + 1;
                    break;
                }
            }
        }

        // Calculate holes
        int holes = 0;
        for (int x = 0; x < bounds.width; x++)
        {
            bool foundTop = false;
            for (int y = bounds.height - 1; y >= 0; y--)
            {
                if (groundBoard[x, y])
                {
                    foundTop = true;
                }
                else if (foundTop)
                {
                    holes++;
                }
            }
        }

        // Calculate covered holes
        int coveredHoles = 0;
        for (int x = 0; x < bounds.width; x++)
        {
            for (int y = 0; y < bounds.height - 1; y++)
            {
                if (!groundBoard[x, y] && groundBoard[x, y + 1])
                {
                    coveredHoles++;
                }
            }
        }

        // Calculate bumpiness
        float bumpiness = 0f;
        for (int x = 0; x < bounds.width - 1; x++)
        {
            bumpiness += Mathf.Abs(heights[x] - heights[x + 1]);
        }

        // Calculate stack height (max height)
        float stackHeight = 0f;
        foreach (int height in heights)
        {
            if (height > stackHeight)
                stackHeight = height;
        }

        // Check if perfect clear
        bool perfectClear = true;
        for (int x = 0; x < bounds.width && perfectClear; x++)
        {
            for (int y = 0; y < bounds.height && perfectClear; y++)
            {
                if (groundBoard[x, y])
                {
                    perfectClear = false;
                }
            }
        }

        // Fill the state data
        stateData.board = boardArray;
        stateData.heights = heights;
        stateData.holesCount = holes;
        stateData.covered = coveredHoles;
        stateData.bumpiness = bumpiness;
        stateData.stackHeight = stackHeight;
        stateData.perfectClear = perfectClear;

        return stateData;
    }
}


