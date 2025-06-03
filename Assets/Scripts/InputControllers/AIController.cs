using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine;
using Unity.MLAgents.Policies;
using System.Collections.Generic;
using System.Linq;
using System.Collections;

public class TetrisMLAgent : Agent, IPlayerInputController
{
    private Board board;
    public Piece currentPiece;
    private MLAgentDebugger debugger;

    // Simplified piece tracking - NO CACHE
    private int lastProcessedPieceInstanceId = -1;
    private bool isExecutingPlacement = false;
    private StatsRecorder m_StatsRecorder;
    private RewardWeights rewardWeights = new RewardWeights();

    [Header("Curriculum Parameters")]
    public int allowedTetrominoTypes = 7;
    public int curriculumBoardPreset;
    public float curriculumBoardHeight = 20f;
    public float curriculumDropSpeed = 0.75f;
    public float curriculumHolePenaltyWeight = 0.5f;
    public bool enableAdvancedMechanics = false;

    [Header("Debug")]
    public bool enableDebugLogs = true;

    // Input flags
    private bool moveLeft;
    private bool moveRight;
    private bool moveDown;
    private bool rotateLeft;
    private bool rotateRight;
    private bool hardDrop;

    public override void Initialize()
    {
        var envParams = Academy.Instance.EnvironmentParameters;
        m_StatsRecorder = Academy.Instance.StatsRecorder;
        
        // Get curriculum parameters
        allowedTetrominoTypes = (int)envParams.GetWithDefault("tetromino_types", 7f);
        curriculumBoardHeight = envParams.GetWithDefault("board_height", 20f);
        curriculumDropSpeed = envParams.GetWithDefault("drop_speed", 0.75f);
        curriculumHolePenaltyWeight = envParams.GetWithDefault("hole_penalty_weight", 0.5f);
        enableAdvancedMechanics = envParams.GetWithDefault("enable_t_spins", 0f) > 0.5f;
        curriculumBoardPreset = (int)envParams.GetWithDefault("board_preset", 6);

        // Apply curriculum settings
        rewardWeights.holeCreationPenalty *= curriculumHolePenaltyWeight;
        rewardWeights.clearReward = envParams.GetWithDefault("clearReward", 5.0f);
        rewardWeights.comboMultiplier = envParams.GetWithDefault("comboMultiplier", 0.5f);
        rewardWeights.perfectClearBonus = envParams.GetWithDefault("perfectClearBonus", 50.0f);

        if (enableDebugLogs)
            Debug.Log($"[TetrisAgent] Initialized - Allowed pieces: {allowedTetrominoTypes}");
    }

    private void Start()
    {
        if (board != null)
        {
            board.inputController = this;
            if (enableDebugLogs)
                Debug.Log("[TetrisAgent] Board input controller set");
        }
    }

    public override void OnEpisodeBegin()
    {
        // COMPLETE RESET - NO CACHE TO CLEAR
        lastProcessedPieceInstanceId = -1;
        isExecutingPlacement = false;
        
        // Update curriculum parameters
        var envParams = Academy.Instance.EnvironmentParameters;
        allowedTetrominoTypes = (int)envParams.GetWithDefault("tetromino_types", 7f);
        curriculumBoardHeight = envParams.GetWithDefault("board_height", 20f);
        curriculumDropSpeed = envParams.GetWithDefault("drop_speed", 0.75f);
        curriculumHolePenaltyWeight = envParams.GetWithDefault("hole_penalty_weight", 0.5f);
        enableAdvancedMechanics = envParams.GetWithDefault("enable_t_spins", 0f) > 0.5f;
        curriculumBoardPreset = (int)envParams.GetWithDefault("board_preset", 6);

        if (enableDebugLogs)
            Debug.Log($"[TetrisAgent] Episode Begin - Curriculum: {allowedTetrominoTypes} tetromino types");
    }

    private void Awake()
    {
        debugger = GetComponent<MLAgentDebugger>();
        if (debugger == null)
        {
            debugger = gameObject.AddComponent<MLAgentDebugger>();
        }

        board = GetComponentInChildren<Board>();

        // Set up behavior parameters
        var behavior = gameObject.GetComponent<BehaviorParameters>();
        if (behavior == null)
        {
            behavior = gameObject.AddComponent<BehaviorParameters>();
        }
        
        behavior.BehaviorName = "TetrisAgent";
        behavior.BrainParameters.VectorObservationSize = 218; // 34*6 + 7 + 7
        behavior.BrainParameters.NumStackedVectorObservations = 1;
        
        // Use exactly 34 actions to match maximum possible placements
        ActionSpec actionSpec = ActionSpec.MakeDiscrete(new int[] { 34 });
        behavior.BrainParameters.ActionSpec = actionSpec;

        // Set up decision requester
        var requestor = gameObject.GetComponent<DecisionRequester>();
        if (requestor == null)
        {
            requestor = gameObject.AddComponent<DecisionRequester>();
        }
        requestor.DecisionPeriod = 999999; // Very large so it doesn't auto-request
        requestor.TakeActionsBetweenDecisions = false;
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        if (currentPiece == null || board == null || isExecutingPlacement)
        {
            if (enableDebugLogs && currentPiece != null)
                Debug.Log($"[TetrisAgent] OnActionReceived blocked - executing: {isExecutingPlacement}");
            return;
        }

        // Check if this is a new piece
        int currentPieceId = currentPiece.GetInstanceID();
        if (currentPieceId == lastProcessedPieceInstanceId)
        {
            if (enableDebugLogs)
                Debug.Log($"[TetrisAgent] Ignoring already processed piece {currentPieceId}");
            return;
        }

        lastProcessedPieceInstanceId = currentPieceId;
        isExecutingPlacement = true;

        if (enableDebugLogs)
            Debug.Log($"[TetrisAgent] Processing new piece: {currentPiece.data.tetromino} (ID: {currentPieceId})");

        // Get valid placements - FRESH CALCULATION, NO CACHE
        int[,] currentBoard = GetBoardState();
        List<PlacementInfo> validPlacements = GenerateAllPossiblePlacements(currentPiece, currentBoard);

        if (validPlacements.Count == 0)
        {
            if (enableDebugLogs)
                Debug.Log("[TetrisAgent] No valid placements - Game Over");
            
            AddReward(rewardWeights.deathPenalty);
            m_StatsRecorder.Add("Tetris/GameOver", 1);
            EndEpisode();
            return;
        }

        // Get selected action
        int selectedAction = Mathf.Clamp(actions.DiscreteActions[0], 0, validPlacements.Count - 1);
        PlacementInfo selectedPlacement = validPlacements[selectedAction];

        if (enableDebugLogs)
            Debug.Log($"[TetrisAgent] Action: {selectedAction}/{validPlacements.Count}, " +
                     $"Target: Column {selectedPlacement.targetColumn}, Rotation {selectedPlacement.targetRotation}");

        // Execute placement
        StartCoroutine(ExecutePlacementCoroutine(selectedPlacement));
    }

    public override void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
    {
        if (currentPiece == null || board == null)
        {
            // Mask all actions if no valid piece
            for (int i = 0; i < 34; i++)
            {
                actionMask.SetActionEnabled(0, i, false);
            }
            return;
        }

        // Generate valid placements - FRESH CALCULATION, NO CACHE
        int[,] currentBoard = GetBoardState();
        List<PlacementInfo> validPlacements = GenerateAllPossiblePlacements(currentPiece, currentBoard);

        // Enable valid actions, disable invalid ones
        for (int i = 0; i < 34; i++)
        {
            actionMask.SetActionEnabled(0, i, i < validPlacements.Count);
        }

        if (enableDebugLogs)
            Debug.Log($"[TetrisAgent] Action mask: {validPlacements.Count}/34 valid placements for {currentPiece.data.tetromino}");
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (currentPiece == null)
        {
            // Add zeros if no current piece
            for (int i = 0; i < 218; i++)
            {
                sensor.AddObservation(0f);
            }
            return;
        }

        // FRESH CALCULATION EVERY TIME - NO CACHE
        int[,] currentBoard = GetBoardState();
        List<PlacementInfo> validPlacements = GenerateAllPossiblePlacements(currentPiece, currentBoard);

        // 1. All possible placements (34 placements × 6 features = 204 observations)
        for (int i = 0; i < 34; i++)
        {
            if (i < validPlacements.Count)
            {
                PlacementInfo placement = validPlacements[i];
                sensor.AddObservation(placement.linesCleared / 4f);
                sensor.AddObservation(placement.aggregateHeight / 200f);
                sensor.AddObservation(placement.maxHeight / 20f);
                sensor.AddObservation(placement.holes / 40f);
                sensor.AddObservation(placement.bumpiness / 100f);
                sensor.AddObservation(placement.wellDepth / 50f);
            }
            else
            {
                // Pad with zeros
                for (int j = 0; j < 6; j++)
                {
                    sensor.AddObservation(0f);
                }
            }
        }

        // 2. Current piece one-hot (7 observations)
        for (int i = 0; i < 7; i++)
        {
            sensor.AddObservation(((int)currentPiece.data.tetromino == i) ? 1f : 0f);
        }

        // 3. Next piece one-hot (7 observations)
        if (board != null && board.nextPieceData.tetromino != Tetromino.I) // Check if next piece is valid
        {
            for (int i = 0; i < 7; i++)
            {
                sensor.AddObservation(((int)board.nextPieceData.tetromino == i) ? 1f : 0f);
            }
        }
        else
        {
            for (int i = 0; i < 7; i++)
            {
                sensor.AddObservation(0f);
            }
        }
    }

    private IEnumerator ExecutePlacementCoroutine(PlacementInfo placement)
    {
        if (enableDebugLogs)
            Debug.Log($"[TetrisAgent] Executing placement - Current pos: {currentPiece.position}, Current rot: {currentPiece.rotationIndex}");

        // Clear all input flags first
        ClearAllInputs();
        yield return new WaitForFixedUpdate();

        // Rotate to target rotation
        int rotationAttempts = 0;
        while (currentPiece != null && currentPiece.rotationIndex != placement.targetRotation && rotationAttempts < 4)
        {
            ClearAllInputs();
            rotateRight = true;
            yield return new WaitForFixedUpdate();
            ClearAllInputs();
            yield return new WaitForFixedUpdate();
            rotationAttempts++;
        }

        if (currentPiece == null)
        {
            if (enableDebugLogs)
                Debug.Log("[TetrisAgent] Piece became null during rotation");
            isExecutingPlacement = false;
            yield break;
        }

        // Move to target column
        int currentCol = currentPiece.position.x - board.Bounds.xMin;
        int moveAttempts = 0;
        while (currentPiece != null && currentCol != placement.targetColumn && moveAttempts < 15)
        {
            ClearAllInputs();
            
            if (currentCol < placement.targetColumn)
            {
                moveRight = true;
            }
            else
            {
                moveLeft = true;
            }
            
            yield return new WaitForFixedUpdate();
            ClearAllInputs();
            yield return new WaitForFixedUpdate();
            
            if (currentPiece != null)
            {
                currentCol = currentPiece.position.x - board.Bounds.xMin;
            }
            moveAttempts++;
        }

        if (currentPiece == null)
        {
            if (enableDebugLogs)
                Debug.Log("[TetrisAgent] Piece became null during movement");
            isExecutingPlacement = false;
            yield break;
        }

        // Hard drop
        ClearAllInputs();
        hardDrop = true;
        yield return new WaitForFixedUpdate();
        ClearAllInputs();

        // Calculate reward
        CalculatePlacementReward(placement);

        if (enableDebugLogs)
            Debug.Log($"[TetrisAgent] Placement complete");

        // Wait a bit before allowing next piece processing
        yield return new WaitForSeconds(0.1f);
        isExecutingPlacement = false;
    }

    private void ClearAllInputs()
    {
        moveLeft = false;
        moveRight = false;
        moveDown = false;
        rotateLeft = false;
        rotateRight = false;
        hardDrop = false;
    }

    private void CalculatePlacementReward(PlacementInfo placement)
    {
        float reward = 0f;

        // Line clear rewards
        reward += placement.linesCleared * rewardWeights.clearReward;
        if (placement.linesCleared == 4)
        {
            reward += rewardWeights.perfectClearBonus; // Tetris bonus
        }

        // Penalties for bad placements
        reward -= placement.holes * rewardWeights.holeCreationPenalty;
        reward -= placement.aggregateHeight * 0.04f;
        reward -= placement.bumpiness * 0.03f;
        reward -= placement.maxHeight * 0.05f; // Penalty for high stacks

        // Small step penalty to encourage faster play
        reward -= 0.001f;

        AddReward(reward);

        // Record stats
        m_StatsRecorder.Add("Tetris/LinesCleared", placement.linesCleared);
        m_StatsRecorder.Add("Tetris/Holes", placement.holes);
        m_StatsRecorder.Add("Tetris/Height", placement.maxHeight);
        m_StatsRecorder.Add("Tetris/PlacementReward", reward);

        if (enableDebugLogs)
            Debug.Log($"[TetrisAgent] Reward: {reward:F3} (Lines: {placement.linesCleared}, Holes: {placement.holes}, Height: {placement.maxHeight})");
    }

    public void SetCurrentPiece(Piece piece)
    {
        if (piece == null) 
        {
            if (enableDebugLogs)
                Debug.Log("[TetrisAgent] SetCurrentPiece called with null");
            return;
        }
        
        currentPiece = piece;
        
        if (enableDebugLogs)
            Debug.Log($"[TetrisAgent] SetCurrentPiece called: {piece.data.tetromino} (ID: {piece.GetInstanceID()})");
        
        // Request decision immediately for new pieces
        int currentPieceId = piece.GetInstanceID();
        if (currentPieceId != lastProcessedPieceInstanceId && !isExecutingPlacement)
        {
            if (enableDebugLogs)
                Debug.Log($"[TetrisAgent] Requesting decision for new piece {currentPieceId}");
            RequestDecision();
        }
    }

    public void OnGameOver()
    {
        if (enableDebugLogs)
            Debug.Log("[TetrisAgent] Game Over");
            
        AddReward(rewardWeights.deathPenalty);
        m_StatsRecorder.Add("Tetris/GameOver", 1);
        EndEpisode();
    }

    // IPlayerInputController implementation
    public bool GetLeft() => moveLeft;
    public bool GetRight() => moveRight;
    public bool GetRotateLeft() => rotateLeft;
    public bool GetRotateRight() => rotateRight;
    public bool GetDown() => moveDown;
    public bool GetHardDrop() => hardDrop;

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;
        
        if (currentPiece == null || board == null)
        {
            discreteActionsOut[0] = 0;
            return;
        }

        // Simple heuristic: choose placement with most line clears, least holes
        int[,] currentBoard = GetBoardState();
        List<PlacementInfo> allPlacements = GenerateAllPossiblePlacements(currentPiece, currentBoard);
        
        if (allPlacements.Count == 0)
        {
            discreteActionsOut[0] = 0;
            return;
        }

        // Find best placement based on simple heuristic
        int bestIndex = 0;
        float bestScore = float.MinValue;
        
        for (int i = 0; i < allPlacements.Count; i++)
        {
            PlacementInfo placement = allPlacements[i];
            float score = placement.linesCleared * 10f - placement.holes * 2f - placement.maxHeight * 0.1f;
            
            if (score > bestScore)
            {
                bestScore = score;
                bestIndex = i;
            }
        }

        discreteActionsOut[0] = bestIndex;
    }

    // All the helper methods remain the same...
    private int[,] GetBoardState()
    {
        var bounds = board.Bounds;
        int[,] boardArray = new int[bounds.height, bounds.width];

        for (int y = 0; y < bounds.height; y++)
        {
            for (int x = 0; x < bounds.width; x++)
            {
                Vector3Int pos = new Vector3Int(bounds.xMin + x, bounds.yMin + y, 0);
                boardArray[bounds.height - 1 - y, x] = board.tilemap.HasTile(pos) ? 1 : 0;
            }
        }

        return boardArray;
    }

    private List<PlacementInfo> GenerateAllPossiblePlacements(Piece piece, int[,] currentBoard)
    {
        List<PlacementInfo> placements = new List<PlacementInfo>();
        List<int[,]> rotations = GetPieceRotations(piece);

        for (int rotation = 0; rotation < 4; rotation++)
        {
            if (rotation >= rotations.Count) continue;

            int[,] pieceShape = rotations[rotation];
            int pieceWidth = pieceShape.GetLength(1);
            int boardWidth = currentBoard.GetLength(1);

            for (int col = 0; col <= boardWidth - pieceWidth; col++)
            {
                if (CanPlacePieceAt(pieceShape, currentBoard, col, out int landingRow))
                {
                    int[,] simulatedBoard = CopyBoard(currentBoard);
                    PlacePieceOnBoard(pieceShape, simulatedBoard, col, landingRow);
                    int linesCleared = ClearLinesAndCount(simulatedBoard);

                    PlacementInfo placement = new PlacementInfo
                    {
                        linesCleared = linesCleared,
                        aggregateHeight = CalculateAggregateHeight(simulatedBoard),
                        maxHeight = CalculateMaxHeight(simulatedBoard),
                        holes = CalculateHoles(simulatedBoard),
                        bumpiness = CalculateBumpiness(simulatedBoard),
                        wellDepth = CalculateWellDepth(simulatedBoard),
                        targetColumn = col,
                        targetRotation = rotation
                    };

                    placements.Add(placement);
                }
            }
        }

        return placements;
    }

    // [Include all other helper methods - keeping them the same as before]
    private List<int[,]> GetPieceRotations(Piece piece)
    {
        List<int[,]> rotations = new List<int[,]>();

        TetrominoData data = piece.data;
        if (data.cells == null || data.cells.Length == 0)
        {
            Debug.LogError("GetPieceRotations: piece.data.cells is null or not initialized.");
            return rotations;
        }

        for (int rotation = 0; rotation < 4; rotation++)
        {
            Vector3Int[] rotatedCells = new Vector3Int[data.cells.Length];

            for (int i = 0; i < data.cells.Length; i++)
            {
                rotatedCells[i] = (Vector3Int)data.cells[i];
            }

            for (int r = 0; r < rotation; r++)
            {
                for (int i = 0; i < rotatedCells.Length; i++)
                {
                    Vector3Int cell = rotatedCells[i];
                    int newX = -cell.y;
                    int newY = cell.x;
                    rotatedCells[i] = new Vector3Int(newX, newY, 0);
                }
            }

            int[,] shapeArray = ConvertCellsToArray(rotatedCells);
            rotations.Add(shapeArray);
        }

        return rotations;
    }

    private int[,] ConvertCellsToArray(Vector3Int[] cells)
    {
        if (cells.Length == 0) return new int[1, 1];

        int minX = cells[0].x, maxX = cells[0].x;
        int minY = cells[0].y, maxY = cells[0].y;

        foreach (var cell in cells)
        {
            minX = Mathf.Min(minX, cell.x);
            maxX = Mathf.Max(maxX, cell.x);
            minY = Mathf.Min(minY, cell.y);
            maxY = Mathf.Max(maxY, cell.y);
        }

        int width = maxX - minX + 1;
        int height = maxY - minY + 1;
        int[,] array = new int[height, width];

        foreach (var cell in cells)
        {
            array[cell.y - minY, cell.x - minX] = 1;
        }

        return array;
    }

    private bool CanPlacePieceAt(int[,] pieceShape, int[,] board, int col, out int landingRow)
    {
        int boardHeight = board.GetLength(0);
        int pieceHeight = pieceShape.GetLength(0);
        landingRow = -1;

        for (int row = 0; row <= boardHeight - pieceHeight; row++)
        {
            if (HasCollision(pieceShape, board, row, col))
            {
                landingRow = row - 1;
                return landingRow >= 0;
            }
        }

        landingRow = boardHeight - pieceHeight;
        return landingRow >= 0;
    }

    private bool HasCollision(int[,] piece, int[,] board, int row, int col)
    {
        int pieceHeight = piece.GetLength(0);
        int pieceWidth = piece.GetLength(1);

        for (int r = 0; r < pieceHeight; r++)
        {
            for (int c = 0; c < pieceWidth; c++)
            {
                if (piece[r, c] == 1)
                {
                    int boardRow = row + r;
                    int boardCol = col + c;

                    if (boardRow >= board.GetLength(0) ||
                        boardCol >= board.GetLength(1) ||
                        board[boardRow, boardCol] == 1)
                    {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    private void PlacePieceOnBoard(int[,] piece, int[,] board, int col, int row)
    {
        int pieceHeight = piece.GetLength(0);
        int pieceWidth = piece.GetLength(1);

        for (int r = 0; r < pieceHeight; r++)
        {
            for (int c = 0; c < pieceWidth; c++)
            {
                if (piece[r, c] == 1)
                {
                    board[row + r, col + c] = 1;
                }
            }
        }
    }

    private int ClearLinesAndCount(int[,] board)
    {
        int linesCleared = 0;
        int height = board.GetLength(0);
        int width = board.GetLength(1);

        for (int row = height - 1; row >= 0; row--)
        {
            bool fullLine = true;
            for (int col = 0; col < width; col++)
            {
                if (board[row, col] == 0)
                {
                    fullLine = false;
                    break;
                }
            }

            if (fullLine)
            {
                linesCleared++;
                for (int moveRow = row; moveRow > 0; moveRow--)
                {
                    for (int col = 0; col < width; col++)
                    {
                        board[moveRow, col] = board[moveRow - 1, col];
                    }
                }
                for (int col = 0; col < width; col++)
                {
                    board[0, col] = 0;
                }
                row++;
            }
        }

        return linesCleared;
    }

    private int CalculateAggregateHeight(int[,] board)
    {
        int height = board.GetLength(0);
        int width = board.GetLength(1);
        int totalHeight = 0;

        for (int col = 0; col < width; col++)
        {
            for (int row = 0; row < height; row++)
            {
                if (board[row, col] == 1)
                {
                    totalHeight += (height - row);
                    break;
                }
            }
        }

        return totalHeight;
    }

    private int CalculateMaxHeight(int[,] board)
    {
        int height = board.GetLength(0);
        int width = board.GetLength(1);
        int maxHeight = 0;

        for (int col = 0; col < width; col++)
        {
            for (int row = 0; row < height; row++)
            {
                if (board[row, col] == 1)
                {
                    maxHeight = Mathf.Max(maxHeight, height - row);
                    break;
                }
            }
        }

        return maxHeight;
    }

    private int CalculateHoles(int[,] board)
    {
        int height = board.GetLength(0);
        int width = board.GetLength(1);
        int holes = 0;

        for (int col = 0; col < width; col++)
        {
            bool blockFound = false;
            for (int row = 0; row < height; row++)
            {
                if (board[row, col] == 1)
                {
                    blockFound = true;
                }
                else if (blockFound)
                {
                    holes++;
                }
            }
        }

        return holes;
    }

    private int CalculateBumpiness(int[,] board)
    {
        int height = board.GetLength(0);
        int width = board.GetLength(1);
        int[] columnHeights = new int[width];

        for (int col = 0; col < width; col++)
        {
            for (int row = 0; row < height; row++)
            {
                if (board[row, col] == 1)
                {
                    columnHeights[col] = height - row;
                    break;
                }
            }
        }

        int bumpiness = 0;
        for (int col = 0; col < width - 1; col++)
        {
            bumpiness += Mathf.Abs(columnHeights[col] - columnHeights[col + 1]);
        }

        return bumpiness;
    }

    private int CalculateWellDepth(int[,] board)
    {
        int height = board.GetLength(0);
        int width = board.GetLength(1);
        int totalWellDepth = 0;

        for (int col = 0; col < width; col++)
        {
            bool leftWall = (col == 0) || HasBlockInColumn(board, col - 1);
            bool rightWall = (col == width - 1) || HasBlockInColumn(board, col + 1);

            if (leftWall && rightWall)
            {
                int wellDepth = 0;
                for (int row = height - 1; row >= 0; row--)
                {
                    if (board[row, col] == 0)
                    {
                        wellDepth++;
                    }
                    else
                    {
                        break;
                    }
                }
                totalWellDepth += wellDepth;
            }
        }

        return totalWellDepth;
    }

    private bool HasBlockInColumn(int[,] board, int col)
    {
        int height = board.GetLength(0);
        for (int row = 0; row < height; row++)
        {
            if (board[row, col] == 1) return true;
        }
        return false;
    }

    private int[,] CopyBoard(int[,] original)
    {
        int height = original.GetLength(0);
        int width = original.GetLength(1);
        int[,] copy = new int[height, width];

        for (int r = 0; r < height; r++)
        {
            for (int c = 0; c < width; c++)
            {
                copy[r, c] = original[r, c];
            }
        }

        return copy;
    }
}

