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

    // Simplified piece tracking
    private Piece lastProcessedPiece;
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

    // Input flags
    private bool moveLeft;
    private bool moveRight;
    private bool moveDown;
    private bool rotateLeft;
    private bool rotateRight;
    private bool hardDrop;

    // Cache for valid placements
    private List<PlacementInfo> cachedValidPlacements = new List<PlacementInfo>();

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
    }

    private void Start()
    {
        if (board != null)
        {
            board.inputController = this;
        }
    }

    public override void OnEpisodeBegin()
    {
        lastProcessedPiece = null;
        isExecutingPlacement = false;
        cachedValidPlacements.Clear();
        
        // Update curriculum parameters
        var envParams = Academy.Instance.EnvironmentParameters;
        allowedTetrominoTypes = (int)envParams.GetWithDefault("tetromino_types", 7f);
        curriculumBoardHeight = envParams.GetWithDefault("board_height", 20f);
        curriculumDropSpeed = envParams.GetWithDefault("drop_speed", 0.75f);
        curriculumHolePenaltyWeight = envParams.GetWithDefault("hole_penalty_weight", 0.5f);
        enableAdvancedMechanics = envParams.GetWithDefault("enable_t_spins", 0f) > 0.5f;
        curriculumBoardPreset = (int)envParams.GetWithDefault("board_preset", 6);
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
        requestor.DecisionPeriod = 1;
        requestor.TakeActionsBetweenDecisions = false;
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        if (currentPiece == null || board == null || isExecutingPlacement)
            return;

        // Check if this is a new piece
        if (currentPiece == lastProcessedPiece)
            return;

        lastProcessedPiece = currentPiece;
        isExecutingPlacement = true;

        // Get valid placements
        int[,] currentBoard = GetBoardState();
        List<PlacementInfo> validPlacements = GenerateAllPossiblePlacements(currentPiece, currentBoard);

        if (validPlacements.Count == 0)
        {
            AddReward(rewardWeights.deathPenalty);
            m_StatsRecorder.Add("Tetris/GameOver", 1);
            EndEpisode();
            return;
        }

        // Get selected action
        int selectedAction = Mathf.Clamp(actions.DiscreteActions[0], 0, validPlacements.Count - 1);
        PlacementInfo selectedPlacement = validPlacements[selectedAction];

        Debug.Log($"Action: {selectedAction}, Column: {selectedPlacement.targetColumn}, Rotation: {selectedPlacement.targetRotation}");

        // Execute placement immediately
        ExecutePlacementDirect(selectedPlacement);
        CalculatePlacementReward(selectedPlacement);
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

        // Generate valid placements
        int[,] currentBoard = GetBoardState();
        List<PlacementInfo> validPlacements = GenerateAllPossiblePlacements(currentPiece, currentBoard);
        cachedValidPlacements = validPlacements; // Cache for observations

        // Enable valid actions, disable invalid ones
        for (int i = 0; i < 34; i++)
        {
            actionMask.SetActionEnabled(0, i, i < validPlacements.Count);
        }
    }

    private void ExecutePlacementDirect(PlacementInfo placement)
    {
        // Direct placement without animation for faster training
        StartCoroutine(ExecutePlacementCoroutine(placement));
    }

    private IEnumerator ExecutePlacementCoroutine(PlacementInfo placement)
    {
        // Rotate to target rotation
        while (currentPiece.rotationIndex != placement.targetRotation)
        {
            rotateRight = true;
            yield return new WaitForFixedUpdate();
            rotateRight = false;
            yield return new WaitForFixedUpdate();
        }

        // Move to target column
        int currentCol = currentPiece.position.x - board.Bounds.xMin;
        int attempts = 0;
        while (currentCol != placement.targetColumn && attempts < 20) // Safety limit
        {
            if (currentCol < placement.targetColumn)
            {
                moveRight = true;
                yield return new WaitForFixedUpdate();
                moveRight = false;
            }
            else
            {
                moveLeft = true;
                yield return new WaitForFixedUpdate();
                moveLeft = false;
            }
            yield return new WaitForFixedUpdate();
            currentCol = currentPiece.position.x - board.Bounds.xMin;
            attempts++;
        }

        // Hard drop
        hardDrop = true;
        yield return new WaitForFixedUpdate();
        hardDrop = false;

        isExecutingPlacement = false;
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

        // Use cached placements from action masking to ensure consistency
        List<PlacementInfo> validPlacements = cachedValidPlacements;
        if (validPlacements.Count == 0)
        {
            int[,] currentBoard = GetBoardState();
            validPlacements = GenerateAllPossiblePlacements(currentPiece, currentBoard);
        }

        // 1. All possible placements (34 placements × 6 features = 204 observations)
        for (int i = 0; i < 34; i++)
        {
            if (i < validPlacements.Count)
            {
                PlacementInfo placement = validPlacements[i];
                sensor.AddObservation(placement.linesCleared / 4f);
                sensor.AddObservation(placement.aggregateHeight / 200f); // Normalized
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
        if (board != null && board.nextPieceData.tetromino != Tetromino.I) // Assuming I is default/invalid
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

    public void SetCurrentPiece(Piece piece)
    {
        if (piece == null) return;
        
        currentPiece = piece;
        
        // Request decision only for new pieces
        if (piece != lastProcessedPiece && !isExecutingPlacement)
        {
            RequestDecision();
        }
    }

    public void OnGameOver()
    {
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

    // Board state analysis methods
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

    // Generate all possible placements for the current piece
    private List<PlacementInfo> GenerateAllPossiblePlacements(Piece piece, int[,] currentBoard)
    {
        List<PlacementInfo> placements = new List<PlacementInfo>();
        List<int[,]> rotations = GetPieceRotations(piece);

        // Generate placements systematically
        for (int rotation = 0; rotation < 4; rotation++)
        {
            if (rotation >= rotations.Count) continue;

            int[,] pieceShape = rotations[rotation];
            int pieceWidth = pieceShape.GetLength(1);
            int boardWidth = currentBoard.GetLength(1);

            // Try each column position (0-9 for standard Tetris)
            for (int col = 0; col <= boardWidth - pieceWidth; col++)
            {
                if (CanPlacePieceAt(pieceShape, currentBoard, col, out int landingRow))
                {
                    // Create simulation
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

    private List<int[,]> GetPieceRotations(Piece piece)
    {
        List<int[,]> rotations = new List<int[,]>();

        TetrominoData data = piece.data;
        if (data.cells == null || data.cells.Length == 0)
        {
            Debug.LogError("GetPieceRotations: piece.data.cells is null or not initialized.");
            return rotations;
        }

        // Convert piece cells to different rotations
        for (int rotation = 0; rotation < 4; rotation++)
        {
            Vector3Int[] rotatedCells = new Vector3Int[data.cells.Length];

            // Copy original cells
            for (int i = 0; i < data.cells.Length; i++)
            {
                rotatedCells[i] = (Vector3Int)data.cells[i];
            }

            // Apply rotation transformations
            for (int r = 0; r < rotation; r++)
            {
                for (int i = 0; i < rotatedCells.Length; i++)
                {
                    Vector3Int cell = rotatedCells[i];
                    // 90-degree clockwise rotation matrix
                    int newX = -cell.y;
                    int newY = cell.x;
                    rotatedCells[i] = new Vector3Int(newX, newY, 0);
                }
            }

            // Convert to 2D array representation
            int[,] shapeArray = ConvertCellsToArray(rotatedCells);
            rotations.Add(shapeArray);
        }

        return rotations;
    }

    // Convert piece cells to 2D array
    private int[,] ConvertCellsToArray(Vector3Int[] cells)
    {
        if (cells.Length == 0) return new int[1, 1];

        // Find bounds
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

    // Check if piece can be placed at specific column
    private bool CanPlacePieceAt(int[,] pieceShape, int[,] board, int col, out int landingRow)
    {
        int boardHeight = board.GetLength(0);
        int pieceHeight = pieceShape.GetLength(0);
        landingRow = -1;

        // Drop piece from top until collision
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

    // Check collision between piece and board
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

    // Place piece on board
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

    // Clear lines and return count
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
                // Shift rows down
                for (int moveRow = row; moveRow > 0; moveRow--)
                {
                    for (int col = 0; col < width; col++)
                    {
                        board[moveRow, col] = board[moveRow - 1, col];
                    }
                }
                // Clear top row
                for (int col = 0; col < width; col++)
                {
                    board[0, col] = 0;
                }
                row++; // Check same row again after shift
            }
        }

        return linesCleared;
    }

    // Calculate aggregate height (sum of all column heights)
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

    // Calculate maximum height
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

    // Calculate number of holes
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

    // Calculate bumpiness (sum of height differences between adjacent columns)
    private int CalculateBumpiness(int[,] board)
    {
        int height = board.GetLength(0);
        int width = board.GetLength(1);
        int[] columnHeights = new int[width];

        // Get height of each column
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

        // Calculate bumpiness
        int bumpiness = 0;
        for (int col = 0; col < width - 1; col++)
        {
            bumpiness += Mathf.Abs(columnHeights[col] - columnHeights[col + 1]);
        }

        return bumpiness;
    }

    // Calculate well depth (depth of single-width wells)
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
                // This is a potential well, calculate its depth
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

    // Helper function to check if column has any blocks
    private bool HasBlockInColumn(int[,] board, int col)
    {
        int height = board.GetLength(0);
        for (int row = 0; row < height; row++)
        {
            if (board[row, col] == 1) return true;
        }
        return false;
    }

    // Copy board for simulation
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

