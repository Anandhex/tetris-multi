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

    private HashSet<int> processedPieceIds = new HashSet<int>();
    private int currentPieceId = -1;
    private bool waitingForDecision = false;
    private bool executingPlacement = false;
    private StatsRecorder m_StatsRecorder;

    private RewardWeights rewardWeights = new RewardWeights();

    [Header("Curriculum Parameters")]
    public int allowedTetrominoTypes = 7;
    public int curriculumBoardPreset;
    public float curriculumBoardHeight = 20f;
    public float curriculumDropSpeed = 0.75f;
    public float curriculumHolePenaltyWeight = 0.5f;
    public bool enableAdvancedMechanics = false;
    private int episodeSteps = 0;
    private ActionSequence queuedActionSequence;
private bool hasQueuedAction = false;
    private bool isExecutingSequence = false;

    // Store valid placements for action mapping
    private List<PlacementInfo> cachedValidPlacements = new List<PlacementInfo>();

    // Execution state
    private enum ExecutionStep
    {
        None,
        Rotate,
        Move,
        Drop,
        Complete
    }
    private ExecutionStep currentStep = ExecutionStep.None;
    private int rotationsRemaining;
    private int executionFrameDelay = 1;
    private int frameCounter = 0;

    public void SetBoard(Board boardRef)
    {
        board = boardRef;
        Debug.Log("TetrisMLAgent: Board reference set");
    }

    public override void Initialize()
    {
        var envParams = Academy.Instance.EnvironmentParameters;
        m_StatsRecorder = Academy.Instance.StatsRecorder;
        curriculumBoardPreset = (int)Academy.Instance.EnvironmentParameters.GetWithDefault("board_preset", 6);

        // Get curriculum parameters
        allowedTetrominoTypes = (int)envParams.GetWithDefault("tetromino_types", 7f);
        curriculumBoardHeight = envParams.GetWithDefault("board_height", 20f);
        curriculumDropSpeed = envParams.GetWithDefault("drop_speed", 0.75f);
        curriculumHolePenaltyWeight = envParams.GetWithDefault("hole_penalty_weight", 0.5f);
        enableAdvancedMechanics = envParams.GetWithDefault("enable_t_spins", 0f) > 0.5f;

        // Apply curriculum settings to reward weights
        rewardWeights.holeCreationPenalty *= curriculumHolePenaltyWeight;
        rewardWeights.clearReward = envParams.GetWithDefault("clearReward", 5.0f);
        rewardWeights.comboMultiplier = envParams.GetWithDefault("comboMultiplier", 0.5f);
        rewardWeights.perfectClearBonus = envParams.GetWithDefault("perfectClearBonus", 50.0f);
    }

    public override void OnEpisodeBegin()
    {
       Debug.Log("TetrisMLAgent: Starting new episode.");
    
    // Clear previous state
    processedPieceIds.Clear();
    currentPieceId = -1;
    waitingForDecision = false;
    executingPlacement = false;
    cachedValidPlacements.Clear();
    isExecutingSequence = false;
    currentStep = ExecutionStep.None;
    queuedActionSequence = default(ActionSequence); // or set to default
    hasQueuedAction = false; // if using the alternative approach
    frameCounter = 0;

        // Get curriculum parameters
        var envParams = Academy.Instance.EnvironmentParameters;
        allowedTetrominoTypes = (int)envParams.GetWithDefault("tetromino_types", 7f);
        curriculumBoardHeight = envParams.GetWithDefault("board_height", 20f);
        curriculumDropSpeed = envParams.GetWithDefault("drop_speed", 0.75f);
        curriculumHolePenaltyWeight = envParams.GetWithDefault("hole_penalty_weight", 0.5f);
        enableAdvancedMechanics = envParams.GetWithDefault("enable_t_spins", 0f) > 0.5f;
        curriculumBoardPreset = (int)envParams.GetWithDefault("board_preset", 6);

        rewardWeights.holeCreationPenalty *= curriculumHolePenaltyWeight;
        episodeSteps = 0;

        // Reset the board if we have one
        if (board != null)
        {
            board.ResetBoard();
        }
    }

    private void Awake()
    {
        debugger = GetComponent<MLAgentDebugger>();
        if (debugger == null)
        {
            debugger = gameObject.AddComponent<MLAgentDebugger>();
        }

        var behavior = gameObject.GetComponent<BehaviorParameters>();
        if (behavior == null)
        {
            behavior = gameObject.AddComponent<BehaviorParameters>();
            behavior.BehaviorName = "TetrisAgent";
            behavior.BrainParameters.VectorObservationSize = 218;
            behavior.BrainParameters.NumStackedVectorObservations = 1;

            // Use single discrete action for placement selection (34 max placements)
            ActionSpec actionSpec = ActionSpec.MakeDiscrete(new int[] { 34 });
            behavior.BrainParameters.ActionSpec = actionSpec;
        }
        else
        {
            behavior.BehaviorName = "TetrisAgent";
            behavior.BrainParameters.VectorObservationSize = 218;
            behavior.BrainParameters.NumStackedVectorObservations = 1;

            ActionSpec actionSpec = ActionSpec.MakeDiscrete(new int[] { 34 });
            behavior.BrainParameters.ActionSpec = actionSpec;
        }

        // Add a decision requester component if it doesn't exist
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
        if (currentPiece == null || board == null || isExecutingSequence)
        {
            return;
        }

        // Generate unique ID for this piece
        int pieceId = GeneratePieceId(currentPiece);

        // Check if we've already processed this specific piece
        if (processedPieceIds.Contains(pieceId))
        {
            return;
        }

        // Mark this piece as processed
        processedPieceIds.Add(pieceId);

        // Use cached placements from CollectObservations
        if (cachedValidPlacements.Count == 0)
        {
            AddReward(rewardWeights.deathPenalty);
            EndEpisode();
            return;
        }

        int selectedAction = actions.DiscreteActions[0];

        // Direct action to placement mapping
        int placementIndex = Mathf.Clamp(selectedAction, 0, cachedValidPlacements.Count - 1);
        PlacementInfo selectedPlacement = cachedValidPlacements[placementIndex];

        Debug.Log($"OnActionReceived: Selected Action: {selectedAction} mapped to Placement Index: {placementIndex}. Target Column: {selectedPlacement.targetColumn}, Rotation: {selectedPlacement.targetRotation}");

        // Queue the action sequence
        ActionSequence sequence = new ActionSequence(
            selectedPlacement.targetColumn,
            selectedPlacement.targetRotation,
            true
        );
        Debug.Log($"OnActionReceived: Queuing action sequence: Target Column={sequence.targetColumn}, Target Rotation={sequence.targetRotation}, Hard Drop={sequence.useHardDrop}");
        QueueActions(sequence);
        CalculatePlacementReward(selectedPlacement);
    }

    private int GeneratePieceId(Piece piece)
    {
        int id = piece.gameObject.GetInstanceID();
        return id;
    }

    // Simplified action masking using cached placements
    public override void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
    {
        if (currentPiece == null || board == null)
        {
            // If no piece, mask all actions
            for (int i = 0; i < 34; i++)
            {
                actionMask.SetActionEnabled(0, i, false);
            }
            return;
        }

        // Use cached placements
        int validPlacementCount = cachedValidPlacements.Count;

        // Enable actions up to the number of valid placements
        for (int i = 0; i < 34; i++)
        {
            bool isEnabled = i < validPlacementCount;
            actionMask.SetActionEnabled(0, i, isEnabled);
        }
    }

    private void CalculatePlacementReward(PlacementInfo placement)
    {
        float reward = 0f;

        reward += placement.linesCleared * rewardWeights.clearReward;
        m_StatsRecorder.Add("Tetris/LinesCleared", placement.linesCleared);

        reward += -placement.holes * rewardWeights.holeCreationPenalty;
        m_StatsRecorder.Add("Tetris/Holes", placement.holes);

        reward += -placement.aggregateHeight * 0.04f;
        m_StatsRecorder.Add("Tetris/AggregateHeight", placement.aggregateHeight);

        reward += -placement.bumpiness * 0.03f;
        m_StatsRecorder.Add("Tetris/Bumpiness", placement.bumpiness);

        if (placement.linesCleared == 4)
        {
            reward += rewardWeights.perfectClearBonus;
            m_StatsRecorder.Add("Tetris/TetrisClears", 1);
        }

        reward += -0.001f; // step penalty
        m_StatsRecorder.Add("Tetris/PlacementReward", reward);
        AddReward(reward);
    }

    public void OnGameOver()
    {
        Debug.Log("TetrisMLAgent: Game Over detected. Ending episode with death penalty.");
        AddReward(rewardWeights.deathPenalty);
        EndEpisode();
    }

    // Cache placements during observation collection
    public override void CollectObservations(VectorSensor sensor)
    {
        if (currentPiece == null || board == null)
        {
            // Add zero observations if no piece
            for (int i = 0; i < 218; i++)
            {
                sensor.AddObservation(0f);
            }
            return;
        }

        // Convert current board state to 2D array for simulation
        int[,] currentBoard = GetBoardState();

        // Generate and cache all possible placements for current piece
        cachedValidPlacements = GenerateAllPossiblePlacements(currentPiece, currentBoard);

        // 1. ALL POSSIBLE PLACEMENTS (34 placements × 6 features = 204 observations)
        for (int i = 0; i < 34; i++)
        {
            if (i < cachedValidPlacements.Count)
            {
                PlacementInfo placement = cachedValidPlacements[i];
                sensor.AddObservation(placement.linesCleared / 4f);      // Normalized (0-1)
                sensor.AddObservation(placement.aggregateHeight / 40f);  // Normalized
                sensor.AddObservation(placement.maxHeight / 20f);        // Normalized
                sensor.AddObservation(placement.holes / 20f);            // Normalized
                sensor.AddObservation(placement.bumpiness / 30f);        // Normalized
                sensor.AddObservation(placement.wellDepth / 15f);        // Normalized
            }
            else
            {
                // Pad with zeros for unused placement slots
                for (int j = 0; j < 6; j++)
                {
                    sensor.AddObservation(0f);
                }
            }
        }

        // 2. CURRENT PIECE ONE-HOT (7 observations)
        for (int i = 0; i < 7; i++)
        {
            sensor.AddObservation(((int)currentPiece.data.tetromino == i) ? 1f : 0f);
        }

        // 3. NEXT PIECE ONE-HOT (7 observations)
        if (board.nextPieceData.tetromino != Tetromino.I) // Check if valid
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

        // Total: 204 + 7 + 7 = 218 observations
    }

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
                int landingRow;
                if (CanPlacePieceAt(pieceShape, currentBoard, col, out landingRow))
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

    // Improved piece detection
    public void SetCurrentPiece(Piece piece)
    {
        if (piece == null)
        {
            currentPiece = null;
            return;
        }

        if (currentPiece != null && piece.gameObject.GetInstanceID() == currentPiece.gameObject.GetInstanceID())
        {
            return;
        }

        currentPiece = piece;
        int pieceId = GeneratePieceId(piece);

        Debug.Log($"SetCurrentPiece: New piece set - {piece.data.tetromino} (ID: {pieceId})");

        // Only request decision for genuinely new pieces
        if (!processedPieceIds.Contains(pieceId) && !waitingForDecision && !isExecutingSequence)
        {
            waitingForDecision = true;
            cachedValidPlacements.Clear(); // Force recalculation
            Debug.Log("SetCurrentPiece: Requesting decision for new piece");
            RequestDecision();
        }
    }

    private int manualActionIndex = 0;
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;

        if (currentPiece == null || board == null || cachedValidPlacements.Count == 0)
        {
            discreteActionsOut[0] = 0;
            return;
        }

        // Use keyboard input to change manualActionIndex
        if (Input.GetKeyDown(KeyCode.UpArrow))
        {
            manualActionIndex = (manualActionIndex + 1) % 34;
            Debug.Log($"ManualActionIndex increased: {manualActionIndex}");
        }
        else if (Input.GetKeyDown(KeyCode.DownArrow))
        {
            manualActionIndex = (manualActionIndex - 1 + 34) % 34;
            Debug.Log($"ManualActionIndex decreased: {manualActionIndex}");
        }

        // Clamp to available valid placements
        int clampedIndex = Mathf.Clamp(manualActionIndex, 0, cachedValidPlacements.Count - 1);

        Debug.Log($"Heuristic: Using manualActionIndex={manualActionIndex}, mapped to valid index={clampedIndex} / {cachedValidPlacements.Count}");
        discreteActionsOut[0] = clampedIndex;
    }

    

   public void QueueActions(ActionSequence sequence)
{
    queuedActionSequence = sequence;
    hasQueuedAction = true;
    isExecutingSequence = true;
    waitingForDecision = true;

    rotationsRemaining = (sequence.targetRotation - currentPiece.rotationIndex + 4) % 4;
    currentStep = ExecutionStep.Rotate;
    frameCounter = 0;

    Debug.Log($"QueueActions: Sequence queued. Rotation: {rotationsRemaining}, Target Column: {sequence.targetColumn}");
}

public void ClearQueue()
{
    queuedActionSequence = default(ActionSequence);
    hasQueuedAction = false;
    isExecutingSequence = false;
    currentStep = ExecutionStep.None;
    Debug.Log("ClearQueue: Action queue cleared.");
}

public bool HasQueuedActions()
{
    return hasQueuedAction;
}

private void Update()
{
    if (!isExecutingSequence || currentStep == ExecutionStep.None || !hasQueuedAction)
        return;

    // Safety check
    if (currentPiece == null || board == null)
    {
        Debug.LogWarning("Update: Missing piece or board, clearing queue");
        ClearQueue();
        return;
    }

    // Optional: Add small frame delay for visual clarity
    frameCounter++;
    if (frameCounter < executionFrameDelay)
        return;
    frameCounter = 0;

    // Now you can access sequence directly
    switch (currentStep)
    {
        case ExecutionStep.Rotate:
            if (rotationsRemaining > 0)
            {
                ExecuteRotation(1);
                rotationsRemaining--;
                Debug.Log($"Update: Rotated. Remaining: {rotationsRemaining}");
            }
            else
            {
                Debug.Log("Update: Rotation complete. Proceeding to movement.");
                currentStep = ExecutionStep.Move;
            }
            break;

        case ExecutionStep.Move:
            int currentCol = currentPiece.position.x - board.Bounds.xMin;
            int targetCol = queuedActionSequence.targetColumn;

            if (currentCol == targetCol)
            {
                Debug.Log("Update: Column reached. Proceeding to drop.");
                currentStep = ExecutionStep.Drop;
            }
            else
            {
                Vector2Int direction = currentCol < targetCol ? Vector2Int.right : Vector2Int.left;
                if (!ExecuteMovement(direction))
                {
                    Debug.LogWarning("Update: Movement blocked. Proceeding to drop.");
                    currentStep = ExecutionStep.Drop;
                }
                else
                {
                    Debug.Log($"Update: Moved to column {currentPiece.position.x - board.Bounds.xMin}.");
                }
            }
            break;
            
        case ExecutionStep.Drop:
            if (queuedActionSequence.useHardDrop)
            {
                Debug.Log("Update: Performing hard drop.");
                ExecuteHardDrop();
            }
            else
            {
                Debug.Log("Update: Hard drop skipped.");
            }
            currentStep = ExecutionStep.Complete;
            break;

        case ExecutionStep.Complete:
            Debug.Log("Update: Sequence complete. Resetting.");
            ClearQueue();
            isExecutingSequence = false;
            waitingForDecision = false;
            currentStep = ExecutionStep.None;
            break;
    }
}
    private void ExecuteRotation(int direction)
    {
        Debug.Log($"ExecuteRotation: Attempting rotation in direction: {direction}. Original Index: {currentPiece.rotationIndex}");
        currentPiece.rotationIndex = ((currentPiece.rotationIndex + direction) + 4) % 4;
        currentPiece.Rotate(direction);
        Debug.Log($"ExecuteRotation: Rotation complete. New Index: {currentPiece.rotationIndex}");
    }

    private bool ExecuteMovement(Vector2Int direction)
    {
        Vector3Int newPosition = currentPiece.position + (Vector3Int)direction;
        Debug.Log($"ExecuteMovement: Trying to move piece {currentPiece.data.tetromino} from {currentPiece.position} to {newPosition}. Direction: {direction}");

        if (board.IsValidPosition(currentPiece, newPosition))
        {
            Debug.Log($"ExecuteMovement: Position {newPosition} is valid. Clearing and setting piece.");
            board.Clear(currentPiece);
            currentPiece.position = newPosition;
            board.Set(currentPiece);
            return true;
        }
        else
        {
            Debug.LogWarning($"ExecuteMovement: Position {newPosition} is INVALID for piece {currentPiece.data.tetromino}. Movement blocked.");
            return false;
        }
    }

    private void ExecuteHardDrop()
    {
        Debug.Log("ExecuteHardDrop: Initiating hard drop.");
        board.Clear(currentPiece);
        Debug.Log("ExecuteHardDrop: Piece cleared from current board position.");

        // Drop until collision
        int initialY = currentPiece.position.y;
        while (true)
        {
            Vector3Int newPosition = currentPiece.position + Vector3Int.down;
            if (board.IsValidPosition(currentPiece, newPosition))
            {
                currentPiece.position = newPosition;
            }
            else
            {
                Debug.Log($"ExecuteHardDrop: Piece stopped at Y: {currentPiece.position.y}. Collision detected below.");
                break;
            }
        }
        Debug.Log($"ExecuteHardDrop: Piece dropped from Y:{initialY} to Y:{currentPiece.position.y}");

        // Lock the piece
        board.Set(currentPiece);
        board.ClearLines();
        board.SpawnPiece();
        Debug.Log("ExecuteHardDrop: Piece locked, lines cleared, new piece spawned.");
    }

   

    // IPlayerInputController implementation - all return false since ML Agent controls actions
    public bool GetLeft() => false;
    public bool GetRight() => false;
    public bool GetRotateLeft() => false;
    public bool GetRotateRight() => false;
    public bool GetDown() => false;
    public bool GetHardDrop() => false;

    #region Simulation Helper Methods
    private List<int[,]> GetPieceRotations(Piece piece)
    {
        List<int[,]> rotations = new List<int[,]>();

        TetrominoData data = piece.data;
        if (data.cells == null || data.cells.Length == 0)
        {
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
        int boardHeight = board.GetLength(0);
        int boardWidth = board.GetLength(1);

        for (int r = 0; r < pieceHeight; r++)
        {
            for (int c = 0; c < pieceWidth; c++)
            {
                if (piece[r, c] == 1)
                {
                    int boardRow = row + r;
                    int boardCol = col + c;

                    if (boardRow >= boardHeight || boardCol >= boardWidth || boardCol < 0)
                    {
                        return true;
                    }

                    if (board[boardRow, boardCol] == 1)
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
        if (col < 0 || col >= board.GetLength(1)) return false;
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
    #endregion
}
