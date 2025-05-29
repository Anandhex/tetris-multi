using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine;
using Unity.MLAgents.Policies;
using System.Collections.Generic;
using System.Linq;

public class TetrisMLAgent : Agent, IPlayerInputController
{
    private Board board;
    private Piece currentPiece;
    private MLAgentDebugger debugger;

    // Action flags
    private bool moveLeft = false;
    private bool moveRight = false;
    private bool rotateLeft = false;
    private bool rotateRight = false;
    private bool moveDown = false;
    private bool hardDrop = false;

    // Fields needed for the improved reward functions
    private float lastSurfaceRoughness = 0f;
    private float previousAccessibility = 0f;
    private int consecutiveClears = 0;

    private int previousHoleCount = 0;

    // Previous state for reward calculation
    private int previousScore = 0;
    private int stepsSinceLastClear = 0;
    private float previousHeight;
    private List<Vector2Int> previousHolePositions = new List<Vector2Int>();
    private RewardWeights rewardWeights = new RewardWeights();
    public override void Initialize()
    {
        var envParams = Academy.Instance.EnvironmentParameters;

        rewardWeights.clearReward = envParams.GetWithDefault("clearReward", 1.0f);
        rewardWeights.comboMultiplier = envParams.GetWithDefault("comboMultiplier", 0.2f);
        rewardWeights.perfectClearBonus = envParams.GetWithDefault("perfectClearBonus", 20.0f);
        rewardWeights.stagnationPenaltyFactor = envParams.GetWithDefault("stagnationPenaltyFactor", 0.01f);
        rewardWeights.roughnessRewardMultiplier = envParams.GetWithDefault("roughnessRewardMultiplier", 0.3f);
        rewardWeights.roughnessPenaltyMultiplier = envParams.GetWithDefault("roughnessPenaltyMultiplier", 0.05f);
        rewardWeights.holeFillReward = envParams.GetWithDefault("holeFillReward", 0.3f);
        rewardWeights.holeCreationPenalty = envParams.GetWithDefault("holeCreationPenalty", 0.2f);
        rewardWeights.wellRewardMultiplier = envParams.GetWithDefault("wellRewardMultiplier", 0.1f);
        rewardWeights.iPieceInWellBonus = envParams.GetWithDefault("iPieceInWellBonus", 0.3f);
        rewardWeights.stackHeightPenalty = envParams.GetWithDefault("stackHeightPenalty", 0.1f);
        rewardWeights.uselessRotationPenalty = envParams.GetWithDefault("uselessRotationPenalty", 0.05f);
        rewardWeights.tSpinReward = envParams.GetWithDefault("tSpinReward", 0.5f);
        rewardWeights.iPieceGapFillBonus = envParams.GetWithDefault("iPieceGapFillBonus", 0.4f);
        rewardWeights.accessibilityRewardMultiplier = envParams.GetWithDefault("accessibilityRewardMultiplier", 0.2f);
        rewardWeights.accessibilityPenaltyMultiplier = envParams.GetWithDefault("accessibilityPenaltyMultiplier", 0.1f);
        rewardWeights.deathPenalty = envParams.GetWithDefault("deathPenalty", 10.0f);
        rewardWeights.idleActionPenalty = envParams.GetWithDefault("idleActionPenalty", 0.01f);
        rewardWeights.moveDownActionReward = envParams.GetWithDefault("moveDownActionReward", 0.01f);
        rewardWeights.hardDropActionReward = envParams.GetWithDefault("hardDropActionReward", 0.025f);
        rewardWeights.doubleLineClearRewardMultiplier = envParams.GetWithDefault("doubleLineClearRewardMultiplier", 3.0f);
        rewardWeights.tripleLineClearRewardMultiplier = envParams.GetWithDefault("tripleLineClearRewardMultiplier", 7.0f);
        rewardWeights.tetrisClearRewardMultiplier = envParams.GetWithDefault("tetrisClearRewardMultiplier", 15.0f);
        rewardWeights.maxWellRewardCap = envParams.GetWithDefault("maxWellRewardCap", 0.5f);
    }
    private void Awake()
    {
        debugger = GetComponent<MLAgentDebugger>();
        if (debugger == null)
        {
            debugger = gameObject.AddComponent<MLAgentDebugger>();
        }

        // Find board in children
        board = GetComponentInChildren<Board>();
        if (board == null)
        {
            Debug.Log("Board not found in children!");
        }
        var behavior = gameObject.GetComponent<BehaviorParameters>();
        if (behavior == null)
        {
            behavior = gameObject.AddComponent<BehaviorParameters>();
            behavior.BehaviorName = "TetrisAgent";
            behavior.BrainParameters.VectorObservationSize = 217;
            behavior.BrainParameters.NumStackedVectorObservations = 1;

            // Set up discrete actions (7 possible actions)
            ActionSpec actionSpec = ActionSpec.MakeDiscrete(new int[] { 7 });
            behavior.BrainParameters.ActionSpec = actionSpec;
        }
        else
        {
            behavior.BehaviorName = "TetrisAgent";
            behavior.BrainParameters.VectorObservationSize = 217;
            behavior.BrainParameters.NumStackedVectorObservations = 1;

            // Set up discrete actions (7 possible actions)
            ActionSpec actionSpec = ActionSpec.MakeDiscrete(new int[] { 7 });
            behavior.BrainParameters.ActionSpec = actionSpec;
        }

        // Add a decision requester component if it doesn't exist
        var requestor = gameObject.GetComponent<DecisionRequester>();
        if (requestor == null)
        {
            requestor = gameObject.AddComponent<DecisionRequester>();
            requestor.DecisionPeriod = 1;  // Request decision every frame
        }

    }

    private void Start()
    {
        // Make sure the board knows we're the input controller
        if (board != null)
        {
            board.inputController = this;
            Debug.Log("TetrisMLAgent set as input controller for board");
        }
    }

    // Called by Board.cs to set the current piece reference
    public void SetCurrentPiece(Piece piece)
    {
        currentPiece = piece;

    }

    public override void OnEpisodeBegin()
    {
        Debug.Log("ML Agent Episode began");
        previousHeight = board.CalculateStackHeight();

        // Reset agent state
        previousScore = 0;
        stepsSinceLastClear = 0;
        previousHoleCount = board.CountHoles();

        // Reset action flags
        moveLeft = false;
        moveRight = false;
        rotateLeft = false;
        rotateRight = false;
        moveDown = false;
        hardDrop = false;
    }

    private void SaveFitnessScore(float score)
    {
        string path = "fitness_score.txt";
        System.IO.File.WriteAllText(path, score.ToString());
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (board == null)
        {
            Debug.LogError("Board is null during observation collection!");
            return;
        }

        int obsCount = 0;

        // 1. Board state (200 observations for 10x20 board)
        RectInt bounds = board.Bounds;
        for (int y = bounds.yMin; y < bounds.yMax; y++)
        {
            for (int x = bounds.xMin; x < bounds.xMax; x++)
            {
                sensor.AddObservation(board.tilemap.HasTile(new Vector3Int(x, y, 0)) ? 1f : 0f);
                obsCount++;
            }
        }

        // 2. Current piece type (7 observations for one-hot encoding of tetromino types)
        if (currentPiece != null && currentPiece.data.tetromino != null)
        {
            int pieceTypeIndex = System.Array.IndexOf(board.tetrominoes, currentPiece.data);
            for (int i = 0; i < board.tetrominoes.Length; i++)
            {
                sensor.AddObservation(i == pieceTypeIndex ? 1.0f : 0.0f);
                obsCount++;
            }
        }
        else
        {
            for (int i = 0; i < 7; i++)
            {
                sensor.AddObservation(0f);
                obsCount++;
            }
        }

        // 3. Current piece position and rotation (3 observations)
        if (currentPiece != null)
        {
            // Normalize position relative to board bounds
            float normalizedX = (currentPiece.position.x - bounds.xMin) / (float)bounds.width;
            float normalizedY = (currentPiece.position.y - bounds.yMin) / (float)bounds.height;

            sensor.AddObservation(normalizedX);
            sensor.AddObservation(normalizedY);
            sensor.AddObservation(currentPiece.rotationIndex / 4.0f); // Normalized rotation 0-1
            obsCount += 3;
        }
        else
        {
            sensor.AddObservation(0.5f); // Default X
            sensor.AddObservation(0.5f); // Default Y
            sensor.AddObservation(0f);   // Default rotation
            obsCount += 3;
        }

        // 4. Next piece type (7 observations)
        int nextPieceTypeIndex = System.Array.IndexOf(board.tetrominoes, board.nextPieceData);
        for (int i = 0; i < board.tetrominoes.Length; i++)
        {
            sensor.AddObservation(i == nextPieceTypeIndex ? 1.0f : 0.0f);
            obsCount++;
        }



    }

    private List<Vector2Int> previousBottomRowHoles = new List<Vector2Int>();
    private int steps = 0;
    public override void OnActionReceived(ActionBuffers actions)
    {
        // Reset action flags
        moveLeft = false;
        moveRight = false;
        rotateLeft = false;
        rotateRight = false;
        moveDown = false;
        hardDrop = false;

        Vector3Int prevPosition = board.activePiece.position;
        Vector3Int[] prevCells = (Vector3Int[])board.activePiece.cells.Clone();
        steps++;

        // Process discrete actions
        int actionIndex = actions.DiscreteActions[0];

        if (debugger != null)
            debugger.SetLastAction(actionIndex);

        // Basic action rewards - simplified
        switch (actionIndex)
        {
            case 0: AddReward(-rewardWeights.idleActionPenalty); break; // Do nothing - small penalty
            case 1: moveLeft = true; break;   // Neutral - let outcomes determine reward
            case 2: moveRight = true; break;  // Neutral - let outcomes determine reward
            case 3: rotateLeft = true; break; // Neutral - let outcomes determine reward
            case 4: rotateRight = true; break; // Neutral - let outcomes determine reward
            case 5: moveDown = true; AddReward(rewardWeights.moveDownActionReward); break; // Small reward for efficiency
            case 6: hardDrop = true; AddReward(rewardWeights.hardDropActionReward); break; // Reduced from 0.05 to prevent premature dropping
        }

        if (board == null)
            return;

        // === LINE CLEAR REWARDS ===
        if (board.playerScore > previousScore)
        {
            int scoreDelta = board.playerScore - previousScore;
            int linesCleared = scoreDelta / 100; // Each line is worth 100 points in Board.cs

            // Exponential reward scaling for multiple lines
            float clearReward = 0f;
            switch (linesCleared)
            {
                case 1: clearReward = rewardWeights.clearReward; break;
                case 2: clearReward = rewardWeights.clearReward * rewardWeights.doubleLineClearRewardMultiplier; break; // Use new multiplier
                case 3: clearReward = rewardWeights.clearReward * rewardWeights.tripleLineClearRewardMultiplier; break; // Use new multiplier
                case 4: clearReward = rewardWeights.clearReward * rewardWeights.tetrisClearRewardMultiplier; break; // Use new multiplier for Tetris
            }

            // Combo system with increasing returns
            if (consecutiveClears > 0)
            {
                clearReward *= (1.0f + (consecutiveClears * rewardWeights.comboMultiplier)); // Scale by combo count
            }
            consecutiveClears++;

            AddReward(clearReward);
            previousScore = board.playerScore;
            stepsSinceLastClear = 0;

            // Perfect clear bonus (keep as is - good reward)
            if (board.IsPerfectClear())
            {
                AddReward(rewardWeights.perfectClearBonus);
            }
        }
        else
        {
            consecutiveClears = 0; // Reset combo counter
            stepsSinceLastClear++;

            // Progressive stagnation penalty
            if (stepsSinceLastClear > 50)
            {
                // Gradually increasing penalty for stagnation
                float stagnationPenalty = Mathf.Min((stepsSinceLastClear - 50) * rewardWeights.stagnationPenaltyFactor, 1.0f);
                AddReward(-stagnationPenalty);
            }
        }

        // === BOARD STATE EVALUATION ===

        // --- Surface Smoothness Reward ---
        float previousRoughness = lastSurfaceRoughness;
        float currentRoughness = CalculateSurfaceRoughness();
        float roughnessDelta = previousRoughness - currentRoughness;

        if (roughnessDelta > 0)
            AddReward(roughnessDelta * rewardWeights.roughnessRewardMultiplier); // Reward smoother surface
        else if (roughnessDelta < -1) // Only penalize significant roughness increases
            AddReward(roughnessDelta * rewardWeights.roughnessPenaltyMultiplier);

        lastSurfaceRoughness = currentRoughness;

        // --- Hole Management ---
        List<Vector2Int> currentHoles = board.GetHolePositions();

        // Reward filling existing holes
        var filledHoles = previousHolePositions.Where(oldPos => !currentHoles.Contains(oldPos)).ToList();
        if (filledHoles.Count > 0)
        {
            // Higher reward for filling holes in lower rows
            float holeReward = 0f;
            foreach (var pos in filledHoles)
            {
                // Scale reward by position - filling deeper holes is better
                float depthFactor = 1f + (pos.y * 0.1f); // Higher reward for deeper holes (0.1f here is a scaling factor, not a reward itself)
                holeReward += rewardWeights.holeFillReward * depthFactor;
            }
            AddReward(holeReward);
        }

        // Penalty for creating new holes - more severe
        var newHoles = currentHoles.Where(newPos => !previousHolePositions.Contains(newPos)).ToList();
        if (newHoles.Count > 0)
        {
            // Higher penalty for creating holes in higher positions
            float holePenalty = 0f;
            foreach (var pos in newHoles)
            {
                // Scale penalty by position - creating higher holes is worse
                int boardHeight = board.boardSize[0];
                float heightFactor = 1f + ((boardHeight - pos.y) * 0.1f); // 0.1f here is a scaling factor, not a reward itself
                holePenalty += rewardWeights.holeCreationPenalty * heightFactor;
            }
            AddReward(-holePenalty);
        }

        previousHolePositions = currentHoles;

        // --- Well Formation Reward ---
        (int wellCol, int wellDepth) = GetDeepestWell();
        if (wellDepth >= 3)
        {
            // Better reward for deeper wells, but cap at reasonable depth
            float wellReward = Mathf.Min(wellDepth * rewardWeights.wellRewardMultiplier, rewardWeights.maxWellRewardCap); // Use new cap
            AddReward(wellReward);

            // Extra reward if the I-piece is next and we have a good well for it
            if (IsIPieceNext() && wellDepth >= 4)
            {
                AddReward(rewardWeights.iPieceInWellBonus);
            }
        }

        // --- Stack Height Management ---
        float currentHeight = board.CalculateStackHeight();
        float heightDelta = previousHeight - currentHeight;

        // Dynamic height penalty based on current stack height
        // Heavier penalty when stack is already high
        if (currentHeight > 10) // Only care about height when it's getting dangerous
        {
            float heightFactor = Mathf.Max(0, (currentHeight - 10) / 10f); // Scale from 0 to 1
            AddReward(-rewardWeights.stackHeightPenalty * heightFactor); // Progressive penalty for having a tall stack
        }

        previousHeight = currentHeight;

        // --- Piece-specific strategy rewards ---
        if (actionIndex == 6) // Hard drop - evaluate final placement
        {
            // Reward good T-piece placements (potential T-spins)
            if (board.activePiece.data.tetromino == Tetromino.T && IsPotentialTSpin())
            {
                AddReward(rewardWeights.tSpinReward);
            }

            // Reward I-piece horizontal placements that fill gaps
            if (board.activePiece.data.tetromino == Tetromino.I &&
                IsHorizontalPiece() && FillsMultipleGaps())
            {
                AddReward(rewardWeights.iPieceGapFillBonus);
            }
        }

        // --- Penalize Inefficient Play ---
        if ((rotateLeft || rotateRight) && board.LastRotationWasUseless(board.activePiece, prevPosition, prevCells))
        {
            AddReward(-rewardWeights.uselessRotationPenalty);  // Penalize useless rotations
        }

        // --- Accessibility Reward ---
        float accessibilityScore = EvaluateAccessibility();
        float accessibilityDelta = accessibilityScore - previousAccessibility;

        if (accessibilityDelta > 0)
            AddReward(accessibilityDelta * rewardWeights.accessibilityRewardMultiplier);
        else if (accessibilityDelta < 0)
            AddReward(accessibilityDelta * rewardWeights.accessibilityPenaltyMultiplier);

        previousAccessibility = accessibilityScore;
    }
    // New helper methods for enhanced rewards

    // Calculate surface roughness by measuring height differences between adjacent columns
    private float CalculateSurfaceRoughness()
    {
        float roughness = 0f;
        int[] columnHeights = GetColumnHeights();

        for (int i = 0; i < columnHeights.Length - 1; i++)
        {
            roughness += Mathf.Abs(columnHeights[i] - columnHeights[i + 1]);
        }

        return roughness;
    }

    // Get heights of each column
    private int[] GetColumnHeights()
    {
        RectInt bounds = board.Bounds;
        int width = bounds.width;
        int[] heights = new int[width];

        for (int x = 0; x < width; x++)
        {
            int columnX = bounds.xMin + x;
            heights[x] = 0;

            for (int y = bounds.yMax - 1; y >= bounds.yMin; y--)
            {
                if (board.tilemap.HasTile(new Vector3Int(columnX, y, 0)))
                {
                    heights[x] = bounds.yMax - y;
                    break;
                }
            }
        }

        return heights;
    }

    // Find the deepest well in the board
    private (int column, int depth) GetDeepestWell()
    {
        int[] columnHeights = GetColumnHeights();
        int deepestWellCol = -1;
        int deepestWellDepth = 0;

        // Check each column except edges
        for (int i = 1; i < columnHeights.Length - 1; i++)
        {
            int leftHeight = columnHeights[i - 1];
            int rightHeight = columnHeights[i + 1];
            int centerHeight = columnHeights[i];

            int wellDepth = Mathf.Min(leftHeight, rightHeight) - centerHeight;

            if (wellDepth > deepestWellDepth && wellDepth >= 2)
            {
                deepestWellDepth = wellDepth;
                deepestWellCol = i;
            }
        }

        // Also check edge columns
        if (columnHeights.Length > 1)
        {
            // Left edge
            int leftEdgeWell = columnHeights[1] - columnHeights[0];
            if (leftEdgeWell > deepestWellDepth && leftEdgeWell >= 2)
            {
                deepestWellDepth = leftEdgeWell;
                deepestWellCol = 0;
            }

            // Right edge
            int rightIdx = columnHeights.Length - 1;
            int rightEdgeWell = columnHeights[rightIdx - 1] - columnHeights[rightIdx];
            if (rightEdgeWell > deepestWellDepth && rightEdgeWell >= 2)
            {
                deepestWellDepth = rightEdgeWell;
                deepestWellCol = rightIdx;
            }
        }

        return (deepestWellCol, deepestWellDepth);
    }

    // Check if next piece is an I tetromino
    private bool IsIPieceNext()
    {
        return board.nextPieceData.tetromino == Tetromino.I;
    }

    // Check if current piece is horizontal I tetromino
    private bool IsHorizontalPiece()
    {
        // For I piece, check if all cells have same y value
        if (board.activePiece.data.tetromino == Tetromino.I)
        {
            int y = board.activePiece.cells[0].y;
            for (int i = 1; i < board.activePiece.cells.Length; i++)
            {
                if (board.activePiece.cells[i].y != y)
                {
                    return false;
                }
            }
            return true;
        }
        return false;
    }

    // Check if horizontal I piece fills multiple gaps
    private bool FillsMultipleGaps()
    {
        if (!IsHorizontalPiece() || board.activePiece.data.tetromino != Tetromino.I)
            return false;

        int gapsFilled = 0;
        Vector3Int pos = board.activePiece.position;

        // Check for cells beneath each cell of the piece
        foreach (Vector3Int cell in board.activePiece.cells)
        {
            Vector3Int posBelow = new Vector3Int(cell.x + pos.x, cell.y + pos.y - 1, 0);
            if (!board.tilemap.HasTile(posBelow) && posBelow.y >= board.Bounds.yMin)
            {
                gapsFilled++;
            }
        }

        return gapsFilled >= 2; // Fills at least 2 gaps
    }

    // Check if T-piece is placed in potential T-spin position
    private bool IsPotentialTSpin()
    {
        if (board.activePiece.data.tetromino != Tetromino.T)
            return false;

        Vector3Int pos = board.activePiece.position;
        int cornersCount = 0;

        // Check corners around T-piece (3 corners filled = potential T-spin)
        Vector3Int[] corners = new Vector3Int[]
        {
        new Vector3Int(-1, -1, 0), // bottom left
        new Vector3Int(1, -1, 0),  // bottom right
        new Vector3Int(-1, 1, 0),  // top left
        new Vector3Int(1, 1, 0)    // top right
        };

        foreach (Vector3Int corner in corners)
        {
            Vector3Int checkPos = pos + corner;
            if (!board.Bounds.Contains((Vector2Int)checkPos) || board.tilemap.HasTile(checkPos))
            {
                cornersCount++;
            }
        }

        return cornersCount >= 3;
    }

    // Evaluate how accessible the board is for future pieces
    private float EvaluateAccessibility()
    {
        int[] colHeights = GetColumnHeights();
        float accessibility = 0f;

        // Reward having space at the top for new pieces
        for (int i = 0; i < colHeights.Length; i++)
        {
            // Give more value to center columns being lower
            float centerFactor = 1.0f - (Mathf.Abs(i - (colHeights.Length / 2)) / (float)colHeights.Length);
            float heightValue = Mathf.Max(0, 20 - colHeights[i]) / 20f; // 0-1 scale
            accessibility += heightValue * (1 + centerFactor);
        }

        return accessibility / colHeights.Length;
    }
    // Check if a move is safe
    // Called by Board.cs when game over occurs
    public void OnGameOver()
    {
        AddReward(-10.0f); // Big penalty for losing
        SaveFitnessScore(GetCumulativeReward());
        EndEpisode();
        Debug.Log("Game over - episode ended");
    }

    // IPlayerInputController implementation
    public bool GetLeft() => moveLeft;
    public bool GetRight() => moveRight;
    public bool GetRotateLeft() => rotateLeft;
    public bool GetRotateRight() => rotateRight;
    public bool GetDown() => moveDown;
    public bool GetHardDrop() => hardDrop;

    // For testing in editor
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;
        if (Input.GetKey(KeyCode.A) || Input.GetKey(KeyCode.LeftArrow))
            discreteActionsOut[0] = 1; // Move left
        else if (Input.GetKey(KeyCode.D) || Input.GetKey(KeyCode.RightArrow))
            discreteActionsOut[0] = 2; // Move right
        else if (Input.GetKey(KeyCode.Q))
            discreteActionsOut[0] = 3; // Rotate left
        else if (Input.GetKey(KeyCode.E) || Input.GetKey(KeyCode.UpArrow))
            discreteActionsOut[0] = 4; // Rotate right
        else if (Input.GetKey(KeyCode.S) || Input.GetKey(KeyCode.DownArrow))
            discreteActionsOut[0] = 5; // Move down
        else if (Input.GetKey(KeyCode.Space))
            discreteActionsOut[0] = 6; // Hard drop
        else
            discreteActionsOut[0] = 0; // Do nothing
    }
}