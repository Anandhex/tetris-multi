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


    // Keep original action flags for IPlayerInputController
    private bool moveLeft = false;
    private bool moveRight = false;
    private bool rotateLeft = false;
    private bool rotateRight = false;
    private bool moveDown = false;
    private bool hardDrop = false;


    // New: Placement control variables
    private bool hasChosenPlacement = false;
    private int chosenColumn = 0;
    private int chosenRotation = 0;
    private bool isExecutingPlacement = false;


    // New: Exploration and balanced placement variables
    private int[] columnUsageCount = new int[10]; // Track usage per column
    private float explorationBonus = 0.2f;
    private int totalPiecesPlaced = 0;
    private float[] columnRewards = new float[10]; // Track rewards per column


    // Fields needed for the improved reward functions
    private StatsRecorder m_StatsRecorder;
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

    [Header("Curriculum Parameters")]
    public int allowedTetrominoTypes = 7;
    public int curriculumBoardPreset;
    public float curriculumBoardHeight = 20f;
    public float curriculumDropSpeed = 0.75f;
    public float curriculumHolePenaltyWeight = 0.5f;
    public bool enableAdvancedMechanics = false;
    private int episodeSteps = 0;


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


        InitializeRewardWeights(envParams);
    }


    private void InitializeRewardWeights(Unity.MLAgents.EnvironmentParameters envParams)
    {
        // REDUCED PENALTIES
        rewardWeights.stagnationPenaltyFactor = envParams.GetWithDefault("stagnationPenaltyFactor", 0.001f);
        rewardWeights.roughnessPenaltyMultiplier = envParams.GetWithDefault("roughnessPenaltyMultiplier", 0.01f);
        rewardWeights.stackHeightPenalty = envParams.GetWithDefault("stackHeightPenalty", 0.02f);
        rewardWeights.uselessRotationPenalty = envParams.GetWithDefault("uselessRotationPenalty", 0.01f);
        rewardWeights.idleActionPenalty = envParams.GetWithDefault("idleActionPenalty", 0.001f);


        // INCREASED POSITIVE REWARDS
        rewardWeights.holeFillReward = envParams.GetWithDefault("holeFillReward", 1.0f);
        rewardWeights.moveDownActionReward = envParams.GetWithDefault("moveDownActionReward", 0.02f);
        rewardWeights.hardDropActionReward = envParams.GetWithDefault("hardDropActionReward", 0.05f);


        // Multipliers
        rewardWeights.doubleLineClearRewardMultiplier = envParams.GetWithDefault("doubleLineClearRewardMultiplier", 3.0f);
        rewardWeights.tripleLineClearRewardMultiplier = envParams.GetWithDefault("tripleLineClearRewardMultiplier", 7.0f);
        rewardWeights.tetrisClearRewardMultiplier = envParams.GetWithDefault("tetrisClearRewardMultiplier", 15.0f);


        // Other rewards
        rewardWeights.wellRewardMultiplier = envParams.GetWithDefault("wellRewardMultiplier", 0.2f);
        rewardWeights.iPieceInWellBonus = envParams.GetWithDefault("iPieceInWellBonus", 0.5f);
        rewardWeights.tSpinReward = envParams.GetWithDefault("tSpinReward", 1.0f);
        rewardWeights.iPieceGapFillBonus = envParams.GetWithDefault("iPieceGapFillBonus", 0.8f);
        rewardWeights.roughnessRewardMultiplier = envParams.GetWithDefault("roughnessRewardMultiplier", 0.3f);
        rewardWeights.accessibilityRewardMultiplier = envParams.GetWithDefault("accessibilityRewardMultiplier", 0.2f);
        rewardWeights.accessibilityPenaltyMultiplier = envParams.GetWithDefault("accessibilityPenaltyMultiplier", 0.05f);
        rewardWeights.deathPenalty = envParams.GetWithDefault("deathPenalty", 2.0f);
        rewardWeights.maxWellRewardCap = envParams.GetWithDefault("maxWellRewardCap", 1.0f);


        rewardWeights.partialRowFillRewardMultiplier = 0.01f;
        rewardWeights.horizontalStackingRewardMultiplier = 0.5f;
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


        var behavior = gameObject.GetComponent<BehaviorParameters>();
        if (behavior == null)
        {
            behavior = gameObject.AddComponent<BehaviorParameters>();
            behavior.BehaviorName = "TetrisAgent";
            behavior.BrainParameters.VectorObservationSize = 228;
            behavior.BrainParameters.NumStackedVectorObservations = 1;


            // Two discrete actions - column (10 options) and rotation (4 options)
            ActionSpec actionSpec = ActionSpec.MakeDiscrete(new int[] { 10, 4 });
            behavior.BrainParameters.ActionSpec = actionSpec;
        }
        else
        {
            behavior.BehaviorName = "TetrisAgent";
            behavior.BrainParameters.VectorObservationSize = 228;
            behavior.BrainParameters.NumStackedVectorObservations = 1;


            // Two discrete actions - column (10 options) and rotation (4 options)
            ActionSpec actionSpec = ActionSpec.MakeDiscrete(new int[] { 10, 4 });
            behavior.BrainParameters.ActionSpec = actionSpec;
        }


        // Add a decision requester component if it doesn't exist
        var requestor = gameObject.GetComponent<DecisionRequester>();
        if (requestor == null)
        {
            requestor = gameObject.AddComponent<DecisionRequester>();
            requestor.DecisionPeriod = 1;
        }


        // Initialize column tracking arrays
        columnUsageCount = new int[10];
        columnRewards = new float[10];
    }


    private void Start()
    {
        if (board != null)
        {
            board.inputController = this;
        }
    }


    // Called by Board.cs to set the current piece reference
    public void SetCurrentPiece(Piece piece)
    {
        currentPiece = piece;
        // Reset placement choice when new piece spawns
        hasChosenPlacement = false;
        isExecutingPlacement = false;
    }


    public override void OnEpisodeBegin()
    {
        var envParams = Academy.Instance.EnvironmentParameters;


        allowedTetrominoTypes = (int)envParams.GetWithDefault("tetromino_types", 7f);
        curriculumBoardHeight = envParams.GetWithDefault("board_height", 20f);
        curriculumDropSpeed = envParams.GetWithDefault("drop_speed", 0.75f);
        curriculumHolePenaltyWeight = envParams.GetWithDefault("hole_penalty_weight", 0.5f);
        enableAdvancedMechanics = envParams.GetWithDefault("enable_t_spins", 0f) > 0.5f;
        curriculumBoardPreset = (int)envParams.GetWithDefault("board_preset", 6);


        rewardWeights.holeCreationPenalty *= curriculumHolePenaltyWeight;
        episodeSteps = 0;
        previousHeight = board.CalculateStackHeight();


        // Reset agent state
        previousScore = 0;
        stepsSinceLastClear = 0;
        previousHoleCount = board.CountHoles();


        // Reset placement flags
        hasChosenPlacement = false;
        chosenColumn = 0;
        chosenRotation = 0;
        isExecutingPlacement = false;


        // Reset exploration tracking
        for (int i = 0; i < 10; i++)
        {
            columnUsageCount[i] = 0;
            columnRewards[i] = 0f;
        }
        totalPiecesPlaced = 0;


        // Reset movement flags
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
            return;
        }


        // 1. Board state (200 observations for 10x20 board)
        RectInt bounds = board.Bounds;
        int maxBoardHeight = 20;
        int boardWidth = 10;


        for (int y = 0; y < maxBoardHeight; y++)
        {
            for (int x = 0; x < boardWidth; x++)
            {
                bool insideCurrentBoard = (y < bounds.height);
                if (insideCurrentBoard)
                {
                    bool tile = board.tilemap.HasTile(new Vector3Int(x, y, 0));
                    sensor.AddObservation(tile ? 1f : 0f);
                }
                else
                {
                    sensor.AddObservation(0f);
                }
            }
        }


        // 2. Current piece type (7 observations for one-hot encoding)
        if (currentPiece != null && currentPiece.data.tetromino != null)
        {
            int pieceTypeIndex = System.Array.IndexOf(board.tetrominoes, currentPiece.data);
            for (int i = 0; i < board.tetrominoes.Length; i++)
            {
                sensor.AddObservation(i == pieceTypeIndex ? 1.0f : 0.0f);
            }
        }
        else
        {
            for (int i = 0; i < 7; i++)
            {
                sensor.AddObservation(0f);
            }
        }


        // 3. Current piece position and rotation (3 observations)
        if (currentPiece != null)
        {
            float normalizedX = (currentPiece.position.x - bounds.xMin) / (float)bounds.width;
            float normalizedY = (currentPiece.position.y - bounds.yMin) / (float)bounds.height;


            sensor.AddObservation(normalizedX);
            sensor.AddObservation(normalizedY);
            sensor.AddObservation(currentPiece.rotationIndex / 4.0f);
        }
        else
        {
            sensor.AddObservation(0.5f);
            sensor.AddObservation(0.5f);
            sensor.AddObservation(0f);
        }


        // 4. Next piece type (7 observations)
        int nextPieceTypeIndex = System.Array.IndexOf(board.tetrominoes, board.nextPieceData);
        for (int i = 0; i < board.tetrominoes.Length; i++)
        {
            sensor.AddObservation(i == nextPieceTypeIndex ? 1.0f : 0.0f);
        }


        // 5. Column Heights (10 observations)
        for (int x = bounds.xMin; x < bounds.xMax; x++)
        {
            int height = 0;
            for (int y = bounds.yMax - 1; y >= bounds.yMin; y--)
            {
                if (board.tilemap.HasTile(new Vector3Int(x, y, 0)))
                {
                    height = y - bounds.yMin + 1;
                    break;
                }
            }
            sensor.AddObservation(height / (float)bounds.height);
        }


        // 6. Hole count observation (1 observation)
        int holeCount = 0;
        for (int x = bounds.xMin; x < bounds.xMax; x++)
        {
            bool blockSeen = false;
            for (int y = bounds.yMax - 1; y >= bounds.yMin; y--)
            {
                if (board.tilemap.HasTile(new Vector3Int(x, y, 0)))
                {
                    blockSeen = true;
                }
                else if (blockSeen)
                {
                    holeCount++;
                }
            }
        }
        sensor.AddObservation(holeCount / 20.0f);
    }


    // Helper function to move tetromino to correct position
    private void UpdatePlacementExecution()
    {
        if (!hasChosenPlacement || !isExecutingPlacement || currentPiece == null)
            return;

        // Reset all movement flags first
        moveLeft = false;
        moveRight = false;
        rotateLeft = false;
        rotateRight = false;
        moveDown = false;
        hardDrop = false;

        bool atCorrectRotation = (currentPiece.rotationIndex == chosenRotation);

        // Step 1: Rotate to correct orientation first
        if (!atCorrectRotation)
        {
            int currentRot = currentPiece.rotationIndex;
            int rotationDiff = (chosenRotation - currentRot + 4) % 4;

            // Choose shortest rotation path
            if (rotationDiff == 1 || rotationDiff == 2)
            {
                rotateRight = true;
            }
            else if (rotationDiff == 3)
            {
                rotateLeft = true;
            }
            return; // Don't move until rotation is correct
        }

        // Step 2: Calculate target position based on piece shape and chosen column
        int targetX = CalculateTargetPosition(chosenColumn, chosenRotation);
        bool atCorrectPosition = (currentPiece.position.x == targetX);

        // Step 3: Move horizontally to correct position
        if (!atCorrectPosition)
        {
            if (currentPiece.position.x < targetX)
            {
                moveRight = true;
            }
            else if (currentPiece.position.x > targetX)
            {
                moveLeft = true;
            }
            return; // Don't drop until position is correct
        }

        // Step 4: Soft drop when in correct position and rotation
        if (atCorrectPosition && atCorrectRotation)
        {
            moveDown = true; // Use soft drop instead of hard drop

            // Check if piece has landed (can't move down anymore)
            if (!CanPieceMoveTo(currentPiece.position + Vector3Int.down, GetRotatedCells(currentPiece.data, currentPiece.rotationIndex)))
            {
                isExecutingPlacement = false; // Placement complete
            }
        }
    }
    private int CalculateTargetPosition(int targetColumn, int targetRotation)
    {
        // Get the piece shape for the target rotation
        Vector2Int[] rotatedCells = GetRotatedCells(currentPiece.data, targetRotation);

        // Find the leftmost cell of the piece
        int minX = rotatedCells.Min(cell => cell.x);

        // Calculate the target position so that the leftmost cell aligns with the target column
        // But ensure the piece doesn't go out of bounds
        int targetPos = targetColumn - minX;

        // Clamp to valid board bounds
        int maxX = rotatedCells.Max(cell => cell.x);
        int pieceWidth = maxX - minX;

        targetPos = Mathf.Max(targetPos, -minX); // Don't go past left edge
        targetPos = Mathf.Min(targetPos, 9 - maxX); // Don't go past right edge (assuming 10-wide board)

        return targetPos;
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        // Base survival reward
        AddReward(0.01f);
        m_StatsRecorder.Add("action-rewarded/survival", 0.01f);

        episodeSteps++;

        if (board == null || currentPiece == null)
            return;

        // Get placement choice from actions (only when new piece spawns)
        if (!hasChosenPlacement)
        {
            int targetColumn = actions.DiscreteActions[0]; // 0-9 for columns
            int targetRotation = actions.DiscreteActions[1]; // 0-3 for rotations

            // Validate and clamp the chosen position
            targetColumn = Mathf.Clamp(targetColumn, 0, 9);
            targetRotation = Mathf.Clamp(targetRotation, 0, 3);

            // Additional validation for piece placement
            if (!IsValidPlacement(targetColumn, targetRotation, currentPiece.data))
            {
                // Find the closest valid placement
                targetColumn = FindClosestValidColumn(targetColumn, targetRotation);

                // Small penalty for invalid placement choice
                AddReward(-0.1f);
                m_StatsRecorder.Add("action-rewarded/invalid-placement", -0.1f);
            }

            chosenColumn = targetColumn;
            chosenRotation = targetRotation;
            hasChosenPlacement = true;
            isExecutingPlacement = true;

            if (debugger != null)
                debugger.SetLastActions(targetColumn, targetRotation);

            // Evaluate and reward the placement choice
            EvaluatePlacementChoice(targetColumn, targetRotation);

            // Track column usage for exploration
            columnUsageCount[targetColumn]++;
            totalPiecesPlaced++;
        }

        // Execute the placement using helper function
        UpdatePlacementExecution();

        // Process all other rewards
        ProcessRewards();
    }

    private int FindClosestValidColumn(int preferredColumn, int rotation)
    {
        // Try to find the closest valid column to the preferred one
        for (int distance = 0; distance <= 5; distance++)
        {
            // Try preferred column + distance
            if (preferredColumn + distance <= 9 && IsValidPlacement(preferredColumn + distance, rotation, currentPiece.data))
            {
                return preferredColumn + distance;
            }

            // Try preferred column - distance
            if (preferredColumn - distance >= 0 && IsValidPlacement(preferredColumn - distance, rotation, currentPiece.data))
            {
                return preferredColumn - distance;
            }
        }

        // Fallback to column 0 (should always be valid for most pieces)
        return 0;
    }
    private int GetValidColumnCount(TetrominoData pieceData, int rotation)
    {
        Vector2Int[] rotatedCells = GetRotatedCells(pieceData, rotation);
        int minX = rotatedCells.Min(cell => cell.x);
        int maxX = rotatedCells.Max(cell => cell.x);

        // For a 10-wide board, calculate how many valid starting positions exist
        int validPositions = 10 - (maxX - minX);
        return Mathf.Max(1, validPositions);
    }


    private bool IsValidPlacement(int column, int rotation, TetrominoData pieceData)
    {
        Vector2Int[] rotatedCells = GetRotatedCells(pieceData, rotation);

        // Find the actual columns this piece would occupy
        int minColumn = column + rotatedCells.Min(cell => cell.x);
        int maxColumn = column + rotatedCells.Max(cell => cell.x);

        // Check if all columns are within bounds (0-9 for standard Tetris)
        return minColumn >= 0 && maxColumn <= 9;
    }


    private void EvaluatePlacementChoice(int targetColumn, int targetRotation)
    {
        // Calculate the actual position the piece will move to
        int actualTargetX = CalculateTargetPosition(targetColumn, targetRotation);
        Vector3Int simulatedPosition = new Vector3Int(actualTargetX, currentPiece.position.y, 0);

        // Calculate where the piece would land
        Vector3Int landingPosition = SimulateDrop(simulatedPosition, targetRotation);

        // Evaluate the quality of this placement
        float placementScore = EvaluatePlacementQuality(landingPosition, targetRotation);

        // Add exploration bonus for balanced column usage (use actual target column)
        Vector2Int[] rotatedCells = GetRotatedCells(currentPiece.data, targetRotation);
        int actualColumn = landingPosition.x + rotatedCells.Min(cell => cell.x);
        float explorationReward = CalculateExplorationBonus(actualColumn);

        // Apply rewards
        float totalReward = placementScore + explorationReward;
        AddReward(totalReward);

        // Track column performance
        if (actualColumn >= 0 && actualColumn < columnRewards.Length)
        {
            columnRewards[actualColumn] += totalReward;
        }

        m_StatsRecorder.Add("action-rewarded/placement-quality", placementScore);
        m_StatsRecorder.Add("action-rewarded/exploration-bonus", explorationReward);

        // Bonus for strategic placements
        if (IsStrategicPlacement(landingPosition, targetRotation))
        {
            AddReward(0.5f);
            m_StatsRecorder.Add("action-rewarded/strategic-placement", 0.5f);
        }
    }



    private float CalculateExplorationBonus(int targetColumn)
    {
        if (totalPiecesPlaced < 20) // Extended early game exploration
            return explorationBonus;


        // Calculate usage distribution
        float averageUsage = totalPiecesPlaced / 10.0f;
        float columnUsage = columnUsageCount[targetColumn];

        // Strong reward for using less-used columns
        if (columnUsage < averageUsage * 0.7f)
        {
            return explorationBonus * 2.0f; // Double bonus for underused columns
        }

        // Moderate reward for balanced usage
        if (columnUsage <= averageUsage)
        {
            return explorationBonus;
        }

        // Penalty for overusing columns
        if (columnUsage > averageUsage * 1.5f)
        {
            return -explorationBonus * 1.5f; // Strong penalty for overuse
        }

        return 0f;
    }


    private Vector3Int SimulateDrop(Vector3Int startPosition, int rotation)
    {
        Vector3Int position = startPosition;

        // Get the piece shape for the target rotation
        Vector2Int[] rotatedCells = GetRotatedCells(currentPiece.data, rotation);

        // Simulate dropping until collision
        while (CanPieceMoveTo(position + Vector3Int.down, rotatedCells))
        {
            position += Vector3Int.down;
        }

        return position;
    }


    private Vector2Int[] GetRotatedCells(TetrominoData pieceData, int targetRotation)
    {
        Vector2Int[] cells = new Vector2Int[pieceData.cells.Length];
        for (int i = 0; i < pieceData.cells.Length; i++)
        {
            cells[i] = pieceData.cells[i];
        }

        // Apply rotation transformations
        for (int rot = 0; rot < targetRotation; rot++)
        {
            // Rotate 90 degrees clockwise
            for (int j = 0; j < cells.Length; j++)
            {
                Vector2Int cell = cells[j];
                cells[j] = new Vector2Int(cell.y, -cell.x);
            }
        }

        return cells;
    }


    private bool CanPieceMoveTo(Vector3Int position, Vector2Int[] cells)
    {
        RectInt bounds = board.Bounds;

        foreach (Vector2Int cell in cells)
        {
            Vector3Int tilePosition = new Vector3Int(cell.x + position.x, cell.y + position.y, 0);

            // Check bounds
            if (tilePosition.x < bounds.xMin || tilePosition.x >= bounds.xMax ||
                tilePosition.y < bounds.yMin)
            {
                return false;
            }

            // Check collision with existing tiles
            if (board.tilemap.HasTile(tilePosition))
            {
                return false;
            }
        }

        return true;
    }


    private float EvaluatePlacementQuality(Vector3Int position, int rotation)
    {
        float score = 0f;
        Vector2Int[] rotatedCells = GetRotatedCells(currentPiece.data, rotation);

        // Factor 1: Height penalty (lower is better, but not too harsh)
        float heightPenalty = (position.y / curriculumBoardHeight) * 0.3f;
        score -= heightPenalty;

        // Factor 2: Line completion potential (major reward)
        float lineBonus = EvaluateLineCompletionPotential(position, rotatedCells);
        score += lineBonus * 4.0f; // Increased weight for line clearing

        // Factor 3: Hole creation penalty
        float holePenalty = EvaluateHoleCreation(position, rotatedCells);
        score -= holePenalty * 2.0f;

        // Factor 4: Surface smoothness
        float smoothnessBonus = EvaluateSurfaceSmoothness(position, rotatedCells);
        score += smoothnessBonus * 1.0f;

        // Factor 5: Center preference for better piece placement options
        float centerBonus = CalculateCenterPreference(position.x);
        score += centerBonus * 0.3f; // Reduced to prevent center bias

        // Factor 6: Wide base building (reward filling bottom rows)
        float baseBonus = EvaluateBaseBuilding(position, rotatedCells);
        score += baseBonus * 2.0f;

        // Factor 7: Even distribution bonus
        float distributionBonus = EvaluateDistribution(position.x);
        score += distributionBonus * 1.0f;

        return Mathf.Clamp(score, -3.0f, 3.0f); // Expanded range for better discrimination
    }


    private float EvaluateLineCompletionPotential(Vector3Int position, Vector2Int[] cells)
    {
        float potential = 0f;

        HashSet<int> affectedRows = new HashSet<int>();
        foreach (Vector2Int cell in cells)
        {
            affectedRows.Add(position.y + cell.y);
        }

        foreach (int row in affectedRows)
        {
            int filledCells = CountFilledCellsInRow(row);
            int cellsFromPiece = CountPieceCellsInRow(position, cells, row);
            int totalAfterPlacement = filledCells + cellsFromPiece;

            if (totalAfterPlacement == 10) // Complete line
            {
                potential += 2.0f; // Major reward for completing lines
            }
            else if (totalAfterPlacement >= 9) // Nearly complete
            {
                potential += 1.0f;
            }
            else if (totalAfterPlacement >= 7) // Good progress
            {
                potential += 0.3f;
            }
        }

        return potential;
    }


    private int CountFilledCellsInRow(int row)
    {
        int count = 0;
        RectInt bounds = board.Bounds;

        for (int x = bounds.xMin; x < bounds.xMax; x++)
        {
            if (board.tilemap.HasTile(new Vector3Int(x, row, 0)))
            {
                count++;
            }
        }

        return count;
    }


    private int CountPieceCellsInRow(Vector3Int position, Vector2Int[] cells, int row)
    {
        int count = 0;

        foreach (Vector2Int cell in cells)
        {
            if (position.y + cell.y == row)
            {
                count++;
            }
        }

        return count;
    }


    private float EvaluateHoleCreation(Vector3Int position, Vector2Int[] cells)
    {
        float holeRisk = 0f;

        foreach (Vector2Int cell in cells)
        {
            Vector3Int cellPos = new Vector3Int(position.x + cell.x, position.y + cell.y, 0);

            // Check if placing this cell creates overhang that might trap empty spaces
            Vector3Int below = cellPos + Vector3Int.down;
            if (below.y >= board.Bounds.yMin && !board.tilemap.HasTile(below))
            {
                // Check if there are blocks to the sides creating a potential hole
                Vector3Int left = below + Vector3Int.left;
                Vector3Int right = below + Vector3Int.right;

                bool leftBlocked = !board.Bounds.Contains((Vector2Int)left) || board.tilemap.HasTile(left);
                bool rightBlocked = !board.Bounds.Contains((Vector2Int)right) || board.tilemap.HasTile(right);

                if (leftBlocked || rightBlocked)
                {
                    holeRisk += 0.5f;
                }
                else
                {
                    holeRisk += 0.1f;
                }
            }
        }

        return holeRisk;
    }


    private float EvaluateSurfaceSmoothness(Vector3Int position, Vector2Int[] cells)
    {
        // Calculate how this placement affects surface smoothness
        int[] currentHeights = GetColumnHeights();
        float currentRoughness = CalculateRoughnessFromHeights(currentHeights);

        // Simulate new heights after placement
        int[] newHeights = new int[currentHeights.Length];
        System.Array.Copy(currentHeights, newHeights, currentHeights.Length);

        foreach (Vector2Int cell in cells)
        {
            int col = position.x + cell.x;
            int newHeight = position.y + cell.y + 1; // +1 because height is 1-indexed

            if (col >= 0 && col < newHeights.Length)
            {
                newHeights[col] = Mathf.Max(newHeights[col], newHeight);
            }
        }

        float newRoughness = CalculateRoughnessFromHeights(newHeights);

        // Return positive value if smoothness improved (roughness decreased)
        return (currentRoughness - newRoughness) * 0.1f;
    }


    private float CalculateRoughnessFromHeights(int[] heights)
    {
        float roughness = 0f;
        for (int i = 0; i < heights.Length - 1; i++)
        {
            roughness += Mathf.Abs(heights[i] - heights[i + 1]);
        }
        return roughness;
    }


    private float CalculateCenterPreference(int column)
    {
        // Mild preference for center columns (3, 4, 5, 6) but not too strong
        float distanceFromCenter = Mathf.Abs(column - 4.5f);
        return (1.0f - (distanceFromCenter / 4.5f)) * 0.2f; // Very mild center bias
    }


    private float EvaluateBaseBuilding(Vector3Int position, Vector2Int[] cells)
    {
        float baseBonus = 0f;
        int boardHeight = (int)curriculumBoardHeight;

        foreach (Vector2Int cell in cells)
        {
            int cellY = position.y + cell.y;

            // Higher reward for placing pieces in bottom third of board
            if (cellY < boardHeight / 3)
            {
                baseBonus += 0.5f;
            }
            else if (cellY < boardHeight / 2)
            {
                baseBonus += 0.2f;
            }
        }

        return baseBonus;
    }


    private float EvaluateDistribution(int column)
    {
        if (totalPiecesPlaced < 10) return 0f;

        // Calculate variance in column usage
        float mean = totalPiecesPlaced / 10.0f;
        float variance = 0f;

        for (int i = 0; i < 10; i++)
        {
            variance += Mathf.Pow(columnUsageCount[i] - mean, 2);
        }
        variance /= 10.0f;

        // Reward using columns that reduce overall variance
        float currentUsage = columnUsageCount[column];
        float futureUsage = currentUsage + 1;

        // Calculate what variance would be if we use this column
        float futureVariance = variance - Mathf.Pow(currentUsage - mean, 2) / 10.0f
                              + Mathf.Pow(futureUsage - (mean + 0.1f), 2) / 10.0f;

        // Reward if this choice reduces variance
        return (variance - futureVariance) * 2.0f;
    }


    private bool IsStrategicPlacement(Vector3Int position, int rotation)
    {
        // Check for strategic placements like T-spins, well formations, etc.

        // T-spin setup
        if (currentPiece.data.tetromino == Tetromino.T && enableAdvancedMechanics)
        {
            return IsTSpinSetup(position, rotation);
        }

        // I-piece in well
        if (currentPiece.data.tetromino == Tetromino.I)
        {
            return IsIPieceInWell(position, rotation);
        }

        return false;
    }


    private bool IsTSpinSetup(Vector3Int position, int rotation)
    {
        // Simplified T-spin detection - check if T-piece fits in a T-spin slot
        Vector2Int[] rotatedCells = GetRotatedCells(currentPiece.data, rotation);
        int cornersBlocked = 0;

        // Check corners around T-piece position
        Vector3Int[] corners = new Vector3Int[]
        {
            new Vector3Int(-1, -1, 0), new Vector3Int(1, -1, 0),
            new Vector3Int(-1, 1, 0), new Vector3Int(1, 1, 0)
        };

        foreach (Vector3Int corner in corners)
        {
            Vector3Int checkPos = position + corner;
            if (!board.Bounds.Contains((Vector2Int)checkPos) || board.tilemap.HasTile(checkPos))
            {
                cornersBlocked++;
            }
        }

        return cornersBlocked >= 3; // T-spin typically requires 3+ corners blocked
    }


    private bool IsIPieceInWell(Vector3Int position, int rotation)
    {
        // Check if I-piece is being placed in a well formation
        (int wellCol, int wellDepth) = GetDeepestWell();

        if (wellDepth >= 4)
        {
            Vector2Int[] rotatedCells = GetRotatedCells(currentPiece.data, rotation);

            // Check if I-piece overlaps with the well column
            foreach (Vector2Int cell in rotatedCells)
            {
                if (position.x + cell.x == wellCol)
                {
                    return true;
                }
            }
        }

        return false;
    }


    private void ProcessRewards()
    {
        // All the existing reward processing logic
        if (board == null)
            return;


        // Partial row fill rewards
        int[] rowFills = board.GetRowFillCounts();
        int maxWidth = board.Bounds.size.x;


        for (int i = 0; i < rowFills.Length; i++)
        {
            float fillRatio = (float)rowFills[i] / maxWidth;


            if (fillRatio > 0.6f && fillRatio < 1.0f)
            {
                float reward = fillRatio * rewardWeights.partialRowFillRewardMultiplier;
                AddReward(reward);
                m_StatsRecorder.Add("action-rewarded/partial-row-fill", reward);
            }
        }


        // === LINE CLEAR REWARDS ===
        if (board.playerScore > previousScore)
        {
            int scoreDelta = board.playerScore - previousScore;
            int linesCleared = scoreDelta / 100;


            float clearReward = 0f;
            switch (linesCleared)
            {
                case 1:
                    clearReward = rewardWeights.clearReward;
                    break;
                case 2:
                    clearReward = rewardWeights.clearReward * rewardWeights.doubleLineClearRewardMultiplier;
                    break;
                case 3:
                    clearReward = rewardWeights.clearReward * rewardWeights.tripleLineClearRewardMultiplier;
                    break;
                case 4:
                    clearReward = rewardWeights.clearReward * rewardWeights.tetrisClearRewardMultiplier;
                    break;
            }


            // Combo system
            if (consecutiveClears > 0)
            {
                clearReward *= (1.0f + (consecutiveClears * rewardWeights.comboMultiplier));
            }
            consecutiveClears++;


            AddReward(clearReward);
            m_StatsRecorder.Add("action-rewarded/clear-reward", clearReward);
            previousScore = board.playerScore;
            stepsSinceLastClear = 0;


            // Perfect clear bonus
            if (board.IsPerfectClear())
            {
                AddReward(rewardWeights.perfectClearBonus);
                m_StatsRecorder.Add("action-rewarded/perfect-clear", rewardWeights.perfectClearBonus);
            }
        }
        else
        {
            consecutiveClears = 0;
            stepsSinceLastClear++;


            // Stagnation penalty
            if (stepsSinceLastClear > 100)
            {
                float stagnationPenalty = Mathf.Min((stepsSinceLastClear - 100) * rewardWeights.stagnationPenaltyFactor, 0.1f);
                AddReward(-stagnationPenalty);
                m_StatsRecorder.Add("action-rewarded/stagnation-penalty", -stagnationPenalty);
            }
        }


        ProcessBoardStateRewards();
        ProcessHoleManagement();
        ProcessWellFormation();
        ProcessStackHeightManagement();
        ProcessAccessibilityEvaluation();
        ProcessCurriculumTracking();
    }


    // Continue with all the existing helper methods...
    private void ProcessBoardStateRewards()
    {
        // Surface Smoothness
        float previousRoughness = lastSurfaceRoughness;
        float currentRoughness = CalculateSurfaceRoughness();


        int maxWidth = board.Bounds.size.x;
        float maxPossibleRoughness = maxWidth * curriculumBoardHeight;
        float normalizedCurrentRoughness = Mathf.Min(currentRoughness / maxPossibleRoughness, 1.0f);
        float normalizedPreviousRoughness = Mathf.Min(previousRoughness / maxPossibleRoughness, 1.0f);


        float roughnessDelta = normalizedPreviousRoughness - normalizedCurrentRoughness;


        if (roughnessDelta > 0.01f)
        {
            float roughnessReward = roughnessDelta * rewardWeights.roughnessRewardMultiplier;
            AddReward(roughnessReward);
            m_StatsRecorder.Add("action-rewarded/roughness-improvement", roughnessReward);
        }
        else if (roughnessDelta < -0.02f)
        {
            float roughnessPenalty = Mathf.Abs(roughnessDelta) * rewardWeights.roughnessPenaltyMultiplier;
            AddReward(-roughnessPenalty);
            m_StatsRecorder.Add("action-rewarded/roughness-penalty", -roughnessPenalty);
        }


        lastSurfaceRoughness = currentRoughness;


        // Wide base building
        int[] rowFills = board.GetRowFillCounts();
        int bottomCheckHeight = Mathf.Max(3, (int)(curriculumBoardHeight / 3));
        int minRowCoverage = Mathf.CeilToInt(maxWidth * 0.65f);


        int wideBaseRows = 0;
        for (int y = 0; y < bottomCheckHeight && y < rowFills.Length; y++)
        {
            int rowFill = rowFills[y];
            if (rowFill >= minRowCoverage)
                wideBaseRows++;
        }


        if (wideBaseRows > 0)
        {
            float stackingReward = wideBaseRows * rewardWeights.horizontalStackingRewardMultiplier;
            AddReward(stackingReward);
            m_StatsRecorder.Add("action-rewarded/horizontal-stack", stackingReward);
        }
    }


    private void ProcessHoleManagement()
    {
        List<Vector2Int> currentHoles = board.GetHolePositions();


        // Reward filling existing holes
        var filledHoles = previousHolePositions.Where(oldPos => !currentHoles.Contains(oldPos)).ToList();
        if (filledHoles.Count > 0)
        {
            float holeReward = 0f;
            foreach (var pos in filledHoles)
            {
                float depthFactor = 1f + (pos.y * 0.1f);
                holeReward += rewardWeights.holeFillReward * depthFactor;
            }
            AddReward(holeReward);
            m_StatsRecorder.Add("action-rewarded/hole-fill", holeReward);
        }


        // Penalty for creating new holes
        var newHoles = currentHoles.Where(newPos => !previousHolePositions.Contains(newPos)).ToList();
        if (newHoles.Count > 0)
        {
            float holePenalty = 0f;
            foreach (var pos in newHoles)
            {
                float heightFactor = 1f + ((curriculumBoardHeight - pos.y) * 0.1f);
                holePenalty += rewardWeights.holeCreationPenalty * heightFactor;
            }
            AddReward(-holePenalty);
            m_StatsRecorder.Add("action-rewarded/hole-creation", -holePenalty);
        }


        previousHolePositions = currentHoles;
    }


    private void ProcessWellFormation()
    {
        (int wellCol, int wellDepth) = GetDeepestWell();
        if (wellDepth >= 3)
        {
            float wellReward = Mathf.Min(wellDepth * rewardWeights.wellRewardMultiplier, rewardWeights.maxWellRewardCap);
            AddReward(wellReward);
            m_StatsRecorder.Add("action-rewarded/well-formation", wellReward);


            if (IsIPieceNext() && wellDepth >= 4)
            {
                AddReward(rewardWeights.iPieceInWellBonus);
                m_StatsRecorder.Add("action-rewarded/i-piece-well-setup", rewardWeights.iPieceInWellBonus);
            }
        }
    }


    private void ProcessStackHeightManagement()
    {
        float currentHeight = board.CalculateStackHeight();


        float heightRatio = currentHeight / curriculumBoardHeight;
        if (heightRatio > 0.5f)
        {
            float heightFactor = Mathf.Pow(heightRatio - 0.5f, 2) * 2.0f;
            float heightPenalty = rewardWeights.stackHeightPenalty * heightFactor;
            AddReward(-heightPenalty);
            m_StatsRecorder.Add("action-rewarded/height-penalty", -heightPenalty);
        }


        previousHeight = currentHeight;
    }


    private void ProcessAccessibilityEvaluation()
    {
        float accessibilityScore = EvaluateAccessibility();
        float accessibilityDelta = accessibilityScore - previousAccessibility;


        const float MAX_ACCESSIBILITY_CHANGE = 0.1f;
        accessibilityDelta = Mathf.Clamp(accessibilityDelta, -MAX_ACCESSIBILITY_CHANGE, MAX_ACCESSIBILITY_CHANGE);


        if (accessibilityDelta > 0.02f)
        {
            float accessibilityReward = accessibilityDelta * rewardWeights.accessibilityRewardMultiplier;
            AddReward(accessibilityReward);
            m_StatsRecorder.Add("action-rewarded/accessibility-improvement", accessibilityReward);
        }
        else if (accessibilityDelta < -0.02f)
        {
            float accessibilityPenalty = Mathf.Abs(accessibilityDelta) * rewardWeights.accessibilityPenaltyMultiplier;
            AddReward(-accessibilityPenalty);
            m_StatsRecorder.Add("action-rewarded/accessibility-loss", -accessibilityPenalty);
        }


        previousAccessibility = accessibilityScore;
    }


    private void ProcessCurriculumTracking()
    {
        if (episodeSteps % 1000 == 0)
        {
            float cumulativeReward = GetCumulativeReward();
            m_StatsRecorder.Add("reward/cumulative", cumulativeReward);


            m_StatsRecorder.Add("curriculum/board-height", curriculumBoardHeight);
            m_StatsRecorder.Add("curriculum/hole-penalty-weight", curriculumHolePenaltyWeight);
            m_StatsRecorder.Add("curriculum/tetromino-types", allowedTetrominoTypes);
            m_StatsRecorder.Add("curriculum/board-preset", curriculumBoardPreset);


            List<Vector2Int> currentHoles = board.GetHolePositions();
            float currentHeight = board.CalculateStackHeight();
            float normalizedCurrentRoughness = CalculateSurfaceRoughness() / (board.Bounds.size.x * curriculumBoardHeight);

            m_StatsRecorder.Add("metrics/current-holes", currentHoles.Count);
            m_StatsRecorder.Add("metrics/stack-height", currentHeight);
            m_StatsRecorder.Add("metrics/surface-roughness", normalizedCurrentRoughness);


            // Track column distribution
            for (int i = 0; i < 10; i++)
            {
                m_StatsRecorder.Add($"column-usage/column-{i}", columnUsageCount[i]);
            }
        }
    }


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


    public float CalculateColumnHeightVariance()
    {
        int[] heights = GetColumnHeights();
        double mean = heights.Average();
        double variance = heights.Select(h => Mathf.Pow((float)(h - mean), 2)).Average();
        return Mathf.Sqrt((float)variance);
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


        if (columnHeights.Length > 1)
        {
            int leftEdgeWell = columnHeights[1] - columnHeights[0];
            if (leftEdgeWell > deepestWellDepth && leftEdgeWell >= 2)
            {
                deepestWellDepth = leftEdgeWell;
                deepestWellCol = 0;
            }


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


    private bool IsIPieceNext()
    {
        return board.nextPieceData.tetromino == Tetromino.I;
    }


    private bool IsHorizontalPiece()
    {
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


    private bool FillsMultipleGaps()
    {
        if (!IsHorizontalPiece() || board.activePiece.data.tetromino != Tetromino.I)
            return false;


        int gapsFilled = 0;
        Vector3Int pos = board.activePiece.position;


        foreach (Vector3Int cell in board.activePiece.cells)
        {
            Vector3Int posBelow = new Vector3Int(cell.x + pos.x, cell.y + pos.y - 1, 0);
            if (!board.tilemap.HasTile(posBelow) && posBelow.y >= board.Bounds.yMin)
            {
                gapsFilled++;
            }
        }


        return gapsFilled >= 2;
    }


    private bool IsPotentialTSpin()
    {
        if (board.activePiece.data.tetromino != Tetromino.T)
            return false;


        Vector3Int pos = board.activePiece.position;
        int cornersCount = 0;


        Vector3Int[] corners = new Vector3Int[]
        {
            new Vector3Int(-1, -1, 0), new Vector3Int(1, -1, 0),
            new Vector3Int(-1, 1, 0), new Vector3Int(1, 1, 0)
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


    private float EvaluateAccessibility()
    {
        int[] colHeights = GetColumnHeights();
        if (colHeights == null || colHeights.Length == 0) return 0f;


        float accessibility = 0f;
        float maxBoardHeight = curriculumBoardHeight;


        for (int i = 0; i < colHeights.Length; i++)
        {
            float centerDistance = Mathf.Abs(i - (colHeights.Length / 2.0f));
            float maxCenterDistance = colHeights.Length / 2.0f;
            float centerFactor = 1.0f - (centerDistance / maxCenterDistance);
            centerFactor = 0.3f + (centerFactor * 0.7f);


            float heightValue = Mathf.Clamp01((maxBoardHeight - colHeights[i]) / maxBoardHeight);


            accessibility += heightValue * centerFactor;
        }


        return accessibility / colHeights.Length;
    }


    public void OnGameOver()
    {
        AddReward(rewardWeights.deathPenalty);
        SaveFitnessScore(GetCumulativeReward());
        EndEpisode();
    }


    // IPlayerInputController implementation - unchanged
    public bool GetLeft() => moveLeft;
    public bool GetRight() => moveRight;
    public bool GetRotateLeft() => rotateLeft;
    public bool GetRotateRight() => rotateRight;
    public bool GetDown() => moveDown;
    public bool GetHardDrop() => hardDrop;


    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;

        // Column selection (0-9)
        if (Input.GetKey(KeyCode.Alpha1)) discreteActionsOut[0] = 0;
        else if (Input.GetKey(KeyCode.Alpha2)) discreteActionsOut[0] = 1;
        else if (Input.GetKey(KeyCode.Alpha3)) discreteActionsOut[0] = 2;
        else if (Input.GetKey(KeyCode.Alpha4)) discreteActionsOut[0] = 3;
        else if (Input.GetKey(KeyCode.Alpha5)) discreteActionsOut[0] = 4;
        else if (Input.GetKey(KeyCode.Alpha6)) discreteActionsOut[0] = 5;
        else if (Input.GetKey(KeyCode.Alpha7)) discreteActionsOut[0] = 6;
        else if (Input.GetKey(KeyCode.Alpha8)) discreteActionsOut[0] = 7;
        else if (Input.GetKey(KeyCode.Alpha9)) discreteActionsOut[0] = 8;
        else if (Input.GetKey(KeyCode.Alpha0)) discreteActionsOut[0] = 9;
        else discreteActionsOut[0] = 4; // Default to middle column

        // Rotation selection (0-3)
        if (Input.GetKey(KeyCode.Q)) discreteActionsOut[1] = 0;
        else if (Input.GetKey(KeyCode.W)) discreteActionsOut[1] = 1;
        else if (Input.GetKey(KeyCode.E)) discreteActionsOut[1] = 2;
        else if (Input.GetKey(KeyCode.R)) discreteActionsOut[1] = 3;
        else discreteActionsOut[1] = 0; // Default to no rotation
    }
}
