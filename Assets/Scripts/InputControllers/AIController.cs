

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

    // Changed: Remove individual movement flags
    private bool hasChosenPlacement = false;
    private int chosenColumn = 0;
    private int chosenRotation = 0;

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

        // Initialize all other reward weights...
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

            // Changed: Two discrete actions - column (10 options) and rotation (4 options)
            ActionSpec actionSpec = ActionSpec.MakeDiscrete(new int[] { 10, 4 });
            behavior.BrainParameters.ActionSpec = actionSpec;
        }
        else
        {
            behavior.BehaviorName = "TetrisAgent";
            behavior.BrainParameters.VectorObservationSize = 228;
            behavior.BrainParameters.NumStackedVectorObservations = 1;

            // Changed: Two discrete actions - column (10 options) and rotation (4 options)
            ActionSpec actionSpec = ActionSpec.MakeDiscrete(new int[] { 10, 4 });
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
        }
    }

    // Called by Board.cs to set the current piece reference
    public void SetCurrentPiece(Piece piece)
    {
        currentPiece = piece;
        // Reset placement choice when new piece spawns
        hasChosenPlacement = false;
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

        // Apply curriculum settings to reward weights
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

        int obsCount = 0;

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
                    sensor.AddObservation(0f); // padded empty row
                }
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
            obsCount++;
        }

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
        sensor.AddObservation(holeCount / 20.0f); // normalize
        obsCount++;
    }

    private List<Vector2Int> previousBottomRowHoles = new List<Vector2Int>();
    private int steps = 0;

    public override void OnActionReceived(ActionBuffers actions)
    {
        // Base survival reward - keep consistent across curriculum
        AddReward(0.01f);
        m_StatsRecorder.Add("action-rewarded/survival", 0.01f);

        episodeSteps++;
        steps++;

        if (board == null || currentPiece == null)
            return;

        // Changed: Get placement choice from actions
        int targetColumn = actions.DiscreteActions[0]; // 0-9 for columns
        int targetRotation = actions.DiscreteActions[1]; // 0-3 for rotations

        if (debugger != null)
            debugger.SetLastAction(targetColumn * 4 + targetRotation); // Combined action for debugging

        // Store the AI's choice
        if (!hasChosenPlacement)
        {
            chosenColumn = targetColumn;
            chosenRotation = targetRotation;
            hasChosenPlacement = true;

            // Evaluate and reward the placement choice
            EvaluatePlacementChoice(targetColumn, targetRotation);
        }

        // Rest of the reward system remains the same...
        ProcessRewards();
    }

    private void EvaluatePlacementChoice(int targetColumn, int targetRotation)
    {
        // Simulate the placement to evaluate its quality
        Vector3Int simulatedPosition = new Vector3Int(targetColumn, currentPiece.position.y, 0);

        // Create a temporary piece data for simulation
        TetrominoData rotatedData = GetRotatedPieceData(currentPiece.data, targetRotation);

        // Calculate where the piece would land
        Vector3Int landingPosition = SimulateDrop(simulatedPosition, rotatedData);

        // Evaluate the quality of this placement
        float placementScore = EvaluatePlacementQuality(landingPosition, rotatedData);

        // Apply rewards based on placement quality
        AddReward(placementScore);
        m_StatsRecorder.Add("action-rewarded/placement-quality", placementScore);

        // Bonus for strategic placements
        if (IsStrategicPlacement(landingPosition, rotatedData))
        {
            AddReward(0.5f);
            m_StatsRecorder.Add("action-rewarded/strategic-placement", 0.5f);
        }
    }

    private TetrominoData GetRotatedPieceData(TetrominoData originalData, int targetRotation)
    {
        // This would need to be implemented based on how your tetromino rotation works
        // For now, return the original data (you'll need to implement proper rotation logic)
        return originalData;
    }

    private Vector3Int SimulateDrop(Vector3Int startPosition, TetrominoData pieceData)
    {
        Vector3Int position = startPosition;

        // Simulate dropping the piece until it can't move down anymore
        while (CanPieceMoveTo(position + Vector3Int.down, pieceData))
        {
            position += Vector3Int.down;
        }

        return position;
    }

    private bool CanPieceMoveTo(Vector3Int position, TetrominoData pieceData)
    {
        // Check if the piece can be placed at this position
        // This would need to check against board bounds and existing tiles
        RectInt bounds = board.Bounds;

        foreach (Vector2Int cell in pieceData.cells)
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

    private float EvaluatePlacementQuality(Vector3Int position, TetrominoData pieceData)
    {
        float score = 0f;

        // Factor 1: Height penalty (lower is better)
        float heightPenalty = position.y * 0.01f;
        score -= heightPenalty;

        // Factor 2: Line completion potential
        score += EvaluateLineCompletionPotential(position, pieceData) * 2.0f;

        // Factor 3: Hole creation penalty
        score -= EvaluateHoleCreation(position, pieceData) * 1.0f;

        // Factor 4: Surface smoothness
        score += EvaluateSurfaceSmoothness(position, pieceData) * 0.5f;

        return Mathf.Clamp(score, -2.0f, 2.0f); // Clamp to reasonable range
    }

    private float EvaluateLineCompletionPotential(Vector3Int position, TetrominoData pieceData)
    {
        // Count how many lines this placement would complete or nearly complete
        float potential = 0f;

        HashSet<int> affectedRows = new HashSet<int>();
        foreach (Vector2Int cell in pieceData.cells)
        {
            affectedRows.Add(position.y + cell.y);
        }

        foreach (int row in affectedRows)
        {
            int filledCells = CountFilledCellsInRow(row);
            int cellsFromPiece = CountPieceCellsInRow(position, pieceData, row);
            int totalAfterPlacement = filledCells + cellsFromPiece;

            if (totalAfterPlacement == 10) // Complete line
            {
                potential += 1.0f;
            }
            else if (totalAfterPlacement >= 8) // Nearly complete
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

    private int CountPieceCellsInRow(Vector3Int position, TetrominoData pieceData, int row)
    {
        int count = 0;

        foreach (Vector2Int cell in pieceData.cells)
        {
            if (position.y + cell.y == row)
            {
                count++;
            }
        }

        return count;
    }

    private float EvaluateHoleCreation(Vector3Int position, TetrominoData pieceData)
    {
        // Estimate how many holes this placement might create
        float holeRisk = 0f;

        foreach (Vector2Int cell in pieceData.cells)
        {
            Vector3Int cellPos = new Vector3Int(position.x + cell.x, position.y + cell.y, 0);

            // Check if placing this cell creates overhang that might trap empty spaces
            Vector3Int below = cellPos + Vector3Int.down;
            if (!board.tilemap.HasTile(below) && below.y >= board.Bounds.yMin)
            {
                holeRisk += 0.2f;
            }
        }

        return holeRisk;
    }

    private float EvaluateSurfaceSmoothness(Vector3Int position, TetrominoData pieceData)
    {
        // Evaluate how this placement affects surface smoothness
        // This is a simplified version - you'd want to calculate actual height differences
        return 0f; // Placeholder
    }

    private bool IsStrategicPlacement(Vector3Int position, TetrominoData pieceData)
    {
        // Check for strategic placements like T-spins, well formations, etc.

        // T-spin setup
        if (currentPiece.data.tetromino == Tetromino.T && enableAdvancedMechanics)
        {
            return IsTSpinSetup(position, pieceData);
        }

        // I-piece in well
        if (currentPiece.data.tetromino == Tetromino.I)
        {
            return IsIPieceInWell(position, pieceData);
        }

        return false;
    }

    private bool IsTSpinSetup(Vector3Int position, TetrominoData pieceData)
    {
        // Check if this T-piece placement sets up a potential T-spin
        // This is a simplified check - implement based on your T-spin detection logic
        return false; // Placeholder
    }

    private bool IsIPieceInWell(Vector3Int position, TetrominoData pieceData)
    {
        // Check if I-piece is being placed in a well formation
        (int wellCol, int wellDepth) = GetDeepestWell();

        if (wellDepth >= 4)
        {
            // Check if I-piece overlaps with the well column
            foreach (Vector2Int cell in pieceData.cells)
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
        // All the existing reward processing logic remains the same
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
            int linesCleared = scoreDelta / 100; // Each line is worth 100 points in Board.cs

            // Line clear rewards with proper multipliers
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

            // Stagnation penalty - curriculum aware through penalty factor
            if (stepsSinceLastClear > 100)
            {
                float stagnationPenalty = Mathf.Min((stepsSinceLastClear - 100) * rewardWeights.stagnationPenaltyFactor, 0.1f);
                AddReward(-stagnationPenalty);
                m_StatsRecorder.Add("action-rewarded/stagnation-penalty", -stagnationPenalty);
            }
        }

        // Continue with all other existing reward logic...
        ProcessBoardStateRewards();
        ProcessHoleManagement();
        ProcessWellFormation();
        ProcessStackHeightManagement();
        ProcessAccessibilityEvaluation();
        ProcessCurriculumTracking();
    }

    // Break down the reward processing into smaller methods for clarity
    private void ProcessBoardStateRewards()
    {
        // Surface Smoothness - Normalized to prevent explosion
        float previousRoughness = lastSurfaceRoughness;
        float currentRoughness = CalculateSurfaceRoughness();

        // Normalize roughness by board dimensions to prevent extreme values
        int maxWidth = board.Bounds.size.x;
        float maxPossibleRoughness = maxWidth * curriculumBoardHeight;
        float normalizedCurrentRoughness = Mathf.Min(currentRoughness / maxPossibleRoughness, 1.0f);
        float normalizedPreviousRoughness = Mathf.Min(previousRoughness / maxPossibleRoughness, 1.0f);

        float roughnessDelta = normalizedPreviousRoughness - normalizedCurrentRoughness;

        if (roughnessDelta > 0.01f) // Significant improvement
        {
            float roughnessReward = roughnessDelta * rewardWeights.roughnessRewardMultiplier;
            AddReward(roughnessReward);
            m_StatsRecorder.Add("action-rewarded/roughness-improvement", roughnessReward);
        }
        else if (roughnessDelta < -0.02f) // Significant worsening
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
                // Scale reward by depth - deeper holes are more valuable to fill
                float depthFactor = 1f + (pos.y * 0.1f);
                holeReward += rewardWeights.holeFillReward * depthFactor;
            }
            AddReward(holeReward);
            m_StatsRecorder.Add("action-rewarded/hole-fill", holeReward);
        }

        // Penalty for creating new holes - uses curriculum-adjusted penalty
        var newHoles = currentHoles.Where(newPos => !previousHolePositions.Contains(newPos)).ToList();
        if (newHoles.Count > 0)
        {
            float holePenalty = 0f;
            foreach (var pos in newHoles)
            {
                // Higher penalty for creating holes higher up
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

            // I-piece in well bonus
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

        // Progressive height penalty based on curriculum board height
        float heightRatio = currentHeight / curriculumBoardHeight;
        if (heightRatio > 0.5f) // Only penalize when stack gets to half the board height
        {
            // Exponential penalty for dangerous heights
            float heightFactor = Mathf.Pow(heightRatio - 0.5f, 2) * 2.0f; // Quadratic scaling
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

        // Cap accessibility changes to prevent reward explosion
        const float MAX_ACCESSIBILITY_CHANGE = 0.1f;
        accessibilityDelta = Mathf.Clamp(accessibilityDelta, -MAX_ACCESSIBILITY_CHANGE, MAX_ACCESSIBILITY_CHANGE);

        if (accessibilityDelta > 0.02f) // Significant improvement
        {
            float accessibilityReward = accessibilityDelta * rewardWeights.accessibilityRewardMultiplier;
            AddReward(accessibilityReward);
            m_StatsRecorder.Add("action-rewarded/accessibility-improvement", accessibilityReward);
        }
        else if (accessibilityDelta < -0.02f) // Significant degradation
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

            // Track curriculum parameters for debugging
            m_StatsRecorder.Add("curriculum/board-height", curriculumBoardHeight);
            m_StatsRecorder.Add("curriculum/hole-penalty-weight", curriculumHolePenaltyWeight);
            m_StatsRecorder.Add("curriculum/tetromino-types", allowedTetrominoTypes);
            m_StatsRecorder.Add("curriculum/board-preset", curriculumBoardPreset);

            // Track key metrics
            List<Vector2Int> currentHoles = board.GetHolePositions();
            float currentHeight = board.CalculateStackHeight();
            float normalizedCurrentRoughness = CalculateSurfaceRoughness() / (board.Bounds.size.x * curriculumBoardHeight);

            m_StatsRecorder.Add("metrics/current-holes", currentHoles.Count);
            m_StatsRecorder.Add("metrics/stack-height", currentHeight);
            m_StatsRecorder.Add("metrics/surface-roughness", normalizedCurrentRoughness);
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
        return Mathf.Sqrt((float)variance); // Standard deviation
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
        if (colHeights == null || colHeights.Length == 0) return 0f;

        float accessibility = 0f;
        float maxBoardHeight = curriculumBoardHeight;

        for (int i = 0; i < colHeights.Length; i++)
        {
            // Center columns are more important for piece placement
            float centerDistance = Mathf.Abs(i - (colHeights.Length / 2.0f));
            float maxCenterDistance = colHeights.Length / 2.0f;
            float centerFactor = 1.0f - (centerDistance / maxCenterDistance);
            centerFactor = 0.3f + (centerFactor * 0.7f); // Scale to 0.3-1.0 range

            // Height accessibility: more open space = better
            float heightValue = Mathf.Clamp01((maxBoardHeight - colHeights[i]) / maxBoardHeight);

            accessibility += heightValue * centerFactor;
        }

        return accessibility / colHeights.Length; // Normalize by column count
    }

    // Called by Board.cs when game over occurs
    public void OnGameOver()
    {
        AddReward(rewardWeights.deathPenalty); // Big penalty for losing
        SaveFitnessScore(GetCumulativeReward());
        EndEpisode();
    }

    // Changed: IPlayerInputController implementation now uses the chosen placement
    public bool GetLeft()
    {
        if (!hasChosenPlacement) return false;

        // Move towards chosen column
        if (currentPiece != null && currentPiece.position.x > chosenColumn)
            return true;

        return false;
    }

    public bool GetRight()
    {
        if (!hasChosenPlacement) return false;

        // Move towards chosen column
        if (currentPiece != null && currentPiece.position.x < chosenColumn)
            return true;

        return false;
    }

    public bool GetRotateLeft()
    {
        if (!hasChosenPlacement) return false;

        // Rotate towards chosen rotation
        if (currentPiece != null && currentPiece.rotationIndex != chosenRotation)
        {
            // Determine if left rotation gets us there faster
            int currentRot = currentPiece.rotationIndex;
            int leftSteps = (currentRot - chosenRotation + 4) % 4;
            int rightSteps = (chosenRotation - currentRot + 4) % 4;
            return leftSteps <= rightSteps;
        }

        return false;
    }

    public bool GetRotateRight()
    {
        if (!hasChosenPlacement) return false;

        // Rotate towards chosen rotation
        if (currentPiece != null && currentPiece.rotationIndex != chosenRotation)
        {
            // Determine if right rotation gets us there faster
            int currentRot = currentPiece.rotationIndex;
            int leftSteps = (currentRot - chosenRotation + 4) % 4;
            int rightSteps = (chosenRotation - currentRot + 4) % 4;
            return rightSteps < leftSteps;
        }

        return false;
    }

    public bool GetDown()
    {
        // Always allow soft drop once positioned correctly
        if (!hasChosenPlacement) return false;

        // Check if we're at the right position and rotation
        if (currentPiece != null &&
            currentPiece.position.x == chosenColumn &&
            currentPiece.rotationIndex == chosenRotation)
        {
            return true;
        }

        return false;
    }

    public bool GetHardDrop()
    {
        // Hard drop when positioned correctly
        if (!hasChosenPlacement) return false;

        // Check if we're at the right position and rotation
        if (currentPiece != null &&
            currentPiece.position.x == chosenColumn &&
            currentPiece.rotationIndex == chosenRotation)
        {
            return true;
        }

        return false;
    }

    // For testing in editor
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





