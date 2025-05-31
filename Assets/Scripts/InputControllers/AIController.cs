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
        rewardWeights.clearReward = envParams.GetWithDefault("clearReward", 5.0f); // Increased from 1.0f
        rewardWeights.comboMultiplier = envParams.GetWithDefault("comboMultiplier", 0.5f); // Increased from 0.2f
        rewardWeights.perfectClearBonus = envParams.GetWithDefault("perfectClearBonus", 50.0f); // Increased from 20.0f

        // REDUCED PENALTIES
        rewardWeights.stagnationPenaltyFactor = envParams.GetWithDefault("stagnationPenaltyFactor", 0.001f); // Reduced from 0.01f
        rewardWeights.roughnessPenaltyMultiplier = envParams.GetWithDefault("roughnessPenaltyMultiplier", 0.01f); // Reduced from 0.05f
        // rewardWeights.holeCreationPenalty = envParams.GetWithDefault("holeCreationPenalty", 0.05f); // Reduced from 0.2f
        rewardWeights.stackHeightPenalty = envParams.GetWithDefault("stackHeightPenalty", 0.02f); // Reduced from 0.1f
        rewardWeights.uselessRotationPenalty = envParams.GetWithDefault("uselessRotationPenalty", 0.01f); // Reduced from 0.05f
        rewardWeights.idleActionPenalty = envParams.GetWithDefault("idleActionPenalty", 0.001f); // Reduced from 0.01f

        // INCREASED POSITIVE REWARDS
        rewardWeights.holeFillReward = envParams.GetWithDefault("holeFillReward", 1.0f); // Increased from 0.3f
        rewardWeights.moveDownActionReward = envParams.GetWithDefault("moveDownActionReward", 0.02f); // Increased from 0.01f
        rewardWeights.hardDropActionReward = envParams.GetWithDefault("hardDropActionReward", 0.05f); // Increased from 0.025f

        // Keep multipliers the same but ensure base rewards are higher
        rewardWeights.doubleLineClearRewardMultiplier = envParams.GetWithDefault("doubleLineClearRewardMultiplier", 3.0f);
        rewardWeights.tripleLineClearRewardMultiplier = envParams.GetWithDefault("tripleLineClearRewardMultiplier", 7.0f);
        rewardWeights.tetrisClearRewardMultiplier = envParams.GetWithDefault("tetrisClearRewardMultiplier", 15.0f);

        // Other rewards stay similar but some increased
        rewardWeights.wellRewardMultiplier = envParams.GetWithDefault("wellRewardMultiplier", 0.2f); // Increased from 0.1f
        rewardWeights.iPieceInWellBonus = envParams.GetWithDefault("iPieceInWellBonus", 0.5f); // Increased from 0.3f
        rewardWeights.tSpinReward = envParams.GetWithDefault("tSpinReward", 1.0f); // Increased from 0.5f
        rewardWeights.iPieceGapFillBonus = envParams.GetWithDefault("iPieceGapFillBonus", 0.8f); // Increased from 0.4f

        rewardWeights.roughnessRewardMultiplier = envParams.GetWithDefault("roughnessRewardMultiplier", 0.3f);
        rewardWeights.accessibilityRewardMultiplier = envParams.GetWithDefault("accessibilityRewardMultiplier", 0.2f);
        rewardWeights.accessibilityPenaltyMultiplier = envParams.GetWithDefault("accessibilityPenaltyMultiplier", 0.05f); // Reduced from 0.1f
        rewardWeights.deathPenalty = envParams.GetWithDefault("deathPenalty", 2.0f); // Reduced from 10.0f
        rewardWeights.maxWellRewardCap = envParams.GetWithDefault("maxWellRewardCap", 1.0f); // Increased from 0.5f

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
        if (board == null)
        {
            // Debug.Log("Board not found in children!");
        }
        var behavior = gameObject.GetComponent<BehaviorParameters>();
        if (behavior == null)
        {
            behavior = gameObject.AddComponent<BehaviorParameters>();
            behavior.BehaviorName = "TetrisAgent";
            behavior.BrainParameters.VectorObservationSize = 228;
            behavior.BrainParameters.NumStackedVectorObservations = 1;

            // Set up discrete actions (7 possible actions)
            ActionSpec actionSpec = ActionSpec.MakeDiscrete(new int[] { 7 });
            behavior.BrainParameters.ActionSpec = actionSpec;
        }
        else
        {
            behavior.BehaviorName = "TetrisAgent";
            behavior.BrainParameters.VectorObservationSize = 228;
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
            // Debug.Log("TetrisMLAgent set as input controller for board");
        }
    }

    // Called by Board.cs to set the current piece reference
    public void SetCurrentPiece(Piece piece)
    {
        currentPiece = piece;

    }

    public override void OnEpisodeBegin()
    {
        // Debug.Log("ML Agent Episode began");
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
            // Debug.LogError("Board is null during observation collection!");
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

        // Action rewards using your reward weights
        switch (actionIndex)
        {
            case 0:
                AddReward(-rewardWeights.idleActionPenalty);
                m_StatsRecorder.Add("action-rewarded/do-nothing", -rewardWeights.idleActionPenalty);
                break;
            case 1: moveLeft = true; break;
            case 2: moveRight = true; break;
            case 3: rotateLeft = true; break;
            case 4: rotateRight = true; break;
            case 5:
                moveDown = true;
                AddReward(rewardWeights.moveDownActionReward);
                m_StatsRecorder.Add("action-rewarded/move-down", rewardWeights.moveDownActionReward);
                break;
            case 6:
                hardDrop = true;
                AddReward(rewardWeights.hardDropActionReward);
                m_StatsRecorder.Add("action-rewarded/hard-drop", rewardWeights.hardDropActionReward);
                break;
        }

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
            m_StatsRecorder.Add("action-rewarded/clear-reward", clearReward); // Fixed: was using base reward instead of actual
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
                m_StatsRecorder.Add("action-rewarded/stagnation-penalty", -stagnationPenalty); // Fixed: was using wrong variable
            }
        }

        // === BOARD STATE EVALUATION ===

        // Surface Smoothness - Normalized to prevent explosion
        float previousRoughness = lastSurfaceRoughness;
        float currentRoughness = CalculateSurfaceRoughness();

        // Normalize roughness by board dimensions to prevent extreme values
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

        // === HOLE MANAGEMENT ===
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

        // === WELL FORMATION ===
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

        // === STACK HEIGHT MANAGEMENT ===
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

        // === PIECE-SPECIFIC STRATEGY REWARDS ===
        if (actionIndex == 6) // Hard drop - evaluate final placement
        {
            // T-spin rewards (only if advanced mechanics are enabled)
            if (enableAdvancedMechanics && board.activePiece.data.tetromino == Tetromino.T && IsPotentialTSpin())
            {
                AddReward(rewardWeights.tSpinReward);
                m_StatsRecorder.Add("action-rewarded/t-spin-setup", rewardWeights.tSpinReward);
            }

            // I-piece gap filling
            if (board.activePiece.data.tetromino == Tetromino.I &&
                IsHorizontalPiece() && FillsMultipleGaps())
            {
                AddReward(rewardWeights.iPieceGapFillBonus);
                m_StatsRecorder.Add("action-rewarded/i-piece-gap-fill", rewardWeights.iPieceGapFillBonus);
            }
        }

        // === EFFICIENCY PENALTIES ===
        if ((rotateLeft || rotateRight) && board.LastRotationWasUseless(board.activePiece, prevPosition, prevCells))
        {
            AddReward(-rewardWeights.uselessRotationPenalty);
            m_StatsRecorder.Add("action-rewarded/useless-rotation", -rewardWeights.uselessRotationPenalty);
        }

        // === ACCESSIBILITY EVALUATION ===
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

        // === CURRICULUM PROGRESS TRACKING ===
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
            m_StatsRecorder.Add("metrics/current-holes", currentHoles.Count);
            m_StatsRecorder.Add("metrics/stack-height", currentHeight);
            m_StatsRecorder.Add("metrics/surface-roughness", normalizedCurrentRoughness);
        }
    } // // New helper methods for enhanced rewards

    // public override void OnActionReceived(ActionBuffers actions)
    // {
    //     episodeSteps++;

    //     // Reset action flags
    //     moveLeft = false;
    //     moveRight = false;
    //     rotateLeft = false;
    //     rotateRight = false;
    //     moveDown = false;
    //     hardDrop = false;

    //     Vector3Int prevPosition = board.activePiece.position;
    //     Vector3Int[] prevCells = (Vector3Int[])board.activePiece.cells.Clone();
    //     steps++;
    //     // Debug.Log("Action received: " + actions.DiscreteActions[0]);

    //     int actionIndex = actions.DiscreteActions[0];

    //     switch (actionIndex)
    //     {
    //         case 1: moveLeft = true; break;
    //         case 2: moveRight = true; break;
    //         case 3: rotateLeft = true; break;
    //         case 4: rotateRight = true; break;
    //         case 5: moveDown = true; break;
    //         case 6: hardDrop = true; break;
    //     }

    //     if (moveLeft)
    //     {
    //         AddReward(1.0f);
    //     }
    //     else
    //     {
    //         AddReward(-1.0f);
    //     }

    //     // === REWARD TRACKING ===
    //     float cumulativeReward = GetCumulativeReward();
    //     m_StatsRecorder.Add("reward/cumulative", cumulativeReward);
    // }

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
    // Check if a move is safe
    // Called by Board.cs when game over occurs
    public void OnGameOver()
    {
        AddReward(rewardWeights.deathPenalty); // Big penalty for losing
        SaveFitnessScore(GetCumulativeReward());
        EndEpisode();
        // Debug.Log("Game over - episode ended");
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