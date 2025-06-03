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
    private bool readyForDecision = true;
    private StatsRecorder m_StatsRecorder;
    private DecisionRequester decisionRequester;

    // Stats tracking variables
    private float lastDecisionTime;
    private float episodeStartTime;
    private int decisionCount = 0;
    private int totalPiecesPlaced = 0;

    private RewardWeights rewardWeights = new RewardWeights();

    [Header("Curriculum Parameters")]
    public int allowedTetrominoTypes = 7;
    public int curriculumBoardPreset;
    public float curriculumBoardHeight = 20f;
    public float curriculumDropSpeed = 0.75f;
    public float curriculumHolePenaltyWeight = 0.5f;
    public bool enableAdvancedMechanics = false;

    // Input flags for IPlayerInputController
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

    private void Start()
    {
        if (board != null)
        {
            board.inputController = this;
        }
    }

    public override void OnEpisodeBegin()
    {
        var envParams = Academy.Instance.EnvironmentParameters;
        processedPieceIds.Clear();
        currentPieceId = -1;
        readyForDecision = true;

        // Reset stats tracking
        decisionCount = 0;
        totalPiecesPlaced = 0;
        episodeStartTime = Time.time;
        lastDecisionTime = Time.time;

        // Update curriculum parameters
        allowedTetrominoTypes = (int)envParams.GetWithDefault("tetromino_types", 7f);
        curriculumBoardHeight = envParams.GetWithDefault("board_height", 20f);
        curriculumDropSpeed = envParams.GetWithDefault("drop_speed", 0.75f);
        curriculumHolePenaltyWeight = envParams.GetWithDefault("hole_penalty_weight", 0.5f);
        enableAdvancedMechanics = envParams.GetWithDefault("enable_t_spins", 0f) > 0.5f;
        curriculumBoardPreset = (int)envParams.GetWithDefault("board_preset", 6);

        rewardWeights.holeCreationPenalty *= curriculumHolePenaltyWeight;

        // Record episode start
        m_StatsRecorder.Add("Tetris/EpisodeStarted", 1);
        m_StatsRecorder.Add("Tetris/EpisodeNumber", CompletedEpisodes + 1);

        Debug.Log($"Episode {CompletedEpisodes + 1} started");
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
        }
        
        behavior.BehaviorName = "TetrisAgent";
        behavior.BrainParameters.VectorObservationSize = 218;
        behavior.BrainParameters.NumStackedVectorObservations = 1;
        ActionSpec actionSpec = ActionSpec.MakeDiscrete(new int[] { 40 });
        behavior.BrainParameters.ActionSpec = actionSpec;

        // Setup DecisionRequester for manual control
        decisionRequester = gameObject.GetComponent<DecisionRequester>();
        if (decisionRequester == null)
        {
            decisionRequester = gameObject.AddComponent<DecisionRequester>();
        }

        // Disable automatic decisions - we control manually
        decisionRequester.DecisionPeriod = 99999;
        decisionRequester.TakeActionsBetweenDecisions = false;
    }

    public void SetCurrentPiece(Piece piece)
    {
        if (piece == null) return;

        currentPiece = piece;
        int pieceId = GeneratePieceId(piece);

        if (!processedPieceIds.Contains(pieceId) && readyForDecision)
        {
            processedPieceIds.Add(pieceId);
            currentPieceId = pieceId;
            readyForDecision = false;
            
            Debug.Log($"New piece detected - ID: {pieceId}, Type: {piece.data.tetromino}");
            
            // Manual decision request with stats tracking
            RequestDecisionWithStats();
        }
    }

    private void RequestDecisionWithStats()
    {
        decisionCount++;
        float currentTime = Time.time;
        
        // Record pre-decision stats
        m_StatsRecorder.Add("Tetris/DecisionRequested", 1);
        m_StatsRecorder.Add("Tetris/DecisionCount", decisionCount);
        m_StatsRecorder.Add("Tetris/CumulativeReward", GetCumulativeReward());
        
        if (decisionCount > 1)
        {
            float timeSinceLastDecision = currentTime - lastDecisionTime;
            m_StatsRecorder.Add("Tetris/DecisionInterval", timeSinceLastDecision);
            m_StatsRecorder.Add("Tetris/DecisionsPerSecond", 1f / timeSinceLastDecision);
        }
        
        lastDecisionTime = currentTime;
        
        // Current game state stats
        RecordGameStateStats();
        
        Debug.Log($"Requesting decision #{decisionCount} at time {currentTime}");
        
        // Actually request the decision
        RequestDecision();
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        // Record that decision was received
        float receptionTime = Time.time;
        m_StatsRecorder.Add("Tetris/DecisionReceived", 1);
        m_StatsRecorder.Add("Tetris/DecisionLatency", receptionTime - lastDecisionTime);
        
        if (currentPiece == null || board == null)
        {
            m_StatsRecorder.Add("Tetris/InvalidDecision_NoPiece", 1);
            Debug.LogWarning("Decision received but no current piece!");
            readyForDecision = true;
            return;
        }

        if (!readyForDecision)
        {
            m_StatsRecorder.Add("Tetris/InvalidDecision_NotReady", 1);
            Debug.LogWarning("Decision received but not ready to process!");
            return;
        }

        // Generate valid placements
        int[,] currentBoard = GetBoardState();
        List<PlacementInfo> allPlacements = GenerateAllPossiblePlacements(currentPiece, currentBoard);

        if (allPlacements.Count == 0)
        {
            m_StatsRecorder.Add("Tetris/GameOver_NoValidPlacements", 1);
            AddReward(rewardWeights.deathPenalty);
            OnGameOver();
            return;
        }

        int selectedAction = actions.DiscreteActions[0];

        // Find the placement that corresponds to this action
        PlacementInfo selectedPlacement = null;
        foreach (var placement in allPlacements)
        {
            int actionIndex = placement.targetRotation * 10 + placement.targetColumn;
            if (actionIndex == selectedAction)
            {
                selectedPlacement = placement;
                break;
            }
        }

        // If invalid action selected, fall back to first valid placement
        if (selectedPlacement == null)
        {
            selectedPlacement = allPlacements[0];
            m_StatsRecorder.Add("Tetris/InvalidAction_Fallback", 1);
            Debug.LogWarning($"Invalid action {selectedAction} selected, falling back to first valid placement");
        }

        Debug.Log($"Selected Action: {selectedAction} → Column: {selectedPlacement.targetColumn}, Rotation: {selectedPlacement.targetRotation}");

        // Record decision stats immediately
        RecordDecisionStats(selectedPlacement, allPlacements);
        
        // Execute placement
        StartCoroutine(ExecutePlacementCoroutine(selectedPlacement));
    }

    private void RecordGameStateStats()
    {
        if (board == null) return;
        
        // Current board state
        int[,] boardState = GetBoardState();
        m_StatsRecorder.Add("Tetris/CurrentHeight", CalculateMaxHeight(boardState));
        m_StatsRecorder.Add("Tetris/CurrentHoles", CalculateHoles(boardState));
        m_StatsRecorder.Add("Tetris/CurrentBumpiness", CalculateBumpiness(boardState));
        m_StatsRecorder.Add("Tetris/CurrentAggregateHeight", CalculateAggregateHeight(boardState));
        
        // Piece information
        if (currentPiece != null)
        {
            m_StatsRecorder.Add("Tetris/CurrentPieceType", (int)currentPiece.data.tetromino);
            m_StatsRecorder.Add("Tetris/PiecePositionX", currentPiece.position.x);
            m_StatsRecorder.Add("Tetris/PiecePositionY", currentPiece.position.y);
            m_StatsRecorder.Add("Tetris/PieceRotation", currentPiece.rotationIndex);
        }
        
        // Game progress
        float episodeTime = Time.time - episodeStartTime;
        m_StatsRecorder.Add("Tetris/EpisodeTime", episodeTime);
        m_StatsRecorder.Add("Tetris/PiecesPlaced", totalPiecesPlaced);
        
        if (episodeTime > 0)
        {
            m_StatsRecorder.Add("Tetris/PiecesPerSecond", totalPiecesPlaced / episodeTime);
        }
    }

    private void RecordDecisionStats(PlacementInfo placement, List<PlacementInfo> allPlacements)
    {
        // Decision analysis
        m_StatsRecorder.Add("Tetris/DecisionProcessed", 1);
        m_StatsRecorder.Add("Tetris/PredictedLinesCleared", placement.linesCleared);
        m_StatsRecorder.Add("Tetris/PredictedHoles", placement.holes);
        m_StatsRecorder.Add("Tetris/PredictedHeight", placement.aggregateHeight);
        m_StatsRecorder.Add("Tetris/PredictedMaxHeight", placement.maxHeight);
        m_StatsRecorder.Add("Tetris/PredictedBumpiness", placement.bumpiness);
        m_StatsRecorder.Add("Tetris/PredictedWellDepth", placement.wellDepth);
        m_StatsRecorder.Add("Tetris/TargetColumn", placement.targetColumn);
        m_StatsRecorder.Add("Tetris/TargetRotation", placement.targetRotation);
        
        // Available options analysis
        m_StatsRecorder.Add("Tetris/AvailablePlacements", allPlacements.Count);
        
        if (allPlacements.Count > 0)
        {
            // Find best placement by simple heuristic
            var bestPlacement = allPlacements.OrderByDescending(p => 
                p.linesCleared * 10 - p.holes * 2 - p.aggregateHeight * 0.1f).First();
            
            bool isOptimalChoice = (placement.targetColumn == bestPlacement.targetColumn && 
                                   placement.targetRotation == bestPlacement.targetRotation);
            m_StatsRecorder.Add("Tetris/OptimalChoice", isOptimalChoice ? 1f : 0f);
            
            // Record best available option stats
            m_StatsRecorder.Add("Tetris/BestAvailableLinesCleared", bestPlacement.linesCleared);
            m_StatsRecorder.Add("Tetris/BestAvailableHoles", bestPlacement.holes);
        }
        
        // Calculate and record immediate reward
        float decisionReward = CalculateDecisionReward(placement);
        AddReward(decisionReward);
        m_StatsRecorder.Add("Tetris/DecisionReward", decisionReward);
    }

    private IEnumerator ExecutePlacementCoroutine(PlacementInfo placement)
    {
        float placementStartTime = Time.time;
        m_StatsRecorder.Add("Tetris/PlacementStarted", 1);
        
        // Store initial board state for comparison
        int[,] initialBoard = GetBoardState();
        int initialHeight = CalculateMaxHeight(initialBoard);
        int initialHoles = CalculateHoles(initialBoard);
        
        const float maxTime = 5f; // Safety timeout
        int rotationAttempts = 0;
        int movementAttempts = 0;
        
        // Rotate to target rotation
        while (currentPiece.rotationIndex != placement.targetRotation && 
               Time.time - placementStartTime < maxTime && rotationAttempts < 8)
        {
            rotateRight = true;
            yield return new WaitForFixedUpdate();
            rotateRight = false;
            yield return new WaitForFixedUpdate();
            rotationAttempts++;
        }
        
        m_StatsRecorder.Add("Tetris/RotationAttempts", rotationAttempts);

        // Move to target column
        int targetAbsoluteX = board.Bounds.xMin + placement.targetColumn;
        while (currentPiece.position.x != targetAbsoluteX && 
               Time.time - placementStartTime < maxTime && movementAttempts < 20)
        {
            if (currentPiece.position.x < targetAbsoluteX)
            {
                moveRight = true;
            }
            else
            {
                moveLeft = true;
            }
            
            yield return new WaitForFixedUpdate();
            moveRight = false;
            moveLeft = false;
            yield return new WaitForFixedUpdate();
            movementAttempts++;
        }
        
        m_StatsRecorder.Add("Tetris/MovementAttempts", movementAttempts);

        // Record final position accuracy
        bool correctRotation = (currentPiece.rotationIndex == placement.targetRotation);
        bool correctColumn = (currentPiece.position.x == targetAbsoluteX);
        m_StatsRecorder.Add("Tetris/CorrectRotation", correctRotation ? 1f : 0f);
        m_StatsRecorder.Add("Tetris/CorrectColumn", correctColumn ? 1f : 0f);
        m_StatsRecorder.Add("Tetris/PerfectPlacement", (correctRotation && correctColumn) ? 1f : 0f);

        // Hard drop
        hardDrop = true;
        yield return new WaitForFixedUpdate();
        hardDrop = false;

        // Wait for piece to settle and lines to clear
        yield return new WaitForSeconds(0.2f);
        
        float placementDuration = Time.time - placementStartTime;
        m_StatsRecorder.Add("Tetris/PlacementDuration", placementDuration);
        
        // Record actual results
        RecordCompletionStats(placement, initialHeight, initialHoles);
        
        totalPiecesPlaced++;
        
        // Ready for next decision
        readyForDecision = true;
        
        Debug.Log($"Placement complete in {placementDuration}s - ready for next decision");
    }

    private void RecordCompletionStats(PlacementInfo predictedPlacement, int initialHeight, int initialHoles)
    {
        // Record actual outcome vs prediction
        int[,] newBoardState = GetBoardState();
        int actualHeight = CalculateMaxHeight(newBoardState);
        int actualHoles = CalculateHoles(newBoardState);
        int actualBumpiness = CalculateBumpiness(newBoardState);
        int actualAggregateHeight = CalculateAggregateHeight(newBoardState);
        
        m_StatsRecorder.Add("Tetris/ActualHeight", actualHeight);
        m_StatsRecorder.Add("Tetris/ActualHoles", actualHoles);
        m_StatsRecorder.Add("Tetris/ActualBumpiness", actualBumpiness);
        m_StatsRecorder.Add("Tetris/ActualAggregateHeight", actualAggregateHeight);
        m_StatsRecorder.Add("Tetris/PlacementCompleted", 1);
        
        // Calculate prediction accuracy
        int heightChange = actualHeight - initialHeight;
        int holeChange = actualHoles - initialHoles;
        
        m_StatsRecorder.Add("Tetris/HeightChange", heightChange);
        m_StatsRecorder.Add("Tetris/HoleChange", holeChange);
        
        // Record prediction errors
        float heightError = Mathf.Abs(actualHeight - predictedPlacement.maxHeight);
        float holeError = Mathf.Abs(actualHoles - predictedPlacement.holes);
        
        m_StatsRecorder.Add("Tetris/HeightPredictionError", heightError);
        m_StatsRecorder.Add("Tetris/HolePredictionError", holeError);
        
        // Calculate final reward for this placement
        float finalReward = CalculateFinalPlacementReward(actualHeight, actualHoles, heightChange, holeChange);
        AddReward(finalReward);
        m_StatsRecorder.Add("Tetris/FinalPlacementReward", finalReward);
    }

    private float CalculateDecisionReward(PlacementInfo placement)
    {
        float reward = 0f;
        
        // Immediate reward based on predicted outcome
        reward += placement.linesCleared * rewardWeights.clearReward;
        reward -= placement.holes * rewardWeights.holeCreationPenalty;
        reward -= placement.aggregateHeight * 0.02f;
        reward -= placement.bumpiness * 0.01f;
        
        if (placement.linesCleared == 4)
        {
            reward += rewardWeights.perfectClearBonus;
        }
        
        return reward;
    }
    
    private float CalculateFinalPlacementReward(int actualHeight, int actualHoles, int heightChange, int holeChange)
    {
        float reward = 0f;
        
        // Reward based on actual outcome
        reward -= holeChange * rewardWeights.holeCreationPenalty;
        reward -= heightChange * 0.1f;
        
        // Small step penalty
        reward -= 0.001f;
        
        return reward;
    }

    public void OnGameOver()
    {
        m_StatsRecorder.Add("Tetris/GameOverTriggered", 1);
        m_StatsRecorder.Add("Tetris/FinalPiecesPlaced", totalPiecesPlaced);
        m_StatsRecorder.Add("Tetris/FinalDecisionCount", decisionCount);
        
        float episodeDuration = Time.time - episodeStartTime;
        m_StatsRecorder.Add("Tetris/FinalEpisodeDuration", episodeDuration);
        
        if (episodeDuration > 0)
        {
            m_StatsRecorder.Add("Tetris/FinalDecisionsPerSecond", decisionCount / episodeDuration);
        }
        
        AddReward(rewardWeights.deathPenalty);
        m_StatsRecorder.Add("Tetris/FinalCumulativeReward", GetCumulativeReward());
        
        EndEpisode();
    }

    // Generate unique ID for piece
    private int GeneratePieceId(Piece piece)
    {
        int hash = ((int)piece.data.tetromino * 1000) +
                   (piece.position.x * 100) +
                   (piece.position.y * 10) +
                   (Time.frameCount % 1000);
        return hash;
    }

    public override void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
    {
        if (currentPiece == null || board == null || !readyForDecision)
        {
            // If not ready, mask all actions
            for (int i = 0; i < 40; i++)
            {
                actionMask.SetActionEnabled(0, i, false);
            }
            return;
        }

        // Generate all possible placements for current piece
        int[,] currentBoard = GetBoardState();
        List<PlacementInfo> validPlacements = GenerateAllPossiblePlacements(currentPiece, currentBoard);

        // Create a set of valid action indices for quick lookup
        HashSet<int> validActionIndices = new HashSet<int>();

        foreach (var placement in validPlacements)
        {
            int actionIndex = placement.targetRotation * 10 + placement.targetColumn;
            validActionIndices.Add(actionIndex);
        }

        // Mask actions: enable valid ones, disable invalid ones
        for (int i = 0; i < 40; i++)
        {
            bool isValid = validActionIndices.Contains(i);
            actionMask.SetActionEnabled(0, i, isValid);
        }
        
        m_StatsRecorder.Add("Tetris/ValidActionsCount", validActionIndices.Count);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (!readyForDecision || currentPiece == null)
        {
            // Pad with zeros if not ready
            for (int i = 0; i < 218; i++)
            {
                sensor.AddObservation(0f);
            }
            return;
        }

        // Convert current board state to 2D array for simulation
        int[,] currentBoard = GetBoardState();

        // Generate all possible placements for current piece
        List<PlacementInfo> allPlacements = GenerateAllPossiblePlacements(currentPiece, currentBoard);

        // 1. ALL POSSIBLE PLACEMENTS (34 placements × 6 features = 204 observations)
        for (int i = 0; i < 34; i++)
        {
            if (i < allPlacements.Count)
            {
                PlacementInfo placement = allPlacements[i];
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

        if (currentPiece == null || board == null || !readyForDecision)
        {
            discreteActionsOut[0] = 0;
            return;
        }

        int[,] currentBoard = GetBoardState();
        List<PlacementInfo> allPlacements = GenerateAllPossiblePlacements(currentPiece, currentBoard);

        if (allPlacements.Count == 0)
        {
            discreteActionsOut[0] = 0;
            return;
        }

        int selectedPlacement = 0; // Default to first placement

        // Allow player to cycle through placements with number keys
        if (Input.GetKey(KeyCode.Alpha1) && allPlacements.Count > 0) selectedPlacement = 0;
        else if (Input.GetKey(KeyCode.Alpha2) && allPlacements.Count > 1) selectedPlacement = 1;
        else if (Input.GetKey(KeyCode.Alpha3) && allPlacements.Count > 2) selectedPlacement = 2;
        else if (Input.GetKey(KeyCode.Alpha4) && allPlacements.Count > 3) selectedPlacement = 3;
        else if (Input.GetKey(KeyCode.Alpha5) && allPlacements.Count > 4) selectedPlacement = 4;
        else if (Input.GetKey(KeyCode.Alpha6) && allPlacements.Count > 5) selectedPlacement = 5;
        else if (Input.GetKey(KeyCode.Alpha7) && allPlacements.Count > 6) selectedPlacement = 6;
        else if (Input.GetKey(KeyCode.Alpha8) && allPlacements.Count > 7) selectedPlacement = 7;
        else if (Input.GetKey(KeyCode.Alpha9) && allPlacements.Count > 8) selectedPlacement = 8;
        else if (Input.GetKey(KeyCode.Alpha0) && allPlacements.Count > 9) selectedPlacement = 9;

        selectedPlacement = Mathf.Clamp(selectedPlacement, 0, allPlacements.Count - 1);
        
        // Convert placement back to action index
        var placement = allPlacements[selectedPlacement];
        int actionIndex = placement.targetRotation * 10 + placement.targetColumn;
        discreteActionsOut[0] = actionIndex;
    }

    // ============================================================================
    // BOARD STATE AND PLACEMENT GENERATION METHODS
    // ============================================================================

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

        // Generate placements in a systematic way that matches action encoding
        for (int rotation = 0; rotation < 4; rotation++)
        {
            if (rotation >= rotations.Count) continue;

            int[,] pieceShape = rotations[rotation];
            int pieceWidth = pieceShape.GetLength(1);
            int boardWidth = currentBoard.GetLength(1);

            // Try each column position (0-9 for standard Tetris)
            for (int col = 0; col < 10; col++)
            {
                // Check if piece fits in this column
                if (col + pieceWidth <= boardWidth)
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

    // ============================================================================
    // BOARD ANALYSIS METHODS
    // ============================================================================

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
