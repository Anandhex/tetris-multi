using UnityEngine;
using System.Collections.Generic;
using System.Linq;
using Unity.Sentis;
using System.Text;
using System.Collections;

/// <summary>
/// Enhanced Tetris AI that integrates both block placement and powerup decisions.
/// First checks for powerup usage, then falls back to block placement if no powerup action is taken.
/// </summary>
public class TetrisSentisAgent : MonoBehaviour, IPlayerInputController
{
    [Header("Sentis Model Asset")]
    [Tooltip("Drag your .sentis ModelAsset here (imported ONNX model)")]

    [Header("Backend Selection")]
    public int allowedTetrominoTypes = 7;

    [SerializeField] private BackendType backendType = BackendType.GPUCompute;

    [Header("Powerup Integration")]
    [SerializeField] private bool enablePowerupDecisions = true;
    [SerializeField] private PowerupTetrisAgent powerupAgent;

    [Header("Logging Settings")]
    [SerializeField] private bool enableDetailedLogging = true;

    // Core Sentis components
    private Model runtimeModel;       // For block placement decisions
    private Model powerupModel;       // For powerup decisions
    private Worker blockWorker;       // Worker for block placement model
    private Worker powerupWorker;     // Worker for powerup model

    // Tetris game references
    private Board board;
    private Piece currentPiece;
    private float lastStateTime = 0f;
    private float stateUpdateInterval = 0.1f;

    // Statistics tracking
    private Dictionary<string, int> actionHistory = new Dictionary<string, int>();
    private int powerupChecksCount = 0;
    private int powerupActionsCount = 0;
    private int blockPlacementsCount = 0;

    void Awake()
    {
        InitializeSentis();
        
        // Find PowerupTetrisAgent if not assigned
        if (powerupAgent == null)
        {
            // powerupAgent = FindObjectOfType<PowerupTetrisAgent>();
            powerupAgent = gameObject.AddComponent<PowerupTetrisAgent>();
            if (powerupAgent != null)
            {
                Debug.Log("TetrisSentisAgent: Found PowerupTetrisAgent automatically");
            }
            else
            {
                Debug.LogWarning("TetrisSentisAgent: No PowerupTetrisAgent found in scene!");
            }
        }
    }

    void OnDestroy()
    {
        CleanupSentis();
    }

    public void InitializeSentis()
    {
        try
        {
            // Load both models
            runtimeModel = ModelLoader.Load(BoardManager.Instance.sentisModelAsset);
            // powerupModel = ModelLoader.Load(BoardManager.Instance.powerupAsset);

            // Create workers for both models
            blockWorker = CreateWorkerWithFallback(runtimeModel, "Block Placement");
            // powerupWorker = CreateWorkerWithFallback(powerupModel, "Powerup");

            if (blockWorker == null)
            {
                Debug.LogError("TetrisSentisAgent: Failed to create block placement worker!");
                return;
            }

            // if (powerupWorker == null)
            // {
            //     Debug.LogWarning("TetrisSentisAgent: Failed to create powerup worker - powerup decisions will be disabled");
            //     enablePowerupDecisions = false;
            // }

            Debug.Log($"TetrisSentisAgent: Successfully initialized with {backendType} backend");
            Debug.Log($"TetrisSentisAgent: Block worker: {(blockWorker != null ? "OK" : "FAILED")}");
            // Debug.Log($"TetrisSentisAgent: Powerup worker: {(powerupWorker != null ? "OK" : "FAILED")}");
            
            // Initialize PowerupTetrisAgent with the powerup worker
            // if (enablePowerupDecisions && powerupAgent != null && powerupWorker != null)
            // {
            //     powerupAgent.SetExternalWorker(powerupWorker, powerupModel);
            //     Debug.Log("TetrisSentisAgent: Powerup integration enabled and configured");
            // }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"TetrisSentisAgent: Failed to initialize Sentis: {e.Message}");
        }
    }

    private Worker CreateWorkerWithFallback(Model model, string modelName)
    {
        // Try the selected backend first
        try
        {
            var worker = new Worker(model, backendType);
            if (worker != null)
            {
                Debug.Log($"TetrisSentisAgent: {modelName} worker created with {backendType} backend");
                return worker;
            }
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"TetrisSentisAgent: Failed to create {modelName} worker with {backendType}: {e.Message}");
        }

        // Fallback sequence
        BackendType[] fallbackOrder = { BackendType.GPUCompute, BackendType.CPU };

        foreach (var backend in fallbackOrder)
        {
            if (backend == backendType) continue; // Already tried

            try
            {
                var worker = new Worker(model, backend);
                if (worker != null)
                {
                    Debug.LogWarning($"TetrisSentisAgent: {modelName} worker fell back to {backend} backend");
                    return worker;
                }
            }
            catch (System.Exception e)
            {
                Debug.LogWarning($"TetrisSentisAgent: {modelName} worker {backend} backend also failed: {e.Message}");
            }
        }

        return null;
    }

    public void CleanupSentis()
    {
        blockWorker?.Dispose();
        // powerupWorker?.Dispose();
        blockWorker = null;
        powerupWorker = null;
        runtimeModel = null;
        // powerupModel = null;
    }

    public void SetBoard(Board gameBoard)
    {
        board = gameBoard;
        if (board != null)
        {
            board.inputController = this;
        }
        lastStateTime = Time.time;
    }

    public void SetCurrentPiece(Piece piece)
    {
        currentPiece = piece;
        
        Debug.Log($"TetrisSentisAgent: SetCurrentPiece called, IsReadyForInference: {IsReadyForInference()}");
        
        if (!IsReadyForInference()) return;
        
        // Start the integrated decision-making process
        StartCoroutine(MakeIntegratedDecision());
    }

    private bool IsReadyForInference()
    {
        bool ready = board != null && currentPiece != null && blockWorker != null && runtimeModel != null;
        
        if (enableDetailedLogging)
        {
            Debug.Log($"TetrisSentisAgent: IsReadyForInference - Board: {board != null}, " +
                     $"Piece: {currentPiece != null}, Worker: {blockWorker != null}, Model: {runtimeModel != null}");
        }
        
        return ready;
    }

    /// <summary>
    /// Integrated decision-making process:
    /// 1. Check if powerups are available and make powerup decision
    /// 2. If powerup decision is "none", proceed with block placement
    /// 3. If powerup is used, wait for next piece/state
    /// </summary>
    private IEnumerator MakeIntegratedDecision()
    {
        if (enableDetailedLogging)
        {
            Debug.Log($"TetrisSentisAgent: Starting integrated decision process");
            Debug.Log($"TetrisSentisAgent: enablePowerupDecisions: {enablePowerupDecisions}");
            Debug.Log($"TetrisSentisAgent: powerupAgent: {powerupAgent != null}");
            Debug.Log($"TetrisSentisAgent: board.powerUpManager: {board?.powerUpManager != null}");
        }

        // Step 1: Check for powerup availability and make powerup decision
        bool powerupActionTaken = false;
        
        // Enhanced debugging for powerup availability check
        bool powerupsAvailable = HasAvailablePowerups();
        
        if (enablePowerupDecisions && powerupAgent != null && powerupsAvailable)
        {
            powerupChecksCount++;
            
            if (enableDetailedLogging)
            {
                Debug.Log("TetrisSentisAgent: Powerups available, consulting powerup agent");
            }

            // Get powerup decision
            var powerupDecision = GetPowerupDecision();
            
            if (powerupDecision.actionType != 0) // Not "none"
            {
                powerupActionTaken = true;
                powerupActionsCount++;
                
                if (enableDetailedLogging)
                {
                    Debug.Log($"TetrisSentisAgent: Powerup action chosen: {powerupDecision.actionName}");
                }

                // Wait a frame for powerup to execute
                yield return null;
                
                // After powerup execution, check if we should continue with block placement
                // Some powerups (like gravity) might change the board state significantly
                if (ShouldContinueAfterPowerup(powerupDecision.actionType))
                {
                    // Re-evaluate the situation after powerup
                    yield return new WaitForSeconds(0.1f);
                    
                    // Recursive call to re-evaluate (could lead to another powerup or block placement)
                    if (currentPiece != null && IsReadyForInference())
                    {
                        StartCoroutine(MakeIntegratedDecision());
                    }
                    yield break;
                }
            }
            else
            {
                if (enableDetailedLogging)
                {
                    Debug.Log("TetrisSentisAgent: Powerup agent decided 'none', proceeding with block placement");
                }
            }
        }
        else
        {
            if (enableDetailedLogging)
            {
                Debug.Log($"TetrisSentisAgent: Skipping powerup check - " +
                         $"Enabled: {enablePowerupDecisions}, Agent: {powerupAgent != null}, Available: {powerupsAvailable}");
            }
        }

        // Step 2: If no powerup action was taken, proceed with block placement
        if (!powerupActionTaken)
        {
            if (enableDetailedLogging)
            {
                Debug.Log("TetrisSentisAgent: Proceeding with block placement decision");
            }
            
            blockPlacementsCount++;
            RunBlockPlacementInference();
        }
    }

    /// <summary>
    /// Check if any powerups are available in the powerUpInventory dictionary
    /// </summary>
    private bool HasAvailablePowerups()
    {
        Debug.Log($"TetrisSentisAgent: HasAvailablePowerups check start");
        Debug.Log($"TetrisSentisAgent: board: {board != null}");
        Debug.Log($"TetrisSentisAgent: powerUpManager: {board?.powerUpManager != null}");
        Debug.Log($"TetrisSentisAgent: powerUpInventory: {board?.powerUpManager?.powerUpInventory != null}");
        
        if (board?.powerUpManager?.powerUpInventory == null)
        {
            Debug.Log("TetrisSentisAgent: No powerup manager or powerUpInventory is null");
            return false;
        }

        var inventory = board.powerUpManager.powerUpInventory;
        Debug.Log($"TetrisSentisAgent: powerUpInventory contents:");
        
        // Log what powerups are available
        foreach (var kvp in inventory)
        {
            Debug.Log($"TetrisSentisAgent: {kvp.Key}: {kvp.Value}");
        }

        // Use powerup agent's more sophisticated availability check if available
        if (powerupAgent != null)
        {
            // Convert dictionary to PowerUp array for the agent
            // var powerUpsArray = ConvertInventoryToPowerUpArray(inventory);
            bool agentCheck = powerupAgent.HasAvailablePowerupsForDecision(inventory, board);
            Debug.Log($"TetrisSentisAgent: PowerupAgent availability check result: {agentCheck}");
            return agentCheck;
        }
        
        // Fallback to simple check - any powerups with count > 0
        bool simpleCheck = inventory.Any(kvp => kvp.Value > 0);
        Debug.Log($"TetrisSentisAgent: Simple availability check result: {simpleCheck}");
        return simpleCheck;
    }

    /// <summary>
    /// Convert PowerUpManager's inventory dictionary to PowerUp array for PowerupTetrisAgent
    /// </summary>
    private PowerUp[] ConvertInventoryToPowerUpArray(Dictionary<PowerUpType, int> inventory)
    {
        var powerUpsList = new List<PowerUp>();
        
        foreach (var kvp in inventory)
        {
            // Add multiple instances based on count
            for (int i = 0; i < kvp.Value; i++)
            {
                // Get the PowerUp data from availablePowerUps array
                var powerUpData = board.powerUpManager.availablePowerUps
                    .FirstOrDefault(p => p.type == kvp.Key);
                
                if (powerUpData != null)
                {
                    powerUpsList.Add(powerUpData);
                }
            }
        }
        
        Debug.Log($"TetrisSentisAgent: Converted inventory to {powerUpsList.Count} PowerUp instances");
        return powerUpsList.ToArray();
    }

    /// <summary>
    /// Get powerup decision from the PowerupTetrisAgent
    /// </summary>
    private PowerupTetrisAgent.WildblockActionResult GetPowerupDecision()
    {
        Debug.Log($"TetrisSentisAgent: GetPowerupDecision called!");
        Debug.Log($"TetrisSentisAgent: powerupAgent: {powerupAgent != null}");
        Debug.Log($"TetrisSentisAgent: board.powerUpManager: {board?.powerUpManager != null}");
        
        if (powerupAgent == null || board?.powerUpManager == null)
        {
            Debug.LogWarning("TetrisSentisAgent: GetPowerupDecision - missing powerupAgent or powerUpManager");
            return new PowerupTetrisAgent.WildblockActionResult
            {
                actionType = 0, // none
                actionName = "none",
                confidence = 1.0f
            };
        }

        try
        {

            Debug.Log("TetrisSentisAgent: Calling powerupAgent.GetPowerupDecisionOnly with inventory...");     

            // Pass the inventory dictionary directly to the PowerupAgent
            var decision = powerupAgent.GetPowerupDecisionOnly(board, board.powerUpManager.powerUpInventory);

            Debug.Log($"TetrisSentisAgent: Powerup decision received - Action: {decision.actionName}, Confidence: {decision.confidence:F2}");

            // If decision is not "none", execute the powerup action and reduce inventory
            if (decision.actionType != 0)
            {
                Debug.Log($"TetrisSentisAgent: Executing powerup action: {decision.actionName}");

                // First, reduce the powerup count in the inventory using PowerUpManager
                var powerupType = GetPowerupTypeFromActionType(decision.actionType);
                if (powerupType.HasValue)
                {
                    Debug.Log($"TetrisSentisAgent: Reducing {powerupType.Value} count in inventory");
                    // board.powerUpManager.UsePowerUp(powerupType.Value);
                    ExecutePowerupWithParameters(powerupType.Value, decision);
                }
                else
                {
                    Debug.LogWarning($"TetrisSentisAgent: Could not determine powerup type for action {decision.actionType}");
                }
            }
            else
            {
                Debug.Log("TetrisSentisAgent: Ai Decided none returning none");
            }
            
            return decision;
        }
        catch (System.Exception e)
        {
            Debug.LogError($"TetrisSentisAgent: Error getting powerup decision: {e.Message}");
            Debug.LogError($"TetrisSentisAgent: Stack trace: {e.StackTrace}");
            return new PowerupTetrisAgent.WildblockActionResult
            {
                actionType = 0,
                actionName = "none", 
                confidence = 1.0f
            };
        }
    }

    private void ExecutePowerupWithParameters(PowerUpType powerupType, PowerupTetrisAgent.WildblockActionResult decision)
    {
        switch (powerupType)
        {
            case PowerUpType.LineBlaster:
                // LineBlaster doesn't need additional parameters
                Debug.Log("TetrisSentisAgent: Executing LineBlaster (no additional parameters)");
                board.powerUpManager.UsePowerUp(PowerUpType.LineBlaster);
                break;
                
            case PowerUpType.Gravity:
                // Gravity doesn't need additional parameters
                Debug.Log("TetrisSentisAgent: Executing Gravity (no additional parameters)");
                board.powerUpManager.UsePowerUp(PowerUpType.Gravity);
                break;
                
            case PowerUpType.Bomb:
                // Bomb needs target column information
                if (decision.targetColumn != -1)
                {
                    Debug.Log($"TetrisSentisAgent: Executing Bomb at column {decision.targetColumn}, row {decision.targetRow}");
                    // You'll need to modify PowerUpManager to accept column parameter
                    board.powerUpManager.UsePowerUp(PowerUpType.Bomb, decision.targetColumn, decision.targetRow);
                }
                else
                {
                    Debug.LogWarning("TetrisSentisAgent: Bomb target column not specified, using default execution");
                    board.powerUpManager.UsePowerUp(PowerUpType.Bomb);
                }
                break;
                
            case PowerUpType.WildCard:
                // WildCard needs target column for opponent board
                if (decision.targetColumn != -1)
                {
                    Debug.Log($"TetrisSentisAgent: Executing WildCard at opponent column {decision.targetColumn}, row {decision.targetRow}");
                    // You'll need to modify PowerUpManager to accept column parameter for WildCard
                    board.powerUpManager.UsePowerUp(PowerUpType.WildCard, decision.targetColumn, decision.targetRow);
                }
                else
                {
                    Debug.LogWarning("TetrisSentisAgent: WildCard target column not specified, using default execution");
                    board.powerUpManager.UsePowerUp(PowerUpType.WildCard);
                }
                break;
                
            default:
                Debug.LogWarning($"TetrisSentisAgent: Unknown powerup type {powerupType}");
                break;
        }
    }

    /// <summary>
    /// Convert PowerupTetrisAgent action type to PowerUpType enum
    /// </summary>
    private PowerUpType? GetPowerupTypeFromActionType(int actionType)
    {
        return actionType switch
        {
            1 => PowerUpType.LineBlaster,   // bottom_clear
            2 => PowerUpType.Gravity,       // gravity
            3 => PowerUpType.Bomb,          // bomb
            4 => PowerUpType.WildCard,      // wildblock
            _ => null                       // none or invalid
        };
    }

    /// <summary>
    /// Determine if we should continue evaluating after a powerup action
    /// This handles different powerup types and their effects on the game state
    /// </summary>
    private bool ShouldContinueAfterPowerup(int powerupType)
    {
        switch (powerupType)
        {
            case 1: // LineBlaster (bottom_clear)
                // Board state changed significantly - bottom line was cleared
                // AI should re-evaluate the new board state to see if more powerups are beneficial
                Debug.Log("TetrisSentisAgent: LineBlaster used - re-evaluating due to board state change");
                return true;
                
            case 2: // Gravity
                // Board state changed dramatically - all floating blocks dropped
                // This can create new line clear opportunities or change strategic positioning
                Debug.Log("TetrisSentisAgent: Gravity used - re-evaluating due to major board restructuring");
                return true;
                
            case 3: // Bomb
                // Bomb affects own board (clears 3x3 area around current piece)
                // Board state changed, but less dramatically than LineBlaster/Gravity
                // Re-evaluate to see if the cleared area creates new opportunities
                Debug.Log("TetrisSentisAgent: Bomb used - re-evaluating due to 3x3 area clearance");
                return true;
                
            case 4: // WildCard/Wildblock
                // Affects opponent board, not own board
                // Own board state unchanged, continue with current piece placement
                Debug.Log("TetrisSentisAgent: WildCard used - continuing with block placement (opponent affected)");
                return true;
                
            default:
                // Unknown powerup type or 'none' - no re-evaluation needed
                Debug.Log($"TetrisSentisAgent: Unknown powerup type {powerupType} - no re-evaluation");
                return false;
        }
    }

    /// <summary>
    /// Original block placement inference logic (now uses blockWorker)
    /// </summary>
    private void RunBlockPlacementInference()
    {
        try
        {
            // Get all possible moves (same as Python trainer)
            var possibleMoves = GetPossibleMoves();

            if (possibleMoves.Count == 0)
            {
                Debug.LogWarning("TetrisSentisAgent: No valid moves available");
                board.GameOver();
                return;
            }

            // Extract features for each move
            var featureList = possibleMoves.Values.ToList();

            if (featureList.Count == 0)
            {
                Debug.LogWarning("TetrisSentisAgent: No features generated");
                return;
            }

            // Create input tensor for the block placement model
            var inputShape = new TensorShape(featureList.Count, 4);
            var features = featureList.SelectMany(f => f).ToArray();
            Tensor<float> inputTensor = new Tensor<float>(inputShape, features);

            // Run inference using the block placement worker
            blockWorker.Schedule(inputTensor);
            var outputTensor = blockWorker.PeekOutput() as Tensor<float>;

            if (outputTensor != null)
            {
                ProcessInferenceOutput(outputTensor, possibleMoves);
            }
            else
            {
                Debug.LogWarning("TetrisSentisAgent: No output tensor received");
            }
            
            inputTensor.Dispose();
            outputTensor.Dispose();
        }
        catch (System.Exception e)
        {
            Debug.LogError($"TetrisSentisAgent: Block placement inference failed: {e.Message}");
        }
    }

    // Rest of the methods remain the same...
    private Dictionary<string, float[]> GetPossibleMoves()
    {
        var moves = new Dictionary<string, float[]>();

        if (board == null || currentPiece == null) return moves;

        try
        {
            var boardData = new BoardData(board);
            var pieceState = new PieceState(currentPiece);

            int w = board.boardSize.x;
            int rotCount = pieceState.data.RotationCount;

            for (int rot = 0; rot < rotCount; rot++)
            {
                for (int col = 0; col < w; col++)
                {
                    var simBoard = boardData.Clone();
                    var simPiece = pieceState.Clone();

                    // Apply rotation
                    int steps = (rot - simPiece.rotationIndex + rotCount) % rotCount;
                    for (int i = 0; i < steps; i++)
                    {
                        simPiece.RotateCW(simBoard);
                    }

                    // Move horizontally
                    if (simPiece.Cells != null && simPiece.Cells.Length > 0)
                    {
                        int minX = simPiece.Cells.Min(c => c.x);
                        var target = new Vector2Int(col + simBoard.xOffset - minX, simPiece.position.y);

                        if (!simBoard.IsValidPosition(simPiece, target)) continue;

                        simPiece.position = target;

                        // Hard drop
                        while (simBoard.IsValidPosition(simPiece, simPiece.position + Vector2Int.down))
                        {
                            simPiece.position += Vector2Int.down;
                        }

                        // Place & clear - get metrics
                        int lines = simBoard.PlaceAndClear(simPiece);
                        float holes = simBoard.CountHoles();
                        float bumpiness = simBoard.GetBumpinessScore();
                        float height = simBoard.CalculateStackHeight();

                        // Store move in same format as Python trainer
                        string moveKey = $"{col}:{rot}";
                        moves[moveKey] = new float[] { lines, holes, bumpiness, height };
                    }
                }
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"TetrisSentisAgent: Error in move simulation: {e.Message}");
        }

        return moves;
    }

    private void ProcessInferenceOutput(Tensor<float> outputTensor, Dictionary<string, float[]> possibleMoves)
    {
        try
        {
            // Download tensor data to CPU
            outputTensor.CompleteAllPendingOperations();
            var cpuTensor = outputTensor.ReadbackAndClone();
            float[] scores = cpuTensor.AsReadOnlyNativeArray().ToArray();

            if (scores.Length == 0)
            {
                Debug.LogWarning("TetrisSentisAgent: Empty output scores");
                return;
            }

            // Find best move
            int bestIndex = GetBestActionIndex(scores);
            float bestScore = scores[bestIndex];

            // Get the corresponding move
            var moveKeys = possibleMoves.Keys.ToList();
            if (bestIndex >= moveKeys.Count)
            {
                Debug.LogWarning($"TetrisSentisAgent: Best index {bestIndex} out of range for {moveKeys.Count} moves");
                return;
            }

            string bestMove = moveKeys[bestIndex];
            var bestFeatures = possibleMoves[bestMove];

            // Execute the move
            StartCoroutine(ExecuteMove(bestMove));

        }
        catch (System.Exception e)
        {
            Debug.LogError($"TetrisSentisAgent: Failed to process output: {e.Message}");
        }
    }

    private int GetBestActionIndex(float[] scores)
    {
        int bestIndex = 0;
        float bestScore = scores[0];

        for (int i = 1; i < scores.Length; i++)
        {
            if (scores[i] > bestScore)
            {
                bestScore = scores[i];
                bestIndex = i;
            }
        }

        return bestIndex;
    }

    private IEnumerator ExecuteMove(string move)
    {
        if (currentPiece?.data == null || board == null)
        {
            Debug.LogWarning("TetrisSentisAgent: Cannot execute move - invalid piece or board");
            yield break;
        }

        var parts = move.Split(':');
        int targetCol = int.Parse(parts[0]);
        int targetRot = int.Parse(parts[1]);

        board.Clear(currentPiece);

        int currentRotation = currentPiece.rotationIndex;
        int rotCount = currentPiece.data.RotationCount;
        int steps = (targetRot - currentRotation + rotCount) % rotCount;

        for (int i = 0; i < steps; i++)
        {
            currentPiece.Rotate(-1);
            board.Set(currentPiece);
            yield return new WaitForSeconds(0.05f);
            board.Clear(currentPiece);
        }

        if (currentPiece.cells != null && currentPiece.cells.Length > 0)
        {
            int minX = currentPiece.cells.Min(c => c.x);
            var newPosition = new Vector3Int(
                targetCol + board.Bounds.xMin - minX,
                currentPiece.position.y,
                currentPiece.position.z
            );

            currentPiece.position = newPosition;
            board.Set(currentPiece);
            yield return new WaitForSeconds(0.05f);
            board.Clear(currentPiece);

            while (board.IsValidPosition2(currentPiece, currentPiece.position + Vector3Int.down))
            {
                currentPiece.position += Vector3Int.down;
                board.Set(currentPiece);
                yield return new WaitForSeconds(0.02f);
                board.Clear(currentPiece);
            }
        }

        board.Set(currentPiece);
        board.ClearLines();
        board.SpawnPiece();
    }

    // IPlayerInputController implementation (unused in AI mode)
    public bool GetLeft() => false;
    public bool GetRight() => false;
    public bool GetDown() => false;
    public bool GetRotateLeft() => false;
    public bool GetRotateRight() => false;
    public bool GetHardDrop() => false;

    [ContextMenu("Log Integration Statistics")]
    public void LogIntegrationStatistics()
    {
        StringBuilder stats = new StringBuilder("TetrisSentisAgent Integration Statistics:\n");
        
        stats.AppendLine($"Powerup checks: {powerupChecksCount}");
        stats.AppendLine($"Powerup actions taken: {powerupActionsCount}");
        stats.AppendLine($"Block placements: {blockPlacementsCount}");
        
        if (powerupChecksCount > 0)
        {
            float powerupUsageRate = (float)powerupActionsCount / powerupChecksCount * 100f;
            stats.AppendLine($"Powerup usage rate: {powerupUsageRate:F1}%");
        }
        
        int totalDecisions = powerupActionsCount + blockPlacementsCount;
        if (totalDecisions > 0)
        {
            float powerupVsBlockRatio = (float)powerupActionsCount / totalDecisions * 100f;
            stats.AppendLine($"Powerup vs Block placement ratio: {powerupVsBlockRatio:F1}% vs {100 - powerupVsBlockRatio:F1}%");
        }

        Debug.Log(stats.ToString());
    }

    [ContextMenu("Test Powerup Availability")]
    public void TestPowerupAvailability()
    {
        Debug.Log("=== TESTING POWERUP AVAILABILITY ===");
        Debug.Log($"enablePowerupDecisions: {enablePowerupDecisions}");
        Debug.Log($"powerupAgent: {powerupAgent != null}");
        Debug.Log($"board: {board != null}");
        Debug.Log($"powerUpManager: {board?.powerUpManager != null}");
        Debug.Log($"availablePowerUps: {board?.powerUpManager?.availablePowerUps != null}");
        
        if (board?.powerUpManager?.availablePowerUps != null)
        {
            var powerUps = board.powerUpManager.availablePowerUps;
            Debug.Log($"PowerUp array length: {powerUps.Length}");
            for (int i = 0; i < powerUps.Length; i++)
            {
                Debug.Log($"PowerUp[{i}]: {(powerUps[i] != null ? powerUps[i].type.ToString() : "NULL")}");
            }
        }
        
        bool hasAvailable = HasAvailablePowerups();
        Debug.Log($"HasAvailablePowerups result: {hasAvailable}");
    }
}