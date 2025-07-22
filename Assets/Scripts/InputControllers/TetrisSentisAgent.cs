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
    public TetrisSentisAgent opponentAgent;
    public TaskQueueRunner taskQueueRunner;
    [SerializeField] private BackendType backendType = BackendType.GPUCompute;

    [Header("Powerup Integration")]
    [SerializeField] private bool enablePowerupDecisions = true;
    [SerializeField] private PowerupTetrisAgent powerupAgent;

    [Header("Logging Settings")]
    [SerializeField] private bool enableDetailedLogging = true;

    // Core Sentis components
    private Model runtimeModel;       // For block placement decisions
    private Worker blockWorker;       // Worker for block placement model

    // Tetris game references
    public Board board;
    private Piece currentPiece;

    // Statistics tracking
    private int powerupChecksCount = 0;
    private int powerupActionsCount = 0;
    private int blockPlacementsCount = 0;
    private PowerupTetrisAgent.WildblockActionResult lastPowerupDecision;

    void Awake()
    {
        taskQueueRunner = gameObject.AddComponent<TaskQueueRunner>();
        InitializeSentis();

        // Find PowerupTetrisAgent if not assigned
        if (powerupAgent == null)
        {
            // powerupAgent = FindObjectOfType<PowerupTetrisAgent>();
            powerupAgent = gameObject.AddComponent<PowerupTetrisAgent>();

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

            // Debug.Log($"TetrisSentisAgent: Successfully initialized with {backendType} backend");
            // Debug.Log($"TetrisSentisAgent: Block worker: {(blockWorker != null ? "OK" : "FAILED")}");
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
                // Debug.Log($"TetrisSentisAgent: {modelName} worker created with {backendType} backend");
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
        // powerupWorker = null;
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
    }

    public void SetCurrentPiece(Piece piece)
    {
        currentPiece = piece;

        // Debug.Log($"TetrisSentisAgent: SetCurrentPiece called, IsReadyForInference: {IsReadyForInference()}");

        if (!IsReadyForInference() || board.isLocked) return;
        Debug.Log($"{board.playerTag} new piece...");

        // Start the integrated decision-making process
        MakeIntegratedDecision();
    }

    private bool IsReadyForInference()
    {
        bool ready = board != null && currentPiece != null && blockWorker != null && runtimeModel != null && powerupAgent != null;



        return ready;
    }

    /// <summary>
    /// Integrated decision-making process:
    /// 1. Check if powerups are available and make powerup decision
    /// 2. If powerup decision is "none", proceed with block placement
    /// 3. If powerup is used, wait for next piece/state
    /// </summary>
    private void MakeIntegratedDecision()
    {
        bool powerupsAvailable = HasAvailablePowerups();

        if (enablePowerupDecisions && powerupAgent != null && powerupsAvailable)
        {
            powerupChecksCount++;

            lastPowerupDecision = powerupAgent.GetPowerupDecisionOnly(board, board.powerUpManager.powerUpInventory);

            if (lastPowerupDecision.actionType == 0)
            {
                // No powerup - just do block placement
                taskQueueRunner.EnqueueTask(new InferenceTask(this, board, GetBesMove()));
                return;
            }

            var powerupType = GetPowerupTypeFromActionType(lastPowerupDecision.actionType);

            if (powerupType.Value == PowerUpType.WildCard && opponentAgent != null)
            {
                board.powerUpManager.powerUpInventory[powerupType.Value]--;

                // Enqueue WildCard on opponent's queue
                taskQueueRunner.EnqueueTask(
                 new WildCardTask(this, opponentAgent, lastPowerupDecision.targetColumn));
                taskQueueRunner.EnqueueTask(new InferenceTask(this, board, GetBesMove()));
                return;
            }

            if (powerupType.Value == PowerUpType.Bomb)
            {
                board.powerUpManager.powerUpInventory[powerupType.Value]--;

                // Chain: Powerup first, then inference
                taskQueueRunner.EnqueueTask(
                    new PowerUpTask(powerupType.Value, lastPowerupDecision.targetColumn, board.powerUpManager, board, true)); // false = don't spawn piece
                return;
            }

            // For other powerups (LineBlaster, Gravity)
            taskQueueRunner.EnqueueTask(
                new PowerUpTask(powerupType.Value, lastPowerupDecision.targetColumn, board.powerUpManager, board, false)); // false = don't spawn piece
            taskQueueRunner.EnqueueTask(new InferenceTask(this, board, GetBesMove()));
            return;
        }

        // No powerups available - just do block placement
        taskQueueRunner.EnqueueTask(new InferenceTask(this, board, GetBesMove()));
    }



    /// <summary>
    /// Check if any powerups are available in the powerUpInventory dictionary
    /// </summary>
    private bool HasAvailablePowerups()
    {


        if (board?.powerUpManager?.powerUpInventory == null)
        {
            return false;
        }

        var inventory = board.powerUpManager.powerUpInventory;



        // Use powerup agent's more sophisticated availability check if available
        if (powerupAgent != null)
        {
            // Convert dictionary to PowerUp array for the agent
            // var powerUpsArray = ConvertInventoryToPowerUpArray(inventory);
            bool agentCheck = powerupAgent.HasAvailablePowerupsForDecision(inventory, board);

            return agentCheck;
        }

        // Fallback to simple check - any powerups with count > 0
        bool simpleCheck = inventory.Any(kvp => kvp.Value > 0);
        return simpleCheck;
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

    public string GetBesMove()
    {
        var possibleMoves = GetPossibleMoves();

        if (possibleMoves.Count == 0)
        {
            Debug.LogWarning("TetrisSentisAgent: No valid moves available");
            board.GameOver();
        }

        // Extract features for each move
        var featureList = possibleMoves.Values.ToList();

        if (featureList.Count == 0)
        {
            Debug.LogWarning("TetrisSentisAgent: No features generated");
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
            outputTensor.CompleteAllPendingOperations();
            var cpuTensor = outputTensor.ReadbackAndClone();
            float[] scores = cpuTensor.AsReadOnlyNativeArray().ToArray();

            if (scores.Length == 0)
            {
                inputTensor.Dispose();
                outputTensor.Dispose();
                return "";
            }

            // Find best move
            int bestIndex = GetBestActionIndex(scores);
            float bestScore = scores[bestIndex];

            // Get the corresponding move
            var moveKeys = possibleMoves.Keys.ToList();
            if (bestIndex >= moveKeys.Count)
            {
                inputTensor.Dispose();
                outputTensor.Dispose();
                return "";
            }

            string bestMove = moveKeys[bestIndex];
            var bestFeatures = possibleMoves[bestMove];
            inputTensor.Dispose();
            outputTensor.Dispose();
            return bestMove;

        }
        else
        {
            inputTensor.Dispose();
            outputTensor.Dispose();
            return "";
        }


    }

    /// <summary>
    /// Original block placement inference logic (now uses blockWorker)
    /// </summary>


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

    public IEnumerator ExecuteMove(string move)
    {
        if (currentPiece?.data == null || board == null)
        {
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

        // ✅ FIXED: Store the return value from ClearLines()
        int linesCleared = board.ClearLines(); // This will internally call PowerUpManager.OnLinesCleared()

        // ✅ ENHANCED LOGGING: Show what happened
        string playerTag = board.playerTag ?? "Unknown";



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
    }
}

public class InferenceTask : ITetrisTask
{
    private TetrisSentisAgent agent;
    private Board board;
    public Board GetBoard() => board;
    public string move;

    public InferenceTask(TetrisSentisAgent agent, Board board, string move)
    {
        this.agent = agent;
        this.board = board;
        this.move = move;
    }

    public IEnumerator Execute()
    {

        board.Lock();
        Debug.Log($"{board.playerTag} Piece placement started...");
        yield return agent.ExecuteMove(this.move);
        board.Unlock();
        Debug.Log($"{board.playerTag} Piece placement completed...");
        board.SpawnPiece();

    }

    public string Description => $"InferenceTask on Board {board.playerTag}";
}

public class PowerUpTask : ITetrisTask
{
    private PowerUpType powerUpType;
    private int column;
    private PowerUpManager powerUpManager;
    private Board board;
    private bool shouldSpawnPiece; // ADDED: Control piece spawning

    public Board GetBoard() => board;

    // FIXED: Added shouldSpawnPiece parameter
    public PowerUpTask(PowerUpType powerUpType, int column, PowerUpManager powerUpManager, Board board, bool shouldSpawnPiece = true)
    {
        this.powerUpType = powerUpType;
        this.column = column;
        this.powerUpManager = powerUpManager;
        this.board = board;
        this.shouldSpawnPiece = shouldSpawnPiece;
    }

    public IEnumerator Execute()
    {
        board.Lock();

        Debug.Log($"{board.playerTag} PowerUp {powerUpType} execution started...");

        switch (powerUpType)
        {
            case PowerUpType.LineBlaster:
            case PowerUpType.Gravity:
                powerUpManager.UsePowerUp(powerUpType);
                break;
            case PowerUpType.Bomb:
                yield return powerUpManager.ExecuteBombAtColumn(column);
                break;

        }

        Debug.Log($"{board.playerTag} PowerUp {powerUpType} execution completed...");
        board.Unlock();

        // FIXED: Only spawn piece if explicitly requested (for standalone powerup tasks)
        if (shouldSpawnPiece)
        {
            board.SpawnPiece();
        }
    }

    public string Description => $"PowerUpTask {powerUpType} at column {column} on Board {board.playerTag}";
}


public class WildCardTask : ITetrisTask
{
    private TetrisSentisAgent sourceAgent;
    private TetrisSentisAgent targetAgent;
    private int targetColumn;

    public Board GetBoard() => sourceAgent.board;

    public WildCardTask(TetrisSentisAgent sourceAgent, TetrisSentisAgent targetAgent, int targetColumn)
    {
        this.sourceAgent = sourceAgent;
        this.targetAgent = targetAgent;
        this.targetColumn = targetColumn;
    }

    public IEnumerator Execute()
    {
        Debug.Log($"{sourceAgent.board.playerTag} executing WildCard on {targetAgent.board.playerTag}");

        // Lock both boards to prevent race conditions
        targetAgent.board.Lock();

        try
        {
            // Execute WildCard on target board
            yield return sourceAgent.board.powerUpManager.DropWildcardOnOpponent(
                targetAgent.board, targetColumn, sourceAgent.board.powerUpManager);

            Debug.Log($"WildCard execution completed");
        }
        finally
        {
            // Always unlock both boards
            targetAgent.board.Unlock();
        }

        // Now spawn a new piece for the source agent and continue their turn

        // IMPORTANT: Don't spawn piece for target - they need to handle their modified piece
        // The target agent will handle their next move when their turn comes
    }

    public string Description => $"WildCardTask from {sourceAgent.board.playerTag} to {targetAgent.board.playerTag}";
}