using UnityEngine;
using System.Collections.Generic;
using System.Linq;
using Unity.Sentis;
using System.Text;
using System.Collections;
using System;

/// <summary>
/// Enhanced ONNX-backed Tetris AI using Unity Sentis for inference with Wildblock support.
/// Updated for CNN DQN wildblock model with dual-board input.
/// Input: (1, 8, 20, 10) - dual boards + powerup context
/// Output: (1, 23) - 5 actions + 10 bomb columns + 8 wildblock columns
/// </summary>
public class PowerupTetrisAgent : MonoBehaviour
{
    [Header("Sentis Model Asset")]
    [SerializeField] private BackendType backendType = BackendType.GPUCompute;

    [Header("Logging Settings")]
    [SerializeField] private bool enableDetailedLogging = true;
    [SerializeField] private bool enableWildblockLogging = true;

    // Core Sentis components
    private Model runtimeModel;
    private Worker worker;
    private bool usingExternalWorker = false;

    // Statistics tracking
    private Dictionary<string, int> actionHistory = new Dictionary<string, int>()
    {
        {"none", 0}, {"bottom_clear", 0}, {"gravity", 0}, {"bomb", 0}, {"wildblock", 0}
    };
    
    private Dictionary<int, int> bombColumnHistory = new Dictionary<int, int>();
    private Dictionary<int, int> wildblockColumnHistory = new Dictionary<int, int>();

    void Awake()
    {
        InitializeSentis();
        InitializeColumnHistories();
    }

    void OnDestroy()
    {
        CleanupSentisIfOwned();
    }

    private void InitializeColumnHistories()
    {
        for (int i = 0; i < 10; i++) bombColumnHistory[i] = 0;
        for (int i = 1; i <= 8; i++) wildblockColumnHistory[i] = 0;
    }

    public void SetExternalWorker(Worker externalWorker, Model externalModel)
    {
        if (externalWorker != null && externalModel != null)
        {
            if (worker != null && !usingExternalWorker)
            {
                worker.Dispose();
            }
            
            worker = externalWorker;
            runtimeModel = externalModel;
            usingExternalWorker = true;
            
            Debug.Log("PowerupTetrisAgent: External worker and model set successfully");
        }
        else
        {
            Debug.LogError("PowerupTetrisAgent: Invalid external worker or model provided");
        }
    }

    public void InitializeSentis()
    {
        try
        {
            runtimeModel = ModelLoader.Load(BoardManager.Instance.powerupAsset);
            worker = CreateWorkerWithFallback();
            usingExternalWorker = false;

            if (worker == null)
            {
                Debug.LogError("PowerupAgent: Failed to create worker with any backend!");
                return;
            }
            

            Debug.Log($"PowerupAgent: Successfully initialized with {backendType} backend");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"PowerupAgent: Failed to initialize Sentis: {e}");
        }
    }

    private Worker CreateWorkerWithFallback()
    {
        try
        {
            var worker = new Worker(runtimeModel, BackendType.GPUCompute);
            if (worker != null) return worker;
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"PowerupAgent: Failed to create {backendType} worker: {e.Message}");
        }

        BackendType[] fallbackOrder = { BackendType.GPUCompute, BackendType.CPU };

        foreach (var backend in fallbackOrder)
        {
            if (backend == backendType) continue;

            try
            {
                var worker = new Worker(runtimeModel, backend);
                if (worker != null)
                {
                    Debug.LogWarning($"PowerupAgent: Fell back to {backend} backend");
                    backendType = backend;
                    return worker;
                }
            }
            catch (System.Exception e)
            {
                Debug.LogWarning($"PowerupAgent: {backend} backend also failed: {e.Message}");
            }
        }

        return null;
    }

    public void CleanupSentisIfOwned()
    {
        if (worker != null && !usingExternalWorker)
        {
            worker.Dispose();
            Debug.Log("PowerupTetrisAgent: Disposed own worker");
        }
        
        worker = null;
        runtimeModel = null;
        usingExternalWorker = false;
    }

    public void CleanupSentis()
    {
        CleanupSentisIfOwned();
    }

    private bool IsReadyForInference()
    {
        Debug.Log($"PowerupAgent IsReadyForInference: {worker} {runtimeModel}");
        return worker != null && runtimeModel != null;
    }

    /// <summary>
    /// Main method called by TetrisSentisAgent - runs full CNN prediction pipeline
    /// </summary>
    public WildblockActionResult GetPowerupDecisionOnly(Board board, PowerUp[] availablePowerUps)
    {
            Debug.Log("powerupAgent getPOwerupDecisionOnly");
        if (!IsReadyForInference())
        {
            return new WildblockActionResult { actionType = 0, actionName = "none", confidence = 1.0f };
        }
        
        try
        {
            // Step 1: Prepare 8-channel input
            float[] inputArray = GetDualBoardStateForCNN(board, availablePowerUps);

            if (inputArray.Length != 1600)
            {
                Debug.LogError($"PowerupAgent: Invalid input shape, expected 1600, got {inputArray.Length}");
                return new WildblockActionResult { actionType = 0, actionName = "none", confidence = 1.0f };
            }

            // Step 2: Run CNN inference
            var inputShape = new TensorShape(1, 8, 20, 10);
            Tensor<float> inputTensor = new Tensor<float>(inputShape, inputArray);
            
            worker.Schedule(inputTensor);
            var outputTensor = worker.PeekOutput() as Tensor<float>;

            // Step 3: Get raw output
            float[] output = GetOutputArray(outputTensor);
            
            if (output.Length != 23)
            {
                Debug.LogError($"PowerupAgent: Invalid output shape, expected 23, got {output.Length}");
                inputTensor.Dispose();
                outputTensor.Dispose();
                return new WildblockActionResult { actionType = 0, actionName = "none", confidence = 1.0f };
            }

            // Step 4: Process output to make decision
            var actionResult = ProcessWildblockCNNOutput(output, board, availablePowerUps);
            
            if (enableDetailedLogging)
            {
                Debug.Log($"PowerupAgent: CNN predicted '{actionResult.actionName}' with confidence {actionResult.confidence:F2}");
            }

            // Cleanup
            inputTensor.Dispose();
            outputTensor.Dispose();
            
            return actionResult;
        }
        catch (System.Exception e)
        {
            Debug.LogError($"PowerupAgent: CNN prediction failed: {e}");
            return new WildblockActionResult { actionType = 0, actionName = "none", confidence = 1.0f };
        }
    }

    public void ExecutePowerupAction(WildblockActionResult actionResult, PowerUpManager powerUpManager, Board board)
    {
        ApplyWildblockAction(actionResult, powerUpManager, board);
    }

    public bool HasAvailablePowerupsForDecision(PowerUp[] availablePowerUps, Board board)
    {
        if (availablePowerUps == null || availablePowerUps.Length == 0) return false;

        bool hasBottomClear = HasPowerupType(availablePowerUps, PowerUpType.LineBlaster);
        bool hasGravity = HasPowerupType(availablePowerUps, PowerUpType.Gravity);
        bool hasBomb = HasPowerupType(availablePowerUps, PowerUpType.Bomb) && FindValidBombColumns(board).Length > 0;
        bool hasWildblock = HasPowerupType(availablePowerUps, PowerUpType.WildCard) && 
                           (board.opponentBoard != null && FindValidWildblockPositions(board.opponentBoard).Count > 0);

        return hasBottomClear || hasGravity || hasBomb || hasWildblock;
    }

    /// <summary>
    /// Check if a specific powerup type exists in the array
    /// </summary>
    private bool HasPowerupType(PowerUp[] availablePowerUps, PowerUpType powerupType)
    {
        if (availablePowerUps == null) return false;
        
        foreach (var powerUp in availablePowerUps)
        {
            if (powerUp != null && powerUp.type == powerupType)
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>
    /// Prepare 8-channel input for dual-board CNN DQN model
    /// </summary>
    private float[] GetDualBoardStateForCNN(Board board, PowerUp[] availablePowerUps)
    {
        float[] inputData = new float[1600]; // 8 * 20 * 10
        int index = 0;

        var bounds = board.Bounds;
        var opponentBoard = board.opponentBoard;
        
        // Get powerup availability directly from array
        bool hasBottomClear = HasPowerupType(availablePowerUps, PowerUpType.LineBlaster);
        bool hasGravity = HasPowerupType(availablePowerUps, PowerUpType.Gravity);
        bool hasBomb = HasPowerupType(availablePowerUps, PowerUpType.Bomb);
        bool hasWildblock = HasPowerupType(availablePowerUps, PowerUpType.WildCard);

        // Channel 0: Self board state (20x10 grid)
        for (int y = 0; y < 20; y++)
        {
            for (int x = 0; x < 10; x++)
            {
                if (y < bounds.height && x < bounds.width)
                {
                    Vector3Int pos = new Vector3Int(bounds.xMin + x, bounds.yMin + y, 0);
                    inputData[index] = board.tilemap.HasTile(pos) ? 1.0f : 0.0f;
                }
                else
                {
                    inputData[index] = 0.0f;
                }
                index++;
            }
        }

        // Channel 1: Opponent board state (20x10 grid)
        for (int y = 0; y < 20; y++)
        {
            for (int x = 0; x < 10; x++)
            {
                if (opponentBoard != null && y < opponentBoard.Bounds.height && x < opponentBoard.Bounds.width)
                {
                    Vector3Int pos = new Vector3Int(opponentBoard.Bounds.xMin + x, opponentBoard.Bounds.yMin + y, 0);
                    inputData[index] = opponentBoard.tilemap.HasTile(pos) ? 1.0f : 0.0f;
                }
                else
                {
                    inputData[index] = 0.0f;
                }
                index++;
            }
        }

        // Channels 2-5: Powerup availability (broadcast to 200 cells each)
        float[] powerupValues = { 
            hasBottomClear ? 1.0f : 0.0f,
            hasGravity ? 1.0f : 0.0f,
            hasBomb ? 1.0f : 0.0f,
            hasWildblock ? 1.0f : 0.0f
        };

        for (int channel = 0; channel < 4; channel++)
        {
            for (int i = 0; i < 200; i++)
            {
                inputData[index++] = powerupValues[channel];
            }
        }

        // Channel 6: Height advantage
        var heightAdvantage = CalculateHeightAdvantage(board, opponentBoard);
        for (int i = 0; i < 200; i++)
        {
            inputData[index++] = heightAdvantage;
        }

        // Channel 7: Threat level
        var threatLevel = CalculateThreatLevel(opponentBoard);
        for (int i = 0; i < 200; i++)
        {
            inputData[index++] = threatLevel;
        }

        return inputData;
    }

    private float CalculateHeightAdvantage(Board selfBoard, Board opponentBoard)
    {
        if (opponentBoard == null) return 0.0f;

        var selfHeights = GetColumnHeights(selfBoard);
        var oppHeights = GetColumnHeights(opponentBoard);
        
        float avgSelfHeight = selfHeights.Average();
        float avgOppHeight = oppHeights.Average();
        
        float advantage = (avgOppHeight - avgSelfHeight) / 20.0f;
        return (float)System.Math.Tanh(advantage);
    }

    private float CalculateThreatLevel(Board opponentBoard)
    {
        if (opponentBoard == null) return 0.0f;
        var heights = GetColumnHeights(opponentBoard);
        return heights.Max() / 20.0f;
    }

    private float[] GetColumnHeights(Board board)
    {
        float[] heights = new float[10];
        var bounds = board.Bounds;
        
        for (int col = 0; col < 10; col++)
        {
            for (int row = 0; row < bounds.height; row++)
            {
                Vector3Int pos = new Vector3Int(bounds.xMin + col, bounds.yMin + row, 0);
                if (board.tilemap.HasTile(pos))
                {
                    heights[col] = bounds.height - row;
                    break;
                }
            }
        }
        
        return heights;
    }

    private float[] GetOutputArray(Tensor<float> outputTensor)
    {
        outputTensor.CompleteAllPendingOperations();
        var cpuTensor = outputTensor.ReadbackAndClone();
        return cpuTensor.AsReadOnlyNativeArray().ToArray();
    }

    [System.Serializable]
    public struct WildblockActionResult
    {
        public int actionType;
        public string actionName;
        public int targetColumn;
        public int targetRow;
        public float confidence;
        public float expectedDamage;
    }

    /// <summary>
    /// Core prediction logic - processes CNN output to make decision
    /// </summary>
    private WildblockActionResult ProcessWildblockCNNOutput(float[] output, Board board, PowerUp[] availablePowerUps)
    {
        Debug.Log("Powerupagent proces cnn");
        // Split output into components
        float[] actionQ = new float[5];
        float[] bombColumnQ = new float[10];
        float[] wildblockColumnQ = new float[8];
        
        Array.Copy(output, 0, actionQ, 0, 5);
        Array.Copy(output, 5, bombColumnQ, 0, 10);
        Array.Copy(output, 15, wildblockColumnQ, 0, 8);

        // Mask invalid actions
        MaskInvalidWildblockActions(actionQ, availablePowerUps, board);

        // Select best action
        int bestAction = ArgMax(actionQ);
        string[] actionNames = {"none", "bottom_clear", "gravity", "bomb", "wildblock"};
        
        var result = new WildblockActionResult
        {
            actionType = bestAction,
            actionName = actionNames[bestAction],
            confidence = Softmax(actionQ)[bestAction],
            targetColumn = -1,
            targetRow = -1,
            expectedDamage = 0.0f
        };

        // Handle targeted actions
        if (bestAction == 3) // Bomb
        {
            var validBombColumns = FindValidBombColumns(board);
            if (validBombColumns.Length > 0)
            {
                MaskInvalidBombColumns(bombColumnQ, validBombColumns);
                int bestColumn = ArgMax(bombColumnQ);
                result.targetColumn = bestColumn;
                result.targetRow = FindSurfaceBlock(board, bestColumn);
            }
        }
        else if (bestAction == 4) // Wildblock
        {
            var validWildblockPositions = FindValidWildblockPositions(board.opponentBoard);
            if (validWildblockPositions.Count > 0)
            {
                MaskInvalidWildblockColumns(wildblockColumnQ, validWildblockPositions.Keys.ToArray());
                int bestColumnIndex = ArgMax(wildblockColumnQ);
                int bestColumn = bestColumnIndex + 1;
                
                if (validWildblockPositions.ContainsKey(bestColumn))
                {
                    result.targetColumn = bestColumn;
                    result.targetRow = validWildblockPositions[bestColumn];
                    result.expectedDamage = CalculateWildblockDamage(board.opponentBoard, result.targetRow, bestColumn);
                }
            }
        }

        return result;
    }

    private void MaskInvalidWildblockActions(float[] actionQ, PowerUp[] availablePowerUps, Board board)
    {
        if (!HasPowerupType(availablePowerUps, PowerUpType.LineBlaster))
            actionQ[1] = float.NegativeInfinity;
        
        if (!HasPowerupType(availablePowerUps, PowerUpType.Gravity))
            actionQ[2] = float.NegativeInfinity;
        
        if (!HasPowerupType(availablePowerUps, PowerUpType.Bomb) || FindValidBombColumns(board).Length == 0)
            actionQ[3] = float.NegativeInfinity;
        
        if (!HasPowerupType(availablePowerUps, PowerUpType.WildCard) || 
            (board.opponentBoard != null && FindValidWildblockPositions(board.opponentBoard).Count == 0))
            actionQ[4] = float.NegativeInfinity;
    }

    private Dictionary<int, int> FindValidWildblockPositions(Board opponentBoard)
    {
        var validPositions = new Dictionary<int, int>();
        if (opponentBoard == null) return validPositions;
        
        for (int centerCol = 1; centerCol <= 8; centerCol++)
        {
            var surfaceRows = new List<int>();
            
            for (int col = centerCol - 1; col <= centerCol + 1; col++)
            {
                int surfaceRow = FindSurfaceBlock(opponentBoard, col);
                surfaceRows.Add(surfaceRow != -1 ? surfaceRow : 20);
            }
            
            int highestSurface = surfaceRows.Min();
            int placementRow = Mathf.Max(0, highestSurface - 1);
            
            if (placementRow >= 0 && placementRow < 19)
            {
                validPositions[centerCol] = placementRow;
            }
        }
        
        return validPositions;
    }

    private float CalculateWildblockDamage(Board opponentBoard, int placementRow, int placementCol)
    {
        if (opponentBoard == null) return 0.0f;
        
        float damage = 0.0f;
        var bounds = opponentBoard.Bounds;
        
        for (int dr = -1; dr <= 1; dr++)
        {
            for (int dc = -1; dc <= 1; dc++)
            {
                int r = placementRow + dr;
                int c = placementCol + dc;
                
                if (r >= 0 && r < bounds.height && c >= 0 && c < bounds.width)
                {
                    Vector3Int pos = new Vector3Int(bounds.xMin + c, bounds.yMin + r, 0);
                    if (!opponentBoard.tilemap.HasTile(pos))
                    {
                        damage += 1.0f;
                    }
                }
            }
        }
        
        var heights = GetColumnHeights(opponentBoard);
        float maxHeight = heights.Max();
        if (maxHeight > 15) damage += (maxHeight - 15) * 2.0f;
        
        return damage;
    }

    private void MaskInvalidBombColumns(float[] bombColumnQ, int[] validColumns)
    {
        for (int i = 0; i < bombColumnQ.Length; i++)
        {
            if (!validColumns.Contains(i))
            {
                bombColumnQ[i] = float.NegativeInfinity;
            }
        }
    }

    private void MaskInvalidWildblockColumns(float[] wildblockColumnQ, int[] validColumns)
    {
        for (int i = 0; i < wildblockColumnQ.Length; i++)
        {
            int columnNumber = i + 1;
            if (!validColumns.Contains(columnNumber))
            {
                wildblockColumnQ[i] = float.NegativeInfinity;
            }
        }
    }

    private int[] FindValidBombColumns(Board board)
    {
        List<int> validColumns = new List<int>();
        
        for (int col = 0; col < 10; col++)
        {
            if (FindSurfaceBlock(board, col) != -1)
            {
                validColumns.Add(col);
            }
        }
        
        return validColumns.ToArray();
    }

    private int FindSurfaceBlock(Board board, int column)
    {
        var bounds = board.Bounds;
        
        for (int row = 0; row < bounds.height; row++)
        {
            Vector3Int pos = new Vector3Int(bounds.xMin + column, bounds.yMin + row, 0);
            if (board.tilemap.HasTile(pos))
            {
                return row;
            }
        }
        
        return -1;
    }

    private int ArgMax(float[] array)
    {
        int bestIndex = 0;
        float bestValue = array[0];

        for (int i = 1; i < array.Length; i++)
        {
            if (array[i] > bestValue)
            {
                bestValue = array[i];
                bestIndex = i;
            }
        }

        return bestIndex;
    }

    private float[] Softmax(float[] array)
    {
        float[] result = new float[array.Length];
        float max = array.Max();
        float sum = 0.0f;
        
        for (int i = 0; i < array.Length; i++)
        {
            if (float.IsNegativeInfinity(array[i]))
            {
                result[i] = 0.0f;
            }
            else
            {
                result[i] = Mathf.Exp(array[i] - max);
                sum += result[i];
            }
        }
        
        if (sum > 0)
        {
            for (int i = 0; i < result.Length; i++)
            {
                result[i] /= sum;
            }
        }
        
        return result;
    }

    private void ApplyWildblockAction(WildblockActionResult actionResult, PowerUpManager powerUpManager, Board board)
    {
        switch (actionResult.actionType)
        {
            case 0:
                UpdateActionHistory("none");
                break;
                
            case 1:
                powerUpManager.ExecuteLineBlaster();
                UpdateActionHistory("bottom_clear");
                break;
                
            case 2:
                powerUpManager.ExecuteGravity();
                UpdateActionHistory("gravity");
                break;
                
            case 3:
                if (actionResult.targetRow != -1)
                {
                    // powerUpManager.ExecuteBomb(actionResult.targetRow, actionResult.targetColumn);
                    UpdateActionHistory("bomb");
                    UpdateBombColumnHistory(actionResult.targetColumn);
                }
                break;
                
            case 4:
                if (actionResult.targetRow != -1 && board.opponentBoard != null)
                {
                    ApplyWildblockToOpponent(board.opponentBoard, actionResult.targetRow, actionResult.targetColumn);
                    // powerUpManager.ExecuteWildblock(actionResult.targetRow, actionResult.targetColumn, board.opponentBoard);
                    UpdateActionHistory("wildblock");
                    UpdateWildblockColumnHistory(actionResult.targetColumn);
                }
                break;
        }
    }

    private void ApplyWildblockToOpponent(Board opponentBoard, int centerRow, int centerCol)
    {
        var bounds = opponentBoard.Bounds;
        
        for (int dr = -1; dr <= 1; dr++)
        {
            for (int dc = -1; dc <= 1; dc++)
            {
                int r = centerRow + dr;
                int c = centerCol + dc;
                
                if (r >= 0 && r < bounds.height && c >= 0 && c < bounds.width)
                {
                    Vector3Int pos = new Vector3Int(bounds.xMin + c, bounds.yMin + r, 0);
                    // opponentBoard.SetTile(pos, someBlockTile);
                }
            }
        }
    }

    private void UpdateActionHistory(string actionName)
    {
        if (actionHistory.ContainsKey(actionName))
        {
            actionHistory[actionName]++;
        }
    }

    private void UpdateBombColumnHistory(int column)
    {
        if (bombColumnHistory.ContainsKey(column))
        {
            bombColumnHistory[column]++;
        }
    }

    private void UpdateWildblockColumnHistory(int column)
    {
        if (wildblockColumnHistory.ContainsKey(column))
        {
            wildblockColumnHistory[column]++;
        }
    }
}