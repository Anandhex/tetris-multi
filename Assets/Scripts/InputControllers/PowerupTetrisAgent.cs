using UnityEngine;
using System.Collections.Generic;
using System.Linq;
using Unity.Sentis;
using System.Text;
using System.Collections;
using System;
using Unity.VisualScripting;

/// <summary>
/// Simplified ONNX-backed Tetris AI using Unity Sentis for inference.
/// Updated for CNN DQN powerup model with surface-only bomb targeting.
/// </summary>
public class PowerupTetrisAgent : MonoBehaviour
{
    [Header("Sentis Model Asset")]
    [Tooltip("Drag your .sentis ModelAsset here (imported ONNX model)")]

    [Header("Backend Selection")]
    public int allowedTetrominoTypes = 7;

    [SerializeField] private BackendType backendType = BackendType.GPUCompute;

    [Header("Logging Settings")]
    [SerializeField] private bool enableDetailedLogging = true;

    // Core Sentis components
    private Model runtimeModel;
    private Worker worker;

    // Tetris game references
    private Piece currentPiece;
    private float lastStateTime = 0f;
    private float stateUpdateInterval = 0.1f;

    // Statistics tracking
    private Dictionary<string, int> actionHistory = new Dictionary<string, int>();

    void Awake()
    {
        InitializeSentis();
    }

    void OnDestroy()
    {
        CleanupSentis();
    }

    public void InitializeSentis()
    {
        try
        {
            // Load the model from the asset
            Debug.Log(BoardManager.Instance.powerupAsset);
            runtimeModel = ModelLoader.Load(BoardManager.Instance.powerupAsset);

            // Create worker with fallback backend selection
            worker = CreateWorkerWithFallback();

            if (worker == null)
            {
                Debug.LogError("PowerupAgent: Failed to create worker with any backend!");
                return;
            }

            Debug.Log($"PowerupAgent: Successfully initialized CNN DQN model with {backendType} backend");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"PowerupAgent: Failed to initialize Sentis: {e}");
        }
    }

    private Worker CreateWorkerWithFallback()
    {
        // Try the selected backend first
        try
        {
            var worker = new Worker(runtimeModel, BackendType.GPUCompute);
            if (worker != null) return worker;
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"PowerupAgent: Failed to create {backendType} worker: {e.Message}");
        }

        // Fallback sequence
        BackendType[] fallbackOrder = { BackendType.GPUCompute, BackendType.CPU };

        foreach (var backend in fallbackOrder)
        {
            if (backend == backendType) continue; // Already tried

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

    public void CleanupSentis()
    {
        worker?.Dispose();
        worker = null;
        runtimeModel = null;
    }

    private bool IsReadyForInference()
    {
        return worker != null && runtimeModel != null;
    }

    public void RunInference(Board board, Dictionary<PowerUpType, int> powerUpInventory, PowerUpManager powerUpManager)
    {
        if (!IsReadyForInference())
        {
            return;
        }
        
        try
        {
            // Prepare 4-channel input for CNN DQN: (1, 4, 20, 10) = 800 floats
            float[] inputArray = GetBoardStateForCNN(board, powerUpInventory);

            if (inputArray.Length != 800)
            {
                Debug.LogError($"PowerupAgent: Invalid input shape, expected 800, got {inputArray.Length}");
                return;
            }

            // Create input tensor with shape (1, 4, 20, 10)
            var inputShape = new TensorShape(1, 4, 20, 10);
            Tensor<float> inputTensor = new Tensor<float>(inputShape, inputArray);
            
            // Run inference
            worker.Schedule(inputTensor);
            var outputTensor = worker.PeekOutput() as Tensor<float>;  // Should be (1, 14)

            // Process output - 14 values: [4 actions + 10 bomb columns]
            float[] output = GetOutputArray(outputTensor);
            
            if (output.Length != 14)
            {
                Debug.LogError($"PowerupAgent: Invalid output shape, expected 14, got {output.Length}");
                inputTensor.Dispose();
                outputTensor.Dispose();
                return;
            }

            // Get action decision
            int chosenAction = ProcessCNNOutput(output, board, powerUpInventory);
            
            if (enableDetailedLogging)
            {
                Debug.Log($"PowerupAgent: CNN DQN predicted action: {chosenAction}");
            }

            // Apply the chosen action
            ApplyPowerupAction(chosenAction, powerUpManager, board);

            // Cleanup
            inputTensor.Dispose();
            outputTensor.Dispose();
        }
        catch (System.Exception e)
        {
            Debug.LogError($"PowerupAgent: Inference failed: {e}");
        }
    }

    /// <summary>
    /// Prepare 4-channel input for CNN DQN model
    /// Channel 0: Board state, Channel 1-3: Powerup availability
    /// </summary>
    private float[] GetBoardStateForCNN(Board board, Dictionary<PowerUpType, int> powerUpInventory)
    {
        float[] inputData = new float[800]; // 4 * 20 * 10 = 800
        int index = 0;

        var bounds = board.Bounds;
        
        // Get powerup availability
        bool hasBottomClear = HasPowerup(powerUpInventory, PowerUpType.LineBlaster);
        bool hasGravity = HasPowerup(powerUpInventory, PowerUpType.Gravity);
        bool hasBomb = HasPowerup(powerUpInventory, PowerUpType.Bomb);

        // Channel 0: Board state (20x10 grid)
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
                    inputData[index] = 0.0f; // Padding
                }
                index++;
            }
        }

        // Channel 1: Bottom clear availability (broadcast to all 200 cells)
        float bottomClearValue = hasBottomClear ? 1.0f : 0.0f;
        for (int i = 0; i < 200; i++)
        {
            inputData[index++] = bottomClearValue;
        }

        // Channel 2: Gravity availability (broadcast to all 200 cells)
        float gravityValue = hasGravity ? 1.0f : 0.0f;
        for (int i = 0; i < 200; i++)
        {
            inputData[index++] = gravityValue;
        }

        // Channel 3: Bomb availability (broadcast to all 200 cells)
        float bombValue = hasBomb ? 1.0f : 0.0f;
        for (int i = 0; i < 200; i++)
        {
            inputData[index++] = bombValue;
        }

        if (enableDetailedLogging)
        {
            Debug.Log($"PowerupAgent: Input prepared - Powerups: [BC:{hasBottomClear}, G:{hasGravity}, B:{hasBomb}]");
        }

        return inputData;
    }

    private bool HasPowerup(Dictionary<PowerUpType, int> inventory, PowerUpType powerupType)
    {
        return inventory.TryGetValue(powerupType, out int count) && count > 0;
    }

    private float[] GetOutputArray(Tensor<float> outputTensor)
    {
        outputTensor.CompleteAllPendingOperations();
        var cpuTensor = outputTensor.ReadbackAndClone();
        return cpuTensor.AsReadOnlyNativeArray().ToArray();
    }

    /// <summary>
    /// Process CNN output to get final action decision
    /// Output format: [4 action Q-values, 10 bomb column Q-values]
    /// </summary>
    private int ProcessCNNOutput(float[] output, Board board, Dictionary<PowerUpType, int> powerUpInventory)
    {
        // Split output into action Q-values and bomb column Q-values
        float[] actionQ = new float[4];
        float[] bombColumnQ = new float[10];
        
        Array.Copy(output, 0, actionQ, 0, 4);      // Actions: [none, bottom_clear, gravity, bomb]
        Array.Copy(output, 4, bombColumnQ, 0, 10); // Bomb columns: [col0, col1, ..., col9]

        // Mask invalid actions
        MaskInvalidActions(actionQ, powerUpInventory, board);

        // Select best action
        int bestAction = ArgMax(actionQ);

        // If bomb was selected, find best column
        if (bestAction == 3) // Bomb action
        {
            int[] validColumns = FindValidBombColumns(board);
            
            if (validColumns.Length > 0)
            {
                // Mask invalid bomb columns
                MaskInvalidBombColumns(bombColumnQ, validColumns);
                
                // Get best column
                int bestColumn = ArgMax(bombColumnQ);
                
                if (enableDetailedLogging)
                {
                    int surfaceRow = FindSurfaceBlock(board, bestColumn);
                    Debug.Log($"PowerupAgent: Bomb selected - Column {bestColumn}, Surface at row {surfaceRow}");
                }
                
                return 3 + bestColumn; // Bomb actions start at index 3, so 3+column
            }
            else
            {
                // No valid bomb targets, fallback to next best action
                actionQ[3] = float.NegativeInfinity;
                bestAction = ArgMax(actionQ);
            }
        }

        return bestAction;
    }

    private void MaskInvalidActions(float[] actionQ, Dictionary<PowerUpType, int> powerUpInventory, Board board)
    {
        // Mask unavailable powerups
        if (!HasPowerup(powerUpInventory, PowerUpType.LineBlaster))
            actionQ[1] = float.NegativeInfinity;
        
        if (!HasPowerup(powerUpInventory, PowerUpType.Gravity))
            actionQ[2] = float.NegativeInfinity;
        
        if (!HasPowerup(powerUpInventory, PowerUpType.Bomb) || FindValidBombColumns(board).Length == 0)
            actionQ[3] = float.NegativeInfinity;
    }

    private void MaskInvalidBombColumns(float[] bombColumnQ, int[] validColumns)
    {
        // Set all to negative infinity first
        for (int i = 0; i < bombColumnQ.Length; i++)
        {
            bombColumnQ[i] = float.NegativeInfinity;
        }
        
        // Restore valid columns to their original values (assume 0 if masked)
        foreach (int col in validColumns)
        {
            if (col >= 0 && col < bombColumnQ.Length)
            {
                bombColumnQ[col] = 0.0f; // Neutral value for valid columns
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
        
        // Find topmost block in column
        for (int row = 0; row < bounds.height; row++)
        {
            Vector3Int pos = new Vector3Int(bounds.xMin + column, bounds.yMin + row, 0);
            if (board.tilemap.HasTile(pos))
            {
                return row;
            }
        }
        
        return -1; // No blocks in column
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

    private void ApplyPowerupAction(int chosenAction, PowerUpManager powerUpManager, Board board)
    {
        if (chosenAction == 0)
        {
            // No action
            if (enableDetailedLogging)
                Debug.Log("PowerupAgent: AI decided to wait");
        }
        else if (chosenAction == 1)
        {
            // Bottom clear
            Debug.Log("PowerupAgent: Executing bottom clear");
            powerUpManager.ExecuteLineBlaster();
            UpdateActionHistory("bottom_clear");
        }
        else if (chosenAction == 2)
        {
            // Gravity
            Debug.Log("PowerupAgent: Executing gravity");
            powerUpManager.ExecuteGravity();
            UpdateActionHistory("gravity");
        }
        else if (chosenAction >= 3 && chosenAction <= 12)
        {
            // Bomb at specific column
            int col = chosenAction - 3;
            int row = FindSurfaceBlock(board, col);
            
            if (row != -1)
            {
                Debug.Log($"PowerupAgent: Executing bomb at column {col}, row {row}");
                // powerUpManager.ExecuteBomb(row, col); // You'll need to implement this
                UpdateActionHistory("bomb");
            }
            else
            {
                Debug.LogWarning($"PowerupAgent: Invalid bomb target at column {col}");
            }
        }
        else
        {
            Debug.LogWarning($"PowerupAgent: Unknown action {chosenAction}");
        }
    }

    private void UpdateActionHistory(string actionName)
    {
        if (!actionHistory.ContainsKey(actionName))
            actionHistory[actionName] = 0;
        
        actionHistory[actionName]++;
    }

    [ContextMenu("Log Action Statistics")]
    public void LogActionStatistics()
    {
        if (actionHistory.Count == 0)
        {
            Debug.Log("PowerupAgent: No actions recorded yet");
            return;
        }

        StringBuilder stats = new StringBuilder("PowerupAgent Action Statistics:\n");
        int total = actionHistory.Values.Sum();
        
        foreach (var kvp in actionHistory)
        {
            float percentage = (float)kvp.Value / total * 100f;
            stats.AppendLine($"  {kvp.Key}: {kvp.Value} times ({percentage:F1}%)");
        }

        Debug.Log(stats.ToString());
    }
}