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
/// Now works with direct {col, rot} actions like the Python trainer.
/// </summary>
public class PowerupTetrisAgent : MonoBehaviour
{
    [Header("Sentis Model Asset")]
    [Tooltip("Drag your .sentis ModelAsset here (imported ONNX model)")]

    [Header("Backend Selection")]
    public int allowedTetrominoTypes = 7;

    [SerializeField] private BackendType backendType = BackendType.GPUCompute;

    [Header("Logging Settings")]

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

            Debug.Log($"PowerupAgent: Successfully initialized with {backendType} backend");
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
            float[] inputArray = GetBoardState(board, powerUpInventory).ToArray(); // Length = 403

            if (inputArray.Length != 404)
            {
                Debug.LogError("Invalid input shape");
                return;
            }

            var inputShape = new TensorShape(1, 404);
            int expectedLength = inputShape.ToArray().Aggregate(1, (a, b) => a * b);

            Tensor<float> inputTensor = new Tensor<float>(inputShape, inputArray);  // [batch, features]
            worker.Schedule(inputTensor);
            var outputTensor = worker.PeekOutput() as Tensor<float>;  // [1, 23]

            int chosenAction = ArgMax(outputTensor);
            Debug.Log($"Predicted action: {chosenAction}");

            ApplyPowerupAction(chosenAction, powerUpManager);  // your implementation

            inputTensor.Dispose();
            outputTensor.Dispose();
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Inference failed: {e}");
        }
    }

    private void ApplyPowerupAction(int chosenAction, PowerUpManager powerUpManager)
    {
        if (chosenAction == 0)
        {
            // No-op

        }
        else if (chosenAction == 1)
        {
            powerUpManager.ExecuteLineBlaster();
        }
        else if (chosenAction == 2)
        {
            powerUpManager.ExecuteGravity();
        }
        else if (chosenAction >= 3 && chosenAction <= 12)
        {
            int col = chosenAction - 3;

        }
        else if (chosenAction >= 13 && chosenAction <= 22)
        {
            int col = chosenAction - 13;

        }
    }
    private int ArgMax(Tensor<float> outputTensor)
    {
        outputTensor.CompleteAllPendingOperations();
        var cpuTensor = outputTensor.ReadbackAndClone();
        float[] data = cpuTensor.AsReadOnlyNativeArray().ToArray();
        int bestIndex = 0;
        float bestValue = float.MinValue;

        for (int i = 0; i < data.Length; i++)
        {
            if (data[i] > bestValue)
            {
                bestValue = data[i];
                bestIndex = i;
            }
        }

        return bestIndex;
    }


    /// <summary>
    /// Get all possible moves in the same format as Python trainer
    /// Returns Dictionary where key is "col:rot" and value is [lines, holes, bumpiness, height]
    /// </summary>
    private List<float> GetBoardState(Board board, Dictionary<PowerUpType, int> powerUpInventory)
    {
        List<float> state = new List<float>();
        try
        {
            var bounds = board.Bounds;

            for (int y = 0; y < bounds.height; y++)
            {
                for (int x = 0; x < bounds.width; x++)
                {
                    Vector3Int pos = new Vector3Int(bounds.xMin + x, bounds.yMin + y, 0);
                    state.Add(board.tilemap.HasTile(pos) ? 1 : 0);
                }
            }

            for (int y = 0; y < bounds.height; y++)
            {
                for (int x = 0; x < bounds.width; x++)
                {
                    Vector3Int pos = new Vector3Int(bounds.xMin + x, bounds.yMin + y, 0);
                    state.Add(0);
                }
            }

            foreach (PowerUpType powerUp in Enum.GetValues(typeof(PowerUpType)))
            {
                bool hasPowerUp = powerUpInventory.TryGetValue(powerUp, out int count) && count > 0;
                state.Add(hasPowerUp ? 1f : 0f);
            }

            state.Add(0);

        }
        catch (System.Exception e)
        {
            Debug.LogError($"Inference failed: {e}");
        }




        return state;
    }







}