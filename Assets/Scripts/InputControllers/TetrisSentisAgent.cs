using UnityEngine;
using System.Collections.Generic;
using System.Linq;
using Unity.Sentis;
using System.Text;
using System.Collections;

/// <summary>
/// Simplified ONNX-backed Tetris AI using Unity Sentis for inference.
/// Now works with direct {col, rot} actions like the Python trainer.
/// </summary>
public class TetrisSentisAgent : MonoBehaviour, IPlayerInputController
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
    private Board board;
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

    private void InitializeSentis()
    {
        try
        {
            // Load the model from the asset
            runtimeModel = ModelLoader.Load(BoardManager.Instance.sentisModelAsset);

            // Create worker with fallback backend selection
            worker = CreateWorkerWithFallback();

            if (worker == null)
            {
                Debug.LogError("TetrisSentisAgent: Failed to create worker with any backend!");
                return;
            }

            Debug.Log($"TetrisSentisAgent: Successfully initialized with {backendType} backend");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"TetrisSentisAgent: Failed to initialize Sentis: {e.Message}");
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
            Debug.LogWarning($"TetrisSentisAgent: Failed to create {backendType} worker: {e.Message}");
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
                    Debug.LogWarning($"TetrisSentisAgent: Fell back to {backend} backend");
                    backendType = backend;
                    return worker;
                }
            }
            catch (System.Exception e)
            {
                Debug.LogWarning($"TetrisSentisAgent: {backend} backend also failed: {e.Message}");
            }
        }

        return null;
    }

    private void CleanupSentis()
    {
        worker?.Dispose();
        worker = null;
        runtimeModel = null;
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
        if (!IsReadyForInference()) return;
        RunInference();
    }

    void Update()
    {
        // if (!IsReadyForInference()) return;
        // if (Time.time - lastStateTime < stateUpdateInterval) return;

        // lastStateTime = Time.time;
        // RunInference();
    }

    private bool IsReadyForInference()
    {
        return board != null && currentPiece != null && worker != null && runtimeModel != null;
    }

    private void RunInference()
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

            // Log move information

            // Create input tensor for the model
            // Each move has 4 features: [lines, holes, bumpiness, height]
            var inputShape = new TensorShape(featureList.Count, 4);
            var features = featureList.SelectMany(f => f).ToArray();
            Tensor<float> inputTensor = new Tensor<float>(inputShape, features);

            // Run inference
            worker.Schedule(inputTensor);
            var outputTensor = worker.PeekOutput() as Tensor<float>;

            if (outputTensor != null)
            {
                ProcessInferenceOutput(outputTensor, possibleMoves);
            }
            else
            {
                Debug.LogWarning("TetrisSentisAgent: No output tensor received");
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"TetrisSentisAgent: Inference failed: {e.Message}");
        }
    }

    /// <summary>
    /// Get all possible moves in the same format as Python trainer
    /// Returns Dictionary where key is "col:rot" and value is [lines, holes, bumpiness, height]
    /// </summary>
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


    private IEnumerator ExecuteMoveWithDelay(string move, float delaySeconds)
    {
        yield return new WaitForSeconds(delaySeconds);
        ExecuteMove(move);
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
            StartCoroutine(ExecuteMoveWithDelay(bestMove, 0.5f));
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

    /// <summary>
    /// Execute move using simple col:rot format (like Python trainer)
    /// </summary>
    private void ExecuteMove(string move)
    {
        if (currentPiece?.data == null || board == null)
        {
            Debug.LogWarning("TetrisSentisAgent: Cannot execute move - invalid piece or board");
            return;
        }

        try
        {
            var parts = move.Split(':');
            int targetCol = int.Parse(parts[0]);
            int targetRot = int.Parse(parts[1]);



            // Clear current piece from board
            board.Clear(currentPiece);

            // Apply rotation
            int currentRotation = currentPiece.rotationIndex;
            int rotCount = currentPiece.data.RotationCount;
            int steps = (targetRot - currentRotation + rotCount) % rotCount;

            for (int i = 0; i < steps; i++)
            {
                currentPiece.Rotate(-1);
            }

            // Move horizontally
            if (currentPiece.cells != null && currentPiece.cells.Length > 0)
            {
                int minX = currentPiece.cells.Min(c => c.x);
                var newPosition = new Vector3Int(
                    targetCol + board.Bounds.xMin - minX,
                    currentPiece.position.y,
                    currentPiece.position.z
                );

                currentPiece.position = newPosition;

                // Hard drop
                while (board.IsValidPosition2(currentPiece, currentPiece.position + Vector3Int.down))
                {
                    currentPiece.position += Vector3Int.down;
                }
            }

            // Place piece on board
            board.Set(currentPiece);
            board.ClearLines();
            board.SpawnPiece();


        }
        catch (System.Exception e)
        {
            Debug.LogError($"TetrisSentisAgent: Error executing move: {e.Message}");
        }
    }



    // IPlayerInputController implementation (unused in AI mode)
    public bool GetLeft() => false;
    public bool GetRight() => false;
    public bool GetDown() => false;
    public bool GetRotateLeft() => false;
    public bool GetRotateRight() => false;
    public bool GetHardDrop() => false;

    private IEnumerator ExecuteMoveWithPause()
    {
        board.Set(currentPiece);

        Debug.Log("[TetrisAI] Waiting for Space key to spawn next piece...");

        yield return new WaitUntil(() => Input.GetKeyDown(KeyCode.Space));

        Debug.Log("[TetrisAI] Space pressed. Spawning next piece.");
        board.SpawnPiece();
    }

}