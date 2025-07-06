using System.Collections.Generic;
using System.Reflection;
using UnityEngine;

/// <summary>
/// Attach this to an empty GameObject in your scene.  
/// It visualizes all placement moves for the current tetromino by drawing colored cubes and heatmap labels.
/// Blue = low holes, Red = high holes.  Height is represented by Y-position of cube centers.
/// </summary>
[ExecuteInEditMode]
public class MoveMetricsVisualizer : MonoBehaviour
{
    [Header("Optional: manually assign")]
    [Tooltip("Your Tetris Board component (auto-detected if left empty)")]
    public Board board;

    [Tooltip("Your SocketTetrisAgent (auto-detected if left empty)")]
    public SocketTetrisAgent agent;

    [Header("Debug Settings")]
    [Tooltip("Index into board.tetrominoes array to select which piece to visualize (0-based)")]
    public int debugIndex = 0;

    [Tooltip("Size of the gizmos representing move positions")]
    public float gizmoSize = 0.9f;

    private MethodInfo getMetricsMethod;
    private Dictionary<(int x, int y), float[]> metrics;

    void OnEnable()
    {
        // Auto-detect board and agent if not assigned
        if (board == null)
            board = FindObjectOfType<Board>();
        if (agent == null)
            agent = FindObjectOfType<SocketTetrisAgent>();

        if (agent != null)
        {
            getMetricsMethod = typeof(SocketTetrisAgent)
                .GetMethod("GetMoveMetricsForCurrentPiece", BindingFlags.NonPublic | BindingFlags.Instance);
            if (getMetricsMethod == null)
                Debug.LogError("MoveMetricsVisualizer: Cannot find GetMoveMetricsForCurrentPiece via reflection.");
        }
    }

    void Update()
    {
        // Ensure we have all references
        if (board == null || agent == null || getMetricsMethod == null)
            return;

        // Clear and reset the board state
        board.ClearBoard();

        // Validate tetromino data
        if (board.tetrominoes == null || board.tetrominoes.Length == 0)
        {
            Debug.LogError("MoveMetricsVisualizer: Board has no tetrominoes configured.");
            return;
        }

        // Clamp debug index and select data
        int idx = Mathf.Clamp(debugIndex, 0, board.tetrominoes.Length - 1);
        var data = board.tetrominoes[idx];

        // Initialize & place the active piece
        board.activePiece.Initialize(board, board.spawnPosition, data, board.inputController);
        agent.SetCurrentPiece(board.activePiece);

        // Invoke the private metrics method
        metrics = (Dictionary<(int, int), float[]>)getMetricsMethod.Invoke(agent, null);
    }

    void OnDrawGizmos()
    {
        if (metrics == null)
            return;

        foreach (var kv in metrics)
        {
            var pos = kv.Key;
            var values = kv.Value;
            // values: [lines, holes, bumpiness, height]
            float height = values.Length > 3 ? values[3] : 0f;
            float holes = values.Length > 1 ? values[1] : 0f;

            // Heatmap from blue (0 holes) to red (max ~5 holes)
            float t = Mathf.Clamp01(holes / 5f);
            Gizmos.color = Color.Lerp(Color.blue, Color.red, t);

            // Draw cube at landing center
            Vector3 center = new Vector3(pos.x + 0.5f, height + 0.5f, pos.y + 0.5f);
            Gizmos.DrawCube(center, Vector3.one * gizmoSize);

            // Draw label for hole count
#if UNITY_EDITOR
            UnityEditor.Handles.Label(
                center + Vector3.up * (gizmoSize * 0.6f),
                holes.ToString("F0")
            );
#endif
        }
    }
}
