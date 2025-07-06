using UnityEngine;
using UnityEngine.Tilemaps;


public enum Tetromino
{
    I, O, T, J, L, S, Z
}


[System.Serializable]
public struct TetrominoData
{
    public Tetromino tetromino;
    public Tile tile;
    public Vector2Int[] cells { get; private set; }
    public Vector2Int[,] wallKicks { get; private set; }
    public int RotationCount => GetRotationCountByType(tetromino);
    private int GetRotationCountByType(Tetromino t) => 4;
    public void Initialize()
    {
        this.cells = Data.Cells[this.tetromino];
        this.wallKicks = Data.WallKicks[this.tetromino];
    }

    /// <summary>
    /// Returns the cell offsets of this tetromino after applying
    /// `rotationIndex` clockwise rotations (0…RotationCount–1).
    /// </summary>
    public Vector2Int[] GetCellsForRotation(int rotationIndex)
    {
        // 1) Copy your base (0°) cells
        Vector2Int[] result = new Vector2Int[cells.Length];
        cells.CopyTo(result, 0);

        // 2) SRS 90° CW rotation matrix
        float m00 = 0, m01 = -1;
        float m10 = 1, m11 = 0;

        // 3) Choose the true SRS pivot
        Vector2 pivot;
        switch (tetromino)
        {
            case Tetromino.I:
                pivot = new Vector2(1.5f, 1.5f);
                break;
            case Tetromino.O:
                pivot = new Vector2(0.5f, 0.5f);
                break;
            default:
                pivot = new Vector2(1f, 1f);
                break;
        }

        // 4) Apply the rotation N times
        for (int r = 0; r < rotationIndex; r++)
        {
            for (int i = 0; i < result.Length; i++)
            {
                // a) translate into pivot‐space
                Vector2 v = (Vector2)result[i] - pivot;

                // b) rotate
                float x = v.x * m00 + v.y * m01;
                float y = v.x * m10 + v.y * m11;

                // c) translate back out
                v = new Vector2(x, y) + pivot;

                // d) snap to nearest integer
                result[i] = new Vector2Int(
                    Mathf.RoundToInt(v.x),
                    Mathf.RoundToInt(v.y)
                );
            }
        }

        return result;
    }


}