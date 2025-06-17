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
    private int GetRotationCountByType(Tetromino t)
    {
        switch (t)
        {
            case Tetromino.O: return 1;
            case Tetromino.I:
            case Tetromino.S:
            case Tetromino.Z: return 2;
            default: return 4;
        }
    }
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
        // start from the base (0-rotation) shape
        Vector2Int[] result = new Vector2Int[cells.Length];
        cells.CopyTo(result, 0);

        // your standard SRS rotation matrix (CW 90°):
        //   [ 0 -1 ]
        //   [ 1  0 ]
        float m00 = 0, m01 = -1, m10 = 1, m11 = 0;

        // apply it rotationIndex times
        for (int r = 0; r < rotationIndex; r++)
        {
            for (int i = 0; i < result.Length; i++)
            {
                Vector2 v = result[i];
                // I and O pieces rotate about their centers
                if (tetromino == Tetromino.I || tetromino == Tetromino.O)
                    v -= new Vector2(0.5f, 0.5f);

                float x = v.x * m00 + v.y * m01;
                float y = v.x * m10 + v.y * m11;

                if (tetromino == Tetromino.I || tetromino == Tetromino.O)
                    result[i] = new Vector2Int(Mathf.CeilToInt(x), Mathf.CeilToInt(y));
                else
                    result[i] = new Vector2Int(Mathf.RoundToInt(x), Mathf.RoundToInt(y));
            }
        }

        return result;
    }

}