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
}