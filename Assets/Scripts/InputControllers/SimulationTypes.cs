using UnityEngine;
using System.Linq;
using System.Text;

/// <summary>
/// Data-only copy of the board's occupancy grid (no Tilemap) for fast simulation.
/// </summary>
public class BoardData
{
    public readonly int width, height;
    public readonly int xOffset, yOffset;
    public bool[,] grid;

    // Construct from real Board, capturing its bounds offsets
    public BoardData(Board real)
    {
        var b = real.Bounds;
        width = real.boardSize.x;
        height = real.boardSize.y;
        xOffset = b.xMin;
        yOffset = b.yMin;
        grid = new bool[width, height];

        // Sample the real tilemap into the bool grid
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                grid[x, y] = real.tilemap.HasTile(
                    new Vector3Int(x + xOffset, y + yOffset, 0)
                );
    }
    /// <summary>
    /// Returns a multi-line string showing the grid.
    /// ‘X’ = filled, ‘.’ = empty. Top row first.
    /// </summary>
    public string DumpToString()
    {
        var sb = new StringBuilder();
        // iterate from top (height-1) down to 0 so the visual matches Unity’s Y-axis
        for (int y = height - 1; y >= 0; y--)
        {
            for (int x = 0; x < width; x++)
            {
                sb.Append(grid[x, y] ? 'X' : '.');
            }
            sb.AppendLine();
        }
        return sb.ToString();
    }

    // Private constructor for cloning
    private BoardData(int w, int h, int xOff, int yOff)
    {
        width = w;
        height = h;
        xOffset = xOff;
        yOffset = yOff;
        grid = new bool[w, h];
    }

    /// <summary>
    /// Deep copy of this BoardData.
    /// </summary>
    public BoardData Clone()
    {
        var copy = new BoardData(width, height, xOffset, yOffset);
        System.Array.Copy(grid, copy.grid, grid.Length);
        return copy;
    }

    /// <summary>
    /// Checks if placing piece p at world position pos is valid (inside bounds & no collision).
    /// </summary>
    public bool IsValidPosition(PieceState p, Vector2Int pos)
    {
        foreach (var c in p.Cells)
        {
            int worldX = pos.x + c.x;
            int worldY = pos.y + c.y;
            int lx = worldX - xOffset;
            int ly = worldY - yOffset;
            if (lx < 0 || lx >= width || ly < 0) return false;
            if (ly >= height)
                continue;

            // 3) Otherwise we're inside [0..height-1] — check for a block
            if (grid[lx, ly])
                return false;
        }
        return true;
    }
    public bool IsGameOverCondition()
    {
        int lastRow = height - 1; // Top row of the visible board
        int secondLastRow = height - 2;
        int center = width / 2;

        // Check if center and adjacent cells in the top row are filled
        bool centerFilled = grid[center, lastRow];
        bool leftFilled = center > 0 ? grid[center - 1, lastRow] : false;
        bool rightFilled = center < width - 1 ? grid[center + 1, lastRow] : false;

        bool centerFilled1 = grid[center, secondLastRow];
        bool leftFilled1 = center > 0 ? grid[center - 1, secondLastRow] : false;
        bool rightFilled1 = center < width - 1 ? grid[center + 1, secondLastRow] : false;

        return centerFilled || leftFilled || rightFilled || centerFilled1 || leftFilled1 || rightFilled1;
    }

    /// <summary>
    /// Locks piece p into the grid, clears full lines, and returns number of lines cleared.
    /// </summary>
    public int PlaceAndClear(PieceState p)
    {
        // Lock piece cells (clamped to the visible grid)
        foreach (var c in p.Cells)
        {
            int worldX = p.position.x + c.x;
            int worldY = p.position.y + c.y;
            int lx = worldX - xOffset;
            int ly = worldY - yOffset;

            // skip any cell outside the grid
            if (lx < 0 || lx >= width || ly < 0 || ly >= height)
                continue;

            grid[lx, ly] = true;
        }

        int cleared = 0;
        // now clear full rows as before...
        for (int y = 0; y < height; y++)
        {
            bool full = true;
            for (int x = 0; x < width; x++)
                if (!grid[x, y]) { full = false; break; }

            if (full)
            {
                cleared++;
                for (int yy = y + 1; yy < height; yy++)
                    for (int x = 0; x < width; x++)
                        grid[x, yy - 1] = grid[x, yy];
                for (int x = 0; x < width; x++)
                    grid[x, height - 1] = false;
                y--;
            }
        }
        return cleared;
    }

    /// <summary>
    /// Counts the number of holes (empty cells with at least one block above).
    /// </summary>
    public int CountHoles()
    {
        int holes = 0;
        for (int x = 0; x < width; x++)
        {
            bool blockSeen = false;
            for (int y = height - 1; y >= 0; y--)
            {
                if (grid[x, y]) blockSeen = true;
                else if (blockSeen) holes++;
            }
        }
        return holes;
    }

    /// <summary>
    /// Returns an array of column heights.
    /// </summary>
    public int[] GetColumnHeights()
    {
        var heights = new int[width];
        for (int x = 0; x < width; x++)
            for (int y = height - 1; y >= 0; y--)
                if (grid[x, y]) { heights[x] = y + 1; break; }
        return heights;
    }

    /// <summary>
    /// Sum of absolute differences between adjacent column heights.
    /// </summary>
    public float GetBumpinessScore()
    {
        var hs = GetColumnHeights();
        float sum = 0f;
        for (int i = 0; i < hs.Length - 1; i++)
            sum += Mathf.Abs(hs[i] - hs[i + 1]);
        return sum;
    }

    /// <summary>
    /// The maximum column height.
    /// </summary>
    public float CalculateStackHeight()
    {
        return GetColumnHeights().Sum();
    }
}

/// <summary>
/// Data-only copy of Piece's state (rotation, position), with computed Cells.
/// </summary>
public struct PieceState
{
    public TetrominoData data;
    public int rotationIndex;
    public Vector2Int position;

    public PieceState(Piece p)
    {
        data = p.data;
        rotationIndex = p.rotationIndex;
        position = (Vector2Int)p.position;
    }

    public PieceState Clone()
    {
        return new PieceState
        {
            data = data,
            rotationIndex = rotationIndex,
            position = position
        };
    }

    public bool RotateCW(BoardData board)
    {


        // look up the SRS kicks for this from→to transition
        int from = rotationIndex;
        int to = (from + 1) % data.RotationCount;  // for CW

        int kickRow;
        if (to == (from + 1) % 4)
        {
            // CW: 0→1,1→2,2→3,3→0  → rows 0–3
            kickRow = from;
        }
        else
        {
            // CCW: 1→0,2→1,3→2,0→3 → rows 4–7
            // map 0→3 into row 7, 1→0 into 4, etc.
            kickRow = 4 + ((to + 3) % 4);
        }
        var kicks = Data.WallKicks[data.tetromino];
        int tests = kicks.GetLength(1);

        for (int t = 0; t < tests; t++)
        {
            var offset = kicks[kickRow, t];

            // Create a test-piece clone
            var testPiece = this.Clone();
            testPiece.rotationIndex = to;
            testPiece.position += offset;

            // Now IsValidPosition wants exactly a PieceState + a pos
            if (board.IsValidPosition(testPiece, testPiece.position))
            {
                // commit the rotation + kick
                rotationIndex = to;
                position += offset;
                return true;
            }
        }

        return false;  // all kicks failed
    }

    /// <summary>
    /// Returns the cell offsets for current rotation.
    /// </summary>
    public Vector2Int[] Cells
    {
        get { return data.GetCellsForRotation(rotationIndex); }
    }
}
