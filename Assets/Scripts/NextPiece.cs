using UnityEngine;
using UnityEngine.Tilemaps;

public class NextPiece : MonoBehaviour
{
    public Tilemap previewTilemap { get; private set; }
    public Vector3Int previewOffset = new Vector3Int(1, 1, 0);


    private Vector3Int[] cells;

    private void Awake()
    {
        this.previewTilemap = GetComponentInChildren<Tilemap>();

        cells = new Vector3Int[4];
    }

    public void DisplayNextPiece(TetrominoData data)
    {
        previewTilemap.ClearAllTiles();

        if (data.Equals(default(TetrominoData)))
        {
            // Debug.LogWarning("NextPiece: Received default TetrominoData.");
            return;
        }


        for (int i = 0; i < data.cells.Length; i++)
        {
            cells[i] = new Vector3Int(data.cells[i].x, data.cells[i].y, 0);
        }

        Vector3Int basePosition = previewOffset;

        for (int i = 0; i < cells.Length; i++)
        {
            Vector3Int tilePosition = basePosition + cells[i];
            previewTilemap.SetTile(tilePosition, data.tile);
        }
    }

}
