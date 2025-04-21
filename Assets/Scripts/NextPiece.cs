using UnityEngine;
using UnityEngine.Tilemaps;

public class NextPiece : MonoBehaviour
{
   public Tilemap previewTilemap;
   public Vector3Int previewOffset = new Vector3Int(1,1,0);


    private TetrominoData nextPieceData;
    private Vector3Int[] cells;

    private void Awake()
    {
        cells = new Vector3Int[4]; 
    }

    public void DisplayNextPiece(TetrominoData data)
    {
        previewTilemap.ClearAllTiles();
        
        nextPieceData = data;
        
        if (data.Equals(default(TetrominoData))) {
            return;
        }
        
        for (int i = 0; i < data.cells.Length; i++) {
            cells[i] = (Vector3Int)data.cells[i];
        }
        
        Vector3Int basePosition = previewOffset;
        
        for (int i = 0; i < cells.Length; i++) {
            Vector3Int tilePosition = cells[i] + basePosition;
            previewTilemap.SetTile(tilePosition, data.tile);
        }
    } 

}
