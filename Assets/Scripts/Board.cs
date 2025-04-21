using UnityEngine;
using UnityEngine.Tilemaps;
using UnityEngine.SceneManagement; 
using TMPro;
using System.Collections;


public class Board : MonoBehaviour
{
   public Tilemap tilemap {get; private set;} 
   public Piece activePiece {get; private set;}
   public TetrominoData nextPieceData {get; private set;}
   public TetrominoData[] tetrominoes;
   public Vector3Int spawnPosition;
   public Vector2Int boardSize = new Vector2Int(10,20);
   private float scoreSpeedBonus = 0f;

   public int playerScore {get; private set;}
   public TMP_Text playerScoreToDisplay; 
   public NextPiece nextPieceDisplay;


   public RectInt Bounds{

       get {
         Vector2Int position = new Vector2Int(-this.boardSize.x/2,-this.boardSize.y/2);
         return new RectInt(position,this.boardSize);
       } 
   } 

public float initialDropRate = 0.75f; // Initial time between drops
public float speedIncreasePerMinute = 0.5f; // How much to decrease drop time per minute
public float minimumDropRate = 0.1f; // Fastest allowed drop rate
private float gameStartTime;

public float CurrentDropRate {
    get {
        float minutesPlayed = (Time.time - gameStartTime) / 60f;
        float timeSpeedDecrease = minutesPlayed * speedIncreasePerMinute;
        
        // Apply all speed increases: time-based, score-based, and temporary boosts
        float totalSpeedDecrease = timeSpeedDecrease + scoreSpeedBonus + temporarySpeedBoost;
        
        return Mathf.Max(initialDropRate - totalSpeedDecrease, minimumDropRate);
    }
}

   private void Awake()
   {
    this.tilemap = GetComponentInChildren<Tilemap>();
    this.activePiece = GetComponentInChildren<Piece>();

    for(int i=0;i<this.tetrominoes.Length;i++){
        this.tetrominoes[i].Initialize();
    }
   }

   private void Start(){
        this.playerScore = 0;
         this.gameStartTime = Time.time;
        SpawnPiece();
   }

   private void Update(){
        this.playerScoreToDisplay.text = this.playerScore.ToString();
   } 

   public void GenerateNextPiece(){
    int random = Random.Range(0,this.tetrominoes.Length);
    this.nextPieceData = this.tetrominoes[random];

    if (nextPieceDisplay != null) {
        nextPieceDisplay.DisplayNextPiece(this.nextPieceData);
    }
   }

   

   public void SpawnPiece(){
    int random = Random.Range(0, this.tetrominoes.Length);
    TetrominoData data = this.tetrominoes[random];
    
    TetrominoData pieceToUse = this.nextPieceData.Equals(default(TetrominoData)) ? data : this.nextPieceData;
    
    this.activePiece.Initialize(this, this.spawnPosition, pieceToUse);
    GenerateNextPiece();


        if(IsValidPosition(this.activePiece,this.spawnPosition)){
            Set(this.activePiece);
        }else{
            Data.PlayerScore = this.playerScore;
            GameOver();
        }
   }
    private void GameOver(){
        SceneManager.LoadScene(2);
    }
   public void Set(Piece piece)
   {
        for(int i=0;i<piece.cells.Length;i++){
            Vector3Int tilePosition = piece.cells[i] + piece.position;
            this.tilemap.SetTile(tilePosition,piece.data.tile);
        }
   }

   public void Clear(Piece piece)
   {
        for(int i=0;i<piece.cells.Length;i++){
            Vector3Int tilePosition = piece.cells[i] + piece.position;
            this.tilemap.SetTile(tilePosition,null);
        }
   }


   public bool IsValidPosition(Piece piece,Vector3Int position){

    RectInt bounds = this.Bounds;

    for(int i=0;i<piece.cells.Length;i++){
        Vector3Int tilePosition = piece.cells[i] + position;

        if(!bounds.Contains((Vector2Int)tilePosition)){
            return false;
        }

        if(this.tilemap.HasTile(tilePosition)){
            return false;
        }
    }
    return true;
   }

 public void ClearLines(){
    RectInt bounds = this.Bounds;
    int row = bounds.yMin;
    int linesCleared = 0;
    
    while(row < bounds.yMax){
        if(IsLineFull(row)){
            LineClear(row);
            playerScore += 100;
            linesCleared++;
        } else {
            row++;
        }
    }
    
    // Calculate score speed bonus based on player score
    // This creates a gradual speed increase as score goes up
    scoreSpeedBonus = Mathf.Min(playerScore / 10000f, 1.0f);

    if (linesCleared >= 4) {
        // Tetris (4 lines) gives temporary significant speed boost
        StartCoroutine(ApplyTemporarySpeedBoost(0.2f, 3f));
    } else if (linesCleared >= 2) {
        // 2-3 lines gives smaller temporary boost
        StartCoroutine(ApplyTemporarySpeedBoost(0.1f, 2f));
    }
}

private float temporarySpeedBoost = 0f;

private IEnumerator ApplyTemporarySpeedBoost(float amount, float duration) {
    temporarySpeedBoost += amount;
    
    yield return new WaitForSeconds(duration);
    
    temporarySpeedBoost -= amount;
}

   private bool IsLineFull(int row){
    RectInt bounds = this.Bounds;
    for(int col = bounds.xMin;col<bounds.xMax;col++){
        Vector3Int position = new Vector3Int(col,row,0);
        if(!this.tilemap.HasTile(position)){
            return false;
        }
    }
    return true;
   }
   private void LineClear(int row){
     RectInt bounds = this.Bounds;
     for(int col = bounds.xMin;col<bounds.xMax;col++){
     Vector3Int position = new Vector3Int(col,row,0);
     this.tilemap.SetTile(position,null);
     }

     while(row<bounds.yMax){
         for(int col = bounds.xMin;col<bounds.xMax;col++){
     Vector3Int position = new Vector3Int(col,row+1,0);
        TileBase above = this.tilemap.GetTile(position);

        position = new Vector3Int(col,row,0);
     this.tilemap.SetTile(position,above);
     }
     row++;
     }
   }
}
