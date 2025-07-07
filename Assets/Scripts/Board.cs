using UnityEngine;
using UnityEngine.Tilemaps;
using UnityEngine.SceneManagement;
using TMPro;
using System.Collections;
using System.Collections.Generic;

public class Board : MonoBehaviour
{
    public Tilemap tilemap { get; private set; }
    public Piece activePiece;
    public TetrominoData nextPieceData { get; private set; }
    public string playerTag;
    private bool gameOverTriggered = false;
    
    [Header("Power-ups")]
    public MonoBehaviour powerUpManagerComponent;
    
    public TetrominoData[] tetrominoes;
    [SerializeField] private GameObject debrisPrefab;

    [Header("Visual Grid")]
    public SpriteRenderer gridSpriteRenderer;

    private int lastBoardHeight = -1;
    public Vector3Int baseSpawnPosition;

    public Vector3Int spawnPosition
    {
        get
        {
            int currentHeight = 20;
            TetrisMLAgent mlAgent = this.inputController as TetrisMLAgent;
            SocketTetrisAgent socketAgent = this.inputController as SocketTetrisAgent;
            TetrisSentisAgent sentisAgent = this.inputController as TetrisSentisAgent;

            if (mlAgent != null)
            {
                currentHeight = (int)mlAgent.curriculumBoardHeight;
            }
            else if (socketAgent != null)
            {
                currentHeight = (int)socketAgent.curriculumBoardHeight;
            }
            else if (sentisAgent != null)
            {
                currentHeight = (int)sentisAgent.curriculumBoardHeight;
            }

            return new Vector3Int(baseSpawnPosition.x, currentHeight / 2 - 2, baseSpawnPosition.z);
        }
    }

    public Vector2Int boardSize = new Vector2Int(10, 70);
    private float scoreSpeedBonus = 0f;

    public IPlayerInputController inputController;

    public int playerScore;
    public TMP_Text playerScoreToDisplay;
    public TMP_Text playerTagHolder;
    public NextPiece nextPieceDisplay;

    public bool isMLTraining = false;

    public RectInt Bounds
    {
        get
        {
            int height = boardSize.y;
            TetrisMLAgent mlAgent = this.inputController as TetrisMLAgent;
            SocketTetrisAgent socketAgent = this.inputController as SocketTetrisAgent;
            TetrisSentisAgent sentisAgent = this.inputController as TetrisSentisAgent;

            if (mlAgent != null)
            {
                height = 20;
            }
            else if (socketAgent != null)
            {
                height = 20;
            }
            else if (sentisAgent != null)
            {
                height = 20;
            }

            Vector2Int position = new Vector2Int(-this.boardSize.x / 2, -height / 2);
            return new RectInt(position, new Vector2Int(boardSize.x, height));
        }
    }

    public float initialDropRate = 0.75f;
    public float speedIncreasePerMinute = 0.5f;
    public float minimumDropRate = 0.1f;
    private float gameStartTime;
    private float temporarySpeedBoost = 0f;

    public float CurrentDropRate
    {
        get
        {
            float minutesPlayed = (Time.time - gameStartTime) / 60f;
            float timeSpeedDecrease = minutesPlayed * speedIncreasePerMinute;
            float totalSpeedDecrease = timeSpeedDecrease + scoreSpeedBonus + temporarySpeedBoost;
            float baseRate = Mathf.Max(initialDropRate - totalSpeedDecrease, minimumDropRate);

            // Power-up effects using reflection to avoid type issues
            if (powerUpManagerComponent != null)
            {
                var speedBoostMethod = powerUpManagerComponent.GetType().GetMethod("IsSpeedBoostActive");
                var frozenMethod = powerUpManagerComponent.GetType().GetMethod("IsFrozen");
                
                if (speedBoostMethod != null && (bool)speedBoostMethod.Invoke(powerUpManagerComponent, null))
                {
                    baseRate *= 2.0f; // Slower for speed boost (more time)
                }
                
                if (frozenMethod != null && (bool)frozenMethod.Invoke(powerUpManagerComponent, null))
                {
                    baseRate *= 2f;
                }
            }

            return baseRate;
        }
    }

    private void Awake()
    {
        this.tilemap = GetComponentInChildren<Tilemap>();
        this.activePiece = GetComponentInChildren<Piece>();

        for (int i = 0; i < this.tetrominoes.Length; i++)
        {
            this.tetrominoes[i].Initialize();
        }
    }

    private void Start()
    {
        this.playerScore = 0;
        this.gameStartTime = Time.time;

        if (inputController is SocketTetrisAgent socketAgent)
        {
            socketAgent.SetBoard(this);
        }
        
        if (inputController is TetrisSentisAgent sentisAgent)
        {
            sentisAgent.SetBoard(this);
        }

        if (playerTagHolder != null)
        {
            this.playerTagHolder.text = playerTag;
        }
        
        Debug.Log("Board Start() completed. Calling SpawnPiece...");
        SpawnPiece();
    }

    private void Update()
    {
        if (playerScoreToDisplay != null)
        {
            this.playerScoreToDisplay.text = this.playerScore.ToString();
        }
    }

    public void GenerateNextPiece()
    {
        int allowedTypes = 7;
        TetrisMLAgent mlAgent = this.inputController as TetrisMLAgent;
        SocketTetrisAgent socketAgent = this.inputController as SocketTetrisAgent;
        TetrisSentisAgent sentisAgent = this.inputController as TetrisSentisAgent;

        if (mlAgent != null)
        {
            allowedTypes = mlAgent.allowedTetrominoTypes;
        }
        else if (socketAgent != null)
        {
            allowedTypes = socketAgent.allowedTetrominoTypes;
        }
        else if (sentisAgent != null)
        {
            allowedTypes = sentisAgent.allowedTetrominoTypes;
        }

        int maxIndex = Mathf.Min(allowedTypes, this.tetrominoes.Length);
        int random = Random.Range(0, maxIndex);
        this.nextPieceData = this.tetrominoes[random];

        if (nextPieceDisplay != null)
        {
            nextPieceDisplay.DisplayNextPiece(this.nextPieceData);
        }
    }

    public void ReadyThePieceForSpwan()
    {
        TetrominoData pieceToUse;

        if (this.nextPieceData.Equals(default(TetrominoData)))
        {
            GenerateNextPiece();
            pieceToUse = this.nextPieceData;
        }
        else
        {
            pieceToUse = this.nextPieceData;
        }

        this.activePiece.Initialize(this, this.spawnPosition, pieceToUse, this.inputController);
        GenerateNextPiece();
        Debug.Log($"Piece ready for spawn: {pieceToUse} at {this.spawnPosition}");
    }

    public void SpawnPiece()
    {
        int random = Random.Range(0, this.tetrominoes.Length);
        TetrominoData data = this.tetrominoes[random];

        TetrominoData pieceToUse = this.nextPieceData.Equals(default(TetrominoData)) ? data : this.nextPieceData;

        this.activePiece.Initialize(this, this.spawnPosition, pieceToUse, this.inputController);

        TetrisMLAgent mlAgent = this.inputController as TetrisMLAgent;
        SocketTetrisAgent socketAgent = this.inputController as SocketTetrisAgent;
        TetrisSentisAgent sentisAgent = this.inputController as TetrisSentisAgent;

        if (mlAgent != null)
        {
            mlAgent.SetCurrentPiece(this.activePiece);
        }
        else if (socketAgent != null)
        {
            socketAgent.SetCurrentPiece(this.activePiece);
        }
        else if (sentisAgent != null)
        {
            sentisAgent.SetCurrentPiece(this.activePiece);
        }

        GenerateNextPiece();

        if (IsValidPosition(this.activePiece, this.spawnPosition))
        {
            Set(this.activePiece);
        }
        else
        {
            Debug.Log("GAME OVER - Cannot spawn new piece!");
            GameOver();
        }
    }

    public void GameOver()
    {
        Debug.Log("=== GAME OVER ===");
        
        // Notify ML agents first (if applicable)
        SocketTetrisAgent socketAgent = this.inputController as SocketTetrisAgent;
        if (socketAgent != null)
        {
            socketAgent.OnGameOver();
            return;
        }

        TetrisMLAgent mlAgent = this.inputController as TetrisMLAgent;
        if (mlAgent != null)
        {
            mlAgent.OnGameOver();
            StartCoroutine(ResetGameForMLTraining());
            return;
        }

        TetrisSentisAgent sentisAgent = this.inputController as TetrisSentisAgent;
        if (sentisAgent != null)
        {
            sentisAgent.OnGameOver();
            StartCoroutine(ResetGameForMLTraining());
            return;
        }

        // For regular game over - simple restart
        Debug.Log($"FINAL SCORE: {this.playerScore}");
        Data.PlayerScore = this.playerScore;
        
        StartCoroutine(GameOverSequence());
    }

    private IEnumerator GameOverSequence()
    {
        Time.timeScale = 0f;
        yield return new WaitForSecondsRealtime(2f);
        Time.timeScale = 1f;
        RestartGame();
    }

    public void RestartGame()
    {
        Debug.Log("Restarting game...");
        
        ClearBoard();
        this.playerScore = 0;
        this.gameStartTime = Time.time;
        
        PowerUpManager powerUpMgr = GetComponent<PowerUpManager>();
        if (powerUpMgr != null)
        {
            powerUpMgr.ClearAllPowerUps();
        }
        
        SpawnPiece();
        Debug.Log("Game restarted!");
    }

    private IEnumerator ResetGameForMLTraining()
    {
        yield return new WaitForSeconds(0.1f);
        ClearBoard();
        this.playerScore = 0;
        this.gameStartTime = Time.time;
        SpawnPiece();
    }

    public void ClearBoard()
    {
        tilemap.ClearAllTiles();
    }

    public void Set(Piece piece)
    {
        for (int i = 0; i < piece.cells.Length; i++)
        {
            Vector3Int tilePosition = piece.cells[i] + piece.position;
            this.tilemap.SetTile(tilePosition, piece.data.tile);
        }
    }

    public void Clear(Piece piece)
    {
        for (int i = 0; i < piece.cells.Length; i++)
        {
            Vector3Int tilePosition = piece.cells[i] + piece.position;
            this.tilemap.SetTile(tilePosition, null);
        }
    }

    public bool IsValidPosition(Piece piece, Vector3Int position)
    {
        RectInt bounds = this.Bounds;

        for (int i = 0; i < piece.cells.Length; i++)
        {
            Vector3Int tilePosition = piece.cells[i] + position;

            if (!bounds.Contains((Vector2Int)tilePosition))
            {
                return false;
            }

            if (powerUpManagerComponent != null)
            {
                var ghostMethod = powerUpManagerComponent.GetType().GetMethod("IsGhostModeActive");
                if (ghostMethod != null && (bool)ghostMethod.Invoke(powerUpManagerComponent, null))
                {
                    continue;
                }
            }

            if (this.tilemap.HasTile(tilePosition))
            {
                return false;
            }
        }
        return true;
    }

    public bool IsValidPosition2(Piece piece, Vector3Int position)
    {
        RectInt bounds = this.Bounds;

        for (int i = 0; i < piece.cells.Length; i++)
        {
            Vector3Int tilePosition = piece.cells[i] + position;

            if (!bounds.Contains((Vector2Int)tilePosition))
            {
                return false;
            }

            if (this.tilemap.HasTile(tilePosition))
            {
                return false;
            }
        }
        return true;
    }

    public void DumpTilemap(RectInt bounds)
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("=== TILEMAP DUMP ===");
        for (int y = bounds.yMax - 1; y >= bounds.yMin; y--)
        {
            for (int x = bounds.xMin; x < bounds.xMax; x++)
            {
                var has = tilemap.HasTile(new Vector3Int(x, y, 0)) ? 'X' : '.';
                sb.Append(has);
            }
            sb.AppendLine();
        }
        Debug.Log(sb.ToString());
    }

    public int ClearLines()
    {
        RectInt bounds = this.Bounds;
        int row = bounds.yMin;
        int linesCleared = 0;

        while (row < bounds.yMax)
        {
            if (IsLineFull(row))
            {
                LineClear(row);
                playerScore += 100;
                linesCleared++;
            }
            else
            {
                row++;
            }
        }

        // Notify PowerUpManager about line clears
        if (linesCleared > 0)
        {
            Debug.Log($"Board: {linesCleared} lines cleared, checking PowerUpManager...");
            PowerUpManager powerUpMgr = GetComponent<PowerUpManager>();
            if (powerUpMgr != null)
            {
                Debug.Log("PowerUpManager found directly, calling OnLinesCleared...");
                powerUpMgr.OnLinesCleared(linesCleared);
                Debug.Log("OnLinesCleared called successfully!");
            }
            else
            {
                Debug.Log("PowerUpManager component NOT found on this GameObject!");
            }
        }

        // Notify ML agents
        if (linesCleared > 0)
        {
            SocketTetrisAgent socketAgent = this.inputController as SocketTetrisAgent;
            if (socketAgent != null)
            {
                socketAgent.OnLinesCleared(linesCleared);
            }

            TetrisMLAgent mlAgent = this.inputController as TetrisMLAgent;
            if (mlAgent != null && mlAgent.GetType().GetMethod("OnLinesCleared") != null)
            {
                System.Reflection.MethodInfo method = mlAgent.GetType().GetMethod("OnLinesCleared");
                method?.Invoke(mlAgent, new object[] { linesCleared });
            }
        }

        scoreSpeedBonus = Mathf.Min(playerScore / 10000f, 1.0f);

        if (linesCleared >= 4)
        {
            StartCoroutine(ApplyTemporarySpeedBoost(0.2f, 3f));
        }
        else if (linesCleared >= 2)
        {
            StartCoroutine(ApplyTemporarySpeedBoost(0.1f, 2f));
        }
        return linesCleared;
    }

    private IEnumerator ApplyTemporarySpeedBoost(float amount, float duration)
    {
        temporarySpeedBoost += amount;
        yield return new WaitForSeconds(duration);
        temporarySpeedBoost -= amount;
    }

    private void SpawnDebris(Vector3Int tilePosition, Color color)
    {
        if (isMLTraining) return;

        Vector3 worldPosition = this.tilemap.CellToWorld(tilePosition) + new Vector3(0.5f, 0.5f, 0);
        GameObject debris = Instantiate(debrisPrefab, worldPosition, Quaternion.identity);
        SpriteRenderer sr = debris.GetComponent<SpriteRenderer>();
        sr.sortingOrder = 200;
        if (sr != null)
        {
            sr.color = color;
        }

        Rigidbody2D rb = debris.GetComponent<Rigidbody2D>();
        if (rb != null)
        {
            float randomForceX = Random.Range(-1f, 1f);
            float randomForceY = Random.Range(1f, 3f);
            rb.AddForce(new Vector2(randomForceX, randomForceY), ForceMode2D.Impulse);
        }

        Destroy(debris, 2f);
    }

    private bool IsLineFull(int row)
    {
        RectInt bounds = this.Bounds;
        for (int col = bounds.xMin; col < bounds.xMax; col++)
        {
            Vector3Int position = new Vector3Int(col, row, 0);
            if (!this.tilemap.HasTile(position))
            {
                return false;
            }
        }
        return true;
    }

    private void LineClear(int row)
    {
        RectInt bounds = this.Bounds;
        for (int col = bounds.xMin; col < bounds.xMax; col++)
        {
            Vector3Int position = new Vector3Int(col, row, 0);
            TileBase tile = this.tilemap.GetTile(position);
            Sprite sprite = this.tilemap.GetSprite(position);
            this.tilemap.SetTile(position, null);
            Color tileColor = Color.white;
            if (sprite != null)
            {
                Texture2D texture = sprite.texture;
                if (texture != null)
                {
                    int centerX = Mathf.FloorToInt(sprite.rect.x + sprite.rect.width / 2f);
                    int centerY = Mathf.FloorToInt(sprite.rect.y + sprite.rect.height / 2f);
                    tileColor = texture.GetPixel(centerX, centerY);
                }
            }
            SpawnDebris(position, tileColor);
            tilemap.SetTile(position, null);
        }

        while (row < bounds.yMax)
        {
            for (int col = bounds.xMin; col < bounds.xMax; col++)
            {
                Vector3Int position = new Vector3Int(col, row + 1, 0);
                TileBase above = this.tilemap.GetTile(position);

                position = new Vector3Int(col, row, 0);
                this.tilemap.SetTile(position, above);
            }
            row++;
        }
    }

    public int ClearLinesCount()
    {
        int cleared = 0;
        int width = boardSize.x;
        int height = boardSize.y;

        for (int y = 0; y < height; y++)
        {
            bool full = true;
            for (int x = 0; x < width; x++)
            {
                if (!tilemap.HasTile(new Vector3Int(x + Bounds.xMin, y + Bounds.yMin, 0)))
                {
                    full = false;
                    break;
                }
            }

            if (full)
            {
                cleared++;

                for (int x = 0; x < width; x++)
                    tilemap.SetTile(new Vector3Int(x + Bounds.xMin, y + Bounds.yMin, 0), null);

                for (int yy = y + 1; yy < height; yy++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        var abovePos = new Vector3Int(x + Bounds.xMin, yy + Bounds.yMin, 0);
                        var belowPos = new Vector3Int(x + Bounds.xMin, yy - 1 + Bounds.yMin, 0);
                        var tile = tilemap.GetTile(abovePos);
                        tilemap.SetTile(belowPos, tile);
                        tilemap.SetTile(abovePos, null);
                    }
                }

                y--;
            }
        }

        return cleared;
    }
}