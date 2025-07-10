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
    // public FireBorderController fireBorderController;
    [SerializeField] private GameObject debrisPrefab;

    [Header("Visual Grid")]
    public SpriteRenderer gridSpriteRenderer;

    private int lastBoardHeight = -1;
    public Vector3Int baseSpawnPosition;

    public Vector3Int spawnPosition
    {
        get
        {
            int currentHeight = 20; // Default height

            // Check for both types of ML agents
            TetrisMLAgent mlAgent = this.inputController as TetrisMLAgent;
            SocketTetrisAgent socketAgent = this.inputController as SocketTetrisAgent;

            if (mlAgent != null)
            {
                currentHeight = (int)mlAgent.curriculumBoardHeight;
            }
            else if (socketAgent != null)
            {
                currentHeight = (int)socketAgent.curriculumBoardHeight;
            }

            // Adjust spawn position to be at the top of the current board height
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

    // Flag to check if we're in ML training mode
    public bool isMLTraining = false;

    public RectInt Bounds
    {
        get
        {
            int height = boardSize.y; // Default height

            // Check for both types of ML agents
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

    public float initialDropRate = 0.75f; // Initial time between drops
    public float speedIncreasePerMinute = 0.5f; // How much to decrease drop time per minute
    public float minimumDropRate = 0.1f; // Fastest allowed drop rate
    private float gameStartTime;

    public float CurrentDropRate
    {
        get
        {
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
            SpawnPiece();
        }
        // ClearBoard();

        // // Apply initial curriculum
        // // ApplyCurriculumBoardPreset();

        // // Only spawn a piece if all components are properly initialized
        if (activePiece != null && tetrominoes != null && tetrominoes.Length > 0 && (inputController is SinglePlayerInputController || inputController is PlayerInputController || inputController is Player1InputController || inputController is Player2InputController))
        {
            SpawnPiece();
        }
        // else
        // {
        // }

        if (playerTagHolder != null)
        {
            this.playerTagHolder.text = playerTag;
        }
    }



    private void Update()
    {
        if (playerScoreToDisplay != null)
        {
            this.playerScoreToDisplay.text = this.playerScore.ToString();
        }

        PowerupTetrisAgent powerupTetrisAgent = FindAnyObjectByType<PowerupTetrisAgent>();
        PowerUpManager powerUpManager = FindAnyObjectByType<PowerUpManager>();
        if (powerupTetrisAgent != null && powerUpManager != null)
        {

            powerupTetrisAgent.RunInference(this, powerUpManager.powerUpInventory, powerUpManager);
        }

        // if (fireBorderController != null)
        // {
        //     fireBorderController.SetGameSpeed(1f / CurrentDropRate);
        // }
    }


    public void GenerateNextPiece()
    {
        int allowedTypes = 7; // Default to all pieces

        // Check for both types of ML agents
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

        // Limit piece selection based on curriculum
        int maxIndex = Mathf.Min(allowedTypes, this.tetrominoes.Length);
        int random = Random.Range(0, maxIndex);
        this.nextPieceData = this.tetrominoes[random];

        if (nextPieceDisplay != null)
        {
            nextPieceDisplay.DisplayNextPiece(this.nextPieceData);
        }
        if ((inputController is SinglePlayerInputController || inputController is PlayerInputController || inputController is Player1InputController || inputController is Player2InputController))
        {
            if (IsValidPosition(this.activePiece, this.spawnPosition))
            {
                Set(this.activePiece);
            }
            else
            {
                Data.PlayerScore = this.playerScore;
                GameOver();
            }
        }
    }


    public void ReadyThePieceForSpwan()
    {
        TetrominoData pieceToUse;

        if (this.nextPieceData.Equals(default(TetrominoData)))
        {
            // No next piece, generate one
            GenerateNextPiece();
            pieceToUse = this.nextPieceData;
        }
        else
        {
            pieceToUse = this.nextPieceData;
        }

        // Initialize the piece but don't place it on the board yet
        this.activePiece.Initialize(this, this.spawnPosition, pieceToUse, this.inputController);

        // Generate the next piece for display
        GenerateNextPiece();
        Debug.Log($"Piece ready for spawn: {pieceToUse} at {this.spawnPosition}");
    }



    public void SpawnPiece()
    {
        int random = Random.Range(0, this.tetrominoes.Length);
        TetrominoData data = this.tetrominoes[random];

        TetrominoData pieceToUse = this.nextPieceData.Equals(default(TetrominoData)) ? data : this.nextPieceData;

        this.activePiece.Initialize(this, this.spawnPosition, pieceToUse, this.inputController);

        // Inform both types of ML agents about the new piece
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
            Data.PlayerScore = this.playerScore;
        }
    }

    public void GameOver()
    {

        // Notify ML agent if this is an ML agent-controlled board
        SocketTetrisAgent socketAgent = this.inputController as SocketTetrisAgent;

        if (isMLTraining)
        {
            ClearBoard();
            this.playerScore = 0;
            this.gameStartTime = Time.time;
            PowerUpManager powerUpMgr = GetComponent<PowerUpManager>();
            if (powerUpMgr != null)
            {
                powerUpMgr.ClearAllPowerUps();
            }
            TetrisSentisAgent tAgent = this.inputController as TetrisSentisAgent;
            tAgent.CleanupSentis();
            tAgent.InitializeSentis();
            SpawnPiece();
            return;
        }
        // new: clear *first*, then notify and reset
        if (socketAgent != null)
        {

            // 1) immediately clear any existing tiles
            // 2) let Python know the game is over on an empty board
            PowerUpManager powerUpMgr = GetComponent<PowerUpManager>();
            if (powerUpMgr != null)
            {
                powerUpMgr.ClearAllPowerUps();
            }
            socketAgent.OnGameOver();

            // 3) schedule the curriculum reset + spawn
            return;
        }


        TetrisMLAgent mlAgent = this.inputController as TetrisMLAgent;
        if (mlAgent != null)
        {
            mlAgent.OnGameOver();
            StartCoroutine(ResetGameForMLTraining());
            PowerUpManager powerUpMgr = GetComponent<PowerUpManager>();
            if (powerUpMgr != null)
            {
                powerUpMgr.ClearAllPowerUps();
            }
            return;
        }

        // Store the score for the game over screen
        Data.PlayerScore = this.playerScore;

        // Load game over scene only if not in ML training
        SceneManager.LoadScene(3);
    }


    private IEnumerator ResetGameForMLTraining()
    {
        // Short delay to ensure ML Agent has processed the game over
        yield return new WaitForSeconds(0.1f);

        // Reset the board
        ClearBoard();
        // ApplyCurriculumBoardPreset();
        this.playerScore = 0;
        this.gameStartTime = Time.time;

        // Spawn a new piece to start the game again
        SpawnPiece();
    }

    public void ClearBoard()
    {
        // Clear the entire tilemap
        // RectInt bounds = this.Bounds;
        // for (int row = bounds.yMin; row < bounds.yMax; row++)
        // {
        //     for (int col = bounds.xMin; col < bounds.xMax; col++)
        //     {
        //         Vector3Int position = new Vector3Int(col, row, 0);
        //         this.tilemap.SetTile(position, null);
        //     }
        // }
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
                // DumpTilemap(bounds);

                return false;
            }

            if (this.tilemap.HasTile(tilePosition))
            {
                Debug.LogError($"Collision at cell #{i} → {tilePosition}. Dumping map:");
                // DumpTilemap(bounds);
                SocketTetrisAgent socketAgent = this.inputController as SocketTetrisAgent;
                if (socketAgent != null)
                {
                    // socketAgent.TriggerGameOver();
                }
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
        if (linesCleared > 0)
        {
            PowerUpManager powerUpMgr = GetComponent<PowerUpManager>();
            if (powerUpMgr != null)
            {
                powerUpMgr.OnLinesCleared(linesCleared);
            }
        }

        // Notify ML agent about line clears


        // Calculate score speed bonus based on player score
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

    private float temporarySpeedBoost = 0f;

    private IEnumerator ApplyTemporarySpeedBoost(float amount, float duration)
    {
        temporarySpeedBoost += amount;

        yield return new WaitForSeconds(duration);

        temporarySpeedBoost -= amount;
    }

    private void SpawnDebris(Vector3Int tilePosition, Color color)
    {
        // Skip debris generation during ML training to improve performance
        if (isMLTraining) return;

        Vector3 worldPosition = this.tilemap.CellToWorld(tilePosition) + new Vector3(0.5f, 0.5f, 0); // center it
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
            float randomForceX = Random.Range(-1f, 1f); // scatter a little
            float randomForceY = Random.Range(1f, 3f);  // upward burst
            rb.AddForce(new Vector2(randomForceX, randomForceY), ForceMode2D.Impulse);
        }

        Destroy(debris, 2f); // destroy after 2 seconds
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
            Color tileColor = Color.white; // fallback
            if (sprite != null)
            {
                Texture2D texture = sprite.texture;
                if (texture != null)
                {
                    // Sample pixel from the center of the sprite's rect
                    int centerX = Mathf.FloorToInt(sprite.rect.x + sprite.rect.width / 2f);
                    int centerY = Mathf.FloorToInt(sprite.rect.y + sprite.rect.height / 2f);

                    // Use GetPixelBilinear if you want normalized 0..1 coords
                    tileColor = texture.GetPixel(centerX, centerY);
                }
            }
            SpawnDebris(position, tileColor); // SPAWN DEBRIS
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


}