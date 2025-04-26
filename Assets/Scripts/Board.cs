using UnityEngine;
using UnityEngine.Tilemaps;
using UnityEngine.SceneManagement;
using TMPro;
using System.Collections;


public class Board : MonoBehaviour
{
    public Tilemap tilemap { get; private set; }
    public Piece activePiece { get; private set; }
    public TetrominoData nextPieceData { get; private set; }
    public string playerTag;
    public TetrominoData[] tetrominoes;
    // public FireBorderController fireBorderController;
    [SerializeField] private GameObject debrisPrefab;

    public Vector3Int spawnPosition;
    public Vector2Int boardSize = new Vector2Int(10, 20);
    private float scoreSpeedBonus = 0f;

    public IPlayerInputController inputController;

    public int playerScore { get; private set; }
    public TMP_Text playerScoreToDisplay;
    public TMP_Text playerTagHolder;
    public NextPiece nextPieceDisplay;


    public RectInt Bounds
    {

        get
        {
            Vector2Int position = new Vector2Int(-this.boardSize.x / 2, -this.boardSize.y / 2);
            return new RectInt(position, this.boardSize);
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

        // Only spawn a piece if all components are properly initialized
        if (activePiece != null && tetrominoes != null && tetrominoes.Length > 0)
        {
            SpawnPiece();
        }
        else
        {
            Debug.LogError("Cannot spawn piece: Required components not initialized");
        }
        this.playerTagHolder.text = playerTag;
    }

    private void Update()
    {
        this.playerScoreToDisplay.text = this.playerScore.ToString();
        // if (fireBorderController != null)
        // {
        //     fireBorderController.SetGameSpeed(1f / CurrentDropRate);
        // }
    }

    public void GenerateNextPiece()
    {
        int random = Random.Range(0, this.tetrominoes.Length);
        this.nextPieceData = this.tetrominoes[random];

        if (nextPieceDisplay != null)
        {
            Debug.Log("called for the next piece");
            nextPieceDisplay.DisplayNextPiece(this.nextPieceData);
        }
    }


    public void SpawnPiece()
    {


        int random = Random.Range(0, this.tetrominoes.Length);
        TetrominoData data = this.tetrominoes[random];

        TetrominoData pieceToUse = this.nextPieceData.Equals(default(TetrominoData)) ? data : this.nextPieceData;

        this.activePiece.Initialize(this, this.spawnPosition, pieceToUse, this.inputController);
        GenerateNextPiece();

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
    private void GameOver()
    {
        SceneManager.LoadScene(2);
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

            if (this.tilemap.HasTile(tilePosition))
            {
                return false;
            }
        }
        return true;
    }

    public void ClearLines()
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

        // Calculate score speed bonus based on player score
        // This creates a gradual speed increase as score goes up
        scoreSpeedBonus = Mathf.Min(playerScore / 10000f, 1.0f);

        if (linesCleared >= 4)
        {
            // Tetris (4 lines) gives temporary significant speed boost
            StartCoroutine(ApplyTemporarySpeedBoost(0.2f, 3f));
        }
        else if (linesCleared >= 2)
        {
            // 2-3 lines gives smaller temporary boost
            StartCoroutine(ApplyTemporarySpeedBoost(0.1f, 2f));
        }
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
