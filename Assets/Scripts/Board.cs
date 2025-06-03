using UnityEngine;
using UnityEngine.Tilemaps;
using UnityEngine.SceneManagement;
using TMPro;
using System.Collections;
using System.Collections.Generic;

public class Board : MonoBehaviour
{
    public Tilemap tilemap { get; private set; }
    public Piece activePiece { get; private set; }
    public TetrominoData nextPieceData { get; private set; }
    public string playerTag;
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
            TetrisMLAgent mlAgent = this.inputController as TetrisMLAgent;
            if (mlAgent != null)
            {
                int currentHeight = (int)mlAgent.curriculumBoardHeight;
                return new Vector3Int(baseSpawnPosition.x, currentHeight / 2 - 2, baseSpawnPosition.z);
            }
            return baseSpawnPosition;
        }
    }
    public Vector2Int boardSize = new Vector2Int(10, 70);
    private float scoreSpeedBonus = 0f;

    public IPlayerInputController inputController;

    public int playerScore { get; private set; }
    public TMP_Text playerScoreToDisplay;
    public TMP_Text playerTagHolder;
    public NextPiece nextPieceDisplay;

    // Flag to check if we're in ML training mode
    private bool isMLTraining = false;

    public RectInt Bounds
    {
        get
        {
            TetrisMLAgent mlAgent = this.inputController as TetrisMLAgent;
            int height = (mlAgent != null) ? (int)mlAgent.curriculumBoardHeight : boardSize.y;

            Vector2Int position = new Vector2Int(-this.boardSize.x / 2, -height / 2);
            return new RectInt(position, new Vector2Int(boardSize.x, height));
        }
    }

    public float initialDropRate = 0.75f;
    public float speedIncreasePerMinute = 0.5f;
    public float minimumDropRate = 0.1f;
    private float gameStartTime;

    public float CurrentDropRate
    {
        get
        {
            float minutesPlayed = (Time.time - gameStartTime) / 60f;
            float timeSpeedDecrease = minutesPlayed * speedIncreasePerMinute;
            float totalSpeedDecrease = timeSpeedDecrease + scoreSpeedBonus + temporarySpeedBonus;
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

        UpdateGridVisualization();

        if (playerTagHolder != null)
        {
            this.playerTagHolder.text = playerTag;
        }

        // Don't spawn piece here - let ML Agent handle initialization
        if (inputController is TetrisMLAgent)
        {
            isMLTraining = true;
            Debug.Log("Board: ML Training mode detected");
        }
        else
        {
            // Non-ML mode - spawn piece normally
            if (activePiece != null && tetrominoes != null && tetrominoes.Length > 0)
            {
                SpawnPiece();
            }
        }
    }

    public void ResetBoard()
    {
        Debug.Log("Board.ResetBoard: Resetting board for new episode");
        
        // Clear everything
        ClearBoard();
        ApplyCurriculumBoardPreset();
        
        // Reset game state
        this.playerScore = 0;
        this.gameStartTime = Time.time;
        this.scoreSpeedBonus = 0f;
        this.temporarySpeedBonus = 0f;
        
        // Clear old piece
        if (activePiece != null)
        {
            Clear(activePiece);
        }
        
        // Update visualization
        UpdateGridVisualization();
        
        // Generate next piece data
        GenerateNextPiece();
        
        // Spawn new piece
        SpawnPiece();
    }

    // ... [Keep all the curriculum preset methods exactly as they are] ...
    public void ApplyCurriculumBoardPreset()
    {
        int preset = 0;
        int boardHeight = 20;

        TetrisMLAgent mlAgent = this.inputController as TetrisMLAgent;
        if (mlAgent != null)
        {
            preset = (int)mlAgent.curriculumBoardPreset;
            boardHeight = (int)mlAgent.curriculumBoardHeight;
        }

        ClearBoard();

        int maxY = Bounds.yMin + boardHeight - 1;

        switch (preset)
        {
            case 0:
                break;

            case 1:
                {
                    int targetRow = boardHeight <= 8 ? Bounds.yMin : Bounds.yMin + 1;
                    int gapStart = Random.Range(Bounds.xMin, Bounds.xMax - 3);
                    int gapEnd = gapStart + 4;

                    for (int col = Bounds.xMin; col < Bounds.xMax; col++)
                    {
                        if (col < gapStart || col >= gapEnd)
                        {
                            SetTile(col, targetRow);
                        }
                    }

                    if (gapEnd - gapStart >= Bounds.xMax - Bounds.xMin)
                    {
                        SetTile(gapStart, targetRow);
                    }
                }
                break;

            case 2:
                {
                    int workingHeight = Mathf.Min(3, boardHeight - 1);

                    int iPieceGap = Random.Range(Bounds.xMin, Bounds.xMax - 3);
                    for (int col = Bounds.xMin; col < Bounds.xMax; col++)
                    {
                        if (col < iPieceGap || col >= iPieceGap + 4)
                        {
                            SetTile(col, Bounds.yMin);
                        }
                    }

                    if (workingHeight >= 2)
                    {
                        int oPieceGap = iPieceGap >= Bounds.xMin + 2 ?
                            Random.Range(Bounds.xMin, iPieceGap - 1) :
                            Random.Range(iPieceGap + 4, Bounds.xMax - 1);

                        for (int col = Bounds.xMin; col < Bounds.xMax; col++)
                        {
                            if (col < oPieceGap || col >= oPieceGap + 2)
                            {
                                SetTile(col, Bounds.yMin + 1);
                            }
                        }
                    }
                }
                break;

            case 3:
                {
                    int workingHeight = Mathf.Min(4, boardHeight - 1);
                    int patternChoice = Random.Range(0, 3);

                    switch (patternChoice)
                    {
                        case 0:
                            {
                                int tCenter = Random.Range(Bounds.xMin + 1, Bounds.xMax - 1);

                                for (int col = Bounds.xMin; col < Bounds.xMax; col++)
                                {
                                    if (col != tCenter)
                                    {
                                        SetTile(col, Bounds.yMin);
                                    }
                                }

                                if (workingHeight >= 2)
                                {
                                    for (int col = Bounds.xMin; col < Bounds.xMax; col++)
                                    {
                                        if (col < tCenter - 1 || col > tCenter + 1)
                                        {
                                            SetTile(col, Bounds.yMin + 1);
                                        }
                                    }
                                }
                            }
                            break;

                        case 1:
                            {
                                int iPieceStart = Random.Range(Bounds.xMin, Bounds.xMax - 3);

                                for (int col = Bounds.xMin; col < Bounds.xMax; col++)
                                {
                                    if (col < iPieceStart || col >= iPieceStart + 4)
                                    {
                                        SetTile(col, Bounds.yMin);
                                    }
                                }

                                if (workingHeight >= 3)
                                {
                                    int oPieceStart = iPieceStart >= Bounds.xMin + 2 ?
                                        Random.Range(Bounds.xMin, iPieceStart - 1) :
                                        Random.Range(iPieceStart + 4, Bounds.xMax - 1);

                                    for (int row = 1; row <= 2; row++)
                                    {
                                        for (int col = Bounds.xMin; col < Bounds.xMax; col++)
                                        {
                                            if (col < oPieceStart || col >= oPieceStart + 2)
                                            {
                                                SetTile(col, Bounds.yMin + row);
                                            }
                                        }
                                    }
                                }
                            }
                            break;

                        case 2:
                            {
                                int iPieceGap = Random.Range(Bounds.xMin, Bounds.xMax - 3);
                                for (int col = Bounds.xMin; col < Bounds.xMax; col++)
                                {
                                    if (col < iPieceGap || col >= iPieceGap + 4)
                                    {
                                        SetTile(col, Bounds.yMin);
                                    }
                                }

                                if (workingHeight >= 2)
                                {
                                    int oPieceGap = Random.Range(Bounds.xMin, Bounds.xMax - 1);
                                    for (int col = Bounds.xMin; col < Bounds.xMax; col++)
                                    {
                                        if (col < oPieceGap || col >= oPieceGap + 2)
                                        {
                                            SetTile(col, Bounds.yMin + 1);
                                        }
                                    }
                                }

                                if (workingHeight >= 4)
                                {
                                    int tCenter = Random.Range(Bounds.xMin + 1, Bounds.xMax - 1);

                                    for (int col = Bounds.xMin; col < Bounds.xMax; col++)
                                    {
                                        if (col != tCenter)
                                        {
                                            SetTile(col, Bounds.yMin + 2);
                                        }
                                    }

                                    for (int col = Bounds.xMin; col < Bounds.xMax; col++)
                                    {
                                        if (col < tCenter - 1 || col > tCenter + 1)
                                        {
                                            SetTile(col, Bounds.yMin + 3);
                                        }
                                    }
                                }
                            }
                            break;
                    }

                    Debug.Log($"[Board] Applied guided_stacking pattern {patternChoice}");
                }
                break;

            case 4:
                {
                    int workingHeight = Mathf.Min(6, boardHeight - 1);

                    for (int row = 0; row < workingHeight; row++)
                    {
                        int currentRow = Bounds.yMin + row;
                        int blocksToPlace = (Bounds.xMax - Bounds.xMin) - (row + 2);

                        if (blocksToPlace > 0)
                        {
                            int gapSize = Random.Range(2, 4);
                            int gapStart = Random.Range(Bounds.xMin, Bounds.xMax - gapSize);

                            for (int col = Bounds.xMin; col < Bounds.xMax; col++)
                            {
                                if (col < gapStart || col >= gapStart + gapSize)
                                {
                                    SetTile(col, currentRow);
                                }
                            }
                        }
                    }
                }
                break;

            case 5:
                {
                    int workingHeight = Mathf.Min(8, boardHeight - 2);

                    for (int col = Bounds.xMin; col < Bounds.xMax; col++)
                    {
                        bool isWell = (col - Bounds.xMin) % 5 == 2;
                        if (!isWell)
                        {
                            for (int wellRow = 0; wellRow < 3 && wellRow < workingHeight; wellRow++)
                            {
                                SetTile(col, Bounds.yMin + wellRow);
                            }
                        }
                    }

                    if (workingHeight >= 5)
                    {
                        for (int col = Bounds.xMin; col < Bounds.xMax; col++)
                        {
                            int pattern = (col - Bounds.xMin) % 4;
                            if (pattern == 0 || pattern == 3)
                            {
                                SetTile(col, Bounds.yMin + 3);
                                if (pattern == 0 && workingHeight >= 6)
                                {
                                    SetTile(col, Bounds.yMin + 4);
                                }
                            }
                        }
                    }

                    if (workingHeight >= 7)
                    {
                        int structures = Random.Range(2, 4);
                        for (int s = 0; s < structures; s++)
                        {
                            int structStart = Random.Range(Bounds.xMin, Bounds.xMax - 2);
                            int structWidth = Random.Range(2, 4);

                            for (int w = 0; w < structWidth && structStart + w < Bounds.xMax; w++)
                            {
                                SetTile(structStart + w, Bounds.yMin + 5 + s);
                            }
                        }
                    }
                }
                break;

            default:
                Debug.LogWarning($"[Board] Unknown board_preset {preset}, using empty board");
                break;
        }

        Debug.Log($"[Board] Applied board_preset {preset} with height {boardHeight}");
    }

    private void SetTile(int x, int y)
    {
        if (x >= Bounds.xMin && x < Bounds.xMax && y >= Bounds.yMin && y < Bounds.yMax)
        {
            tilemap.SetTile(new Vector3Int(x, y, 0), tetrominoes[0].tile);
        }
    }

    private void Update()
    {
        if (playerScoreToDisplay != null)
        {
            this.playerScoreToDisplay.text = this.playerScore.ToString();
        }

        CheckForBoardHeightChange();
    }

    [Header("Curriculum Settings")]
    public int maxTetrominoTypes = 7;
    public float curriculumBoardHeight = 20f;

    public void GenerateNextPiece()
    {
        TetrisMLAgent mlAgent = this.inputController as TetrisMLAgent;
        int allowedTypes = (mlAgent != null) ? mlAgent.allowedTetrominoTypes : 7;

        int maxIndex = Mathf.Min(allowedTypes, this.tetrominoes.Length);
        int random = Random.Range(0, maxIndex);
        this.nextPieceData = this.tetrominoes[random];

        if (nextPieceDisplay != null)
        {
            nextPieceDisplay.DisplayNextPiece(this.nextPieceData);
        }
    }

    private void CheckForBoardHeightChange()
    {
        TetrisMLAgent mlAgent = this.inputController as TetrisMLAgent;
        if (mlAgent != null)
        {
            int currentHeight = (int)mlAgent.curriculumBoardHeight;

            if (currentHeight != lastBoardHeight)
            {
                Debug.Log($"Board height changing from {lastBoardHeight} to {currentHeight}");
                lastBoardHeight = currentHeight;

                UpdateGridVisualization();
                ClearBoard();
                ApplyCurriculumBoardPreset();

                // Clear the old piece completely
                if (activePiece != null)
                {
                    Clear(activePiece);
                    // Reset the ML agent's piece reference
                    mlAgent.SetCurrentPiece(null);
                }

                // Spawn a new piece with the updated parameters
                SpawnPiece();
            }
        }
    }

    private void UpdateGridVisualization()
    {
        if (gridSpriteRenderer == null) return;

        RectInt bounds = this.Bounds;

        float originalWidth = 10f;
        float originalHeight = 20f;

        float scaleX = boardSize.x / originalWidth;
        float scaleY = bounds.height / originalHeight;

        gridSpriteRenderer.transform.localScale = new Vector3(scaleX, scaleY, 1f);

        Vector3 gridCenter = new Vector3(bounds.center.x, bounds.center.y, gridSpriteRenderer.transform.position.z);
        gridSpriteRenderer.transform.position = gridCenter;
    }

    public void SpawnPiece()
    {
        int random = Random.Range(0, this.tetrominoes.Length);
        TetrominoData data = this.tetrominoes[random];

        TetrominoData pieceToUse = this.nextPieceData.Equals(default(TetrominoData)) ? data : this.nextPieceData;

        // Clear the old piece reference first
        if (this.activePiece != null)
        {
            Clear(this.activePiece);
        }

        this.activePiece.Initialize(this, this.spawnPosition, pieceToUse, this.inputController);

        GenerateNextPiece();

        if (IsValidPosition(this.activePiece, this.spawnPosition))
        {
            Set(this.activePiece);
            
            // Inform ML agent about the new piece AFTER it's been set on the board
            TetrisMLAgent mlAgent = this.inputController as TetrisMLAgent;
            if (mlAgent != null)
            {
                Debug.Log($"Board.SpawnPiece: Notifying ML Agent of new piece {pieceToUse.tetromino}");
                mlAgent.SetCurrentPiece(this.activePiece);
            }
        }
        else
        {
            Data.PlayerScore = this.playerScore;
            GameOver();
        }
    }

    private void GameOver()
    {
        TetrisMLAgent mlAgent = this.inputController as TetrisMLAgent;
        if (mlAgent != null)
        {
            Debug.Log("Board.GameOver: Notifying ML Agent of game over");
            mlAgent.OnGameOver();
            return;
        }

        // Non-ML game over logic
        Data.PlayerScore = this.playerScore;
        SceneManager.LoadScene(2);
    }

    private void ClearBoard()
    {
        RectInt bounds = this.Bounds;
        for (int row = bounds.yMin; row < bounds.yMax; row++)
        {
            for (int col = bounds.xMin; col < bounds.xMax; col++)
            {
                Vector3Int position = new Vector3Int(col, row, 0);
                this.tilemap.SetTile(position, null);
            }
        }
    }

    // ... [Keep all other methods exactly as they are] ...
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

        scoreSpeedBonus = Mathf.Min(playerScore / 10000f, 1.0f);

        if (linesCleared >= 4)
        {
            StartCoroutine(ApplyTemporarySpeedBoost(0.2f, 3f));
        }
        else if (linesCleared >= 2)
        {
            StartCoroutine(ApplyTemporarySpeedBoost(0.1f, 2f));
        }
    }

    private float temporarySpeedBonus = 0f;

    private IEnumerator ApplyTemporarySpeedBoost(float amount, float duration)
    {
        temporarySpeedBonus += amount;
        yield return new WaitForSeconds(duration);
        temporarySpeedBonus -= amount;
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

    // ... [Include all other helper methods like CountHoles, GetHolePositions, etc.] ...
    public float CalculateStackHeight()
    {
        int maxHeight = 0;
        for (int x = Bounds.xMin; x < Bounds.xMax; x++)
        {
            for (int y = Bounds.yMax - 1; y >= Bounds.yMin; y--)
            {
                if (tilemap.HasTile(new Vector3Int(x, y, 0)))
                {
                    maxHeight = Mathf.Max(maxHeight, Bounds.yMax - y);
                    break;
                }
            }
        }
        return maxHeight;
    }

    public List<Vector2Int> GetHolePositions()
    {
        List<Vector2Int> holes = new List<Vector2Int>();
        RectInt bounds = this.Bounds;

        for (int x = bounds.xMin; x < bounds.xMax; x++)
        {
            bool blockAbove = false;
            for (int y = bounds.yMax - 1; y >= bounds.yMin; y--)
            {
                if (tilemap.HasTile(new Vector3Int(x, y, 0)))
                {
                    blockAbove = true;
                }
                else if (blockAbove)
                {
                    holes.Add(new Vector2Int(x, y));
                }
            }
        }
        return holes;
    }

    public int CountHoles()
    {
        int holes = 0;
        for (int x = Bounds.xMin; x < Bounds.xMax; x++)
        {
            bool blockFound = false;
            for (int y = Bounds.yMax - 1; y >= Bounds.yMin; y--)
            {
                if (tilemap.HasTile(new Vector3Int(x, y, 0)))
                {
                    blockFound = true;
                }
                else if (blockFound)
                {
                    holes++;
                }
            }
        }
        return holes;
    }

    public int[] GetRowFillCounts()
    {
        int[] rowFills = new int[Bounds.size.y];

        for (int y = Bounds.yMin; y < Bounds.yMax; y++)
        {
            int fillCount = 0;
            for (int x = Bounds.xMin; x < Bounds.xMax; x++)
            {
                if (tilemap.HasTile(new Vector3Int(x, y, 0)))
                {
                    fillCount++;
                }
            }
            rowFills[y - Bounds.yMin] = fillCount;
        }

        return rowFills;
    }

    public bool IsPerfectClear()
    {
        RectInt bounds = this.Bounds;

        for (int x = bounds.xMin; x < bounds.xMax; x++)
        {
            for (int y = bounds.yMin; y < bounds.yMax; y++)
            {
                if (tilemap.HasTile(new Vector3Int(x, y, 0)))
                {
                    return false;
                }
            }
        }

        return true;
    }

    public bool LastRotationWasUseless(Piece piece, Vector3Int originalPosition, Vector3Int[] originalCells)
    {
        Vector3Int[] rotatedCells = piece.cells;
        for (int i = 0; i < originalCells.Length; i++)
        {
            if (originalCells[i] + originalPosition != rotatedCells[i] + piece.position)
            {
                return false;
            }
        }
        return true;
    }

    public bool HasDeepWell(int depthThreshold = 4)
    {
        RectInt bounds = this.Bounds;

        for (int x = bounds.xMin + 1; x < bounds.xMax - 1; x++)
        {
            int currentDepth = 0;
            for (int y = bounds.yMin; y < bounds.yMax; y++)
            {
                bool centerEmpty = !tilemap.HasTile(new Vector3Int(x, y, 0));
                bool leftFilled = tilemap.HasTile(new Vector3Int(x - 1, y, 0));
                bool rightFilled = tilemap.HasTile(new Vector3Int(x + 1, y, 0));

                if (centerEmpty && leftFilled && rightFilled)
                {
                    currentDepth++;
                    if (currentDepth >= depthThreshold)
                    {
                        return true;
                    }
                }
                else
                {
                    currentDepth = 0;
                }
            }
        }

        return false;
    }
}