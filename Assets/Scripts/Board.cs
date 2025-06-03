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
            TetrisMLAgent mlAgent = this.inputController as TetrisMLAgent;
            if (mlAgent != null)
            {
                int currentHeight = (int)mlAgent.curriculumBoardHeight;
                // Adjust spawn position to be at the top of the current board height
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

        // Check if we're using ML-Agent as input controller
    }

    private void Start()
    {

        this.playerScore = 0;
        this.gameStartTime = Time.time;

        UpdateGridVisualization();

        // Only spawn a piece if all components are properly initialized
        if (activePiece != null && tetrominoes != null && tetrominoes.Length > 0)
        {
            SpawnPiece();
        }
        else
        {
            // Debug.LogError("Cannot spawn piece: Required components not initialized");
        }

        if (playerTagHolder != null)
        {
            this.playerTagHolder.text = playerTag;
        }
    }

    // Your board class
    public void ApplyCurriculumBoardPreset()
    {
        int preset = 0; // Default to empty board
        int boardHeight = 20; // Default board height

        TetrisMLAgent mlAgent = this.inputController as TetrisMLAgent;
        if (mlAgent != null)
        {
            preset = (int)mlAgent.curriculumBoardPreset;
            boardHeight = (int)mlAgent.curriculumBoardHeight;
        }

        ClearBoard(); // Always start clean

        // Adjust bounds based on curriculum board height
        int maxY = Bounds.yMin + boardHeight - 1;

        switch (preset)
        {
            case 0: // empty_board - Full Tetris game
                    // Empty board - no pre-configuration
                break;

            case 1: // minimal_pre_config - Single obvious I-piece placement
                {
                    // For small boards (6-8 height), use bottom row
                    // For larger boards, place higher to avoid immediate danger
                    int targetRow = boardHeight <= 8 ? Bounds.yMin : Bounds.yMin + 1;

                    // Create I-piece gap (4 spaces) - NEVER fill completely
                    int gapStart = Random.Range(Bounds.xMin, Bounds.xMax - 3);
                    int gapEnd = gapStart + 4;

                    for (int col = Bounds.xMin; col < Bounds.xMax; col++)
                    {
                        if (col < gapStart || col >= gapEnd)
                        {
                            SetTile(col, targetRow);
                        }
                    }

                    // Ensure we never create a complete line
                    if (gapEnd - gapStart >= Bounds.xMax - Bounds.xMin)
                    {
                        // If gap would be entire row, add one tile
                        SetTile(gapStart, targetRow);
                    }
                }
                break;

            case 2: // basic_placement - Two-piece scenarios
                {
                    int workingHeight = Mathf.Min(3, boardHeight - 1);

                    // Bottom row: I-piece opportunity (4 gaps)
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
                        // Second row: O-piece opportunity (2 gaps) - different position
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

            case 3: // guided_stacking - Multi-piece practice (I, O, T pieces)
                {
                    int workingHeight = Mathf.Min(4, boardHeight - 1);

                    // Create multiple placement opportunities for different piece types
                    int patternChoice = Random.Range(0, 3);

                    switch (patternChoice)
                    {
                        case 0: // T-piece focused pattern
                            {
                                int tCenter = Random.Range(Bounds.xMin + 1, Bounds.xMax - 1);

                                // Bottom row: fill everything except center gap for T-piece stem
                                for (int col = Bounds.xMin; col < Bounds.xMax; col++)
                                {
                                    if (col != tCenter)
                                    {
                                        SetTile(col, Bounds.yMin);
                                    }
                                }

                                if (workingHeight >= 2)
                                {
                                    // Second row: leave 3-wide gap for T-piece arms
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

                        case 1: // I-piece + O-piece pattern
                            {
                                // Create both a 4-wide gap (I-piece) and 2x2 area (O-piece)
                                int iPieceStart = Random.Range(Bounds.xMin, Bounds.xMax - 3);

                                // Bottom row: I-piece gap
                                for (int col = Bounds.xMin; col < Bounds.xMax; col++)
                                {
                                    if (col < iPieceStart || col >= iPieceStart + 4)
                                    {
                                        SetTile(col, Bounds.yMin);
                                    }
                                }

                                if (workingHeight >= 3)
                                {
                                    // Create O-piece opportunity in a different area
                                    int oPieceStart = iPieceStart >= Bounds.xMin + 2 ?
                                        Random.Range(Bounds.xMin, iPieceStart - 1) :
                                        Random.Range(iPieceStart + 4, Bounds.xMax - 1);

                                    // Rows 1 and 2: create 2x2 gap for O-piece
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

                        case 2: // Mixed opportunities - all three pieces
                            {
                                // Bottom: partial fill with I-piece gap
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
                                    // Middle: O-piece opportunity
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
                                    // Top: T-piece opportunity
                                    int tCenter = Random.Range(Bounds.xMin + 1, Bounds.xMax - 1);

                                    // Create inverted T cavity
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

            case 4: // structured_challenge - Multi-piece strategy
                {
                    int workingHeight = Mathf.Min(6, boardHeight - 1);

                    // Create stepped structure with placement opportunities
                    for (int row = 0; row < workingHeight; row++)
                    {
                        int currentRow = Bounds.yMin + row;
                        int blocksToPlace = (Bounds.xMax - Bounds.xMin) - (row + 2); // Fewer blocks each row up

                        if (blocksToPlace > 0)
                        {
                            // Distribute blocks with strategic gaps
                            int gapSize = Random.Range(2, 4); // Gap for different pieces
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

            case 5: // complex_scenario - Advanced multi-level challenge
                {
                    int workingHeight = Mathf.Min(8, boardHeight - 2); // Leave room at top

                    // Create complex but solvable structure
                    // Bottom foundation with wells
                    for (int col = Bounds.xMin; col < Bounds.xMax; col++)
                    {
                        // Create wells every 5 columns for I-pieces
                        bool isWell = (col - Bounds.xMin) % 5 == 2;
                        if (!isWell)
                        {
                            // Fill bottom 3 rows of non-well columns
                            for (int wellRow = 0; wellRow < 3 && wellRow < workingHeight; wellRow++)
                            {
                                SetTile(col, Bounds.yMin + wellRow);
                            }
                        }
                    }

                    // Mid-level irregular pattern
                    if (workingHeight >= 5)
                    {
                        for (int col = Bounds.xMin; col < Bounds.xMax; col++)
                        {
                            int pattern = (col - Bounds.xMin) % 4;
                            if (pattern == 0 || pattern == 3) // Irregular placement
                            {
                                SetTile(col, Bounds.yMin + 3);
                                if (pattern == 0 && workingHeight >= 6)
                                {
                                    SetTile(col, Bounds.yMin + 4);
                                }
                            }
                        }
                    }

                    // Top level structures
                    if (workingHeight >= 7)
                    {
                        // Create scattered placement opportunities
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
        // if (fireBorderController != null)
        // {
        //     fireBorderController.SetGameSpeed(1f / CurrentDropRate);
        // }
    }

    [Header("Curriculum Settings")]
    public int maxTetrominoTypes = 7;
    public float curriculumBoardHeight = 20f;

    public void GenerateNextPiece()
    {
        // Get curriculum parameters from ML agent
        TetrisMLAgent mlAgent = this.inputController as TetrisMLAgent;
        int allowedTypes = (mlAgent != null) ? mlAgent.allowedTetrominoTypes : 7;

        // Limit piece selection based on curriculum
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
            // Debug.Log("Curriculum Board Height: " + mlAgent.curriculumBoardHeight);
            int currentHeight = (int)mlAgent.curriculumBoardHeight;
            // Debug.Log("Here:" + currentHeight + ":" + lastBoardHeight);

            if (currentHeight != lastBoardHeight)
            {
                // Debug.Log("In Here");

                // Debug.Log("LastBoardHeight:" + lastBoardHeight);
                lastBoardHeight = currentHeight;
                // Debug.Log("CurrentBoardHeight:" + lastBoardHeight);

                UpdateGridVisualization();

                ClearBoard();
                ApplyCurriculumBoardPreset();

                // Reset the active piece to reflect curriculum change:
                if (activePiece != null)
                {
                    // Clear the piece tiles from tilemap
                    Clear(activePiece);
                }

                // Spawn a new piece using updated curriculum parameters
                SpawnPiece();
            }
        }
    }
    private void UpdateGridVisualization()
    {
        if (gridSpriteRenderer == null) return;

        RectInt bounds = this.Bounds;

        // Calculate the scale needed to match the current board size
        // Assuming the original grid sprite is designed for 10x20
        float originalWidth = 10f;
        float originalHeight = 20f;

        float scaleX = boardSize.x / originalWidth;
        float scaleY = bounds.height / originalHeight;

        // Apply the new scale
        gridSpriteRenderer.transform.localScale = new Vector3(scaleX, scaleY, 1f);

        // Position the grid to match the board bounds
        Vector3 gridCenter = new Vector3(bounds.center.x, bounds.center.y, gridSpriteRenderer.transform.position.z);
        gridSpriteRenderer.transform.position = gridCenter;
    }


    public void SpawnPiece()
    {
        int random = Random.Range(0, this.tetrominoes.Length);
        TetrominoData data = this.tetrominoes[random];

        TetrominoData pieceToUse = this.nextPieceData.Equals(default(TetrominoData)) ? data : this.nextPieceData;

        this.activePiece.Initialize(this, this.spawnPosition, pieceToUse, this.inputController);

        // Inform ML agent about the new piece if applicable
        TetrisMLAgent mlAgent = this.inputController as TetrisMLAgent;
        if (mlAgent != null)
        {
            mlAgent.SetCurrentPiece(this.activePiece);
        }

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

    public float CalculateStackHeight()
    {
        int maxHeight = 0;
        for (int x = Bounds.xMin; x < Bounds.xMax; x++)
        {
            for (int y = Bounds.yMax - 1; y >= Bounds.yMin; y--)
            {
                if (tilemap.HasTile(new Vector3Int(x, y, 0))) // Found a filled cell
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
                    // Empty cell below a block is a hole
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
    private void GameOver()
    {
        // Notify ML agent if this is an ML agent-controlled board
        TetrisMLAgent mlAgent = this.inputController as TetrisMLAgent;
        if (mlAgent != null)
        {
            mlAgent.OnGameOver();

            // If in ML training mode, reset the game instead of loading the game over scene

            StartCoroutine(ResetGameForMLTraining());
            return;
        }

        // Store the score for the game over screen
        Data.PlayerScore = this.playerScore;

        // Load game over scene only if not in ML training
        SceneManager.LoadScene(2);
    }

    private IEnumerator ResetGameForMLTraining()
    {
        // Short delay to ensure ML Agent has processed the game over
        yield return new WaitForSeconds(0.1f);

        // Reset the board
        ClearBoard();
        ApplyCurriculumBoardPreset();
        this.playerScore = 0;
        this.gameStartTime = Time.time;

        // Spawn a new piece to start the game again
        SpawnPiece();
    }

    private void ClearBoard()
    {
        // Clear the entire tilemap
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
                Debug.LogWarning($"IsValidPosition: Out of bounds {tilePosition}");
                return false;
            }

            if (this.tilemap.HasTile(tilePosition))
            {
                Debug.LogWarning($"IsValidPosition: Tile already occupied at {tilePosition}");
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