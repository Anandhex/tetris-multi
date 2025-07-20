using UnityEngine;
using System.Collections.Generic;
using System.Linq;
using UnityEngine.Tilemaps;
using TMPro;
using System.Collections;

public class PowerUpManager : MonoBehaviour
{
    [Header("Power-up Settings")]
    public PowerUp[] availablePowerUps;
    public float powerUpChance = 0.8f;

    [Header("Time Challenge Settings")]
    public float timeWindowMinutes = 4f; // 4 minutes window
    public int requiredLinesInWindow = 8; // 8 lines needed in 4 minutes

    [Header("UI References")]
    public GameObject powerUpSlotPrefab;
    public Transform powerUpInventoryParent;

    [Header("Power-up HUD UI")]
    public TextMeshProUGUI inventoryText;
    public TextMeshProUGUI progressText;
    public TextMeshProUGUI keysText;

    [Header("Audio")]
    public AudioClip powerUpObtainedSound;
    public AudioClip powerUpUsedSound;

    // Inventory by power-up type
    public Dictionary<PowerUpType, int> powerUpInventory = new Dictionary<PowerUpType, int>();
    private Board ownerBoard;

    // Time challenge tracking
    private List<float> linesClearedTimes = new List<float>();
    private float gameStartTime;

    public List<PowerupKeyMapping> powerupKeyMappings;

    // Logging control
    private float lastLogTime = 0f;
    private float logInterval = 2f; // Log every 2 seconds
    private int lastLinesCount = 0;
    private int bombTestColumn = -5;
    private int wildcardTestColumn = -4;

    private Board opponentBoard;

    public void SetupPowerupManager(Board userBoard, Board opponentBoard, List<PowerupKeyMapping> powerupKeyMappings)
    {
        this.ownerBoard = userBoard;
        this.opponentBoard = opponentBoard;
        this.powerupKeyMappings = powerupKeyMappings;

        gameStartTime = Time.time;

        string playerTag = ownerBoard != null ? ownerBoard.playerTag : "Unknown";
        Debug.Log($"🎮 === POWER-UP MANAGER INITIALIZED (Player: {playerTag}) ===");
        Debug.Log($"⚙️ Settings: PowerUp Chance = {powerUpChance * 100}%");
        Debug.Log($"⏰ Time Window: {timeWindowMinutes} minutes for {requiredLinesInWindow} lines");
        Debug.Log($"🎯 Constraint 1: Clear 2+ lines = {powerUpChance * 100}% base chance");
        Debug.Log($"🎯 Constraint 2: Clear {requiredLinesInWindow} lines in {timeWindowMinutes} minutes = GUARANTEED");
        Debug.Log($"🔗 Opponent Board: {(opponentBoard != null ? "Available" : "None")}");

        // Initialize inventory
        InitializeInventory();

        // Initialize UI
        InitializeUI();

        // Initial status log
        LogCurrentStatus();
    }

    private void InitializeInventory()
    {
        string playerTag = ownerBoard != null ? ownerBoard.playerTag : "Unknown";
        Debug.Log($"📦 === INITIALIZING INVENTORY (Player: {playerTag}) ===");
        
        powerUpInventory[PowerUpType.LineBlaster] = 0;
        powerUpInventory[PowerUpType.Gravity] = 0;
        powerUpInventory[PowerUpType.Bomb] = 0;
        
        if (opponentBoard != null)
        {
            powerUpInventory[PowerUpType.WildCard] = 0;
            Debug.Log("🃏 WildCard initialized (opponent board available)");
        }
        else
        {
            Debug.Log("❌ WildCard NOT initialized (no opponent board)");
        }

        LogFullInventory("After initialization");
    }

    private void InitializeUI()
    {
        // Set up the keys instruction (this won't change)
        if (keysText != null)
        {
            keysText.text = "Keys:\n1=LineBlaster\n2=Gravity\n3=Bomb";
            Debug.Log("✅ Keys UI initialized");
            if (opponentBoard != null)
            {
                keysText.text += "\n4=WildCard";
            }
        }
        else
        {
            Debug.LogWarning("⚠️ Keys UI not connected!");
        }

        // Update the dynamic UI elements
        UpdateUI();

        // Check UI connections
        Debug.Log($"🔗 UI Connections: Inventory={inventoryText != null}, Progress={progressText != null}, Keys={keysText != null}");
    }

    private void Update()
    {
        // ADD NULL CHECK HERE:
        if (this.powerupKeyMappings != null)
        {
            foreach (var mapping in this.powerupKeyMappings)
            {
                if (Input.GetKeyDown(mapping.key))
                {
                    Debug.Log($"🔑 {mapping.key} pressed - attempting {mapping.powerupType}");
                    UsePowerUp(mapping.powerupType);
                }
            }
        }

        if (Input.GetKeyDown(KeyCode.Semicolon)) // ; key for bomb
        {
            Debug.Log("🧪 TESTING: Bomb at column 0");
            UsePowerUp(PowerUpType.Bomb, 0); // ← CHANGED: Use merged function with column parameter
        }

        if (Input.GetKeyDown(KeyCode.Quote)) // ' key for wildcard
        {
            Debug.Log($"🧪 TESTING: Wildcard at column {wildcardTestColumn}");
            if (opponentBoard != null)
            {
                Debug.Log($"🔍 Opponent board bounds: {opponentBoard.Bounds}");
            }
            UsePowerUp(PowerUpType.WildCard, wildcardTestColumn); // ← CHANGED: Use merged function with column parameter
            
            wildcardTestColumn++;
            if (wildcardTestColumn > 3) wildcardTestColumn = -4;
        }

        // Clean up old line clear times (outside the time window)
        CleanupOldLineTimes();

        // Update UI every frame
        UpdateUI();

        // Periodic status logging
        if (Time.time - lastLogTime > logInterval)
        {
            LogCurrentStatus();
            lastLogTime = Time.time;
        }

        // Check for lines count change
        if (linesClearedTimes.Count != lastLinesCount)
        {
            Debug.Log($"📊 Lines count changed: {lastLinesCount} → {linesClearedTimes.Count}");
            lastLinesCount = linesClearedTimes.Count;
        }
    }

    private void LogCurrentStatus()
    {
        string playerTag = ownerBoard != null ? ownerBoard.playerTag : "Unknown";
        float timeElapsed = Time.time - gameStartTime;
        float timeRemaining = (timeWindowMinutes * 60f) - timeElapsed;
        int linesInWindow = linesClearedTimes.Count;

        Debug.Log($"📋 === CURRENT STATUS (Player: {playerTag}) ===");
        Debug.Log($"⏰ Time: {timeElapsed:F1}s elapsed, {Mathf.Max(0, timeRemaining):F1}s remaining");
        Debug.Log($"📊 Progress: {linesInWindow}/{requiredLinesInWindow} lines in window");
        
        LogFullInventory("Current status");
        
        // Calculate progress percentage
        float progressPercent = (float)linesInWindow / requiredLinesInWindow * 100f;
        Debug.Log($"📈 Progress: {progressPercent:F1}% towards time challenge");

        // Time window status
        if (timeRemaining > 0)
        {
            Debug.Log($"⏳ Time Challenge: {timeRemaining:F0}s remaining");
        }
        else
        {
            Debug.Log("🔄 Time window expired - ready for reset");
        }

        Debug.Log("=========================");
    }

    private void CleanupOldLineTimes()
    {
        float currentTime = Time.time;
        float timeWindowSeconds = timeWindowMinutes * 60f;

        int beforeCount = linesClearedTimes.Count;
        linesClearedTimes.RemoveAll(time => currentTime - time > timeWindowSeconds);
        int afterCount = linesClearedTimes.Count;

        if (beforeCount != afterCount)
        {
            string playerTag = ownerBoard != null ? ownerBoard.playerTag : "Unknown";
            Debug.Log($"🧹 Cleaned up {beforeCount - afterCount} old line records (Player: {playerTag})");
            Debug.Log($"📊 Current window now has {afterCount} lines");
        }
    }

    private void UpdateUI()
    {
        // Update inventory display
        if (inventoryText != null)
        {
            // Safe dictionary access with default values
            int lineBlasterCount = powerUpInventory.ContainsKey(PowerUpType.LineBlaster) ? powerUpInventory[PowerUpType.LineBlaster] : 0;
            int gravityCount = powerUpInventory.ContainsKey(PowerUpType.Gravity) ? powerUpInventory[PowerUpType.Gravity] : 0;
            int bombCount = powerUpInventory.ContainsKey(PowerUpType.Bomb) ? powerUpInventory[PowerUpType.Bomb] : 0;
            
            string inventoryDisplay = $"Power-ups:\n[LineBlaster:{lineBlasterCount}]\n[Gravity:{gravityCount}]\n[Bomb:{bombCount}]";
            
            if (opponentBoard != null)
            {
                int wildcardCount = powerUpInventory.ContainsKey(PowerUpType.WildCard) ? powerUpInventory[PowerUpType.WildCard] : 0;
                inventoryDisplay += $"\n[WildCard:{wildcardCount}]";
            }
            
            inventoryText.text = inventoryDisplay;
        }

        // Update progress display
        if (progressText != null)
        {
            float timeRemaining = (gameStartTime + timeWindowMinutes * 60f) - Time.time;
            int linesInWindow = linesClearedTimes.Count;

            string progressDisplay = $"Progress: {linesInWindow}/{requiredLinesInWindow} lines";

            if (timeRemaining > 0)
            {
                int minutesLeft = Mathf.FloorToInt(timeRemaining / 60f);
                int secondsLeft = Mathf.FloorToInt(timeRemaining % 60f);
                progressDisplay += $"\n({minutesLeft}:{secondsLeft:00} remaining)";
            }
            else
            {
                progressDisplay += "\n(New window started)";
                // Auto-reset if time window expired
                if (linesInWindow == 0)
                {
                    Debug.Log("🔄 Auto-resetting time window");
                    gameStartTime = Time.time;
                }
            }

            // Add progress bar visual
            float progressPercentage = (float)linesInWindow / requiredLinesInWindow;
            string progressBar = CreateProgressBar(progressPercentage);
            progressDisplay = progressBar + "\n\n " + progressDisplay;

            progressText.text = progressDisplay;
        }
    }

    private string CreateProgressBar(float percentage)
    {
        int totalBars = 10;
        int filledBars = Mathf.FloorToInt(percentage * totalBars);

        string bar = "[";
        for (int i = 0; i < totalBars; i++)
        {
            if (i < filledBars)
                bar += "█";
            else
                bar += "░";
        }
        bar += "]";

        return bar;
    }

    public void UsePowerUp(PowerUpType type, int targetColumn = -1, int targetRow = -1)
    {
        string playerTag = ownerBoard != null ? ownerBoard.playerTag : "Unknown";
        Debug.Log($"🎯 === ATTEMPTING TO USE {type.ToString().ToUpper()} (Player: {playerTag}) ===");
        
        // Add column info to log if specified
        if (targetColumn != -1)
        {
            Debug.Log($"🎯 Target Column: {targetColumn}");
        }

        if (powerUpInventory.ContainsKey(type) && powerUpInventory[type] > 0)
        {
            int beforeCount = powerUpInventory[type];
            powerUpInventory[type]--;
            int afterCount = powerUpInventory[type];

            Debug.Log($"✅ {type} used successfully by {playerTag}! Count: {beforeCount} → {afterCount}");

            SafeExecutePowerUp(type, targetColumn);
            UpdatePowerUpUI();

            // Log updated inventory
            LogFullInventory($"After using {type}");
        }
        else
        {
            int currentCount = powerUpInventory.ContainsKey(type) ? powerUpInventory[type] : 0;
            Debug.Log($"❌ {playerTag} cannot use {type}! Current count: {currentCount}");
            LogFullInventory($"Failed attempt to use {type}");
        }

        Debug.Log($"=== END POWER-UP USAGE ({playerTag}) ===");
    }

    private void SafeExecutePowerUp(PowerUpType type, int targetColumn = -1)
    {
        string playerTag = ownerBoard != null ? ownerBoard.playerTag : "Unknown";
        Debug.Log($"🔧 {playerTag} executing {type} power-up...");

        // Add column info to log if specified
        if (targetColumn != -1)
        {
            Debug.Log($"🔧 {playerTag} target Column: {targetColumn}");
        }

        // ✅ UPDATED: Enhanced bomb handling with detailed logging
        if (type == PowerUpType.Bomb)
        {
            Debug.Log($"💣 {playerTag} bomb powerup detected!");
            if (targetColumn != -1)
            {
                Debug.Log($"💣 {playerTag} using bomb at specific column: {targetColumn}");
                ExecuteBombAtColumn(targetColumn);
            }
            else
            {
                Debug.Log($"💣 {playerTag} using bomb at active piece position");
                ExecuteBombImproved(); // Uses active piece position
            }
            return;
        }

        if (type == PowerUpType.WildCard && this.opponentBoard != null)
        {
            if (targetColumn != -1)
            {
                ExecuteWildCardAtColumn(targetColumn);
            }
            else
            {
                ReplaceOpponentPieceWithWildcard(); // Uses active piece
            }
            return;
        }

        // For other power-ups, clear active piece temporarily
        bool hadActivePiece = ownerBoard.activePiece != null;
        if (hadActivePiece)
        {
            Debug.Log("🔄 Temporarily clearing active piece");
            ownerBoard.Clear(ownerBoard.activePiece);
        }

        // Execute the power-up (LineBlaster and Gravity don't need column targeting)
        switch (type)
        {
            case PowerUpType.LineBlaster:
                ExecuteLineBlaster();
                break;
            case PowerUpType.Gravity:
                ExecuteGravity();
                break;
        }

        // Put active piece back
        if (hadActivePiece && ownerBoard.activePiece != null)
        {
            if (ownerBoard.IsValidPosition(ownerBoard.activePiece, ownerBoard.activePiece.position))
            {
                ownerBoard.Set(ownerBoard.activePiece);
                Debug.Log("✅ Active piece restored to original position");
            }
            else
            {
                ownerBoard.activePiece.position = new Vector3Int(ownerBoard.activePiece.position.x, ownerBoard.activePiece.position.y + 1, 0);
                if (ownerBoard.IsValidPosition(ownerBoard.activePiece, ownerBoard.activePiece.position))
                {
                    ownerBoard.Set(ownerBoard.activePiece);
                    Debug.Log("✅ Active piece restored to adjusted position");
                }
                else
                {
                    Debug.Log("⚠️ Could not restore active piece - position invalid");
                }
            }
        }
    }

    // ✅ FIXED BOMB METHODS
    public void ExecuteBombAtColumn(int targetColumn)
    {
        string playerTag = ownerBoard != null ? ownerBoard.playerTag : "Unknown";
        Debug.Log($"💣 === {playerTag} EXECUTING BOMB AT COLUMN {targetColumn} ===");

        // Validate column
        RectInt bounds = ownerBoard.Bounds;
        if (targetColumn < bounds.xMin || targetColumn >= bounds.xMax)
        {
            Debug.LogError($"❌ Invalid column {targetColumn}. Valid range: {bounds.xMin} to {bounds.xMax - 1}");
            return;
        }

        // Find where the bomb should actually land
        Vector3Int bombPosition = FindBombLandingPosition(targetColumn);

        if (bombPosition.y < bounds.yMin)
        {
            Debug.LogWarning($"⚠️ Cannot place bomb in column {targetColumn} - column might be full");
            return;
        }

        Debug.Log($"💥 {playerTag} bomb will land at position: {bombPosition}");

        // Create and explode bomb immediately
        CreateAndExplodeBomb(bombPosition);
        
        Debug.Log($"💣 {playerTag} bomb execution complete!");
    }

    public void ExecuteBombImproved()
    {
        string playerTag = ownerBoard != null ? ownerBoard.playerTag : "Unknown";
        Debug.Log($"💣 === {playerTag} EXECUTING BOMB POWER-UP (Active Piece) ===");

        if (ownerBoard.activePiece != null)
        {
            Vector3Int bombCenter = ownerBoard.activePiece.position;
            Debug.Log($"💥 {playerTag} bomb centered on active piece at: {bombCenter}");

            // Clear the active piece first
            ownerBoard.Clear(ownerBoard.activePiece);

            // Count tiles before explosion
            int tilesBefore = CountTilesInArea(bombCenter);
            Debug.Log($"💣 {playerTag} tiles before explosion: {tilesBefore}");

            // Use the Board's explosion method
            ownerBoard.ExecuteBombExplosion(bombCenter);
            
            // Count tiles after explosion
            int tilesAfter = CountTilesInArea(bombCenter);
            int tilesCleared = tilesBefore - tilesAfter;
            
            Debug.Log($"💥 {playerTag} bomb explosion complete! Cleared {tilesCleared} tiles");
            
            // Spawn a new piece since we cleared the active one
            ownerBoard.SpawnPiece();
        }
        else
        {
            Debug.LogWarning($"⚠️ {playerTag} has no active piece to center bomb on!");
        }
    }

    private void CreateAndExplodeBomb(Vector3Int position)
    {
        string playerTag = ownerBoard != null ? ownerBoard.playerTag : "Unknown";
        RectInt bounds = ownerBoard.Bounds;
        
        // Make sure position is within bounds
        if (position.x < bounds.xMin || position.x >= bounds.xMax || 
            position.y < bounds.yMin || position.y >= bounds.yMax)
        {
            Debug.LogError($"💣 {playerTag} bomb position {position} is out of bounds {bounds}");
            return;
        }

        Debug.Log($"💥 {playerTag} placing and exploding bomb at {position}");
        
        // Count tiles before explosion
        int tilesBefore = CountTilesInArea(position);
        Debug.Log($"💣 {playerTag} tiles before explosion: {tilesBefore}");
        
        // Call the explosion function from Board
        ownerBoard.ExecuteBombExplosion(position);
        
        // Count tiles after explosion
        int tilesAfter = CountTilesInArea(position);
        int tilesCleared = tilesBefore - tilesAfter;
        
        Debug.Log($"💥 {playerTag} bomb explosion complete! Cleared {tilesCleared} tiles");
        Debug.Log($"📊 {playerTag} tiles after explosion: {tilesAfter}");
    }

    private Vector3Int FindBombLandingPosition(int targetColumn)
    {
        RectInt bounds = ownerBoard.Bounds;

        // Start from the top and find where the bomb would land if dropped
        for (int y = bounds.yMax - 1; y >= bounds.yMin; y--)
        {
            Vector3Int checkPos = new Vector3Int(targetColumn, y, 0);
            
            // If this position has a tile, bomb lands on top of it
            if (ownerBoard.tilemap.HasTile(checkPos))
            {
                Vector3Int landingPos = new Vector3Int(targetColumn, y + 1, 0);
                Debug.Log($"💣 Bomb lands on top of tile at {checkPos} → landing at {landingPos}");
                return landingPos;
            }
        }

        // If no tiles found, bomb lands at the bottom
        Vector3Int bottomPos = new Vector3Int(targetColumn, bounds.yMin, 0);
        Debug.Log($"💣 Empty column - bomb lands at bottom: {bottomPos}");
        return bottomPos;
    }

    private int CountTilesInArea(Vector3Int center)
    {
        int count = 0;
        for (int x = -1; x <= 1; x++)
        {
            for (int y = -1; y <= 1; y++)
            {
                Vector3Int pos = center + new Vector3Int(x, y, 0);
                if (ownerBoard.tilemap.HasTile(pos))
                {
                    count++;
                }
            }
        }
        return count;
    }

    // ✅ WILDCARD METHODS (unchanged - these work fine)
    public void ExecuteWildCardAtColumn(int targetColumn)
    {
        Debug.Log($"🃏 === EXECUTING WILDCARD AT COLUMN {targetColumn} ===");

        if (opponentBoard == null)
        {
            Debug.LogError("❌ Cannot use WildCard - no opponent board available");
            return;
        }

        // Validate column for opponent board
        RectInt bounds = opponentBoard.Bounds;
        if (targetColumn < bounds.xMin || targetColumn >= bounds.xMax)
        {
            Debug.LogError($"❌ Invalid column {targetColumn}. Valid range: {bounds.xMin} to {bounds.xMax - 1}");
            return;
        }

        // Find suitable position for 3x3 wildcard block
        Vector3Int wildcardPosition = FindWildcardPlacementPosition(targetColumn);

        if (wildcardPosition.y < bounds.yMin)
        {
            Debug.LogWarning($"⚠️ Cannot place wildcard in column {targetColumn} - not enough space");
            return;
        }

        Debug.Log($"🃏 Placing wildcard at position: {wildcardPosition}");

        // Create and place wildcard
        CreateAndPlaceWildcard(wildcardPosition);
    }

    private Vector3Int FindWildcardPlacementPosition(int targetColumn)
    {
        RectInt bounds = opponentBoard.Bounds;
        
        // Adjust center column to fit 3x3 within bounds
        int centerX = targetColumn;
        if (centerX - 1 < bounds.xMin) centerX = bounds.xMin + 1;
        if (centerX + 1 >= bounds.xMax) centerX = bounds.xMax - 2;
        
        // Find the actual surface height in the area where we want to place the 3x3 wildcard
        int highestSurface = bounds.yMin; // Start from bottom
        
        // Check the 3 columns where the wildcard will be placed
        for (int dx = -1; dx <= 1; dx++)
        {
            int checkColumn = centerX + dx;
            
            // Make sure column is within bounds
            if (checkColumn >= bounds.xMin && checkColumn < bounds.xMax)
            {
                // Find the highest tile in this column
                for (int y = bounds.yMax - 1; y >= bounds.yMin; y--)
                {
                    Vector3Int checkPos = new Vector3Int(checkColumn, y, 0);
                    if (opponentBoard.tilemap.HasTile(checkPos))
                    {
                        // Found a tile, so surface is one position above
                        highestSurface = Mathf.Max(highestSurface, y + 1);
                        break;
                    }
                }
            }
        }
        
        // Place the wildcard at the surface level (bottom of the 3x3 block)
        // Make sure it doesn't go above the board
        int placementY = Mathf.Min(highestSurface, bounds.yMax - 3);
        
        Vector3Int result = new Vector3Int(centerX, placementY, 0);
        Debug.Log($"🃏 Wildcard position calculated: {result}, surface height: {highestSurface}, bounds: {bounds}");
        
        return result;
    }

    private void CreateAndPlaceWildcard(Vector3Int centerPosition)
    {
        if (opponentBoard == null)
        {
            Debug.LogWarning("⚠️ No opponent board for wildcard");
            return;
        }
        
        RectInt bounds = opponentBoard.Bounds;
        int placed = 0;
        List<Vector3Int> placedPositions = new List<Vector3Int>();
        
        // Place 3x3 wildcard tiles with better bounds checking
        for (int dx = -1; dx <= 1; dx++)
        {
            for (int dy = -1; dy <= 1; dy++)
            {
                Vector3Int pos = centerPosition + new Vector3Int(dx, dy, 0);
                
                // Only place if within bounds and not too high
                if (pos.x >= bounds.xMin && pos.x < bounds.xMax && 
                    pos.y >= bounds.yMin && pos.y < bounds.yMax - 2) // Leave some space at top
                {
                    opponentBoard.tilemap.SetTile(pos, opponentBoard.bombTile);
                    placedPositions.Add(pos);
                    placed++;
                }
                else
                {
                    Debug.Log($"🚫 Skipping wildcard tile at {pos} - out of bounds or too high");
                }
            }
        }
        
        Debug.Log($"🃏 Wildcard placed {placed}/9 tiles at {centerPosition}");
        Debug.Log($"🎯 Placed at positions: {string.Join(", ", placedPositions)}");
        
        // Apply gravity effect to make wildcard settle properly
        if (placed > 0)
        {
            StartCoroutine(SettleWildcardBlocks(placedPositions));
        }
    }

    private System.Collections.IEnumerator SettleWildcardBlocks(List<Vector3Int> positions)
    {
        yield return new WaitForSeconds(0.1f); // Small delay

        bool blocksMoving = true;
        int maxIterations = 20; // Prevent infinite loops
        int iterations = 0;

        while (blocksMoving && iterations < maxIterations)
        {
            blocksMoving = false;
            iterations++;

            // Process blocks from bottom to top to avoid conflicts
            var sortedPositions = positions.OrderBy(p => p.y).ToList();

            for (int i = 0; i < sortedPositions.Count; i++)
            {
                Vector3Int currentPos = sortedPositions[i];
                Vector3Int belowPos = currentPos + Vector3Int.down;

                // Check if block can move down
                if (opponentBoard.tilemap.HasTile(currentPos) &&
                    !opponentBoard.tilemap.HasTile(belowPos) &&
                    belowPos.y >= opponentBoard.Bounds.yMin)
                {
                    // Move the tile down
                    TileBase tile = opponentBoard.tilemap.GetTile(currentPos);
                    opponentBoard.tilemap.SetTile(currentPos, null);
                    opponentBoard.tilemap.SetTile(belowPos, tile);

                    // Update position in our list
                    sortedPositions[i] = belowPos;
                    blocksMoving = true;

                    Debug.Log($"🃏 Wildcard block settled: {currentPos} → {belowPos}");
                }
            }

            // Update the main positions list
            positions.Clear();
            positions.AddRange(sortedPositions);

            yield return new WaitForSeconds(0.05f); // Small delay between settling steps
        }

        Debug.Log($"🃏 Wildcard settling complete after {iterations} iterations");
    }

    public void ReplaceOpponentPieceWithWildcard()
    {
        Piece oldPiece = opponentBoard.activePiece;
        Piece userPiece = ownerBoard.activePiece;
        if (oldPiece == null)
        {
            Debug.LogWarning("Opponent has no active piece to replace.");
            return;
        }

        // Save the input controllers
        var userInput = ownerBoard.inputController;
        var opponentInput = opponentBoard.inputController;

        // Clear old piece from tilemap
        opponentBoard.Clear(oldPiece);

        // Override cells with wildcard
        oldPiece.SetCells(Data.WildcardCells); // You may need to expose this
        oldPiece.tile = opponentBoard.bombTile;
        oldPiece.isBomb = false;
        oldPiece.inputController = userInput;
        userPiece.inputController = null;

        // Change input control: opponent board is now player-controlled
        opponentBoard.inputController = userInput;
        ownerBoard.inputController = null; // disable own input

        // When wildcard locks, restore everything
        oldPiece.OnLockComplete = () =>
        {
            opponentBoard.inputController = opponentInput;
            ownerBoard.inputController = userInput;
            userPiece.inputController = userInput;
        };

        // Redraw
        opponentBoard.Set(oldPiece);
    }

    // ✅ OTHER POWERUP METHODS (unchanged)
    public void ExecuteLineBlaster()
    {
        Debug.Log("⚡ === EXECUTING LINE BLASTER POWER-UP ===");

        RectInt bounds = ownerBoard.Bounds;
        Debug.Log($"🎯 Searching for bottom line in bounds: {bounds}");

        bool lineFound = false;
        for (int y = bounds.yMin; y < bounds.yMax; y++)
        {
            bool hasBlocks = false;
            int blockCount = 0;

            for (int x = bounds.xMin; x < bounds.xMax; x++)
            {
                if (ownerBoard.tilemap.HasTile(new Vector3Int(x, y, 0)))
                {
                    hasBlocks = true;
                    blockCount++;
                }
            }

            if (hasBlocks)
            {
                Debug.Log($"⚡ Found bottom line at y={y} with {blockCount} blocks");
                ClearLine(y);
                Debug.Log($"✅ LineBlaster cleared line {y}");
                lineFound = true;
                break;
            }
        }

        if (!lineFound)
        {
            Debug.Log("❌ No lines found to clear with LineBlaster");
        }
    }

    public void ExecuteGravity()
    {
        Debug.Log("🌍 === EXECUTING GRAVITY POWER-UP ===");

        RectInt bounds = ownerBoard.Bounds;
        int totalMoved = 0;

        Debug.Log($"🎯 Processing columns in bounds: {bounds}");

        for (int x = bounds.xMin; x < bounds.xMax; x++)
        {
            List<TileBase> column = new List<TileBase>();
            int originalTiles = 0;

            // Collect tiles in column
            for (int y = bounds.yMin; y < bounds.yMax; y++)
            {
                Vector3Int pos = new Vector3Int(x, y, 0);
                TileBase tile = ownerBoard.tilemap.GetTile(pos);
                if (tile != null)
                {
                    column.Add(tile);
                    originalTiles++;
                }
                ownerBoard.tilemap.SetTile(pos, null);
            }

            // Place tiles at bottom
            for (int i = 0; i < column.Count; i++)
            {
                Vector3Int pos = new Vector3Int(x, bounds.yMin + i, 0);
                ownerBoard.tilemap.SetTile(pos, column[i]);
                totalMoved++;
            }

            if (originalTiles > 0)
            {
                Debug.Log($"🌍 Column {x}: {originalTiles} tiles → compacted to bottom");
            }
        }

        Debug.Log($"✅ Gravity completed: {totalMoved} tiles moved");
    }

    private void ClearLine(int row)
    {
        Debug.Log($"🧹 Clearing line {row}");

        RectInt bounds = ownerBoard.Bounds;
        int clearedTiles = 0;

        // Clear the line
        for (int x = bounds.xMin; x < bounds.xMax; x++)
        {
            Vector3Int pos = new Vector3Int(x, row, 0);
            if (ownerBoard.tilemap.HasTile(pos))
            {
                ownerBoard.tilemap.SetTile(pos, null);
                clearedTiles++;
            }
        }

        Debug.Log($"🧹 Cleared {clearedTiles} tiles from line {row}");

        // Move lines down
        int movedTiles = 0;
        for (int y = row + 1; y < bounds.yMax; y++)
        {
            for (int x = bounds.xMin; x < bounds.xMax; x++)
            {
                Vector3Int above = new Vector3Int(x, y, 0);
                Vector3Int below = new Vector3Int(x, y - 1, 0);
                TileBase tile = ownerBoard.tilemap.GetTile(above);
                ownerBoard.tilemap.SetTile(below, tile);
                ownerBoard.tilemap.SetTile(above, null);

                if (tile != null) movedTiles++;
            }
        }

        Debug.Log($"🔄 Moved {movedTiles} tiles down after line clear");
    }

    // ✅ POWER-UP GENERATION AND MANAGEMENT (unchanged)
    public void OnLinesCleared(int lineCount)
    {
        string playerTag = ownerBoard != null ? ownerBoard.playerTag : "Unknown";
        Debug.Log($"🎯 === LINES CLEARED EVENT: {lineCount} LINES (Player: {playerTag}) ===");

        // Add lines to time tracking (each line gets a timestamp)
        float currentTime = Time.time;
        Debug.Log($"⏰ Adding {lineCount} line timestamps at time {currentTime:F1}");

        for (int i = 0; i < lineCount; i++)
        {
            linesClearedTimes.Add(currentTime);
        }

        int totalLinesInWindow = linesClearedTimes.Count;
        Debug.Log($"📊 Total lines in current window: {totalLinesInWindow}");

        // Check both constraints for power-up earning
        Debug.Log("🔍 Checking power-up constraints...");

        bool earnedFromLineCount = CheckLineCountConstraint(lineCount);
        bool earnedFromTimeChallenge = CheckTimeConstraint();

        Debug.Log($"📋 Constraint Results for {playerTag}:");
        Debug.Log($"  🎲 Line Count ({lineCount} lines): {(earnedFromLineCount ? "✅ PASSED" : "❌ FAILED")}");
        Debug.Log($"  ⏰ Time Challenge ({totalLinesInWindow}/{requiredLinesInWindow}): {(earnedFromTimeChallenge ? "✅ PASSED" : "❌ FAILED")}");

        // Award power-ups
        int powerUpsAwarded = 0;

        if (earnedFromLineCount)
        {
            Debug.Log($"🎁 Awarding power-up to {playerTag} from line count constraint!");
            GenerateRandomPowerUp();
            powerUpsAwarded++;
        }

        if (earnedFromTimeChallenge)
        {
            Debug.Log($"🎁 Awarding power-up to {playerTag} from time challenge constraint!");
            GenerateRandomPowerUp();
            powerUpsAwarded++;
            ResetTimeChallenge();
        }

        if (powerUpsAwarded == 0)
        {
            Debug.Log($"💔 No power-ups awarded to {playerTag} this time");
        }
        else
        {
            Debug.Log($"🎉 TOTAL POWER-UPS AWARDED TO {playerTag}: {powerUpsAwarded}");
        }

        LogFullInventory($"After processing {lineCount} lines for {playerTag}");
        Debug.Log($"=== END LINES CLEARED PROCESSING FOR {playerTag} ===");
    }

    private bool CheckLineCountConstraint(int lineCount)
    {
        string playerTag = ownerBoard != null ? ownerBoard.playerTag : "Unknown";
        Debug.Log($"🔍 === CHECKING LINE COUNT CONSTRAINT (Player: {playerTag}) ===");
        Debug.Log($"📊 Lines cleared this turn: {lineCount}");

        // Constraint 1: 2+ lines cleared = chance for power-up
        if (lineCount < 2)
        {
            Debug.Log($"❌ Insufficient lines: {lineCount} < 2 (minimum required)");
            return false;
        }

        float baseChance = powerUpChance;
        float multiplier = lineCount switch
        {
            2 => 1.0f,
            3 => 1.5f,
            4 => 2.0f, // Tetris bonus
            _ => 1.0f
        };

        float finalChance = baseChance * multiplier;
        float randomRoll = Random.Range(0f, 1f);
        bool success = randomRoll < finalChance;

        Debug.Log($"🎲 Calculation Details for {playerTag}:");
        Debug.Log($"  Base Chance: {baseChance * 100:F1}%");
        Debug.Log($"  Multiplier: {multiplier}x (for {lineCount} lines)");
        Debug.Log($"  Final Chance: {finalChance * 100:F1}%");
        Debug.Log($"  Random Roll: {randomRoll:F3}");
        Debug.Log($"  Required: < {finalChance:F3}");
        Debug.Log($"  Result: {(success ? "✅ SUCCESS!" : "❌ FAILED")}");

        return success;
    }

    private bool CheckTimeConstraint()
    {
        string playerTag = ownerBoard != null ? ownerBoard.playerTag : "Unknown";
        Debug.Log($"🔍 === CHECKING TIME CHALLENGE CONSTRAINT (Player: {playerTag}) ===");

        // Constraint 2: X lines in Y minutes = guaranteed power-up
        int linesInWindow = linesClearedTimes.Count;
        float timeElapsed = Time.time - gameStartTime;
        float timeRemaining = (timeWindowMinutes * 60f) - timeElapsed;

        Debug.Log($"📊 Challenge Details for {playerTag}:");
        Debug.Log($"  Lines in window: {linesInWindow}");
        Debug.Log($"  Required lines: {requiredLinesInWindow}");
        Debug.Log($"  Time elapsed: {timeElapsed:F1}s");
        Debug.Log($"  Time remaining: {timeRemaining:F1}s");
        Debug.Log($"  Window duration: {timeWindowMinutes * 60f}s");

        bool success = linesInWindow >= requiredLinesInWindow;

        if (success)
        {
            Debug.Log($"✅ TIME CHALLENGE COMPLETED by {playerTag}!");
            Debug.Log($"  Achievement: {linesInWindow}/{requiredLinesInWindow} lines");
            Debug.Log($"  Time used: {timeElapsed:F1}s of {timeWindowMinutes * 60f}s");
        }
        else
        {
            int linesNeeded = requiredLinesInWindow - linesInWindow;
            Debug.Log($"❌ Challenge not completed yet by {playerTag}");
            Debug.Log($"  Still need: {linesNeeded} more lines");
            Debug.Log($"  Time left: {Mathf.Max(0, timeRemaining):F1}s");
        }

        return success;
    }

    private void ResetTimeChallenge()
    {
        string playerTag = ownerBoard != null ? ownerBoard.playerTag : "Unknown";
        Debug.Log($"🔄 === RESETTING TIME CHALLENGE (Player: {playerTag}) ===");
        Debug.Log($"📊 Previous window stats:");
        Debug.Log($"  Lines cleared: {linesClearedTimes.Count}");
        Debug.Log($"  Time taken: {Time.time - gameStartTime:F1}s");

        linesClearedTimes.Clear();
        gameStartTime = Time.time;

        Debug.Log($"✅ New time challenge window started for {playerTag}!");
        Debug.Log($"  Target: {requiredLinesInWindow} lines in {timeWindowMinutes} minutes");
    }

    private void GenerateRandomPowerUp()
    {
        string playerTag = ownerBoard != null ? ownerBoard.playerTag : "Unknown";
        Debug.Log($"🎲 === GENERATING RANDOM POWER-UP (Player: {playerTag}) ===");

        // Base types
        List<PowerUpType> availableTypes = new List<PowerUpType>
        {
            PowerUpType.LineBlaster,
            PowerUpType.Gravity,
            PowerUpType.Bomb
        };

        // Corresponding weights
        List<float> weights = new List<float> { 3f, 2f, 1f };

        // Add WildCard if opponentBoard exists
        if (opponentBoard != null)
        {
            availableTypes.Add(PowerUpType.WildCard);
            weights.Add(1.5f);
            Debug.Log("🃏 WildCard available (opponent board exists)");
        }
        else
        {
            Debug.Log("❌ WildCard NOT available (no opponent board)");
        }

        Debug.Log($"📋 Available types: {string.Join(", ", availableTypes)}");
        Debug.Log($"⚖️ Weights: {string.Join(", ", weights)}");

        // Weighted selection
        float totalWeight = weights.Sum();
        float randomValue = Random.Range(0f, totalWeight);
        float currentWeight = 0f;

        Debug.Log($"🎲 Random roll: {randomValue:F3} / {totalWeight:F3}");

        for (int i = 0; i < availableTypes.Count; i++)
        {
            currentWeight += weights[i];
            Debug.Log($"🎯 Checking {availableTypes[i]}: weight range {currentWeight - weights[i]:F3} - {currentWeight:F3}");

            if (randomValue <= currentWeight)
            {
                Debug.Log($"🎉 Selected: {availableTypes[i]} for {playerTag}");
                AddPowerUp(availableTypes[i]);
                break;
            }
        }
        
        Debug.Log("=== END POWER-UP GENERATION ===");
    }

    public void AddPowerUp(PowerUpType type)
    {
        string playerTag = ownerBoard != null ? ownerBoard.playerTag : "Unknown";
        Debug.Log($"🎁 === ADDING POWER-UP: {type.ToString().ToUpper()} (Player: {playerTag}) ===");

        if (powerUpInventory.ContainsKey(type))
        {
            int beforeCount = powerUpInventory[type];
            powerUpInventory[type]++;
            int afterCount = powerUpInventory[type];

            Debug.Log($"✅ {type} added successfully to {playerTag}!");
            Debug.Log($"📊 Count: {beforeCount} → {afterCount}");

            UpdatePowerUpUI();

            LogFullInventory($"After adding {type}");
            Debug.Log($"🎮 Player {playerTag} now has {afterCount} {type}(s)!");
        }
        else
        {
            Debug.LogError($"❌ Error: {type} not found in inventory dictionary for {playerTag}!");
            Debug.LogError($"📦 Available keys: {string.Join(", ", powerUpInventory.Keys)}");
        }
        
        Debug.Log("=== END POWER-UP ADDITION ===");
    }

    private void LogFullInventory(string context)
    {
        string playerTag = ownerBoard != null ? ownerBoard.playerTag : "Unknown";
        Debug.Log($"📦 === FULL INVENTORY ({context}) - Player: {playerTag} ===");
        
        foreach (var kvp in powerUpInventory)
        {
            Debug.Log($"  {kvp.Key}: {kvp.Value}");
        }
        
        int totalPowerUps = powerUpInventory.Values.Sum();
        Debug.Log($"📊 Total Power-ups: {totalPowerUps}");
        Debug.Log("=== END INVENTORY LOG ===");
    }

    public void ClearAllPowerUps()
    {
        string playerTag = ownerBoard != null ? ownerBoard.playerTag : "Unknown";
        Debug.Log($"🧹 === CLEARING ALL POWER-UPS (Player: {playerTag}) ===");
        
        LogFullInventory("Before clear");
        Debug.Log($"📊 Lines in window: {linesClearedTimes.Count}");

        InitializeInventory();
        linesClearedTimes.Clear();
        gameStartTime = Time.time;
        UpdatePowerUpUI();

        Debug.Log($"✅ All power-ups and progress cleared for {playerTag}");
        Debug.Log($"🔄 New game session started for {playerTag}");
    }

    private void UpdatePowerUpUI()
    {
        if (powerUpInventoryParent != null)
        {
            // Count existing UI elements
            int existingSlots = powerUpInventoryParent.childCount;

            foreach (Transform child in powerUpInventoryParent)
            {
                Destroy(child.gameObject);
            }

            // Create UI for each power-up type
            int totalSlots = 0;
            foreach (var kvp in powerUpInventory)
            {
                if (kvp.Value > 0 && powerUpSlotPrefab != null)
                {
                    for (int i = 0; i < kvp.Value; i++)
                    {
                        GameObject slot = Instantiate(powerUpSlotPrefab, powerUpInventoryParent);
                        slot.name = $"PowerUp_{kvp.Key}_{i}";
                        totalSlots++;
                    }
                }
            }

            if (existingSlots != totalSlots)
            {
                string playerTag = ownerBoard != null ? ownerBoard.playerTag : "Unknown";
                Debug.Log($"🎨 UI updated for {playerTag}: {existingSlots} → {totalSlots} power-up slots");
            }
        }
    }

    private PowerUp GetPowerUpData(PowerUpType type)
    {
        return availablePowerUps.FirstOrDefault(p => p.type == type);
    }

    public int GetPowerUpCount()
    {
        return powerUpInventory.Values.Sum();
    }

    public int GetPowerUpCount(PowerUpType type)
    {
        return powerUpInventory.ContainsKey(type) ? powerUpInventory[type] : 0;
    }

    public bool HasPowerUp(PowerUpType type)
    {
        return GetPowerUpCount(type) > 0;
    }
}

public class PowerupKeyMapping
{
    public KeyCode key;
    public PowerUpType powerupType;
}