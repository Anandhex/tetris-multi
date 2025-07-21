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



        // Initialize inventory
        InitializeInventory();

        // Initialize UI
        InitializeUI();

        // Initial status log
    }

    private void InitializeInventory()
    {

        powerUpInventory[PowerUpType.LineBlaster] = 0;
        powerUpInventory[PowerUpType.Gravity] = 0;
        powerUpInventory[PowerUpType.Bomb] = 0;

        if (opponentBoard != null)
        {
            powerUpInventory[PowerUpType.WildCard] = 0;
        }
        else
        {
            // Debug.Log("❌ WildCard NOT initialized (no opponent board)");
        }

    }

    private void InitializeUI()
    {
        // Set up the keys instruction (this won't change)
        if (keysText != null)
        {
            keysText.text = "Keys:\n1=LineBlaster\n2=Gravity\n3=Bomb";
            // Debug.Log("✅ Keys UI initialized");
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
        // Debug.Log($"🔗 UI Connections: Inventory={inventoryText != null}, Progress={progressText != null}, Keys={keysText != null}");
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

        // if (Input.GetKeyDown(KeyCode.Semicolon)) // ; key for bomb
        // {
        //     UsePowerUp(PowerUpType.Bomb, 0);
        // }

        // if (Input.GetKeyDown(KeyCode.Quote)) // ' key for wildcard
        // {
        //     Debug.Log($"🧪 TESTING: Wildcard at column {wildcardTestColumn}");
        //     if (opponentBoard != null)
        //     {
        //         Debug.Log($"🔍 Opponent board bounds: {opponentBoard.Bounds}");
        //     }
        //     UsePowerUp(PowerUpType.WildCard, wildcardTestColumn);
        //     wildcardTestColumn++;
        //     if (wildcardTestColumn > 3) wildcardTestColumn = -4;
        // }
        // Clean up old line clear times (outside the time window)
        CleanupOldLineTimes();

        // Update UI every frame
        UpdateUI();

        // Periodic status logging
        if (Time.time - lastLogTime > logInterval)
        {
            lastLogTime = Time.time;
        }

        // Check for lines count change
        if (linesClearedTimes.Count != lastLinesCount)
        {
            // Debug.Log($"📊 Lines count changed: {lastLinesCount} → {linesClearedTimes.Count}");
            lastLinesCount = linesClearedTimes.Count;
        }
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

    public void UsePowerUp(PowerUpType type)
    {
        if (powerUpInventory.ContainsKey(type) && powerUpInventory[type] > 0)
        {
            powerUpInventory[type]--;
            SafeExecutePowerUp(type);
            UpdatePowerUpUI();
        }
    }

    private void SafeExecutePowerUp(PowerUpType type)
    {
        string playerTag = ownerBoard != null ? ownerBoard.playerTag : "Unknown";



        // ✅ Bomb handling
        if (type == PowerUpType.Bomb)
        {

            ExecuteBombImproved();
            return;
        }

        // ✅ WildCard handling
        if (type == PowerUpType.WildCard && opponentBoard != null)
        {

            ReplaceOpponentPieceWithWildcard();
        }

        // ✅ Temporarily clear active piece
        bool hadActivePiece = ownerBoard.activePiece != null;
        if (hadActivePiece)
        {
            ownerBoard.Clear(ownerBoard.activePiece);
        }

        // ✅ Execute power-up
        switch (type)
        {
            case PowerUpType.LineBlaster:
                ExecuteLineBlaster();
                break;
            case PowerUpType.Gravity:
                ExecuteGravity();
                break;
        }

        // ✅ Restore active piece
        if (hadActivePiece && ownerBoard.activePiece != null)
        {
            if (ownerBoard.IsValidPosition(ownerBoard.activePiece, ownerBoard.activePiece.position))
            {
                ownerBoard.Set(ownerBoard.activePiece);
                Debug.Log("✅ Active piece restored to original position");
            }
            else
            {
                ownerBoard.activePiece.position += Vector3Int.up;
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

        return;
    }


    // ✅ FIXED BOMB METHODS


    public void ExecuteBombImproved()
    {
        if (ownerBoard.activePiece != null)
        {
            {
                ownerBoard.Clear(ownerBoard.activePiece);

                ownerBoard.activePiece.SetCells(new Vector3Int[] { Vector3Int.zero }); // single cell at center
                ownerBoard.activePiece.tile = ownerBoard.bombTile;
                ownerBoard.activePiece.isBomb = true;

                ownerBoard.Set(ownerBoard.activePiece);
            }
        }
    }

    public IEnumerator ExecuteBombAtColumn(int targetColumn)
    {
        yield return new WaitForSeconds(0.05f);
        Piece activePiece = ownerBoard.activePiece;



        // Clear old piece from the board
        ownerBoard.Clear(activePiece);

        // Set wildcard shape and properties
        ownerBoard.activePiece.SetCells(new Vector3Int[] { Vector3Int.zero }); // single cell at center
        ownerBoard.activePiece.tile = ownerBoard.bombTile;
        ownerBoard.activePiece.isBomb = true;

        ownerBoard.Set(activePiece);
        yield return new WaitForSeconds(0.05f);
        ownerBoard.Clear(activePiece);

        // Set piece on the board to update tilemap


        // Move horizontally toward target column
        Vector3Int pos = activePiece.position;
        int delta = targetColumn - pos.x;
        int dir = delta > 0 ? 1 : -1;

        for (int i = 0; i < Mathf.Abs(delta); i++)
        {
            pos.x += dir;
            if (!ownerBoard.IsValidPosition2(activePiece, pos))
            {
                pos.x -= dir; // revert if invalid
                break;
            }
            activePiece.position = pos;
            ownerBoard.Set(activePiece);
            yield return new WaitForSeconds(0.05f);
            ownerBoard.Clear(activePiece);


        }

        // Drop vertically until collision
        while (ownerBoard.IsValidPosition2(activePiece, pos + Vector3Int.down))
        {
            pos += Vector3Int.down;
            activePiece.position = pos;
            ownerBoard.Set(activePiece);
            yield return new WaitForSeconds(0.02f);
            ownerBoard.Clear(activePiece);
        }

        // Final placement and cleanup
        ownerBoard.Set(activePiece);
        activePiece.Lock();
        ownerBoard.ExecuteBombExplosion(activePiece.position);

        // Decrement powerup count and update UI - you may need to pass or access powerUpInventory differently here
        // Example:
        ownerBoard.powerUpManager.powerUpInventory[PowerUpType.Bomb]--;
        ownerBoard.powerUpManager.UpdatePowerUpUI();
    }



    // ✅ WILDCARD METHODS (unchanged - these work fine)
    public IEnumerator DropWildcardOnOpponent(Board targetBoard, int targetColumn, PowerUpManager powerUpManager)
    {
        yield return new WaitForSeconds(0.05f);
        // Wait until the target board is free (unlocked)


        Piece activePiece = targetBoard.activePiece;
        if (activePiece == null)
        {
            Debug.LogWarning("⚠️ Target board has no active piece to replace with wildcard");
            targetBoard.Unlock();
            yield break;
        }


        // Clear old piece from the board
        targetBoard.Clear(activePiece);

        // Set wildcard shape and properties
        activePiece.SetCells(Data.WildcardCells); // 3x3 wildcard shape
        activePiece.tile = targetBoard.bombTile;
        activePiece.isBomb = false;
        targetBoard.Set(activePiece);
        yield return new WaitForSeconds(0.05f);
        targetBoard.Clear(activePiece);


        // Set piece on the board to update tilemap


        // Move horizontally toward target column
        Vector3Int pos = activePiece.position;
        int delta = targetColumn - pos.x;
        int dir = delta > 0 ? 1 : -1;

        for (int i = 0; i < Mathf.Abs(delta); i++)
        {
            pos.x += dir;
            if (!targetBoard.IsValidPosition2(activePiece, pos))
            {
                pos.x -= dir; // revert if invalid
                break;
            }
            activePiece.position = pos;
            targetBoard.Set(activePiece);
            yield return new WaitForSeconds(0.05f);
            targetBoard.Clear(activePiece);
        }

        // Drop vertically until collision
        while (targetBoard.IsValidPosition2(activePiece, pos + Vector3Int.down))
        {
            pos += Vector3Int.down;
            activePiece.position = pos;
            targetBoard.Set(activePiece);
            yield return new WaitForSeconds(0.02f);
            targetBoard.Clear(activePiece);
        }

        // Final placement and cleanup
        targetBoard.Set(activePiece);

        // Decrement powerup count and update UI - you may need to pass or access powerUpInventory differently here
        // Example:
        powerUpManager.powerUpInventory[PowerUpType.WildCard]--;
        powerUpManager.UpdatePowerUpUI();
        targetBoard.SpawnPiece();
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

    }

    // ✅ POWER-UP GENERATION AND MANAGEMENT (unchanged)
    public void OnLinesCleared(int lineCount)
    {
        string playerTag = ownerBoard != null ? ownerBoard.playerTag : "Unknown";

        // Add lines to time tracking (each line gets a timestamp)
        float currentTime = Time.time;

        for (int i = 0; i < lineCount; i++)
        {
            linesClearedTimes.Add(currentTime);
        }


        // Check both constraints for power-up earning

        bool earnedFromLineCount = CheckLineCountConstraint(lineCount);
        bool earnedFromTimeChallenge = CheckTimeConstraint();


        // Award power-ups
        int powerUpsAwarded = 0;

        if (earnedFromLineCount)
        {
            GenerateRandomPowerUp();
            powerUpsAwarded++;
        }

        if (earnedFromTimeChallenge)
        {
            GenerateRandomPowerUp();
            powerUpsAwarded++;
            ResetTimeChallenge();
        }

        if (powerUpsAwarded == 0)
        {
            // Debug.Log($"💔 No power-ups awarded to {playerTag} this time");
        }
        else
        {
            // Debug.Log($"🎉 TOTAL POWER-UPS AWARDED TO {playerTag}: {powerUpsAwarded}");
        }

    }

    private bool CheckLineCountConstraint(int lineCount)
    {
        string playerTag = ownerBoard != null ? ownerBoard.playerTag : "Unknown";


        // Constraint 1: 2+ lines cleared = chance for power-up
        if (lineCount < 2)
        {
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



        return success;
    }

    private bool CheckTimeConstraint()
    {
        string playerTag = ownerBoard != null ? ownerBoard.playerTag : "Unknown";

        // Constraint 2: X lines in Y minutes = guaranteed power-up
        int linesInWindow = linesClearedTimes.Count;
        float timeElapsed = Time.time - gameStartTime;

        // Debug.Log($"📊 Challenge Details for {playerTag}:");
        // Debug.Log($"  Lines in window: {linesInWindow}");
        // Debug.Log($"  Required lines: {requiredLinesInWindow}");
        // Debug.Log($"  Time elapsed: {timeElapsed:F1}s");
        // Debug.Log($"  Time remaining: {timeRemaining:F1}s");
        // Debug.Log($"  Window duration: {timeWindowMinutes * 60f}s");

        bool success = linesInWindow >= requiredLinesInWindow;



        return success;
    }

    private void ResetTimeChallenge()
    {
        string playerTag = ownerBoard != null ? ownerBoard.playerTag : "Unknown";


        linesClearedTimes.Clear();
        gameStartTime = Time.time;


    }

    private void GenerateRandomPowerUp()
    {
        string playerTag = ownerBoard != null ? ownerBoard.playerTag : "Unknown";

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
        }




        // Weighted selection
        float totalWeight = weights.Sum();
        float randomValue = Random.Range(0f, totalWeight);
        float currentWeight = 0f;


        for (int i = 0; i < availableTypes.Count; i++)
        {
            currentWeight += weights[i];

            if (randomValue <= currentWeight)
            {
                AddPowerUp(availableTypes[i]);
                break;
            }
        }

    }

    public void AddPowerUp(PowerUpType type)
    {

        if (powerUpInventory.ContainsKey(type))
        {
            int beforeCount = powerUpInventory[type];
            powerUpInventory[type]++;
            int afterCount = powerUpInventory[type];


            UpdatePowerUpUI();

        }
        else
        {
        }

    }

    // private void LogFullInventory(string context)
    // {
    //     string playerTag = ownerBoard != null ? ownerBoard.playerTag : "Unknown";
    //     Debug.Log($"📦 === FULL INVENTORY ({context}) - Player: {playerTag} ===");

    //     foreach (var kvp in powerUpInventory)
    //     {
    //         Debug.Log($"  {kvp.Key}: {kvp.Value}");
    //     }

    //     int totalPowerUps = powerUpInventory.Values.Sum();
    //     Debug.Log($"📊 Total Power-ups: {totalPowerUps}");
    //     Debug.Log("=== END INVENTORY LOG ===");
    // }

    public void ClearAllPowerUps()
    {
        string playerTag = ownerBoard != null ? ownerBoard.playerTag : "Unknown";
        // Debug.Log($"🧹 === CLEARING ALL POWER-UPS (Player: {playerTag}) ===");

        // LogFullInventory("Before clear");
        // Debug.Log($"📊 Lines in window: {linesClearedTimes.Count}");

        InitializeInventory();
        linesClearedTimes.Clear();
        gameStartTime = Time.time;
        UpdatePowerUpUI();

        // Debug.Log($"✅ All power-ups and progress cleared for {playerTag}");
        // Debug.Log($"🔄 New game session started for {playerTag}");
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