using UnityEngine;
using System.Collections.Generic;
using System.Linq;
using UnityEngine.Tilemaps;
using TMPro;

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
    // private AudioSource audioSource;

    // Time challenge tracking
    private List<float> linesClearedTimes = new List<float>();
    private float gameStartTime;

    public List<PowerupKeyMapping> powerupKeyMappings;

    // Logging control
    private float lastLogTime = 0f;
    private float logInterval = 2f; // Log every 2 seconds
    private int lastLinesCount = 0;

    private Board opponentBoard;

    // private void Start()
    // {
    //     ownerBoard = GetComponent<Board>();
    //     // audioSource = GetComponent<AudioSource>();
    //     gameStartTime = Time.time;

    //     // Debug.Log("🎮 === POWER-UP MANAGER INITIALIZED ===");
    //     // Debug.Log($"⚙️ Settings: PowerUp Chance = {powerUpChance * 100}%");
    //     // Debug.Log($"⏰ Time Window: {timeWindowMinutes} minutes for {requiredLinesInWindow} lines");
    //     // Debug.Log($"🎯 Constraint 1: Clear 2+ lines = {powerUpChance * 100}% base chance");
    //     // Debug.Log($"🎯 Constraint 2: Clear {requiredLinesInWindow} lines in {timeWindowMinutes} minutes = GUARANTEED");

    //     // Initialize inventory
    //     InitializeInventory();

    //     // Initialize UI
    //     InitializeUI();

    //     // Initial status log
    //     LogCurrentStatus();
    // }

    public void SetupPowerupManager(Board userBoard, Board opponentBoard, List<PowerupKeyMapping> powerupKeyMappings)
    {
        this.ownerBoard = userBoard;
        this.opponentBoard = opponentBoard;
        this.powerupKeyMappings = powerupKeyMappings;

        gameStartTime = Time.time;

        // Debug.Log("🎮 === POWER-UP MANAGER INITIALIZED ===");
        // Debug.Log($"⚙️ Settings: PowerUp Chance = {powerUpChance * 100}%");
        // Debug.Log($"⏰ Time Window: {timeWindowMinutes} minutes for {requiredLinesInWindow} lines");
        // Debug.Log($"🎯 Constraint 1: Clear 2+ lines = {powerUpChance * 100}% base chance");
        // Debug.Log($"🎯 Constraint 2: Clear {requiredLinesInWindow} lines in {timeWindowMinutes} minutes = GUARANTEED");

        // Initialize inventory
        InitializeInventory();

        // Initialize UI
        InitializeUI();

        // Initial status log
        LogCurrentStatus();

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

        // Debug.Log("📦 Inventory initialized: LineBlaster=0, Gravity=0, Bomb=0");
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
            // Debug.LogWarning("⚠️ Keys UI not connected!");
        }

        // Update the dynamic UI elements
        UpdateUI();

        // Check UI connections
        // Debug.Log($"🔗 UI Connections: Inventory={inventoryText != null}, Progress={progressText != null}, Keys={keysText != null}");
    }

    private void Update()
    {
        // INDIVIDUAL KEYS FOR EACH POWER-UP (AI-Friendly!)
        foreach (var mapping in this.powerupKeyMappings)
        {
            if (Input.GetKeyDown(mapping.key))
            {
                Debug.Log($"🔑 {mapping.key} pressed - attempting {mapping.powerupType}");
                UsePowerUp(mapping.powerupType);
            }
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
            // Debug.Log($"📊 Lines count changed: {lastLinesCount} → {linesClearedTimes.Count}");
            lastLinesCount = linesClearedTimes.Count;
        }
    }

    private void LogCurrentStatus()
    {
        // float timeElapsed = Time.time - gameStartTime;
        // float timeRemaining = (timeWindowMinutes * 60f) - timeElapsed;
        // int linesInWindow = linesClearedTimes.Count;

        // Debug.Log("📋 === CURRENT STATUS ===");
        // Debug.Log($"⏰ Time: {timeElapsed:F1}s elapsed, {Mathf.Max(0, timeRemaining):F1}s remaining");
        // Debug.Log($"📊 Progress: {linesInWindow}/{requiredLinesInWindow} lines in window");
        // Debug.Log($"🎒 Inventory: L={powerUpInventory[PowerUpType.LineBlaster]}, G={powerUpInventory[PowerUpType.Gravity]}, B={powerUpInventory[PowerUpType.Bomb]}");

        // // Calculate progress percentage
        // float progressPercent = (float)linesInWindow / requiredLinesInWindow * 100f;
        // Debug.Log($"📈 Progress: {progressPercent:F1}% towards time challenge");

        // // Time window status
        // if (timeRemaining > 0)
        // {
        //     Debug.Log($"⏳ Time Challenge: {timeRemaining:F0}s remaining");
        // }
        // else
        // {
        //     Debug.Log("🔄 Time window expired - ready for reset");
        // }

        // Debug.Log("=========================");
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
            // Debug.Log($"🧹 Cleaned up {beforeCount - afterCount} old line records");
            // Debug.Log($"📊 Current window now has {afterCount} lines");
        }
    }

    private void UpdateUI()
    {
        // Update inventory display
        if (inventoryText != null)
        {
            string inventoryDisplay = $"Power-ups:\n[LineBlaster:{powerUpInventory[PowerUpType.LineBlaster]}]\n[Gravity:{powerUpInventory[PowerUpType.Gravity]}]\n[Bomb:{powerUpInventory[PowerUpType.Bomb]}]";
            if (opponentBoard != null)
            {
                inventoryDisplay += $"\n[WildCard:{powerUpInventory[PowerUpType.WildCard]}]";
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
                    // Debug.Log("🔄 Auto-resetting time window");
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
        Debug.Log($"🎯Powerup === ATTEMPTING TO USE {type.ToString().ToUpper()} ===");

        if (powerUpInventory.ContainsKey(type) && powerUpInventory[type] > 0)
        {
            int beforeCount = powerUpInventory[type];
            powerUpInventory[type]--;
            int afterCount = powerUpInventory[type];

            Debug.Log($"✅ {type} used successfully! Count: {beforeCount} → {afterCount}");

            SafeExecutePowerUp(type);
            UpdatePowerUpUI();
            // PlaySound(powerUpUsedSound);

            // Log updated inventory
            // Debug.Log($"📦 New inventory: L={powerUpInventory[PowerUpType.LineBlaster]}, G={powerUpInventory[PowerUpType.Gravity]}, B={powerUpInventory[PowerUpType.Bomb]}");
        }
        else
        {
            int currentCount = powerUpInventory.ContainsKey(type) ? powerUpInventory[type] : 0;
            Debug.Log($"❌ Cannot use {type}! Current count: {currentCount}");
            // Debug.Log($"📦 Available: L={powerUpInventory[PowerUpType.LineBlaster]}, G={powerUpInventory[PowerUpType.Gravity]}, B={powerUpInventory[PowerUpType.Bomb]}");
        }

        Debug.Log("=== END POWER-UP USAGE ===");
    }

    private void SafeExecutePowerUp(PowerUpType type)
    {
        Debug.Log($"🔧 Executing {type} power-up...");

        // For bomb, we need special handling
        if (type == PowerUpType.Bomb)
        {
            ExecuteBombImproved();
            return;
        }

        if (type == PowerUpType.WildCard && this.opponentBoard != null)
        {
            ReplaceOpponentPieceWithWildcard();
            return;
        }

        // For other power-ups, clear active piece temporarily
        bool hadActivePiece = ownerBoard.activePiece != null;
        if (hadActivePiece)
        {
            Debug.Log("🔄 Temporarily clearing active piece");
            ownerBoard.Clear(ownerBoard.activePiece);
        }

        // Execute the power-up
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
                // Debug.Log("✅ Active piece restored to original position");
            }
            else
            {
                ownerBoard.activePiece.position = new Vector3Int(ownerBoard.activePiece.position.x, ownerBoard.activePiece.position.y + 1, 0);
                if (ownerBoard.IsValidPosition(ownerBoard.activePiece, ownerBoard.activePiece.position))
                {
                    ownerBoard.Set(ownerBoard.activePiece);
                    // Debug.Log("✅ Active piece restored to adjusted position");
                }
                else
                {
                    // Debug.Log("⚠️ Could not restore active piece - position invalid");
                }
            }
        }
    }

    public void OnLinesCleared(int lineCount)
    {
        // Debug.Log($"🎯 === LINES CLEARED EVENT: {lineCount} LINES ===");

        // Add lines to time tracking (each line gets a timestamp)
        float currentTime = Time.time;
        // Debug.Log($"⏰ Adding {lineCount} line timestamps at time {currentTime:F1}");

        for (int i = 0; i < lineCount; i++)
        {
            linesClearedTimes.Add(currentTime);
        }

        int totalLinesInWindow = linesClearedTimes.Count;
        // Debug.Log($"📊 Total lines in current window: {totalLinesInWindow}");

        // Check both constraints for power-up earning
        // Debug.Log("🔍 Checking power-up constraints...");

        bool earnedFromLineCount = CheckLineCountConstraint(lineCount);
        bool earnedFromTimeChallenge = CheckTimeConstraint();

        // Debug.Log($"📋 Constraint Results:");
        // Debug.Log($"  🎲 Line Count: {(earnedFromLineCount ? "✅ PASSED" : "❌ FAILED")}");
        // Debug.Log($"  ⏰ Time Challenge: {(earnedFromTimeChallenge ? "✅ PASSED" : "❌ FAILED")}");

        // Award power-ups
        int powerUpsAwarded = 0;

        if (earnedFromLineCount)
        {
            // Debug.Log("🎁 Awarding power-up from line count constraint!");
            GenerateRandomPowerUp();
            powerUpsAwarded++;
        }

        if (earnedFromTimeChallenge)
        {
            // Debug.Log("🎁 Awarding power-up from time challenge constraint!");
            GenerateRandomPowerUp();
            powerUpsAwarded++;
            ResetTimeChallenge();
        }

        if (powerUpsAwarded == 0)
        {
            // Debug.Log("💔 No power-ups awarded this time");
        }
        else
        {
            // Debug.Log($"🎉 TOTAL POWER-UPS AWARDED: {powerUpsAwarded}");
        }

        // Debug.Log("=== END LINES CLEARED PROCESSING ===");
    }

    private bool CheckLineCountConstraint(int lineCount)
    {
        // Debug.Log($"🔍 === CHECKING LINE COUNT CONSTRAINT ===");
        // Debug.Log($"📊 Lines cleared this turn: {lineCount}");

        // Constraint 1: 2+ lines cleared = chance for power-up
        if (lineCount < 2)
        {
            // Debug.Log($"❌ Insufficient lines: {lineCount} < 2 (minimum required)");
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

        // Debug.Log($"🎲 Calculation Details:");
        // Debug.Log($"  Base Chance: {baseChance * 100:F1}%");
        // Debug.Log($"  Multiplier: {multiplier}x (for {lineCount} lines)");
        // Debug.Log($"  Final Chance: {finalChance * 100:F1}%");
        // Debug.Log($"  Random Roll: {randomRoll:F3}");
        // Debug.Log($"  Required: < {finalChance:F3}");
        // Debug.Log($"  Result: {(success ? "✅ SUCCESS!" : "❌ FAILED")}");

        return success;
    }

    private bool CheckTimeConstraint()
    {
        // Debug.Log($"🔍 === CHECKING TIME CHALLENGE CONSTRAINT ===");

        // Constraint 2: X lines in Y minutes = guaranteed power-up
        int linesInWindow = linesClearedTimes.Count;
        float timeElapsed = Time.time - gameStartTime;
        float timeRemaining = (timeWindowMinutes * 60f) - timeElapsed;

        // Debug.Log($"📊 Challenge Details:");
        // Debug.Log($"  Lines in window: {linesInWindow}");
        // Debug.Log($"  Required lines: {requiredLinesInWindow}");
        // Debug.Log($"  Time elapsed: {timeElapsed:F1}s");
        // Debug.Log($"  Time remaining: {timeRemaining:F1}s");
        // Debug.Log($"  Window duration: {timeWindowMinutes * 60f}s");

        bool success = linesInWindow >= requiredLinesInWindow;

        if (success)
        {
            // Debug.Log($"✅ TIME CHALLENGE COMPLETED!");
            // Debug.Log($"  Achievement: {linesInWindow}/{requiredLinesInWindow} lines");
            // Debug.Log($"  Time used: {timeElapsed:F1}s of {timeWindowMinutes * 60f}s");
        }
        else
        {
            int linesNeeded = requiredLinesInWindow - linesInWindow;
            // Debug.Log($"❌ Challenge not completed yet");
            // Debug.Log($"  Still need: {linesNeeded} more lines");
            // Debug.Log($"  Time left: {Mathf.Max(0, timeRemaining):F1}s");
        }

        return success;
    }

    private void ResetTimeChallenge()
    {
        // Debug.Log("🔄 === RESETTING TIME CHALLENGE ===");
        // Debug.Log($"📊 Previous window stats:");
        // Debug.Log($"  Lines cleared: {linesClearedTimes.Count}");
        // Debug.Log($"  Time taken: {Time.time - gameStartTime:F1}s");

        linesClearedTimes.Clear();
        gameStartTime = Time.time;

        // Debug.Log("✅ New time challenge window started!");
        // Debug.Log($"  Target: {requiredLinesInWindow} lines in {timeWindowMinutes} minutes");
    }

    private void GenerateRandomPowerUp()
    {
        // Base types
        List<PowerUpType> availableTypes = new List<PowerUpType>
    {
        PowerUpType.LineBlaster,
        PowerUpType.Gravity,
        PowerUpType.Bomb
    };

        // Add WildCard if opponentBoard exists
        if (opponentBoard != null)
        {
            // availableTypes.Add(PowerUpType.WildCard);
        }

        // Corresponding weights (adjusted if WildCard is added)
        List<float> weights = new List<float> { 3f, 2f, 1f };

        if (opponentBoard != null)
        {
            weights.Add(1.5f); // or any weight you prefer for WildCard
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
        // Debug.Log($"🎁 === ADDING POWER-UP: {type.ToString().ToUpper()} ===");

        if (powerUpInventory.ContainsKey(type))
        {
            int beforeCount = powerUpInventory[type];
            powerUpInventory[type]++;
            int afterCount = powerUpInventory[type];

            Debug.Log($"✅ Powerup added {type} successfully!");
            // Debug.Log($"  Count: {beforeCount} → {afterCount}");

            UpdatePowerUpUI();
            // PlaySound(powerUpObtainedSound);

            // Log full inventory
            Debug.Log($"📦Powerup Full inventory: L={powerUpInventory[PowerUpType.LineBlaster]}, G={powerUpInventory[PowerUpType.Gravity]}, B={powerUpInventory[PowerUpType.Bomb]}");

            string playerTag = ownerBoard != null ? ownerBoard.playerTag : "Unknown";
            // Debug.Log($"🎮 Player {playerTag} received {type}!");
        }
        else
        {
            // Debug.LogError($"❌ Error: {type} not found in inventory dictionary!");
        }
    }

    public void ClearAllPowerUps()
    {
        // Debug.Log("🧹 === CLEARING ALL POWER-UPS ===");
        // Debug.Log($"📊 Before clear: L={powerUpInventory[PowerUpType.LineBlaster]}, G={powerUpInventory[PowerUpType.Gravity]}, B={powerUpInventory[PowerUpType.Bomb]}");
        // Debug.Log($"📊 Lines in window: {linesClearedTimes.Count}");

        InitializeInventory();
        linesClearedTimes.Clear();
        gameStartTime = Time.time;
        UpdatePowerUpUI();

        // Debug.Log("✅ All power-ups and progress cleared");
        // Debug.Log("🔄 New game session started");
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
                // Debug.Log($"🎨 UI updated: {existingSlots} → {totalSlots} power-up slots");
            }
        }
    }

    public void ExecuteBombImproved()
    {
        Debug.Log("💣 === EXECUTING BOMB POWER-UP ===");

        // if (ownerBoard.activePiece != null)
        // {
        //     Vector3Int center = ownerBoard.activePiece.position;
        //     Debug.Log($"💥 Bomb center: {center}");

        //     ownerBoard.Clear(ownerBoard.activePiece);
        //     int clearedCount = 0;
        //     List<Vector3Int> clearedPositions = new List<Vector3Int>();

        //     // Clear 3x3 area
        //     for (int x = -1; x <= 1; x++)
        //     {
        //         for (int y = -1; y <= 1; y++)
        //         {
        //             Vector3Int pos = center + new Vector3Int(x, y, 0);
        //             if (ownerBoard.tilemap.HasTile(pos))
        //             {
        //                 ownerBoard.tilemap.SetTile(pos, null);
        //                 clearedPositions.Add(pos);
        //                 clearedCount++;
        //             }
        //         }
        //     }

        //     // Clear falling piece cells
        //     foreach (Vector3Int cell in ownerBoard.activePiece.cells)
        //     {
        //         Vector3Int pos = cell + center;
        //         if (ownerBoard.tilemap.HasTile(pos))
        //         {
        //             ownerBoard.tilemap.SetTile(pos, null);
        //             clearedPositions.Add(pos);
        //             clearedCount++;
        //         }
        //     }

        //     Debug.Log($"💥 Bomb cleared {clearedCount} tiles at positions:");
        //     foreach (var pos in clearedPositions)
        //     {
        //         Debug.Log($"  - {pos}");
        //     }

        //     // Try to place piece back safely
        //     bool placed = false;
        //     for (int yOffset = 0; yOffset < 5; yOffset++)
        //     {
        //         Vector3Int newPos = new Vector3Int(center.x, center.y + yOffset, center.z);
        //         if (ownerBoard.IsValidPosition(ownerBoard.activePiece, newPos))
        //         {
        //             ownerBoard.activePiece.position = newPos;
        //             ownerBoard.Set(ownerBoard.activePiece);
        //             placed = true;
        //             Debug.Log($"✅ Active piece repositioned to {newPos}");
        //             break;
        //         }
        //     }

        //     if (!placed)
        //     {
        //         Debug.Log("🔄 Spawning new piece after bomb (couldn't reposition)");
        //         ownerBoard.SpawnPiece();
        //     }
        // }
        // else
        // {
        //     Debug.LogWarning("⚠️ No active piece to center bomb on!");
        // }

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
        // oldPiece.isWildcard = true; // ← Add a new bool if you want to distinguish it

        // Reset position (centered)
        // oldPiece.position = new Vector3Int(opponentBoard.width / 2, opponentBoard.height - 2, 0);

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

    private PowerUp GetPowerUpData(PowerUpType type)
    {
        return availablePowerUps.FirstOrDefault(p => p.type == type);
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

    // private void PlaySound(AudioClip clip)
    // {
    //     if (audioSource != null && clip != null)
    //     {
    //         audioSource.PlayOneShot(clip);
    //         // Debug.Log($"🔊 Playing sound: {clip.name}");
    //     }
    //     else
    //     {
    //         // Debug.Log($"🔇 Cannot play sound: AudioSource={audioSource != null}, Clip={clip != null}");
    //     }
    // }

    public int GetPowerUpCount()
    {
        return powerUpInventory.Values.Sum();
    }

    // Public methods for AI agents to check inventory
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