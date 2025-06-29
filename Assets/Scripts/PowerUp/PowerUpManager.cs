using UnityEngine;
using System.Collections.Generic;
using System.Linq; 
using UnityEngine.Tilemaps;
public class PowerUpManager : MonoBehaviour
{
    [Header("Power-up Settings")]
    public PowerUp[] availablePowerUps;
    public float powerUpChance = 0.3f;
    
    [Header("UI References")]
    public GameObject powerUpSlotPrefab;
    public Transform powerUpInventoryParent;

    [Header("Audio")]
    public AudioClip powerUpObtainedSound;
    public AudioClip powerUpUsedSound;
    
    private List<PowerUpInstance> playerPowerUps = new List<PowerUpInstance>();
    private Board ownerBoard;
    private AudioSource audioSource;
    
    // Active power-up effects
    private Dictionary<PowerUpType, float> activePowerUps = new Dictionary<PowerUpType, float>();
    
    private void Start()
    {
        ownerBoard = GetComponent<Board>();
        audioSource = GetComponent<AudioSource>();
    }

private void Update()
{
    UpdateActivePowerUps();
    
    // MANUAL TESTING - SAFELY EXECUTE power-ups
    if (Input.GetKeyDown(KeyCode.Alpha1))
    {
        Debug.Log("Manual execute: LineClear");
        SafeExecutePowerUp(PowerUpType.LineClear);
    }
    if (Input.GetKeyDown(KeyCode.Alpha2))
    {
        Debug.Log("Manual execute: LineBlaster");
        SafeExecutePowerUp(PowerUpType.LineBlaster);
    }
    if (Input.GetKeyDown(KeyCode.Alpha3))
    {
        Debug.Log("Manual execute: Bomb");
        SafeExecutePowerUp(PowerUpType.Bomb);
    }
    if (Input.GetKeyDown(KeyCode.Alpha4))
    {
        Debug.Log("Manual execute: Gravity");
        SafeExecutePowerUp(PowerUpType.Gravity);
    }
    if (Input.GetKeyDown(KeyCode.Alpha5))
    {
        Debug.Log("Manual execute: SpeedBoost");
        ExecuteSpeedBoost(); // SpeedBoost doesn't affect board, so it's safe
    }
}

    private void SafeExecutePowerUp(PowerUpType type)
    {
        // Clear active piece temporarily
        if (ownerBoard.activePiece != null)
        {
            ownerBoard.Clear(ownerBoard.activePiece);
        }
        
        // Execute the power-up
        switch (type)
        {
            case PowerUpType.LineClear:
                ExecuteLineClear();
                break;
            case PowerUpType.LineBlaster:
                ExecuteLineBlaster();
                break;
            case PowerUpType.Bomb:
                ExecuteBomb();
                break;
            case PowerUpType.Gravity:
                ExecuteGravity();
                break;
        }
        
        // Put active piece back
        if (ownerBoard.activePiece != null)
        {
            // Check if position is still valid after power-up
            if (ownerBoard.IsValidPosition(ownerBoard.activePiece, ownerBoard.activePiece.position))
            {
                ownerBoard.Set(ownerBoard.activePiece);
            }
            else
            {
                // If not valid, try to move it to a safe position
                ownerBoard.activePiece.position = new Vector3Int(ownerBoard.activePiece.position.x, ownerBoard.activePiece.position.y + 1, 0);
                if (ownerBoard.IsValidPosition(ownerBoard.activePiece, ownerBoard.activePiece.position))
                {
                    ownerBoard.Set(ownerBoard.activePiece);
                }
            }
        }
    }
    
    private void UpdatePowerUpUI()
    {
        // Clear existing UI
        if (powerUpInventoryParent != null)
        {
            foreach (Transform child in powerUpInventoryParent)
            {
                Destroy(child.gameObject);
            }

            // Create UI for each power-up
            foreach (PowerUpInstance powerUp in playerPowerUps)
            {
                if (powerUpSlotPrefab != null)
                {
                    GameObject slot = Instantiate(powerUpSlotPrefab, powerUpInventoryParent);
                    // Basic setup - you can enhance this later
                    slot.name = $"PowerUp_{powerUp.type}";
                }
            }
        }
    }
    
    public void OnLinesCleared(int lineCount)
    {
        Debug.Log($"PowerUpManager: OnLinesCleared called with {lineCount} lines!");
        // Increased chance for better line clears
        float baseChance = powerUpChance;
        float multiplier = lineCount switch
        {
            1 => 0.5f,
            2 => 1.0f,
            3 => 1.5f,
            4 => 2.5f, // Tetris bonus
            _ => 0f
        };

        float finalChance = baseChance * multiplier;

        if (Random.Range(0f, 1f) < finalChance)
        {
            GenerateRandomPowerUp();
        }
    }
    
    private void GenerateRandomPowerUp()
    {
        if (availablePowerUps.Length == 0) return;
        
        // Weight-based selection
        float totalWeight = 0f;
        foreach (var powerUp in availablePowerUps)
        {
            totalWeight += powerUp.spawnWeight;
        }
        
        float randomValue = Random.Range(0f, totalWeight);
        float currentWeight = 0f;
        
        foreach (var powerUp in availablePowerUps)
        {
            currentWeight += powerUp.spawnWeight;
            if (randomValue <= currentWeight)
            {
                AddPowerUp(powerUp.type);
                break;
            }
        }
    }
    
    public void AddPowerUp(PowerUpType type)
    {
        var powerUpData = GetPowerUpData(type);
        if (powerUpData != null)
        {
            playerPowerUps.Add(new PowerUpInstance(type, powerUpData));
            UpdatePowerUpUI();
            PlaySound(powerUpObtainedSound);
            
            // Visual feedback
            ShowPowerUpNotification(powerUpData);
            
            Debug.Log($"Player {ownerBoard.playerTag} received {type} power-up!");
        }
    }
    
    public bool UsePowerUp(PowerUpType type)
    {
        var powerUpInstance = playerPowerUps.FirstOrDefault(p => p.type == type);
        if (powerUpInstance != null)
        {
            playerPowerUps.Remove(powerUpInstance);
            ExecutePowerUp(type);
            UpdatePowerUpUI();
            PlaySound(powerUpUsedSound);
            return true;
        }
        return false;
    }
    
    private void ExecutePowerUp(PowerUpType type)
    {
        Debug.Log($"=== ExecutePowerUp called with type: {type} ===");
        
        switch (type)
        {
            case PowerUpType.LineClear:
                Debug.Log("About to call ExecuteLineClear...");
                ExecuteLineClear();
                Debug.Log("ExecuteLineClear completed.");
                break;
                
            case PowerUpType.LineBlaster:
                Debug.Log("About to call ExecuteLineBlaster...");
                ExecuteLineBlaster();
                Debug.Log("ExecuteLineBlaster completed.");
                break;
                
            case PowerUpType.Bomb:
                Debug.Log("About to call ExecuteBomb...");
                ExecuteBomb();
                Debug.Log("ExecuteBomb completed.");
                break;
                
            case PowerUpType.Gravity:
                Debug.Log("About to call ExecuteGravity...");
                ExecuteGravity();
                Debug.Log("ExecuteGravity completed.");
                break;
                
            case PowerUpType.SpeedBoost:
                Debug.Log("About to call ExecuteSpeedBoost...");
                ExecuteSpeedBoost();
                Debug.Log("ExecuteSpeedBoost completed.");
                break;
                
            default:
                Debug.Log($"No implementation for power-up type: {type}");
                break;
        }
    }
    
    private void ExecuteLineBlaster()
    {
        
        // Clear the bottom-most line with blocks
        RectInt bounds = ownerBoard.Bounds;
        for (int y = bounds.yMin; y < bounds.yMax; y++)
        {
            bool hasBlocks = false;
            for (int x = bounds.xMin; x < bounds.xMax; x++)
            {
                if (ownerBoard.tilemap.HasTile(new Vector3Int(x, y, 0)))
                {
                    hasBlocks = true;
                    break;
                }
            }
            
            if (hasBlocks)
            {
                ClearLine(y);
                break;
            }
        }
    }
    
    private void ExecuteFreeze()
    {
        activePowerUps[PowerUpType.Freeze] = 5f; // 5 seconds
        // This would affect the opponent's board - implement multiplayer logic
    }
    
    private void ExecuteSpeedBoost()
    {
        activePowerUps[PowerUpType.SpeedBoost] = 10f; // 10 seconds
    }
    
    private void ExecuteGhostMode()
    {
        activePowerUps[PowerUpType.GhostMode] = 15f; // 15 seconds
    }
    
    private void ExecuteLineClear()
    {
        Debug.Log("=== ExecuteLineClear Starting ===");
        if (ownerBoard == null)
           {
                Debug.LogError("ownerBoard is NULL! Cannot execute LineClear.");
                return;
            }
            
            Debug.Log($"ownerBoard found: {ownerBoard.name}");
    
        if (ownerBoard.tilemap == null)
        {
            Debug.LogError("ownerBoard.tilemap is NULL!");
            return;
        }
        
        Debug.Log("Tilemap found, proceeding with LineClear...");
        
        // Find a random line with blocks and clear it
        RectInt bounds = ownerBoard.Bounds;
        Debug.Log($"Board bounds: {bounds}");
        
        List<int> linesWithBlocks = new List<int>();
        
        for (int y = bounds.yMin; y < bounds.yMax; y++)
        {
            for (int x = bounds.xMin; x < bounds.xMax; x++)
            {
                if (ownerBoard.tilemap.HasTile(new Vector3Int(x, y, 0)))
                {
                    linesWithBlocks.Add(y);
                    break;
                }
            }
        }
        
        Debug.Log($"Found {linesWithBlocks.Count} lines with blocks");
        
        if (linesWithBlocks.Count > 0)
        {
            int randomLine = linesWithBlocks[Random.Range(0, linesWithBlocks.Count)];
            Debug.Log($"Clearing line: {randomLine}");
            ClearLine(randomLine);
        }
        else
        {
            Debug.Log("No lines with blocks found!");
        }
    }
    
    private void ExecuteBomb()
    {
        Debug.Log("=== ExecuteBomb Starting ===");
        
        // Clear a 3x3 area around the active piece
        if (ownerBoard.activePiece != null)
        {
            Vector3Int center = ownerBoard.activePiece.position;
            Debug.Log($"Bomb center position: {center}");
            
            int clearedCount = 0;
            for (int x = -1; x <= 1; x++)
            {
                for (int y = -1; y <= 1; y++)
                {
                    Vector3Int pos = center + new Vector3Int(x, y, 0);
                    
                    if (ownerBoard.tilemap.HasTile(pos))
                    {
                        Debug.Log($"Clearing tile at: {pos}");
                        ownerBoard.tilemap.SetTile(pos, null);
                        clearedCount++;
                    }
                }
            }
            Debug.Log($"Bomb cleared {clearedCount} tiles");
        }
        else
        {
            Debug.LogError("No active piece found for bomb!");
        }
    }
    
    private void ExecuteGravity()
    {
        Debug.Log("=== ExecuteGravity Starting ===");
        
        // Drop all floating blocks
        RectInt bounds = ownerBoard.Bounds;
        Debug.Log($"Gravity bounds: {bounds}");
        
        int totalMoved = 0;
        
        for (int x = bounds.xMin; x < bounds.xMax; x++)
        {
            List<TileBase> column = new List<TileBase>();
            
            // Collect all tiles in column
            for (int y = bounds.yMin; y < bounds.yMax; y++)
            {
                Vector3Int pos = new Vector3Int(x, y, 0);
                TileBase tile = ownerBoard.tilemap.GetTile(pos);
                if (tile != null)
                {
                    column.Add(tile);
                }
                ownerBoard.tilemap.SetTile(pos, null);
            }
            
            Debug.Log($"Column {x}: Found {column.Count} tiles");
            
            // Place them at bottom
            for (int i = 0; i < column.Count; i++)
            {
                Vector3Int pos = new Vector3Int(x, bounds.yMin + i, 0);
                ownerBoard.tilemap.SetTile(pos, column[i]);
                totalMoved++;
            }
        }
        
        Debug.Log($"Gravity moved {totalMoved} total tiles");
    }
    private void UpdateActivePowerUps()
    {
        var keysToRemove = new List<PowerUpType>();
        
        foreach (var kvp in activePowerUps.ToList())
        {
            activePowerUps[kvp.Key] -= Time.deltaTime;
            if (activePowerUps[kvp.Key] <= 0)
            {
                keysToRemove.Add(kvp.Key);
            }
        }
        
        foreach (var key in keysToRemove)
        {
            activePowerUps.Remove(key);
            OnPowerUpExpired(key);
        }
    }
    
    // Public getters for active power-ups
    public bool IsSpeedBoostActive() => activePowerUps.ContainsKey(PowerUpType.SpeedBoost);
    public bool IsGhostModeActive() => activePowerUps.ContainsKey(PowerUpType.GhostMode);
    public bool IsFrozen() => activePowerUps.ContainsKey(PowerUpType.Freeze);
    
    // Helper methods
    private PowerUp GetPowerUpData(PowerUpType type)
    {
        return availablePowerUps.FirstOrDefault(p => p.type == type);
    }
    
    private void ClearLine(int row)
    {
        RectInt bounds = ownerBoard.Bounds;
        for (int x = bounds.xMin; x < bounds.xMax; x++)
        {
            ownerBoard.tilemap.SetTile(new Vector3Int(x, row, 0), null);
        }
        
        // Drop everything above
        for (int y = row + 1; y < bounds.yMax; y++)
        {
            for (int x = bounds.xMin; x < bounds.xMax; x++)
            {
                Vector3Int above = new Vector3Int(x, y, 0);
                Vector3Int below = new Vector3Int(x, y - 1, 0);
                TileBase tile = ownerBoard.tilemap.GetTile(above);
                ownerBoard.tilemap.SetTile(below, tile);
                ownerBoard.tilemap.SetTile(above, null);
            }
        }
    }
    
    private void ShowPowerUpNotification(PowerUp powerUp)
    {
        // Implement UI notification system
        // You could create a popup or add to a notification queue
    }
    
    private void PlaySound(AudioClip clip)
    {
        if (audioSource != null && clip != null)
        {
            audioSource.PlayOneShot(clip);
        }
    }
    
    private void OnPowerUpExpired(PowerUpType type)
    {
        Debug.Log($"Power-up {type} expired for player {ownerBoard.playerTag}");
        // Add visual feedback for expiration
    }
}

