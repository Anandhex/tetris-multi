using UnityEngine;
using System.Collections.Generic;
using System.Linq; 
using UnityEngine.Tilemaps;

public class PowerUpManager : MonoBehaviour
{
    [Header("Power-up Settings")]
    public PowerUp[] availablePowerUps;
    public float powerUpChance = 0.8f;
    
    [Header("UI References")]
    public GameObject powerUpSlotPrefab;
    public Transform powerUpInventoryParent;

    [Header("Audio")]
    public AudioClip powerUpObtainedSound;
    public AudioClip powerUpUsedSound;
    
    private List<PowerUpInstance> playerPowerUps = new List<PowerUpInstance>();
    private Board ownerBoard;
    private AudioSource audioSource;
    
    private void Start()
    {
        ownerBoard = GetComponent<Board>();
        audioSource = GetComponent<AudioSource>();
    }

    private void Update()
    {
        // SINGLE KEY TO USE POWER-UPS FROM INVENTORY
        if (Input.GetKeyDown(KeyCode.Alpha1))
        {
            UseNextPowerUp();
        }
    }

    private void UseNextPowerUp()
    {
        if (playerPowerUps.Count > 0)
        {
            PowerUpInstance powerUpToUse = playerPowerUps[0];
            Debug.Log($"Using power-up from inventory: {powerUpToUse.type}");
            
            playerPowerUps.RemoveAt(0);
            SafeExecutePowerUp(powerUpToUse.type);
            UpdatePowerUpUI();
            PlaySound(powerUpUsedSound);
        }
        else
        {
            Debug.Log("No power-ups in inventory!");
        }
    }

    private void SafeExecutePowerUp(PowerUpType type)
    {
        // For bomb, we need special handling
        if (type == PowerUpType.Bomb)
        {
            ExecuteBombImproved();
            return;
        }
        
        // For other power-ups, clear active piece temporarily
        if (ownerBoard.activePiece != null)
        {
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
        if (ownerBoard.activePiece != null)
        {
            if (ownerBoard.IsValidPosition(ownerBoard.activePiece, ownerBoard.activePiece.position))
            {
                ownerBoard.Set(ownerBoard.activePiece);
            }
            else
            {
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
        if (powerUpInventoryParent != null)
        {
            foreach (Transform child in powerUpInventoryParent)
            {
                Destroy(child.gameObject);
            }

            foreach (PowerUpInstance powerUp in playerPowerUps)
            {
                if (powerUpSlotPrefab != null)
                {
                    GameObject slot = Instantiate(powerUpSlotPrefab, powerUpInventoryParent);
                    slot.name = $"PowerUp_{powerUp.type}";
                }
            }
        }
        
        Debug.Log($"Power-up inventory: {playerPowerUps.Count} items");
        if (playerPowerUps.Count > 0)
        {
            Debug.Log($"Next power-up to use: {playerPowerUps[0].type}");
        }
    }
    
    public void OnLinesCleared(int lineCount)
    {
        Debug.Log($"PowerUpManager: OnLinesCleared called with {lineCount} lines!");
        
        // Only give power-ups for 2+ lines cleared
        if (lineCount < 2)
        {
            Debug.Log("Not enough lines cleared for power-up (need 2+)");
            return;
        }
        
        // RANDOM POWER-UP ASSIGNMENT (not specific to line count)
        float baseChance = powerUpChance;
        float multiplier = lineCount switch
        {
            2 => 1.0f,
            3 => 1.5f,
            4 => 2.0f, // Tetris gets higher chance, but still random power-up
            _ => 0f
        };

        float finalChance = baseChance * multiplier;
        Debug.Log($"Power-up chance: {finalChance * 100}%");

        if (Random.Range(0f, 1f) < finalChance)
        {
            // Generate COMPLETELY RANDOM power-up regardless of line count
            GenerateRandomPowerUp();
        }
    }
    // Add this method to PowerUpManager.cs
    public void ClearAllPowerUps()
    {
        playerPowerUps.Clear();
        UpdatePowerUpUI();
        Debug.Log("All power-ups cleared from inventory");
    }

    private void GenerateRandomPowerUp()
    {
        if (availablePowerUps.Length == 0) return;

        // PURE RANDOM SELECTION based on weights only
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
                Debug.Log($"Randomly selected power-up: {powerUp.type}");
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
            
            Debug.Log($"Player {ownerBoard.playerTag} received {type} power-up! (Total: {playerPowerUps.Count})");
        }
    }
    
    private void ExecuteBombImproved()
    {
        Debug.Log("=== ExecuteBombImproved: Clearing 3x3 area ===");
        
        if (ownerBoard.activePiece != null)
        {
            Vector3Int center = ownerBoard.activePiece.position;
            ownerBoard.Clear(ownerBoard.activePiece);
            
            int clearedCount = 0;
            
            // Clear 3x3 area
            for (int x = -1; x <= 1; x++)
            {
                for (int y = -1; y <= 1; y++)
                {
                    Vector3Int pos = center + new Vector3Int(x, y, 0);
                    if (ownerBoard.tilemap.HasTile(pos))
                    {
                        ownerBoard.tilemap.SetTile(pos, null);
                        clearedCount++;
                    }
                }
            }
            
            // Clear falling piece cells
            foreach (Vector3Int cell in ownerBoard.activePiece.cells)
            {
                Vector3Int pos = cell + center;
                if (ownerBoard.tilemap.HasTile(pos))
                {
                    ownerBoard.tilemap.SetTile(pos, null);
                    clearedCount++;
                }
            }
            
            Debug.Log($"Bomb cleared {clearedCount} tiles");
            
            // Try to place piece back safely
            bool placed = false;
            for (int yOffset = 0; yOffset < 5; yOffset++)
            {
                Vector3Int newPos = new Vector3Int(center.x, center.y + yOffset, center.z);
                if (ownerBoard.IsValidPosition(ownerBoard.activePiece, newPos))
                {
                    ownerBoard.activePiece.position = newPos;
                    ownerBoard.Set(ownerBoard.activePiece);
                    placed = true;
                    break;
                }
            }
            
            if (!placed)
            {
                Debug.Log("Spawning new piece after bomb");
                ownerBoard.SpawnPiece();
            }
        }
    }
    
    private void ExecuteLineBlaster()
    {
        Debug.Log("=== LineBlaster: Clearing bottom line ===");
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
                Debug.Log($"LineBlaster cleared line {y}");
                break;
            }
        }
    }
    
    private void ExecuteGravity()
    {
        Debug.Log("=== Gravity: Dropping floating blocks ===");
        
        RectInt bounds = ownerBoard.Bounds;
        int totalMoved = 0;
        
        for (int x = bounds.xMin; x < bounds.xMax; x++)
        {
            List<TileBase> column = new List<TileBase>();
            
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
            
            for (int i = 0; i < column.Count; i++)
            {
                Vector3Int pos = new Vector3Int(x, bounds.yMin + i, 0);
                ownerBoard.tilemap.SetTile(pos, column[i]);
                totalMoved++;
            }
        }
        
        Debug.Log($"Gravity moved {totalMoved} tiles");
    }
    
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
    
    private void PlaySound(AudioClip clip)
    {
        if (audioSource != null && clip != null)
        {
            audioSource.PlayOneShot(clip);
        }
    }

    public int GetPowerUpCount()
    {
        return playerPowerUps.Count;
    }
}