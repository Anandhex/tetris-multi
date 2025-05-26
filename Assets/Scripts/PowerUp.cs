using System.Collections;
using UnityEngine;

public enum PowerUpType
{
    // Self-Benefits
    ClearRow,           // Clear a row from your board
    SlowDown,           // Slow down your game temporarily
    BonusPoints,        // Get bonus points
    BlockFreeze,        // Current piece stops falling for a few seconds
    ExtraPiece,         // Store a piece for later use

    // Opponent Effects
    AddGarbage,         // Add garbage rows to opponent
    SpeedUp,            // Increase opponent's speed temporarily
    Confusion,          // Rotate opponent's controls temporarily
    BlockFlip,          // Rotate opponent's current piece randomly
    BlindMode,          // Temporarily hide opponent's board partially
    SwapPiece           // Swap current piece with opponent
}

[System.Serializable]
public class PowerUp
{
    public PowerUpType type;
    public string name;
    public string description;
    public Sprite icon;
    public float duration = 5f;
    public float cooldown = 30f;
    public Color effectColor = Color.white;
    public bool isAvailable = true;

    public PowerUp(PowerUpType type, string name, string description)
    {
        this.type = type;
        this.name = name;
        this.description = description;
    }
}

