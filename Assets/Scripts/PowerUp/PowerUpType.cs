using UnityEngine;

public enum PowerUpType
{
    Wild,
    LineBlaster,
    Freeze,
    SpeedBoost,
    GhostMode,
    LineClear,
    Scramble,
    Shield,
    Bomb,
    Gravity
}

[System.Serializable]
public class PowerUp
{
    public PowerUpType type;
    public string name;
    public string description;
    public Sprite icon;
    public float spawnWeight = 1f;
}

[System.Serializable]
public class PowerUpInstance
{
    public PowerUpType type;
    public PowerUp data;
    public float timeObtained;
    
    public PowerUpInstance(PowerUpType type, PowerUp data)
    {
        this.type = type;
        this.data = data;
        this.timeObtained = Time.time;
    }
}