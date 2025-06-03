using UnityEngine;

[System.Serializable]
public class GameCommand
{
    public string type; // "action", "reset", "curriculum_change"
    public ActionData action;
    public CurriculumData curriculum;
    public ResetData reset;
}

[System.Serializable]
public class ActionData
{
    public int actionIndex; // 0-39 for your 40 discrete actions
}

[System.Serializable]
public class CurriculumData
{
    public int boardHeight = 20;
    public int boardPreset = 0;
    public int allowedTetrominoTypes = 7;
}

[System.Serializable]
public class ResetData
{
    public bool resetBoard = true;
}

[System.Serializable]
public class GameState
{
    public string type = "game_state";
    public float[] board; // Flattened board state
    public int[] currentPiece; // Current piece info [type, rotation, x, y]
    public int[] nextPiece; // Next piece info [type]
    public Vector2Int piecePosition;
    public int score;
    public bool gameOver;
    public float reward;
    public bool episodeEnd;
    
    // Action space information
    public int actionSpaceSize = 40;
    public string actionSpaceType = "column_rotation"; // 10 columns × 4 rotations
    public bool isExecutingAction = false;
    public bool waitingForAction = true;
    
    // Additional metrics
    public int linesCleared;
    public int holesCount;
    public float stackHeight;
    public bool perfectClear;
}