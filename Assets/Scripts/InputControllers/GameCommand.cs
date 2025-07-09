using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public class GameCommand
{
    public string type; // "action", "reset", "curriculum_change", "hold_powerup", "execute_bomb_drop", "execute_gravity", "execute_bottom_clear"
    public ActionData action;
    public CurriculumData curriculum;
    public ResetData reset;
    
    // PowerUp command fields
    public string powerup_type;
    public float ai_confidence;
    public float timestamp;
    
    public BombData bomb;
    public GravityData gravity;
    public BottomClearData bottom_clear;
}

[System.Serializable]
public class ActionData
{
    public int actionIndex; // 0-39 for your 40 discrete actions
    public int col;
    public int rot;
}

[System.Serializable]
public class CurriculumData
{
    public int boardHeight = 20;
    public int boardPreset = 0;
    public int allowedTetrominoTypes = 7;

    public string stageName;
}

[System.Serializable]
public class ResetData
{
    public bool resetBoard = true;
    public bool clearPowerups = true;
    public float timestamp;
}

// PowerUp command data structures
[System.Serializable]
public class BombData
{
    public int column;
    public float predicted_impact;
    public float ai_confidence;
    public float timestamp;
}

[System.Serializable]
public class GravityData
{
    public float predicted_impact;
    public float ai_confidence;
    public float timestamp;
}

[System.Serializable]
public class BottomClearData
{
    public float predicted_impact;
    public float ai_confidence;
    public float timestamp;
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
    // Board metrics
    public int holesCount = 0;
    public float stackHeight = 0f;
    public bool perfectClear = false;
    public int linesCleared = 0;

    // int lines = simBoard.PlaceAndClear(simPiece);
    public float holes;
    public float height;


    public float curriculumBoardHeight = 20f;
    public int curriculumBoardPreset = 0;
    public int allowedTetrominoTypes = 7;
    public bool curriculumConfirmed = false;

    public List<int> validActions;
    public float bumpiness;
    public int covered;
    public int[] heights;
    
    // PowerUp related fields
    public string currentPowerupType = "none";
    public bool hasPowerup = false;
    public int powerupCount = 0;
}