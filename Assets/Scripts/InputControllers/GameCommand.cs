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

    public string stageName;
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
    // Board metrics
    public int holesCount = 0;
    public float stackHeight = 0f;
    public bool perfectClear = false;
    public int linesCleared = 0;


    public float curriculumBoardHeight = 20f;
    public int curriculumBoardPreset = 0;
    public int allowedTetrominoTypes = 7;
    public bool curriculumConfirmed = false;
    //peice metric
    public int currentPieceType;
    public int currentPieceX;
    public int currentPieceY;
    public int currentPieceRotation;
    public int nextPieceType;

      public float bumpiness;
    public int wells;
    public float averageHoleDepth;
    public int potentialLineClears;
    public float boardDensity;
    public bool tSpinOpportunity;
    public float efficiencyScore;
    
    // Action validation
    public bool[] validActions; // A
}