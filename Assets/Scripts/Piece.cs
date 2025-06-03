using UnityEngine;

public class Piece : MonoBehaviour
{
    public Board board { get; private set; }
    public TetrominoData data { get; private set; }
    public Vector3Int[] cells { get; private set; }
    public Vector3Int position;
    public int rotationIndex;
    private IPlayerInputController inputController;

    public float stepDelay = 1f;
    public bool isAgentControlled { get; set; } = false;

    public float moveDelay = 0.1f;
    public float lockDelay = 0.5f;

    private float stepTime;
    private float moveTime;
    private float lockTime;

    public void Initialize(Board board, Vector3Int position, TetrominoData data, IPlayerInputController controller)
    {
        if (board == null || controller == null)
        {
            Debug.LogError("Piece.Initialize: Board or Controller is null!");
            return;
        }

        this.data = data;
        this.board = board;
        this.position = position;
        this.inputController = controller;

        rotationIndex = 0;
        moveTime = Time.time + moveDelay;
        lockTime = 0f;

        UpdateStepDelay();

        if (cells == null)
        {
            cells = new Vector3Int[data.cells.Length];
        }

        for (int i = 0; i < cells.Length; i++)
        {
            cells[i] = (Vector3Int)data.cells[i];
        }

        Debug.Log($"Piece: Initialized {data.tetromino} at {position}, StepDelay: {stepDelay}");
    }

    private void UpdateStepDelay()
    {
        stepDelay = board != null ? board.CurrentDropRate : 1f;
        stepTime = Time.time + stepDelay;
    }

    private void Update()
    {
        if (board == null || inputController == null)
            return;

        board.Clear(this);
        lockTime += Time.deltaTime;

        if (inputController.GetRotateLeft())
            Rotate(-1);
        else if (inputController.GetRotateRight())
            Rotate(1);

        if (inputController.GetHardDrop())
            HardDrop();

        if (Time.time > moveTime)
            HandleMoveInputs();

        if (Time.time > stepTime)
            Step();

        board.Set(this);
    }

    private void HandleMoveInputs()
    {
        if (inputController.GetDown())
        {
            if (Move(Vector2Int.down))
                stepTime = Time.time + stepDelay;
        }

        if (inputController.GetLeft())
            Move(Vector2Int.left);
        else if (inputController.GetRight())
            Move(Vector2Int.right);

        moveTime = Time.time + moveDelay;
    }

    private void Step()
    {
        UpdateStepDelay();
        stepTime = Time.time + stepDelay;

        if (!Move(Vector2Int.down))
        {
            if (lockTime >= lockDelay)
            {
                Debug.Log($"Piece: Locking due to no downward move after {lockTime:F2}s");
                Lock();
            }
        }
        else
        {
            lockTime = 0f;
        }
    }

    private void HardDrop()
    {
        Debug.Log("Piece: Performing HardDrop");
        while (Move(Vector2Int.down)) { }
        Lock();
    }

    private void Lock()
    {
        board.Set(this);
        board.ClearLines();
        board.SpawnPiece();
        Debug.Log("Piece: Locked and spawned new piece");
    }

    private bool Move(Vector2Int translation)
    {
        Vector3Int newPosition = position + new Vector3Int(translation.x, translation.y, 0);
        bool valid = board.IsValidPosition(this, newPosition);

        if (valid)
        {
            position = newPosition;
            moveTime = Time.time + moveDelay;
            if (translation == Vector2Int.down)
                lockTime = 0f;
        }
        // Optional: Keep for debugging collisions
        // else
        // {
        //     Debug.LogWarning($"Piece: Invalid move to {newPosition}");
        // }

        return valid;
    }

    public void Rotate(int direction)
    {
        int originalRotation = rotationIndex;
        rotationIndex = Wrap(rotationIndex + direction, 0, 4);
        ApplyRotationMatrix(direction);

        if (!TestWallKicks(rotationIndex, direction))
        {
            rotationIndex = originalRotation;
            ApplyRotationMatrix(-direction);
        }
        else
        {
            Debug.Log($"Piece: Rotated to index {rotationIndex}");
        }
    }

    private void ApplyRotationMatrix(int direction)
    {
        float[] matrix = Data.RotationMatrix;

        for (int i = 0; i < cells.Length; i++)
        {
            Vector3 cell = cells[i];
            int x, y;

            switch (data.tetromino)
            {
                case Tetromino.I:
                case Tetromino.O:
                    cell.x -= 0.5f;
                    cell.y -= 0.5f;
                    x = Mathf.CeilToInt((cell.x * matrix[0] * direction) + (cell.y * matrix[1] * direction));
                    y = Mathf.CeilToInt((cell.x * matrix[2] * direction) + (cell.y * matrix[3] * direction));
                    break;
                default:
                    x = Mathf.RoundToInt((cell.x * matrix[0] * direction) + (cell.y * matrix[1] * direction));
                    y = Mathf.RoundToInt((cell.x * matrix[2] * direction) + (cell.y * matrix[3] * direction));
                    break;
            }

            cells[i] = new Vector3Int(x, y, 0);
        }
    }

    private bool TestWallKicks(int rotationIndex, int rotationDirection)
    {
        int wallKickIndex = GetWallKickIndex(rotationIndex, rotationDirection);

        for (int i = 0; i < data.wallKicks.GetLength(1); i++)
        {
            Vector2Int translation = data.wallKicks[wallKickIndex, i];
            if (Move(translation))
                return true;
        }

        return false;
    }

    private int GetWallKickIndex(int rotationIndex, int rotationDirection)
    {
        int wallKickIndex = rotationIndex * 2;
        if (rotationDirection < 0)
            wallKickIndex--;
        return Wrap(wallKickIndex, 0, data.wallKicks.GetLength(0));
    }

    private int Wrap(int input, int min, int max)
    {
        return input < min
            ? max - (min - input) % (max - min)
            : min + (input - min) % (max - min);
    }
}
