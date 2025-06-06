using UnityEngine;

public class Piece : MonoBehaviour
{
    public Board board { get; private set; }
    public TetrominoData data { get; private set; }
    public Vector3Int[] cells { get; private set; }
    public Vector3Int position { get; private set; }
    public int rotationIndex { get; private set; }
    private IPlayerInputController inputController;


    public float stepDelay = 1f;
    public float moveDelay = 0.1f;
    public float lockDelay = 0.5f;

    private float stepTime;
    private float moveTime;
    private float lockTime;

    // Existing code...

    private bool isFrozen = false;

    // Add this method for power-up system
    public void SetFrozen(bool frozen)
    {
        this.isFrozen = frozen;
    }





    public void Initialize(Board board, Vector3Int position, TetrominoData data, IPlayerInputController controller)
    {

        if (board == null)
        {
            Debug.LogError("Board is null in Piece.Initialize");
            return;
        }

        if (controller == null)
        {
            Debug.LogError("InputController is null in Piece.Initialize");
            return; // Don't continue with initialization if controller is null
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
    }

    private void UpdateStepDelay()
    {
        stepDelay = board.CurrentDropRate;
        stepTime = Time.time + stepDelay;
    }


    private void Update()
    {
        if (board == null || inputController == null)
        {
            Debug.LogWarning("Piece not properly initialized, skipping update");
            return;
        }

        board.Clear(this);

        // Only proceed with normal piece update if not frozen
        if (!isFrozen)
        {
            lockTime += Time.deltaTime;

            if (this.inputController.GetRotateLeft())
            {
                Rotate(-1);
            }
            else if (this.inputController.GetRotateRight())
            {
                Rotate(1);
            }

            if (this.inputController.GetHardDrop())
            {
                HardDrop();
            }

            if (Time.time > moveTime)
            {
                HandleMoveInputs();
            }

            if (Time.time > stepTime)
            {
                Step();
            }
        }

        board.Set(this);
    }

    private void HandleMoveInputs()
    {
        if (this.inputController.GetDown())
        {
            if (Move(Vector2Int.down))
            {
                stepTime = Time.time + stepDelay;
            }
        }

        if (this.inputController.GetLeft())
        {
            Move(Vector2Int.left);
        }
        else if (this.inputController.GetRight())
        {
            Move(Vector2Int.right);
        }
    }

    private void Step()
    {
        UpdateStepDelay();
        stepTime = Time.time + stepDelay;

        Move(Vector2Int.down);

        if (lockTime >= lockDelay)
        {
            Lock();
        }
    }

    private void HardDrop()
    {
        while (Move(Vector2Int.down))
        {
            continue;
        }

        Lock();
    }

    private void Lock()
    {
        board.Set(this);
        board.ClearLines();
        board.SpawnPiece();
    }

    private bool Move(Vector2Int translation)
    {
        Vector3Int newPosition = position;
        newPosition.x += translation.x;
        newPosition.y += translation.y;

        bool valid = board.IsValidPosition(this, newPosition);

        if (valid)
        {
            position = newPosition;
            moveTime = Time.time + moveDelay;
            lockTime = 0f; // reset
        }

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
            {
                return true;
            }
        }

        return false;
    }

    private int GetWallKickIndex(int rotationIndex, int rotationDirection)
    {
        int wallKickIndex = rotationIndex * 2;

        if (rotationDirection < 0)
        {
            wallKickIndex--;
        }

        return Wrap(wallKickIndex, 0, data.wallKicks.GetLength(0));
    }

    private int Wrap(int input, int min, int max)
    {
        if (input < min)
        {
            return max - (min - input) % (max - min);
        }
        else
        {
            return min + (input - min) % (max - min);
        }
    }




}