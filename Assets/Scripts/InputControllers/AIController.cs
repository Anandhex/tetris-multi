using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine;
using Unity.MLAgents.Policies;
using System.Collections.Generic;
using System.Linq;
using System.Collections;


public class TetrisMLAgent : Agent, IPlayerInputController
{
    private Board board;
    public Piece currentPiece;
    private MLAgentDebugger debugger;


    private HashSet<Piece> processedPieces = new HashSet<Piece>();

    private StatsRecorder m_StatsRecorder;

    private RewardWeights rewardWeights = new RewardWeights();

    [Header("Curriculum Parameters")]
    public int allowedTetrominoTypes = 7;
    public int curriculumBoardPreset;
    public float curriculumBoardHeight = 20f;
    public float curriculumDropSpeed = 0.75f;
    public float curriculumHolePenaltyWeight = 0.5f;
    public bool enableAdvancedMechanics = false;
    private int episodeSteps = 0;
    private bool moveLeft;
    private bool moveRight;
    private bool moveDown;
    private bool rotateLeft;
    private bool rotateRight;
    private bool hardDrop;
    private int lastPlacementIndex;

    public override void Initialize()
    {
        var envParams = Academy.Instance.EnvironmentParameters;
        m_StatsRecorder = Academy.Instance.StatsRecorder;
        curriculumBoardPreset = (int)Academy.Instance.EnvironmentParameters.GetWithDefault("board_preset", 6);


        // Get curriculum parameters
        allowedTetrominoTypes = (int)envParams.GetWithDefault("tetromino_types", 7f);
        curriculumBoardHeight = envParams.GetWithDefault("board_height", 20f);
        curriculumDropSpeed = envParams.GetWithDefault("drop_speed", 0.75f);
        curriculumHolePenaltyWeight = envParams.GetWithDefault("hole_penalty_weight", 0.5f);
        enableAdvancedMechanics = envParams.GetWithDefault("enable_t_spins", 0f) > 0.5f;


        // Apply curriculum settings to reward weights
        rewardWeights.holeCreationPenalty *= curriculumHolePenaltyWeight;
        rewardWeights.clearReward = envParams.GetWithDefault("clearReward", 5.0f);
        rewardWeights.comboMultiplier = envParams.GetWithDefault("comboMultiplier", 0.5f);
        rewardWeights.perfectClearBonus = envParams.GetWithDefault("perfectClearBonus", 50.0f);


    }


    private void Start()
    {
        if (board != null)
        {
            board.inputController = this;
        }
    }



    // Called by Board.cs to set the current piece reference



    public override void OnEpisodeBegin()
    {
        var envParams = Academy.Instance.EnvironmentParameters;
        processedPieces.Clear();


        allowedTetrominoTypes = (int)envParams.GetWithDefault("tetromino_types", 7f);
        curriculumBoardHeight = envParams.GetWithDefault("board_height", 20f);
        curriculumDropSpeed = envParams.GetWithDefault("drop_speed", 0.75f);
        curriculumHolePenaltyWeight = envParams.GetWithDefault("hole_penalty_weight", 0.5f);
        enableAdvancedMechanics = envParams.GetWithDefault("enable_t_spins", 0f) > 0.5f;
        curriculumBoardPreset = (int)envParams.GetWithDefault("board_preset", 6);


        rewardWeights.holeCreationPenalty *= curriculumHolePenaltyWeight;
        episodeSteps = 0;
        m_StatsRecorder.Add("Episode/Started", 1f);

    }

    private void Awake()
    {
        debugger = GetComponent<MLAgentDebugger>();
        if (debugger == null)
        {
            debugger = gameObject.AddComponent<MLAgentDebugger>();
        }


        // Find board in children
        board = GetComponentInChildren<Board>();


        var behavior = gameObject.GetComponent<BehaviorParameters>();
        if (behavior == null)
        {
            behavior = gameObject.AddComponent<BehaviorParameters>();
            behavior.BehaviorName = "TetrisAgent";
            behavior.BrainParameters.VectorObservationSize = 218;
            behavior.BrainParameters.NumStackedVectorObservations = 1;


            // Two discrete actions - column (10 options) and rotation (4 options)
            ActionSpec actionSpec = ActionSpec.MakeDiscrete(new int[] { 40 });
            behavior.BrainParameters.ActionSpec = actionSpec;
        }
        else
        {
            behavior.BehaviorName = "TetrisAgent";
            behavior.BrainParameters.VectorObservationSize = 218;
            behavior.BrainParameters.NumStackedVectorObservations = 1;


            // Two discrete actions - column (10 options) and rotation (4 options)
            ActionSpec actionSpec = ActionSpec.MakeDiscrete(new int[] { 40 });
            behavior.BrainParameters.ActionSpec = actionSpec;
        }


        // Add a decision requester component if it doesn't exist
        var requestor = gameObject.GetComponent<DecisionRequester>();
        if (requestor == null)
        {
            requestor = gameObject.AddComponent<DecisionRequester>();
        }

        requestor.DecisionPeriod = 1000000; // Very large number so it doesn't auto-request
        requestor.TakeActionsBetweenDecisions = false;


    }





    // Keep the simple placement logic without complex timing
    public override void OnActionReceived(ActionBuffers actions)
    {
        if (currentPiece == null || board == null)
            return;
        episodeSteps++;
        // Check if we've already processed this piece
        if (processedPieces.Contains(currentPiece))
        {
            return; // Already handled this piece
        }

        // Mark piece as processed
        processedPieces.Add(currentPiece);

        // Generate and execute placement
        int[,] currentBoard = GetBoardState();
        List<PlacementInfo> allPlacements = GenerateAllPossiblePlacements(currentPiece, currentBoard);

        if (allPlacements.Count == 0)
        {
            AddReward(rewardWeights.deathPenalty);
            OnGameOver();
            return;
        }

        int selectedPlacement = actions.DiscreteActions[0];
        selectedPlacement = Mathf.Clamp(selectedPlacement, 0, allPlacements.Count - 1);
        if (debugger != null)
        {
            debugger.SetLastPlacementAction(selectedPlacement);
        }

        ExecutePlacement(allPlacements[selectedPlacement]);
        CalculatePlacementReward(allPlacements[selectedPlacement]);

    }
    private void ExecutePlacement(PlacementInfo placement)
    {
        // Move piece to target position and rotation
        // This would involve setting the piece's position and rotation
        // then dropping it using hard drop

        // For now, simulate the action by directly placing
        StartCoroutine(ExecutePlacementCoroutine(placement));
    }

    private IEnumerator ExecutePlacementCoroutine(PlacementInfo placement)
    {
        // Rotate piece to target rotation
        while (currentPiece.rotationIndex != placement.targetRotation)
        {
            rotateRight = true;
            yield return new WaitForFixedUpdate();
            rotateRight = false;
            yield return new WaitForFixedUpdate();
        }

        // Move piece to target column
        int currentCol = currentPiece.position.x - board.Bounds.xMin;
        while (currentCol != placement.targetColumn)
        {
            if (currentCol < placement.targetColumn)
            {
                moveRight = true;
                yield return new WaitForFixedUpdate();
                moveRight = false;
            }
            else
            {
                moveLeft = true;
                yield return new WaitForFixedUpdate();
                moveLeft = false;
            }
            yield return new WaitForFixedUpdate();
            currentCol = currentPiece.position.x - board.Bounds.xMin;
        }

        // Hard drop
        hardDrop = true;
        yield return new WaitForFixedUpdate();
        hardDrop = false;
    }

    private void CalculatePlacementReward(PlacementInfo placement)
    {
        // Reward based on placement quality
        AddReward(placement.linesCleared * rewardWeights.clearReward);
        AddReward(-placement.holes * rewardWeights.holeCreationPenalty);
        AddReward(-placement.aggregateHeight * 0.01f);
        AddReward(-placement.bumpiness * 0.01f);

        // Bonus for Tetris (4 lines)
        if (placement.linesCleared == 4)
        {
            AddReward(rewardWeights.perfectClearBonus);
        }

        // Small step penalty to encourage faster play
        AddReward(-0.001f);

        RecordStats(placement);

        // Notify debugger
        if (debugger != null)
        {
            // debugger.OnPlacementMade(lastPlacementIndex, placement);
        }

    }


    public void OnGameOver()
    {
        m_StatsRecorder.Add("Episode/Length", episodeSteps);
        m_StatsRecorder.Add("Episode/Final Reward", GetCumulativeReward());
        // ADD THIS LINE:
        if (debugger != null)
        {
            debugger.OnEpisodeEnd();
        }

        EndEpisode();

    }


    // IPlayerInputController implementation - unchanged
    public bool GetLeft() => moveLeft;
    public bool GetRight() => moveRight;
    public bool GetRotateLeft() => rotateLeft;
    public bool GetRotateRight() => rotateRight;
    public bool GetDown() => moveDown;
    public bool GetHardDrop() => hardDrop;



    private void RecordStats(PlacementInfo placement)
    {
        // Record key metrics to TensorBoard
        m_StatsRecorder.Add("Tetris/Lines Cleared", placement.linesCleared);
        m_StatsRecorder.Add("Tetris/Board Height", placement.maxHeight);
        m_StatsRecorder.Add("Tetris/Holes Created", placement.holes);
        m_StatsRecorder.Add("Tetris/Aggregate Height", placement.aggregateHeight);
        m_StatsRecorder.Add("Tetris/Bumpiness", placement.bumpiness);
        m_StatsRecorder.Add("Tetris/Episode Steps", episodeSteps);

        // Record curriculum parameters
        m_StatsRecorder.Add("Curriculum/Board Height", curriculumBoardHeight);
        m_StatsRecorder.Add("Curriculum/Tetromino Types", allowedTetrominoTypes);
        m_StatsRecorder.Add("Curriculum/Board Preset", curriculumBoardPreset);
    }

    public override void CollectObservations(VectorSensor sensor)
    {


        // Convert current board state to 2D array for simulation
        int[,] currentBoard = GetBoardState();

        // Generate all possible placements for current piece
        List<PlacementInfo> allPlacements = GenerateAllPossiblePlacements(currentPiece, currentBoard);

        // 1. ALL POSSIBLE PLACEMENTS (34 placements × 6 features = 204 observations)
        for (int i = 0; i < 34; i++)
        {
            if (i < allPlacements.Count)
            {
                PlacementInfo placement = allPlacements[i];
                sensor.AddObservation(placement.linesCleared / 4f);      // Normalized (0-1)
                sensor.AddObservation(placement.aggregateHeight / 40f);  // Normalized
                sensor.AddObservation(placement.maxHeight / 20f);        // Normalized
                sensor.AddObservation(placement.holes / 20f);            // Normalized
                sensor.AddObservation(placement.bumpiness / 30f);        // Normalized
                sensor.AddObservation(placement.wellDepth / 15f);        // Normalized
            }
            else
            {
                // Pad with zeros for unused placement slots
                for (int j = 0; j < 6; j++)
                {
                    sensor.AddObservation(0f);
                }
            }
        }

        // 2. CURRENT PIECE ONE-HOT (7 observations)
        for (int i = 0; i < 7; i++)
        {
            sensor.AddObservation(((int)currentPiece.data.tetromino == i) ? 1f : 0f);
        }

        // 3. NEXT PIECE ONE-HOT (7 observations)
        if (board.nextPieceData.tetromino != Tetromino.I) // Check if valid
        {
            for (int i = 0; i < 7; i++)
            {
                sensor.AddObservation(((int)board.nextPieceData.tetromino == i) ? 1f : 0f);
            }
        }
        else
        {
            for (int i = 0; i < 7; i++)
            {
                sensor.AddObservation(0f);
            }
        }

        // Total: 204 + 7 + 7 = 218 observations
    }

    private int[,] GetBoardState()
    {
        var bounds = board.Bounds;
        int[,] boardArray = new int[bounds.height, bounds.width];

        for (int y = 0; y < bounds.height; y++)
        {
            for (int x = 0; x < bounds.width; x++)
            {
                Vector3Int pos = new Vector3Int(bounds.xMin + x, bounds.yMin + y, 0);
                boardArray[bounds.height - 1 - y, x] = board.tilemap.HasTile(pos) ? 1 : 0;
            }
        }

        return boardArray;
    }

    // Generate all possible placements for the current piece
    private List<PlacementInfo> GenerateAllPossiblePlacements(Piece piece, int[,] currentBoard)
    {
        List<PlacementInfo> placements = new List<PlacementInfo>();

        // Get all rotations of the current piece
        List<int[,]> rotations = GetPieceRotations(piece);

        for (int rotation = 0; rotation < rotations.Count; rotation++)
        {
            int[,] pieceShape = rotations[rotation];
            int pieceWidth = pieceShape.GetLength(1);
            int boardWidth = currentBoard.GetLength(1);

            // Try each column position
            for (int col = 0; col <= boardWidth - pieceWidth; col++)
            {
                if (CanPlacePieceAt(pieceShape, currentBoard, col, out int landingRow))
                {
                    // Create a copy of the board for simulation
                    int[,] simulatedBoard = CopyBoard(currentBoard);

                    // Place the piece
                    PlacePieceOnBoard(pieceShape, simulatedBoard, col, landingRow);

                    // Clear lines and get count
                    int linesCleared = ClearLinesAndCount(simulatedBoard);

                    // Calculate features
                    PlacementInfo placement = new PlacementInfo
                    {
                        linesCleared = linesCleared,
                        aggregateHeight = CalculateAggregateHeight(simulatedBoard),
                        maxHeight = CalculateMaxHeight(simulatedBoard),
                        holes = CalculateHoles(simulatedBoard),
                        bumpiness = CalculateBumpiness(simulatedBoard),
                        wellDepth = CalculateWellDepth(simulatedBoard),
                        targetColumn = col,
                        targetRotation = rotation
                    };

                    placements.Add(placement);
                }
            }
        }

        return placements;
    }
    private List<int[,]> GetPieceRotations(Piece piece)
    {
        List<int[,]> rotations = new List<int[,]>();
        TetrominoData data = piece.data;

        // Convert piece cells to different rotations
        for (int rotation = 0; rotation < 4; rotation++)
        {
            Vector3Int[] rotatedCells = new Vector3Int[data.cells.Length];

            // Copy original cells
            for (int i = 0; i < data.cells.Length; i++)
            {
                rotatedCells[i] = (Vector3Int)data.cells[i];
            }

            // Apply rotation transformations
            for (int r = 0; r < rotation; r++)
            {
                for (int i = 0; i < rotatedCells.Length; i++)
                {
                    Vector3Int cell = rotatedCells[i];
                    // 90-degree clockwise rotation matrix
                    int newX = -cell.y;
                    int newY = cell.x;
                    rotatedCells[i] = new Vector3Int(newX, newY, 0);
                }
            }

            // Convert to 2D array representation
            int[,] shapeArray = ConvertCellsToArray(rotatedCells);
            rotations.Add(shapeArray);
        }

        return rotations;
    }

    // Convert piece cells to 2D array
    private int[,] ConvertCellsToArray(Vector3Int[] cells)
    {
        if (cells.Length == 0) return new int[1, 1];

        // Find bounds
        int minX = cells[0].x, maxX = cells[0].x;
        int minY = cells[0].y, maxY = cells[0].y;

        foreach (var cell in cells)
        {
            minX = Mathf.Min(minX, cell.x);
            maxX = Mathf.Max(maxX, cell.x);
            minY = Mathf.Min(minY, cell.y);
            maxY = Mathf.Max(maxY, cell.y);
        }

        int width = maxX - minX + 1;
        int height = maxY - minY + 1;
        int[,] array = new int[height, width];

        foreach (var cell in cells)
        {
            array[cell.y - minY, cell.x - minX] = 1;
        }

        return array;
    }

    // Check if piece can be placed at specific column
    private bool CanPlacePieceAt(int[,] pieceShape, int[,] board, int col, out int landingRow)
    {
        int boardHeight = board.GetLength(0);
        int pieceHeight = pieceShape.GetLength(0);
        landingRow = -1;

        // Drop piece from top until collision
        for (int row = 0; row <= boardHeight - pieceHeight; row++)
        {
            if (HasCollision(pieceShape, board, row, col))
            {
                landingRow = row - 1;
                return landingRow >= 0;
            }
        }

        landingRow = boardHeight - pieceHeight;
        return landingRow >= 0;
    }

    // Check collision between piece and board
    private bool HasCollision(int[,] piece, int[,] board, int row, int col)
    {
        int pieceHeight = piece.GetLength(0);
        int pieceWidth = piece.GetLength(1);

        for (int r = 0; r < pieceHeight; r++)
        {
            for (int c = 0; c < pieceWidth; c++)
            {
                if (piece[r, c] == 1)
                {
                    int boardRow = row + r;
                    int boardCol = col + c;

                    if (boardRow >= board.GetLength(0) ||
                        boardCol >= board.GetLength(1) ||
                        board[boardRow, boardCol] == 1)
                    {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    // Place piece on board
    private void PlacePieceOnBoard(int[,] piece, int[,] board, int col, int row)
    {
        int pieceHeight = piece.GetLength(0);
        int pieceWidth = piece.GetLength(1);

        for (int r = 0; r < pieceHeight; r++)
        {
            for (int c = 0; c < pieceWidth; c++)
            {
                if (piece[r, c] == 1)
                {
                    board[row + r, col + c] = 1;
                }
            }
        }
    }

    // Clear lines and return count
    private int ClearLinesAndCount(int[,] board)
    {
        int linesCleared = 0;
        int height = board.GetLength(0);
        int width = board.GetLength(1);

        for (int row = height - 1; row >= 0; row--)
        {
            bool fullLine = true;
            for (int col = 0; col < width; col++)
            {
                if (board[row, col] == 0)
                {
                    fullLine = false;
                    break;
                }
            }

            if (fullLine)
            {
                linesCleared++;
                // Shift rows down
                for (int moveRow = row; moveRow > 0; moveRow--)
                {
                    for (int col = 0; col < width; col++)
                    {
                        board[moveRow, col] = board[moveRow - 1, col];
                    }
                }
                // Clear top row
                for (int col = 0; col < width; col++)
                {
                    board[0, col] = 0;
                }
                row++; // Check same row again after shift
            }
        }

        return linesCleared;
    }

    // Calculate aggregate height (sum of all column heights)
    private int CalculateAggregateHeight(int[,] board)
    {
        int height = board.GetLength(0);
        int width = board.GetLength(1);
        int totalHeight = 0;

        for (int col = 0; col < width; col++)
        {
            for (int row = 0; row < height; row++)
            {
                if (board[row, col] == 1)
                {
                    totalHeight += (height - row);
                    break;
                }
            }
        }

        return totalHeight;
    }

    // Calculate maximum height
    private int CalculateMaxHeight(int[,] board)
    {
        int height = board.GetLength(0);
        int width = board.GetLength(1);
        int maxHeight = 0;

        for (int col = 0; col < width; col++)
        {
            for (int row = 0; row < height; row++)
            {
                if (board[row, col] == 1)
                {
                    maxHeight = Mathf.Max(maxHeight, height - row);
                    break;
                }
            }
        }

        return maxHeight;
    }

    // Calculate number of holes
    private int CalculateHoles(int[,] board)
    {
        int height = board.GetLength(0);
        int width = board.GetLength(1);
        int holes = 0;

        for (int col = 0; col < width; col++)
        {
            bool blockFound = false;
            for (int row = 0; row < height; row++)
            {
                if (board[row, col] == 1)
                {
                    blockFound = true;
                }
                else if (blockFound)
                {
                    holes++;
                }
            }
        }

        return holes;
    }

    // Calculate bumpiness (sum of height differences between adjacent columns)
    private int CalculateBumpiness(int[,] board)
    {
        int height = board.GetLength(0);
        int width = board.GetLength(1);
        int[] columnHeights = new int[width];

        // Get height of each column
        for (int col = 0; col < width; col++)
        {
            for (int row = 0; row < height; row++)
            {
                if (board[row, col] == 1)
                {
                    columnHeights[col] = height - row;
                    break;
                }
            }
        }

        // Calculate bumpiness
        int bumpiness = 0;
        for (int col = 0; col < width - 1; col++)
        {
            bumpiness += Mathf.Abs(columnHeights[col] - columnHeights[col + 1]);
        }

        return bumpiness;
    }

    // Calculate well depth (depth of single-width wells)
    private int CalculateWellDepth(int[,] board)
    {
        int height = board.GetLength(0);
        int width = board.GetLength(1);
        int totalWellDepth = 0;

        for (int col = 0; col < width; col++)
        {
            bool leftWall = (col == 0) || HasBlockInColumn(board, col - 1);
            bool rightWall = (col == width - 1) || HasBlockInColumn(board, col + 1);

            if (leftWall && rightWall)
            {
                // This is a potential well, calculate its depth
                int wellDepth = 0;
                for (int row = height - 1; row >= 0; row--)
                {
                    if (board[row, col] == 0)
                    {
                        wellDepth++;
                    }
                    else
                    {
                        break;
                    }
                }
                totalWellDepth += wellDepth;
            }
        }

        return totalWellDepth;
    }

    // Helper function to check if column has any blocks
    private bool HasBlockInColumn(int[,] board, int col)
    {
        int height = board.GetLength(0);
        for (int row = 0; row < height; row++)
        {
            if (board[row, col] == 1) return true;
        }
        return false;
    }

    // Copy board for simulation
    private int[,] CopyBoard(int[,] original)
    {
        int height = original.GetLength(0);
        int width = original.GetLength(1);
        int[,] copy = new int[height, width];

        for (int r = 0; r < height; r++)
        {
            for (int c = 0; c < width; c++)
            {
                copy[r, c] = original[r, c];
            }
        }

        return copy;
    }

    public void SetCurrentPiece(Piece piece)
    {
        currentPiece = piece;

        processedPieces.RemoveWhere(p => p == null);

        // Request decision only for new pieces
        if (!processedPieces.Contains(piece))
        {
            RequestDecision();
        }

    }


    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;

        // Column selection (0-9)
        if (Input.GetKey(KeyCode.Alpha1)) discreteActionsOut[0] = 0;
        else if (Input.GetKey(KeyCode.Alpha2)) discreteActionsOut[0] = 1;
        else if (Input.GetKey(KeyCode.Alpha3)) discreteActionsOut[0] = 2;
        else if (Input.GetKey(KeyCode.Alpha4)) discreteActionsOut[0] = 3;
        else if (Input.GetKey(KeyCode.Alpha5)) discreteActionsOut[0] = 4;
        else if (Input.GetKey(KeyCode.Alpha6)) discreteActionsOut[0] = 5;
        else if (Input.GetKey(KeyCode.Alpha7)) discreteActionsOut[0] = 6;
        else if (Input.GetKey(KeyCode.Alpha8)) discreteActionsOut[0] = 7;
        else if (Input.GetKey(KeyCode.Alpha9)) discreteActionsOut[0] = 8;
        else if (Input.GetKey(KeyCode.Alpha0)) discreteActionsOut[0] = 9;
        else discreteActionsOut[0] = 4; // Default to middle column

        // Rotation selection (0-3)
        if (Input.GetKey(KeyCode.Q)) discreteActionsOut[1] = 0;
        else if (Input.GetKey(KeyCode.W)) discreteActionsOut[1] = 1;
        else if (Input.GetKey(KeyCode.E)) discreteActionsOut[1] = 2;
        else if (Input.GetKey(KeyCode.R)) discreteActionsOut[1] = 3;
        else discreteActionsOut[1] = 0; // Default to no rotation
    }
}
