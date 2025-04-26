using UnityEngine;

public class BoardManager : MonoBehaviour
{
    public GameObject boardPrefab;  // Assign your BoardResuse prefab here

    public enum GameMode { SinglePlayer, TwoPlayer, VsAI, AIVsAI, AI };
    private GameMode currentMode = Data.gameMode;

    public Vector3Int singlePlayerPosition = new Vector3Int(0, 0, 0);
    public Vector3Int player1Position = new Vector3Int(-10, 0, 0);
    public Vector3Int player2Position = new Vector3Int(10, 0, 0);

    private Board[] activeBoards;

    void Start()
    {
        if (boardPrefab == null)
        {
            Debug.LogError("Board Prefab is not assigned in the inspector!");
            return;
        }

        // Find the Board component in the prefab's hierarchy
        Board prefabBoard = boardPrefab.GetComponentInChildren<Board>();
        if (prefabBoard == null)
        {
            Debug.LogError("No Board component found in the prefab hierarchy! Make sure Board script is attached to a GameObject in the prefab.");
            return;
        }

        SetupGame();
    }

    public void SetupGame()
    {
        // Clear existing boards
        if (activeBoards != null)
        {
            foreach (Board board in activeBoards)
            {
                if (board != null)
                {
                    Destroy(board.transform.parent.gameObject); // Destroy the parent container
                }
            }
        }

        if (currentMode == GameMode.SinglePlayer)
        {
            activeBoards = new Board[1];
            activeBoards[0] = CreateBoard(singlePlayerPosition, new SinglePlayerInputController(), "Player");
        }
        else if (currentMode == GameMode.VsAI)
        {
            activeBoards = new Board[2];
            activeBoards[0] = CreateBoard(player1Position, new Player1InputController(), "Human Player");
            activeBoards[1] = CreateBoard(player2Position, new AIController(), "AI Player");
        }
        else if (currentMode == GameMode.AI)
        {
            activeBoards = new Board[1];
            activeBoards[0] = CreateBoard(singlePlayerPosition, new AIController(), "AI Player");
        }
        else if (currentMode == GameMode.AIVsAI)
        {
            activeBoards = new Board[2];
            activeBoards[0] = CreateBoard(player1Position, new AIController(), "AI Player 1");
            activeBoards[0] = CreateBoard(player2Position, new AIController(), "AI Player 2");
        }
        else
        {
            activeBoards = new Board[2];
            activeBoards[0] = CreateBoard(player1Position, new Player1InputController(), "Player 1");
            activeBoards[1] = CreateBoard(player2Position, new Player2InputController(), "Player 2");
        }
    }

    Board CreateBoard(Vector3Int position, IPlayerInputController input, string playerLabel)
    {
        Debug.Log($"Creating board for {playerLabel} at position {position}");

        // Create a container for the board
        GameObject container = new GameObject(playerLabel + " Container");
        container.transform.position = position;

        // Instantiate the board prefab as a child of the container
        GameObject boardObj = Instantiate(boardPrefab, container.transform, false);
        boardObj.name = playerLabel + " Board";

        // Find the Board component in the hierarchy
        Board board = boardObj.GetComponentInChildren<Board>();
        if (board == null)
        {
            Debug.LogError("Board component not found in instantiated prefab hierarchy!");
            Destroy(container);
            return null;
        }

        // Set the input controller
        board.inputController = input;
        board.playerTag = playerLabel;



        return board;
    }
    // Method to switch game modes
    public void SetGameMode(GameMode mode)
    {
        currentMode = mode;
        SetupGame();
    }

    // Reset the current game
    public void ResetGame()
    {
        SetupGame();
    }
}
