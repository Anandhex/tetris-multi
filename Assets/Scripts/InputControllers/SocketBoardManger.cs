using UnityEngine;

public class SocketBoardManager : MonoBehaviour
{
    public GameObject boardPrefab;
    public Vector3Int boardPosition = new Vector3Int(0, 0, 0);
    public bool useDirectPlacement = true; // Toggle between direct placement and traditional input

    private Board activeBoard;
    private IPlayerInputController inputController;

    void Start()
    {
        SetupSocketGame();
    }

    void SetupSocketGame()
    {
        // Create container
        GameObject container = new GameObject("Socket Tetris Game");
        container.transform.position = boardPosition;

        // Create board
        GameObject boardObj = Instantiate(boardPrefab, container.transform);
        boardObj.transform.localPosition = Vector3.zero;

        activeBoard = boardObj.GetComponentInChildren<Board>();
        if (activeBoard == null)
        {
            Debug.LogError("Board component missing from prefab!");
            return;
        }

        // Add appropriate input controller
        if (useDirectPlacement)
        {
            var socketAgent = container.AddComponent<SocketTetrisAgent>();
            inputController = socketAgent;
            socketAgent.SetBoard(activeBoard);
            socketAgent.SetCurrentPiece(activeBoard.activePiece);
        }
        else
        {
            var socketInput = container.AddComponent<SocketInputController>();
            inputController = socketInput;
        }

        // Connect to board
        activeBoard.inputController = inputController;

        Debug.Log($"Socket-based Tetris setup complete. Mode: {(useDirectPlacement ? "Direct Placement" : "Traditional Input")}");
    }
}