using UnityEngine;

public class PowerupBoardManager : MonoBehaviour
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
        GameObject container = new GameObject("Powerup Tetris Game");
        container.transform.position = boardPosition;

        // Create board

        Board[] activeBoard = FindObjectsOfType<Board>();
        if (activeBoard[0] == null)
        {
            Debug.LogError("Board component missing from prefab!");
            return;
        }

        // Add appropriate input controller
        if (useDirectPlacement)
        {
            var socketAgent = container.AddComponent<PowerupTetrisAgent>();
            socketAgent.SetBoard(activeBoard[0]);
            activeBoard[0].isMLTraining = true;
            // socketAgent.SetCurrentPiece(activeBoard.activePiece);
        }
        else
        {
            var socketInput = container.AddComponent<SocketInputController>();
            inputController = socketInput;
        }

        // Connect to board
        activeBoard[0].inputController = inputController;

        Debug.Log($"Socket-based Tetris setup complete. Mode: {(useDirectPlacement ? "Direct Placement" : "Traditional Input")}");
    }
}