using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.Sentis;
using UnityEngine;

public class BoardManager1 : MonoBehaviour
{
    public GameObject boardPrefab;
    public GameMode currentMode = GameMode.AI;
    public Vector3Int singlePlayerPosition = new Vector3Int(0, 0, 0);
    public ModelAsset nNModelAsset;

    private Board[] activeBoards;

    public enum GameMode { AI }
    void Awake()
    {
        Application.runInBackground = true;
    }


    void Start()
    {
        SetupGame();
    }

    void SetupGame()
    {
        ClearBoards();

        if (currentMode == GameMode.AI)
        {
            // Create a container for the board and ML agent
            GameObject container = new GameObject("ML Agent Container");
            container.transform.position = singlePlayerPosition;

            // Instantiate the board as a child of the container
            GameObject boardObj = Instantiate(boardPrefab, container.transform);
            boardObj.transform.localPosition = Vector3.zero;

            // Get the Board component
            var board = boardObj.GetComponentInChildren<Board>();
            if (board == null)
            {
                // Debug.LogError("Board component missing from prefab!");
                return;
            }

            // Add the ML Agent
            var agent = container.AddComponent<TetrisMLAgent>();


            // Connect the agent to the board
            board.inputController = agent;
            agent.currentPiece = board.activePiece;

            // Store reference to the board
            activeBoards = new[] { board };

            // agent.SetModel("TetrisAgent", nNModelAsset, InferenceDevice.Default);
            // Debug.Log("AI Board setup complete");
        }
    }

    void ClearBoards()
    {
        if (activeBoards == null) return;

        foreach (var board in activeBoards)
        {
            if (board != null)
            {
                Destroy(board.transform.parent.gameObject);
            }
        }

        activeBoards = null;
    }

    private void OnApplicationQuit()
    {
        ClearBoards();
    }
}