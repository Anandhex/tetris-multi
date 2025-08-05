using System.Collections.Generic;
using Unity.Sentis;
using UnityEngine;
using UnityEngine.SceneManagement;

public class BoardManager : MonoBehaviour
{
    public GameObject boardPrefab;  // Assign your BoardResuse prefab here
    public static BoardManager Instance { get; private set; }
    public enum GameMode { SinglePlayer, TwoPlayer, VsAI, AIVsAI, AI };
    public GameMode currentMode = GameMode.AIVsAI;

    public Vector3Int singlePlayerPosition = new Vector3Int(0, 0, 0);
    public ModelAsset sentisModelAsset;
    public ModelAsset powerupAsset;

    public Vector3Int player1Position = new Vector3Int(-10, 0, 0);
    public Vector3Int player2Position = new Vector3Int(10, 0, 0);

    private Board[] activeBoards;

    void Awake()
    {
        if (Instance != null && Instance != this) Destroy(gameObject);
        else Instance = this;
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Escape))
        {
            SceneManager.LoadScene(0);
        }
    }
    void Start()
    {
        if (boardPrefab == null)
        {
            // Debug.LogError("Board Prefab is not assigned in the inspector!");
            return;
        }

        // Find the Board component in the prefab's hierarchy
        Board prefabBoard = boardPrefab.GetComponentInChildren<Board>();
        if (prefabBoard == null)
        {
            // Debug.LogError("No Board component found in the prefab hierarchy! Make sure Board script is attached to a GameObject in the prefab.");
            return;
        }

        SetupGame();
    }

    public void SetupGame()
    {

        var player1KeyMapping = new List<PowerupKeyMapping>
    {
        new PowerupKeyMapping { key = KeyCode.Alpha1, powerupType = PowerUpType.LineBlaster },
        new PowerupKeyMapping { key = KeyCode.Alpha2, powerupType = PowerUpType.Gravity },
        new PowerupKeyMapping { key = KeyCode.Alpha3, powerupType = PowerUpType.Bomb }
    };
        var player2KeyMapping = new List<PowerupKeyMapping>
    {
        new PowerupKeyMapping { key = KeyCode.Alpha7, powerupType = PowerUpType.LineBlaster },
        new PowerupKeyMapping { key = KeyCode.Alpha8, powerupType = PowerUpType.Gravity },
        new PowerupKeyMapping { key = KeyCode.Alpha9, powerupType = PowerUpType.Bomb }
    };
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

        if (Data.gameMode == GameMode.SinglePlayer)
        {
            activeBoards = new Board[1];
            activeBoards[0] = CreateBoard(singlePlayerPosition, new SinglePlayerInputController(), "Player");
            activeBoards[0].powerupKeyMapping = player1KeyMapping;
        }
        else if (Data.gameMode == GameMode.VsAI)
        {
            player1KeyMapping.Add(new PowerupKeyMapping { key = KeyCode.Alpha4, powerupType = PowerUpType.WildCard });
            player2KeyMapping.Add(new PowerupKeyMapping { key = KeyCode.Alpha0, powerupType = PowerUpType.WildCard });
            activeBoards = new Board[2];
            activeBoards[0] = CreateBoard(player1Position, new Player1InputController(), "Human Player");

            // Use TetrisMLAgent instead of AIController
            TetrisSentisAgent mlAgent = gameObject.AddComponent<TetrisSentisAgent>();
            activeBoards[1] = CreateBoard(player2Position, mlAgent, "ML Player");

            activeBoards[0].opponentBoard = activeBoards[1];
            activeBoards[1].opponentBoard = activeBoards[0];
            activeBoards[0].powerupKeyMapping = player1KeyMapping;
            activeBoards[1].powerupKeyMapping = player2KeyMapping;

        }
        else if (Data.gameMode == GameMode.AI)
        {
            activeBoards = new Board[1];
            TetrisSentisAgent[] existingAgents = FindObjectsOfType<TetrisSentisAgent>();
            foreach (TetrisSentisAgent agent in existingAgents)
            {
                if (agent != null)
                {
                    // Debug.Log($"Destroying existing agent: {agent.GetInstanceID()}");
                    Destroy(agent);
                }
            }

            // Use TetrisMLAgent instead of AIController
            TetrisSentisAgent mlAgent = gameObject.AddComponent<TetrisSentisAgent>();
            gameObject.AddComponent<PowerupTetrisAgent>();
            activeBoards[0] = CreateBoard(singlePlayerPosition, mlAgent, "ML Player");
            activeBoards[0].isMLTraining = false;
            activeBoards[0].powerupKeyMapping = player1KeyMapping;

        }
        else if (Data.gameMode == GameMode.AIVsAI)
        {
            activeBoards = new Board[2];
            player1KeyMapping.Add(new PowerupKeyMapping { key = KeyCode.Alpha4, powerupType = PowerUpType.WildCard });
            player2KeyMapping.Add(new PowerupKeyMapping { key = KeyCode.Alpha0, powerupType = PowerUpType.WildCard });

            // Use TetrisMLAgents for both players
            TetrisSentisAgent mlAgent1 = gameObject.AddComponent<TetrisSentisAgent>();
            TetrisSentisAgent mlAgent2 = gameObject.AddComponent<TetrisSentisAgent>();

            activeBoards[0] = CreateBoard(player1Position, mlAgent1, "ML Player 1");
            activeBoards[1] = CreateBoard(player2Position, mlAgent2, "ML Player 2");
            activeBoards[0].opponentBoard = activeBoards[1];
            activeBoards[1].opponentBoard = activeBoards[0];
            activeBoards[0].powerupKeyMapping = player1KeyMapping;
            activeBoards[1].powerupKeyMapping = player2KeyMapping;
            mlAgent1.opponentAgent = mlAgent2;
            mlAgent2.opponentAgent = mlAgent1;


        }
        else
        {
            player1KeyMapping.Add(new PowerupKeyMapping { key = KeyCode.Alpha4, powerupType = PowerUpType.WildCard });
            player2KeyMapping.Add(new PowerupKeyMapping { key = KeyCode.Alpha0, powerupType = PowerUpType.WildCard });
            activeBoards = new Board[2];
            activeBoards[0] = CreateBoard(player1Position, new Player1InputController(), "Player 1");
            activeBoards[1] = CreateBoard(player2Position, new Player2InputController(), "Player 2");
            activeBoards[0].opponentBoard = activeBoards[1];
            activeBoards[1].opponentBoard = activeBoards[0];
            activeBoards[0].powerupKeyMapping = player1KeyMapping;
            activeBoards[1].powerupKeyMapping = player2KeyMapping;


        }
    }
    Board CreateBoard(Vector3Int position, IPlayerInputController input, string playerLabel)
    {
        // Debug.Log($"Creating board for {playerLabel} at position {position}");

        // Create a container for the board
        GameObject container = new GameObject(playerLabel + " Container");
        container.transform.position = position;

        // Instantiate the board prefab as a child of the container
        GameObject boardObj = Instantiate(boardPrefab, container.transform, false);
        boardObj.transform.localPosition = Vector3.zero;

        boardObj.name = playerLabel + " Board";

        // Find the Board component in the hierarchy
        Board board = boardObj.GetComponentInChildren<Board>();
        if (board == null)
        {
            // Debug.LogError("Board component not found in instantiated prefab hierarchy!");
            Destroy(container);
            return null;
        }

        // Set the input controller
        board.inputController = input;
        // Debug.Log($"Assigned input controller: {input} to board: {board.playerTag}");

        // Special handling for ML agent
        // if (input is TetrisMLAgent mlAgent)
        // {
        //     mlAgent.Initialize(board);
        // }

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
