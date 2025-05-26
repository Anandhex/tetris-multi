using TMPro;
using UnityEngine;
using UnityEngine.UI; // Added for Image component

public class BoardManager : MonoBehaviour
{
    public GameObject boardPrefab;  // Assign your BoardResuse prefab here
    public Canvas uiCanvas; // Reference to your UI Canvas

    public enum GameMode { SinglePlayer, TwoPlayer, VsAI, AIVsAI, AI };
    private GameMode currentMode = Data.gameMode;

    public Vector3Int singlePlayerPosition = new Vector3Int(0, 0, 0);
    public Vector3Int player1Position = new Vector3Int(-10, 0, 0);
    public Vector3Int player2Position = new Vector3Int(10, 0, 0);

    public Board[] activeBoards { get; private set; }

    // Reference to the PowerUp UI container we create
    private GameObject powerUpUIContainer;

    public Board GetPlayerBoard(int playerIndex)
    {
        if (activeBoards != null && playerIndex >= 0 && playerIndex < activeBoards.Length)
        {
            return activeBoards[playerIndex];
        }
        return null;
    }


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

        // Check if we have the UI Canvas
        if (uiCanvas == null)
        {
            // Try to find canvas in scene
            uiCanvas = FindObjectOfType<Canvas>();
            if (uiCanvas == null)
            {
                Debug.LogError("UI Canvas is not assigned and could not be found in the scene!");
                return;
            }
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
            activeBoards[1] = CreateBoard(player2Position, new AIController(), "AI Player 2");
        }
        else
        {
            activeBoards = new Board[2];
            activeBoards[0] = CreateBoard(player1Position, new Player1InputController(), "Player 1");
            activeBoards[1] = CreateBoard(player2Position, new Player2InputController(), "Player 2");
        }

        // Setup UI elements for power-ups
        SetupPowerUpUI();
        AddPowerUpInstructions();
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

    private void SetupPowerUpUI()
    {
        // Clear any existing power-up UI
        Transform existingContainer = uiCanvas.transform.Find("PowerUpUI");
        if (existingContainer != null)
        {
            Destroy(existingContainer.gameObject);
        }

        // Create a power-up UI container
        powerUpUIContainer = new GameObject("PowerUpUI");
        powerUpUIContainer.transform.SetParent(uiCanvas.transform);
        RectTransform containerRect = powerUpUIContainer.AddComponent<RectTransform>();
        containerRect.anchorMin = new Vector2(0.5f, 0);
        containerRect.anchorMax = new Vector2(0.5f, 0);
        containerRect.pivot = new Vector2(0.5f, 0);
        containerRect.anchoredPosition = new Vector2(0, 100);
        containerRect.sizeDelta = new Vector2(400, 100);

        // Create power-up slot prefabs
        PowerUpUI[] slots = new PowerUpUI[3];
        for (int i = 0; i < 3; i++)
        {
            GameObject slotObj = new GameObject("PowerUpSlot" + i);
            slotObj.transform.SetParent(powerUpUIContainer.transform);

            RectTransform slotRect = slotObj.AddComponent<RectTransform>();
            slotRect.anchorMin = new Vector2(0, 0);
            slotRect.anchorMax = new Vector2(0, 0);
            slotRect.pivot = new Vector2(0.5f, 0.5f);
            slotRect.anchoredPosition = new Vector2(100 + i * 120, 50);
            slotRect.sizeDelta = new Vector2(100, 100);

            // Add UI elements
            Image frameImage = slotObj.AddComponent<Image>();
            frameImage.color = Color.gray;

            GameObject iconObj = new GameObject("Icon");
            iconObj.transform.SetParent(slotObj.transform);
            RectTransform iconRect = iconObj.AddComponent<RectTransform>();
            iconRect.anchorMin = Vector2.zero;
            iconRect.anchorMax = Vector2.one;
            iconRect.offsetMin = new Vector2(10, 10);
            iconRect.offsetMax = new Vector2(-10, -30);
            Image iconImage = iconObj.AddComponent<Image>();

            GameObject textObj = new GameObject("Text");
            textObj.transform.SetParent(slotObj.transform);
            RectTransform textRect = textObj.AddComponent<RectTransform>();
            textRect.anchorMin = new Vector2(0, 0);
            textRect.anchorMax = new Vector2(1, 0);
            textRect.offsetMin = new Vector2(5, 0);
            textRect.offsetMax = new Vector2(-5, 25);
            TextMeshProUGUI nameText = textObj.AddComponent<TextMeshProUGUI>();
            nameText.fontSize = 14;
            nameText.alignment = TextAlignmentOptions.Center;
            nameText.text = "Empty";

            // Create PowerUpUI component
            PowerUpUI powerUpUI = slotObj.AddComponent<PowerUpUI>();
            powerUpUI.iconImage = iconImage;
            powerUpUI.nameText = nameText;
            powerUpUI.frameImage = frameImage;

            slots[i] = powerUpUI;
        }
        Debug.Log(PowerUpManager.Instance);
        // Register slots with PowerUpManager
        if (PowerUpManager.Instance != null)
        {
            // Pass both the container and slots to PowerUpManager
            PowerUpManager.Instance.SetupUIContainer(powerUpUIContainer, slots);
        }
        else
        {
            Debug.LogWarning("PowerUpManager instance not found! Make sure it's initialized before BoardManager.");
        }
    }

    private void AddPowerUpInstructions()
    {
        // Remove any existing instructions
        Transform existingInstructions = uiCanvas.transform.Find("PowerUpInstructions");
        if (existingInstructions != null)
        {
            Destroy(existingInstructions.gameObject);
        }

        GameObject instructionsObj = new GameObject("PowerUpInstructions");
        instructionsObj.transform.SetParent(uiCanvas.transform);
        RectTransform rectTransform = instructionsObj.AddComponent<RectTransform>();
        rectTransform.anchorMin = new Vector2(0.5f, 0);
        rectTransform.anchorMax = new Vector2(0.5f, 0);
        rectTransform.pivot = new Vector2(0.5f, 0);
        rectTransform.anchoredPosition = new Vector2(0, 220);
        rectTransform.sizeDelta = new Vector2(500, 100);

        TextMeshProUGUI text = instructionsObj.AddComponent<TextMeshProUGUI>();
        text.fontSize = 16;
        text.alignment = TextAlignmentOptions.Center;
        text.text = "POWER-UPS\n" +
                    "Player 1: Press 1 + slot number (1-3) to use\n" +
                    "Player 2: Press 2 + slot number (1-3) to use";

        // Make background semi-transparent
        Image bg = instructionsObj.AddComponent<Image>();
        bg.color = new Color(0, 0, 0, 0.5f);
    }
}