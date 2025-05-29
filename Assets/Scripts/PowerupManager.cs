using System.Collections;
using UnityEngine;
using TMPro;
using UnityEngine.UI;

public class PowerUpManager : MonoBehaviour
{
    public static PowerUpManager Instance { get; private set; }

    public PowerUp[] availablePowerUps;
    public PowerUp[] activePowerUps = new PowerUp[3]; // Currently available power-ups

    [Header("PowerUp UI")]
    public GameObject powerUpUIContainer;  // Can be assigned in inspector or created by BoardManager
    public PowerUpUI[] powerUpUISlots;

    [Header("PowerUp Settings")]
    public float powerUpGenerationInterval = 30f; // Generate a new power-up every 30 seconds
    public float powerUpChance = 0.3f; // 30% chance of getting a power-up when clearing multiple lines

    // Add UI for player selection
    private GameObject playerSelectionPanel;
    private bool waitingForPlayerSelection = false;
    private int pendingPowerUpIndex = -1;

    private TextMeshProUGUI instructionsText;

    private bool isInitialized = false;

    private void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject); // Optional: Keep PowerUpManager across scenes
        }
        else if (Instance != this)
        {
            Destroy(gameObject);
            return;
        }

        InitializePowerUps();
    }

    private void Start()
    {
        StartCoroutine(GeneratePowerUpsRoutine());

        // Only refresh UI if we have a container
        if (powerUpUIContainer != null && powerUpUISlots != null && powerUpUISlots.Length > 0)
        {
            RefreshPowerUpUI();
        }

        // Create player selection UI
        CreatePlayerSelectionPanel();

        isInitialized = true;
    }

    private void Update()
    {
        // Check for keyboard shortcuts to use power-ups directly
        if (!waitingForPlayerSelection)
        {
            CheckPowerUpHotkeys();
        }
        else
        {
            CheckPlayerSelectionInput();
        }
    }

    private void CheckPowerUpHotkeys()
    {
        // Player 1 shortcuts: Alt+1, Alt+2, Alt+3
        if (Input.GetKey(KeyCode.LeftAlt) || Input.GetKey(KeyCode.RightAlt))
        {
            for (int i = 0; i < 3; i++)
            {
                if (Input.GetKeyDown(KeyCode.Alpha1 + i) || Input.GetKeyDown(KeyCode.Keypad1 + i))
                {
                    // Check if the power-up exists in this slot
                    if (i < activePowerUps.Length && activePowerUps[i] != null)
                    {
                        UsePowerUpForPlayer(i, 0); // Player 1 (index 0)
                    }
                }
            }
        }

        // Player 2 shortcuts: Ctrl+1, Ctrl+2, Ctrl+3
        if (Input.GetKey(KeyCode.LeftControl) || Input.GetKey(KeyCode.RightControl))
        {
            for (int i = 0; i < 3; i++)
            {
                if (Input.GetKeyDown(KeyCode.Alpha1 + i) || Input.GetKeyDown(KeyCode.Keypad1 + i))
                {
                    // Check if the power-up exists in this slot
                    if (i < activePowerUps.Length && activePowerUps[i] != null)
                    {
                        UsePowerUpForPlayer(i, 1); // Player 2 (index 1)
                    }
                }
            }
        }
    }

    // This method can be called by BoardManager to provide a container
    public void SetupUIContainer(GameObject container, PowerUpUI[] slots)
    {
        powerUpUIContainer = container;
        powerUpUISlots = slots;

        // Set slot numbers for each PowerUpUI
        for (int i = 0; i < slots.Length; i++)
        {
            slots[i].slotNumber = i + 1;
        }

        if (isInitialized)
        {
            RefreshPowerUpUI();
        }

        // Create player selection UI
        CreatePlayerSelectionPanel();
    }

    private void InitializePowerUps()
    {
        if (availablePowerUps == null || availablePowerUps.Length == 0)
        {
            // Simplified to just 5 essential power-ups
            availablePowerUps = new PowerUp[]
            {
                // Self-benefit power-ups (Green)
                new PowerUp(PowerUpType.ClearRow, "Clear Row", "Clear a row from your board"),
                new PowerUp(PowerUpType.BonusPoints, "Bonus Points", "Get 500 bonus points"),
                new PowerUp(PowerUpType.SlowDown, "Slow Down", "Slow down your game temporarily"),
                
                // Opponent-affecting power-ups (Red)
                new PowerUp(PowerUpType.AddGarbage, "Add Garbage", "Add garbage rows to opponent"),
                new PowerUp(PowerUpType.SpeedUp, "Speed Up", "Increase opponent's drop speed")
            };

            // Assign default icons
            foreach (PowerUp powerUp in availablePowerUps)
            {
                if (powerUp.icon == null)
                {
                    // Create a simple colored sprite icon based on self/opponent effect
                    bool isSelfBenefit = powerUp.type == PowerUpType.ClearRow ||
                                        powerUp.type == PowerUpType.SlowDown ||
                                        powerUp.type == PowerUpType.BonusPoints;

                    Color iconColor = isSelfBenefit ? Color.green : Color.red;
                    powerUp.icon = CreateSquareSprite(iconColor);
                    powerUp.effectColor = iconColor;
                }
            }
        }

        // Initialize the active power-ups with empty slots
        for (int i = 0; i < activePowerUps.Length; i++)
        {
            activePowerUps[i] = null;
        }
    }

    private IEnumerator GeneratePowerUpsRoutine()
    {
        while (true)
        {
            yield return new WaitForSeconds(powerUpGenerationInterval);

            // Find an empty slot for a new power-up
            int emptySlot = -1;
            for (int i = 0; i < activePowerUps.Length; i++)
            {
                if (activePowerUps[i] == null)
                {
                    emptySlot = i;
                    break;
                }
            }

            // If we found an empty slot, add a random power-up
            if (emptySlot >= 0)
            {
                int randomIndex = Random.Range(0, availablePowerUps.Length);
                activePowerUps[emptySlot] = availablePowerUps[randomIndex];
                RefreshPowerUpUI();
            }
        }
    }

    public void RefreshPowerUpUI()
    {
        if (powerUpUISlots == null || powerUpUISlots.Length == 0)
        {
            Debug.LogWarning("PowerUpManager: UI slots not set up!");
            return;
        }

        for (int i = 0; i < powerUpUISlots.Length; i++)
        {
            if (i < activePowerUps.Length && powerUpUISlots[i] != null)
            {
                powerUpUISlots[i].SetPowerUp(activePowerUps[i], i);
            }
        }
    }

    // New method to handle player clicking on a power-up slot
    public void PowerUpSlotClicked(int index)
    {
        if (index < 0 || index >= activePowerUps.Length || activePowerUps[index] == null)
        {
            Debug.LogWarning($"Invalid power-up slot clicked: {index}");
            return;
        }

        Debug.Log($"PowerUp slot {index + 1} clicked with power-up: {activePowerUps[index].name}");

        // Get the BoardManager to check game mode
        BoardManager boardManager = FindObjectOfType<BoardManager>();
        if (boardManager == null)
        {
            Debug.LogError("BoardManager not found!");
            return;
        }

        if (boardManager.activeBoards == null)
        {
            Debug.LogError("BoardManager.activeBoards is null!");
            return;
        }

        if (boardManager.activeBoards.Length == 0)
        {
            Debug.LogError("No active boards found!");
            return;
        }

        // Single player mode - use immediately
        if (boardManager.activeBoards.Length == 1)
        {
            UsePowerUpForPlayer(index, 0); // Use for the single player
            return;
        }

        // Two player mode - show player selection panel
        pendingPowerUpIndex = index;
        ShowPlayerSelectionPanel();
    }

    private void CreatePlayerSelectionPanel()
    {
        // Get the canvas
        Canvas canvas = FindObjectOfType<Canvas>();
        if (canvas == null)
        {
            Debug.LogError("Canvas not found for player selection panel");
            return;
        }

        // Create panel
        playerSelectionPanel = new GameObject("PlayerSelectionPanel");
        playerSelectionPanel.transform.SetParent(canvas.transform);

        RectTransform panelRect = playerSelectionPanel.AddComponent<RectTransform>();
        panelRect.anchorMin = new Vector2(0.5f, 0.5f);
        panelRect.anchorMax = new Vector2(0.5f, 0.5f);
        panelRect.pivot = new Vector2(0.5f, 0.5f);
        panelRect.sizeDelta = new Vector2(300, 150);
        panelRect.anchoredPosition = Vector2.zero;

        // Add background image
        Image panelImage = playerSelectionPanel.AddComponent<Image>();
        panelImage.color = new Color(0, 0, 0, 0.8f);

        // Add instructions text
        GameObject textObj = new GameObject("Instructions");
        textObj.transform.SetParent(playerSelectionPanel.transform);

        RectTransform textRect = textObj.AddComponent<RectTransform>();
        textRect.anchorMin = new Vector2(0, 0);
        textRect.anchorMax = new Vector2(1, 1);
        textRect.offsetMin = new Vector2(10, 10);
        textRect.offsetMax = new Vector2(-10, -10);

        instructionsText = textObj.AddComponent<TextMeshProUGUI>();
        instructionsText.fontSize = 20;
        instructionsText.alignment = TextAlignmentOptions.Center;
        instructionsText.text = "Who should use this Power-Up?\n\nPress 1 for Player 1\nPress 2 for Player 2\n\nPress ESC to cancel";

        // Hide the panel by default
        playerSelectionPanel.SetActive(false);
    }

    private void ShowPlayerSelectionPanel()
    {
        if (playerSelectionPanel != null)
        {
            waitingForPlayerSelection = true;
            playerSelectionPanel.SetActive(true);

            // Update text to show which power-up is being used
            if (pendingPowerUpIndex >= 0 && pendingPowerUpIndex < activePowerUps.Length && activePowerUps[pendingPowerUpIndex] != null)
            {
                PowerUp powerUp = activePowerUps[pendingPowerUpIndex];
                instructionsText.text = $"Using Power-Up: {powerUp.name}\n\nPress 1 for Player 1\nPress 2 for Player 2\n\nPress ESC to cancel";
            }
        }
    }

    private void HidePlayerSelectionPanel()
    {
        if (playerSelectionPanel != null)
        {
            waitingForPlayerSelection = false;
            playerSelectionPanel.SetActive(false);
            pendingPowerUpIndex = -1;
        }
    }

    private void CheckPlayerSelectionInput()
    {
        if (Input.GetKeyDown(KeyCode.Alpha1) || Input.GetKeyDown(KeyCode.Keypad1))
        {
            if (pendingPowerUpIndex >= 0)
            {
                UsePowerUpForPlayer(pendingPowerUpIndex, 0); // Player 1
                HidePlayerSelectionPanel();
            }
        }
        else if (Input.GetKeyDown(KeyCode.Alpha2) || Input.GetKeyDown(KeyCode.Keypad2))
        {
            if (pendingPowerUpIndex >= 0)
            {
                UsePowerUpForPlayer(pendingPowerUpIndex, 1); // Player 2
                HidePlayerSelectionPanel();
            }
        }
        else if (Input.GetKeyDown(KeyCode.Escape))
        {
            HidePlayerSelectionPanel();
        }
    }

    private void UsePowerUpForPlayer(int powerUpIndex, int playerIndex)
    {
        // Add comprehensive null checks
        if (powerUpIndex < 0 || powerUpIndex >= activePowerUps.Length)
        {
            Debug.LogError($"Invalid powerUpIndex: {powerUpIndex}");
            return;
        }

        if (activePowerUps[powerUpIndex] == null)
        {
            Debug.LogError($"PowerUp at index {powerUpIndex} is null");
            return;
        }

        // Store the power-up name BEFORE using it (since UsePowerUp will null it out)
        string powerUpName = activePowerUps[powerUpIndex].name;

        BoardManager boardManager = FindObjectOfType<BoardManager>();
        if (boardManager == null)
        {
            Debug.LogError("BoardManager not found!");
            return;
        }

        if (boardManager.activeBoards == null)
        {
            Debug.LogError("BoardManager.activeBoards is null!");
            return;
        }

        if (boardManager.activeBoards.Length == 0)
        {
            Debug.LogError("No active boards found!");
            return;
        }

        // Get the board for the current player (who activated the power-up)
        Board userBoard = null;
        Board targetBoard = null;

        // Single player mode
        if (boardManager.activeBoards.Length == 1)
        {
            userBoard = boardManager.activeBoards[0];
            targetBoard = userBoard; // Target self in single player

            if (userBoard == null)
            {
                Debug.LogError("Single player board is null!");
                return;
            }
        }
        // Two player mode
        else if (boardManager.activeBoards.Length > 1)
        {
            if (playerIndex == 0) // Player 1
            {
                if (boardManager.activeBoards.Length > 0)
                    userBoard = boardManager.activeBoards[0];
                if (boardManager.activeBoards.Length > 1)
                    targetBoard = boardManager.activeBoards[1];
            }
            else // Player 2
            {
                if (boardManager.activeBoards.Length > 1)
                    userBoard = boardManager.activeBoards[1];
                if (boardManager.activeBoards.Length > 0)
                    targetBoard = boardManager.activeBoards[0];
            }

            if (userBoard == null)
            {
                Debug.LogError($"User board for player {playerIndex + 1} is null!");
                return;
            }

            if (targetBoard == null)
            {
                Debug.LogError($"Target board for player {(playerIndex == 0 ? 2 : 1)} is null!");
                return;
            }
        }

        if (userBoard != null && targetBoard != null)
        {
            UsePowerUp(powerUpIndex, userBoard, targetBoard);
            // Use the stored name instead of accessing the now-null power-up
            Debug.Log($"Power-up {powerUpName} used by player {playerIndex + 1}");
        }
        else
        {
            Debug.LogError("Failed to get valid user and target boards");
        }
    }

    public void UsePowerUp(int index, Board userBoard, Board targetBoard)
    {
        if (index < 0 || index >= activePowerUps.Length || activePowerUps[index] == null)
        {
            Debug.LogError($"Invalid power-up usage attempt at index {index}");
            return;
        }

        if (userBoard == null)
        {
            Debug.LogError("User board is null in UsePowerUp");
            return;
        }

        if (targetBoard == null)
        {
            Debug.LogError("Target board is null in UsePowerUp");
            return;
        }

        PowerUp powerUp = activePowerUps[index];

        Debug.Log($"Using power-up: {powerUp.name}");
        // Apply the power-up effect
        ApplyPowerUpEffect(powerUp, userBoard, targetBoard);

        // Remove the used power-up
        activePowerUps[index] = null;
        RefreshPowerUpUI();
    }

    private void ApplyPowerUpEffect(PowerUp powerUp, Board userBoard, Board targetBoard)
    {
        Debug.Log("In here");
        if (powerUp == null)
        {
            Debug.LogError("PowerUp is null in ApplyPowerUpEffect");
            return;
        }
        if (userBoard == null)
        {
            Debug.LogError("userBoard is null in ApplyPowerUpEffect");
            return;
        }
        if (targetBoard == null)
        {
            Debug.LogError("targetBoard is null in ApplyPowerUpEffect");
            return;
        }

        switch (powerUp.type)
        {
            // Self-benefit power-ups
            case PowerUpType.ClearRow:
                StartCoroutine(ClearRandomRow(userBoard));
                break;

            case PowerUpType.SlowDown:
                StartCoroutine(ModifyDropRate(userBoard, 0.5f, 10f)); // 10 second duration
                break;

            case PowerUpType.BonusPoints:
                userBoard.AddPoints(500);
                break;

            // Opponent-affecting power-ups
            case PowerUpType.AddGarbage:
                StartCoroutine(AddGarbageRows(targetBoard, 2));
                break;

            case PowerUpType.SpeedUp:
                StartCoroutine(ModifyDropRate(targetBoard, -0.3f, 10f)); // 10 second duration
                break;
        }

        // Visual feedback
        ShowPowerUpEffect(userBoard, targetBoard, powerUp);
    }

    // PowerUp effect implementations
    private IEnumerator ClearRandomRow(Board board)
    {
        if (board == null)
        {
            Debug.LogError("Board is null in ClearRandomRow");
            yield break;
        }

        // Randomly select a row to clear
        RectInt bounds = board.Bounds;
        int randomRow = Random.Range(bounds.yMin, bounds.yMax);

        // Visual effect before clearing
        yield return StartCoroutine(HighlightRow(board, randomRow, Color.green, 0.5f));

        // Clear the row (similar to LineClear logic)
        if (board != null) // Double-check in case board was destroyed during coroutine
        {
            board.ClearSpecificRow(randomRow);
        }
    }

    private IEnumerator ModifyDropRate(Board board, float modifier, float duration)
    {
        if (board == null)
        {
            Debug.LogError("Board is null in ModifyDropRate");
            yield break;
        }

        // Store the original drop rate
        float originalDropRateModifier = board.temporarySpeedBoost;

        // Apply the modifier
        board.temporarySpeedBoost += modifier;

        // Wait for the duration
        yield return new WaitForSeconds(duration);

        // Restore the original drop rate
        if (board != null) // Check if board still exists
        {
            board.temporarySpeedBoost = originalDropRateModifier;
        }
    }

    private IEnumerator AddGarbageRows(Board board, int rowCount)
    {
        if (board == null)
        {
            Debug.LogError("Board is null in AddGarbageRows");
            yield break;
        }

        // Visual effect before adding garbage
        yield return StartCoroutine(FlashEffect(board.gameObject, Color.red, 0.5f));

        if (board != null) // Check if board still exists
        {
            board.AddGarbageRows(rowCount);
        }
    }

    // Visual effect helpers
    private IEnumerator HighlightRow(Board board, int row, Color color, float duration)
    {
        if (board == null || board.tilemap == null)
        {
            Debug.LogError("Board or tilemap is null in HighlightRow");
            yield break;
        }

        RectInt bounds = board.Bounds;
        GameObject[] highlights = new GameObject[bounds.width];

        for (int col = bounds.xMin; col < bounds.xMax; col++)
        {
            Vector3Int position = new Vector3Int(col, row, 0);
            Vector3 worldPos = board.tilemap.CellToWorld(position) + new Vector3(0.5f, 0.5f, 0);

            GameObject highlight = new GameObject("Highlight");
            highlight.transform.SetParent(board.transform);
            highlight.transform.position = worldPos;

            SpriteRenderer renderer = highlight.AddComponent<SpriteRenderer>();
            renderer.color = color;
            renderer.sortingOrder = 10;

            // Use a square sprite or create one programmatically
            renderer.sprite = CreateSquareSprite(color);

            highlights[col - bounds.xMin] = highlight;
        }

        yield return new WaitForSeconds(duration);

        foreach (GameObject highlight in highlights)
        {
            if (highlight != null) Destroy(highlight);
        }
    }

    private IEnumerator FlashEffect(GameObject target, Color color, float duration)
    {
        if (target == null)
        {
            Debug.LogError("Target is null in FlashEffect");
            yield break;
        }

        GameObject flashObj = new GameObject("FlashEffect");
        flashObj.transform.SetParent(target.transform);
        flashObj.transform.localPosition = Vector3.zero;

        SpriteRenderer renderer = flashObj.AddComponent<SpriteRenderer>();
        renderer.color = color;
        renderer.sortingOrder = 200;

        // Adjust size to cover the target
        Bounds bounds = new Bounds();
        foreach (Renderer r in target.GetComponentsInChildren<Renderer>())
        {
            bounds.Encapsulate(r.bounds);
        }

        flashObj.transform.localScale = bounds.size;

        // Flash effect
        float flashSpeed = 4f;
        float startTime = Time.time;

        while (Time.time - startTime < duration && flashObj != null)
        {
            float alpha = Mathf.PingPong((Time.time - startTime) * flashSpeed, 0.5f);
            if (renderer != null)
            {
                renderer.color = new Color(color.r, color.g, color.b, alpha);
            }
            yield return null;
        }

        if (flashObj != null)
        {
            Destroy(flashObj);
        }
    }

    private void ShowPowerUpEffect(Board userBoard, Board targetBoard, PowerUp powerUp)
    {
        if (powerUp == null)
        {
            Debug.LogError("PowerUp is null in ShowPowerUpEffect");
            return;
        }

        Board affectedBoard = IsSelfBenefit(powerUp.type) ? userBoard : targetBoard;

        if (affectedBoard != null && affectedBoard.gameObject != null)
        {
            StartCoroutine(FlashEffect(affectedBoard.gameObject, powerUp.effectColor, 0.5f));
        }
    }

    private bool IsSelfBenefit(PowerUpType type)
    {
        return type == PowerUpType.ClearRow ||
               type == PowerUpType.SlowDown ||
               type == PowerUpType.BonusPoints;
    }

    private Sprite CreateSquareSprite(Color color)
    {
        Texture2D texture = new Texture2D(32, 32);
        Color[] colors = new Color[32 * 32];

        for (int i = 0; i < colors.Length; i++)
        {
            colors[i] = color;
        }

        texture.SetPixels(colors);
        texture.Apply();

        return Sprite.Create(texture, new Rect(0, 0, 32, 32), Vector2.one * 0.5f);
    }
}