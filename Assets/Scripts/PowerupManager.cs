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
            availablePowerUps = new PowerUp[]
            {
                new PowerUp(PowerUpType.ClearRow, "Clear Row", "Clear a row from your board"),
                new PowerUp(PowerUpType.SlowDown, "Slow Down", "Slow down your game temporarily"),
                new PowerUp(PowerUpType.BonusPoints, "Bonus Points", "Get 500 bonus points"),
                new PowerUp(PowerUpType.BlockFreeze, "Freeze Block", "Current piece stops falling"),
                new PowerUp(PowerUpType.ExtraPiece, "Store Piece", "Store current piece for later"),
                new PowerUp(PowerUpType.AddGarbage, "Add Garbage", "Add garbage rows to opponent"),
                new PowerUp(PowerUpType.SpeedUp, "Speed Up", "Increase opponent's speed"),
                new PowerUp(PowerUpType.Confusion, "Confusion", "Rotate opponent's controls"),
                new PowerUp(PowerUpType.BlockFlip, "Block Flip", "Rotate opponent's current piece"),
                new PowerUp(PowerUpType.BlindMode, "Blind Mode", "Hide opponent's board partially"),
                new PowerUp(PowerUpType.SwapPiece, "Swap Piece", "Swap current piece with opponent")
            };

            // Assign default icons until proper ones are set
            foreach (PowerUp powerUp in availablePowerUps)
            {
                if (powerUp.icon == null)
                {
                    // Create a simple colored sprite icon based on self/opponent effect
                    bool isSelfBenefit = powerUp.type == PowerUpType.ClearRow ||
                                        powerUp.type == PowerUpType.SlowDown ||
                                        powerUp.type == PowerUpType.BonusPoints ||
                                        powerUp.type == PowerUpType.BlockFreeze ||
                                        powerUp.type == PowerUpType.ExtraPiece;

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
            return;

        Debug.Log($"PowerUp slot {index + 1} clicked with power-up: {activePowerUps[index].name}");

        // Get the BoardManager to check game mode
        BoardManager boardManager = FindObjectOfType<BoardManager>();
        if (boardManager == null || boardManager.activeBoards == null)
        {
            Debug.LogWarning("BoardManager not found or no active boards");
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
            PowerUp powerUp = activePowerUps[pendingPowerUpIndex];
            instructionsText.text = $"Using Power-Up: {powerUp.name}\n\nPress 1 for Player 1\nPress 2 for Player 2\n\nPress ESC to cancel";
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
        BoardManager boardManager = FindObjectOfType<BoardManager>();
        if (boardManager == null || boardManager.activeBoards == null)
        {
            Debug.LogWarning("BoardManager not found or no active boards");
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
        }
        // Two player mode
        else if (boardManager.activeBoards.Length > 1)
        {
            if (playerIndex == 0) // Player 1
            {
                userBoard = boardManager.activeBoards[0];
                targetBoard = boardManager.activeBoards[1];
            }
            else // Player 2
            {
                userBoard = boardManager.activeBoards[1];
                targetBoard = boardManager.activeBoards[0];
            }
        }

        if (userBoard != null && targetBoard != null)
        {
            UsePowerUp(powerUpIndex, userBoard, targetBoard);
            Debug.Log($"Power-up {activePowerUps[powerUpIndex].name} used by player {playerIndex + 1}");
        }
    }

    public void UsePowerUp(int index, Board userBoard, Board targetBoard)
    {
        if (index < 0 || index >= activePowerUps.Length || activePowerUps[index] == null)
            return;

        PowerUp powerUp = activePowerUps[index];

        // Apply the power-up effect
        ApplyPowerUpEffect(powerUp, userBoard, targetBoard);

        // Remove the used power-up
        activePowerUps[index] = null;
        RefreshPowerUpUI();
    }

    private void ApplyPowerUpEffect(PowerUp powerUp, Board userBoard, Board targetBoard)
    {
        switch (powerUp.type)
        {
            // Self-benefit power-ups
            case PowerUpType.ClearRow:
                StartCoroutine(ClearRandomRow(userBoard));
                break;

            case PowerUpType.SlowDown:
                StartCoroutine(ModifyDropRate(userBoard, 0.5f, powerUp.duration));
                break;

            case PowerUpType.BonusPoints:
                userBoard.AddPoints(500);
                break;

            case PowerUpType.BlockFreeze:
                StartCoroutine(FreezePiece(userBoard, powerUp.duration));
                break;

            case PowerUpType.ExtraPiece:
                // Implementation depends on your hold piece system
                break;

            // Opponent-affecting power-ups
            case PowerUpType.AddGarbage:
                StartCoroutine(AddGarbageRows(targetBoard, 2));
                break;

            case PowerUpType.SpeedUp:
                StartCoroutine(ModifyDropRate(targetBoard, -0.3f, powerUp.duration));
                break;

            case PowerUpType.Confusion:
                StartCoroutine(ConfuseControls(targetBoard, powerUp.duration));
                break;

            case PowerUpType.BlockFlip:
                FlipCurrentPiece(targetBoard);
                break;

            case PowerUpType.BlindMode:
                StartCoroutine(EnableBlindMode(targetBoard, powerUp.duration));
                break;

            case PowerUpType.SwapPiece:
                SwapPieces(userBoard, targetBoard);
                break;
        }

        // Visual feedback
        ShowPowerUpEffect(userBoard, targetBoard, powerUp);
    }

    // PowerUp effect implementations
    private IEnumerator ClearRandomRow(Board board)
    {
        // Randomly select a row to clear
        RectInt bounds = board.Bounds;
        int randomRow = Random.Range(bounds.yMin, bounds.yMax);

        // Visual effect before clearing
        yield return StartCoroutine(HighlightRow(board, randomRow, Color.green, 0.5f));

        // Clear the row (similar to LineClear logic)
        board.ClearSpecificRow(randomRow);
    }

    private IEnumerator ModifyDropRate(Board board, float modifier, float duration)
    {
        // Store the original drop rate
        float originalDropRateModifier = board.temporarySpeedBoost;

        // Apply the modifier
        board.temporarySpeedBoost += modifier;

        // Wait for the duration
        yield return new WaitForSeconds(duration);

        // Restore the original drop rate
        board.temporarySpeedBoost = originalDropRateModifier;
    }

    private IEnumerator FreezePiece(Board board, float duration)
    {
        if (board.activePiece != null)
        {
            board.activePiece.SetFrozen(true);
            yield return new WaitForSeconds(duration);
            board.activePiece.SetFrozen(false);
        }
        else
        {
            yield return null;
        }
    }

    private IEnumerator AddGarbageRows(Board board, int rowCount)
    {
        // Visual effect before adding garbage
        yield return StartCoroutine(FlashEffect(board.gameObject, Color.red, 0.5f));

        board.AddGarbageRows(rowCount);
    }

    private IEnumerator ConfuseControls(Board board, float duration)
    {
        if (board.inputController != null)
        {
            // Implement control inversion in your input controller
            board.inputController.InvertControls(true);
            yield return new WaitForSeconds(duration);
            board.inputController.InvertControls(false);
        }
        else
        {
            yield return null;
        }
    }

    private void FlipCurrentPiece(Board board)
    {
        if (board.activePiece != null)
        {
            // Randomly rotate the piece 1-3 times
            int rotations = Random.Range(1, 4);
            for (int i = 0; i < rotations; i++)
            {
                board.activePiece.Rotate(1);
            }
        }
    }

    private IEnumerator EnableBlindMode(Board board, float duration)
    {
        // Create a semi-transparent overlay on the board
        GameObject overlay = new GameObject("BlindOverlay");
        overlay.transform.SetParent(board.transform);
        SpriteRenderer renderer = overlay.AddComponent<SpriteRenderer>();

        // Set up the overlay
        renderer.color = new Color(0, 0, 0, 0.7f);
        renderer.sortingOrder = 100;

        // Scale and position to cover the board
        Vector3 boardSize = board.GetBoardWorldSize();
        overlay.transform.localScale = new Vector3(boardSize.x, boardSize.y, 1);
        overlay.transform.localPosition = Vector3.zero;

        yield return new WaitForSeconds(duration);

        Destroy(overlay);
    }

    private void SwapPieces(Board userBoard, Board targetBoard)
    {
        if (userBoard.activePiece != null && targetBoard.activePiece != null)
        {
            // Store piece data
            TetrominoData userPieceData = userBoard.activePiece.data;
            TetrominoData targetPieceData = targetBoard.activePiece.data;

            // Clear current pieces
            userBoard.Clear(userBoard.activePiece);
            targetBoard.Clear(targetBoard.activePiece);

            // Respawn with swapped data
            Vector3Int userPosition = userBoard.activePiece.position;
            Vector3Int targetPosition = targetBoard.activePiece.position;

            userBoard.activePiece.Initialize(userBoard, userPosition, targetPieceData, userBoard.inputController);
            targetBoard.activePiece.Initialize(targetBoard, targetPosition, userPieceData, targetBoard.inputController);

            // Set the pieces to make them visible
            userBoard.Set(userBoard.activePiece);
            targetBoard.Set(targetBoard.activePiece);
        }
    }

    // Visual effect helpers
    private IEnumerator HighlightRow(Board board, int row, Color color, float duration)
    {
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

        while (Time.time - startTime < duration)
        {
            float alpha = Mathf.PingPong((Time.time - startTime) * flashSpeed, 0.5f);
            renderer.color = new Color(color.r, color.g, color.b, alpha);
            yield return null;
        }

        Destroy(flashObj);
    }

    private void ShowPowerUpEffect(Board userBoard, Board targetBoard, PowerUp powerUp)
    {
        Board affectedBoard = IsSelfBenefit(powerUp.type) ? userBoard : targetBoard;

        StartCoroutine(FlashEffect(affectedBoard.gameObject, powerUp.effectColor, 0.5f));
    }

    private bool IsSelfBenefit(PowerUpType type)
    {
        return type == PowerUpType.ClearRow ||
               type == PowerUpType.SlowDown ||
               type == PowerUpType.BonusPoints ||
               type == PowerUpType.BlockFreeze ||
               type == PowerUpType.ExtraPiece;
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