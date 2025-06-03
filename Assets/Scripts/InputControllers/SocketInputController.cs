using UnityEngine;

public class SocketInputController : MonoBehaviour, IPlayerInputController
{
    private bool leftPressed = false;
    private bool rightPressed = false;
    private bool downPressed = false;
    private bool rotateLeftPressed = false;
    private bool rotateRightPressed = false;
    private bool hardDropPressed = false;

    // Action execution tracking
    private int pendingAction = -1;
    private float actionCooldown = 0.1f;
    private float lastActionTime = 0f;

    void Start()
    {
        // Subscribe to socket events
        if (SocketManager.Instance != null)
        {
            SocketManager.Instance.OnCommandReceived += HandleCommand;
        }
    }

    void Update()
    {
        // Reset input flags each frame
        ResetInputFlags();

        // Process pending action if cooldown has passed
        if (pendingAction >= 0 && Time.time - lastActionTime >= actionCooldown)
        {
            ExecuteAction(pendingAction);
            pendingAction = -1;
            lastActionTime = Time.time;
        }
    }

    void ResetInputFlags()
    {
        leftPressed = false;
        rightPressed = false;
        downPressed = false;
        rotateLeftPressed = false;
        rotateRightPressed = false;
        hardDropPressed = false;
    }

    void HandleCommand(GameCommand command)
    {
        if (command.type == "action" && command.action != null)
        {
            pendingAction = command.action.actionIndex;
        }
    }

    void ExecuteAction(int actionIndex)
    {
        if (actionIndex < 0 || actionIndex >= 40)
        {
            return;
        }

        // Decode action: action = column * 4 + rotation
        int targetColumn = actionIndex / 4;  // 0-9
        int targetRotation = actionIndex % 4; // 0-3

        // For traditional input, we'll need to simulate the sequence of moves
        // This is a simplified version - you might want to implement a more sophisticated system

        // Set rotation input based on target rotation (simplified)
        if (targetRotation == 1 || targetRotation == 3)
        {
            rotateRightPressed = true;
        }
        else if (targetRotation == 2)
        {
            rotateRightPressed = true; // Will need to be called twice
        }

        // Set movement input based on target column (simplified)
        // This assumes piece starts at column 4-5 (center)
        int currentColumn = 4; // Assume center start
        if (targetColumn < currentColumn)
        {
            leftPressed = true;
        }
        else if (targetColumn > currentColumn)
        {
            rightPressed = true;
        }

        // Always drop after positioning
        hardDropPressed = true;
    }

    // IPlayerInputController implementation with correct method names
    public bool GetLeft()
    {
        return leftPressed;
    }

    public bool GetRight()
    {
        return rightPressed;
    }

    public bool GetDown()
    {
        return downPressed;
    }

    public bool GetRotateLeft()
    {
        return rotateLeftPressed;
    }

    public bool GetRotateRight()
    {
        return rotateRightPressed;
    }

    public bool GetHardDrop()
    {
        return hardDropPressed;
    }

    void OnDestroy()
    {
        if (SocketManager.Instance != null)
        {
            SocketManager.Instance.OnCommandReceived -= HandleCommand;
        }
    }
}
