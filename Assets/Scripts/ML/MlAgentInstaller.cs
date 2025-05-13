using UnityEngine;

public class MLAgentInstaller : MonoBehaviour
{
    [SerializeField] private BoardManager boardManager;

    void Start()
    {
        // Make sure we have a reference to the BoardManager
        if (boardManager == null)
        {
            boardManager = FindObjectOfType<BoardManager>();
            if (boardManager == null)
            {
                Debug.LogError("MLAgentInstaller: Could not find BoardManager!");
                return;
            }
        }

        // Set the game mode to AI
        boardManager.SetGameMode(BoardManager.GameMode.AI);
    }

    // Method to reset the game for a new training episode
    public void ResetEnvironment()
    {
        if (boardManager != null)
        {
            boardManager.ResetGame();
        }
    }
}