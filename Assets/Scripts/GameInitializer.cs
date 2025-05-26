using UnityEngine;

// This class manages all necessary game initializations
public class GameInitializer : MonoBehaviour
{
    public GameObject powerUpManagerPrefab; // Drag your PowerUpManager prefab here
    public GameObject boardManagerPrefab;   // Drag your BoardManager prefab here

    private void Awake()
    {
        // Step 1: Ensure PowerUpManager exists
        if (PowerUpManager.Instance == null)
        {
            if (powerUpManagerPrefab != null)
            {
                Instantiate(powerUpManagerPrefab);
                Debug.Log("PowerUpManager instantiated by GameInitializer");
            }
            else
            {
                GameObject powerUpObj = new GameObject("PowerUpManager");
                powerUpObj.AddComponent<PowerUpManager>();
                Debug.Log("PowerUpManager created by GameInitializer");
            }
        }

        // Step 2: Create BoardManager after PowerUpManager is initialized
        if (boardManagerPrefab != null)
        {
            Instantiate(boardManagerPrefab);
            Debug.Log("BoardManager instantiated by GameInitializer");
        }
    }
}