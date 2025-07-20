using UnityEngine;
public class PowerUpTester : MonoBehaviour
{
    public PowerUpManager powerUpManager;

    void Update()
    {
        // Test different columns with number keys
        for (int i = 0; i <= 9; i++)
        {
            if (Input.GetKeyDown(KeyCode.Alpha0 + i))
            {
                Debug.Log($"Testing bomb at column {i}");
                powerUpManager.UsePowerUpAtColumn(PowerUpType.Bomb, i);
            }
        }
    }
}