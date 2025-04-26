using UnityEngine;

public class FireBorderController : MonoBehaviour
{
    public ParticleSystem fireParticles; // Reference to your particle system
    public float minSpeed = 0.5f;       // Minimum speed of the fire
    public float maxSpeed = 2.0f;       // Maximum speed of the fire
    private ParticleSystem.MainModule mainModule; // To modify the particle system

    private void Start()
    {
        // Cache the MainModule for easier access
        if (fireParticles != null)
        {
            mainModule = fireParticles.main;
        }
    }

    // Call this method to adjust fire speed based on game speed
    public void SetGameSpeed(float gameSpeed)
    {
        if (fireParticles != null)
        {
            // Scale the fire speed with game speed
            mainModule.simulationSpeed = Mathf.Lerp(minSpeed, maxSpeed, gameSpeed);
        }
    }
}
