using System.Collections;
using UnityEngine;
using Unity.MLAgents;

public class TetrisTrainingManager : MonoBehaviour
{
    [SerializeField] private MLAgentInstaller agentInstaller;
    [SerializeField] private int trainingEpisodes = 1000;
    [SerializeField] private float episodeTimeout = 300f; // 5 minutes max per episode

    private int completedEpisodes = 0;
    private TetrisMLAgent currentAgent;
    private float episodeStartTime;

    private void Start()
    {
        // Find references if not set
        if (agentInstaller == null)
        {
            agentInstaller = FindObjectOfType<MLAgentInstaller>();
            if (agentInstaller == null)
            {
                Debug.LogError("TetrisTrainingManager: Could not find MLAgentInstaller!");
                return;
            }
        }

        // Start training coroutine
        StartCoroutine(TrainingLoop());
    }

    private IEnumerator TrainingLoop()
    {
        Debug.Log("Starting training loop...");

        for (completedEpisodes = 0; completedEpisodes < trainingEpisodes; completedEpisodes++)
        {
            Debug.Log($"Starting episode {completedEpisodes + 1}/{trainingEpisodes}");

            // Reset environment for new episode
            agentInstaller.ResetEnvironment();

            // Find the current ML agent
            currentAgent = FindObjectOfType<TetrisMLAgent>();
            if (currentAgent == null)
            {
                Debug.LogError("Could not find TetrisMLAgent after environment reset!");
                yield break;
            }

            // Record episode start time
            episodeStartTime = Time.time;

            // Wait until episode ends (either by game over or timeout)
            while (IsEpisodeActive())
            {
                yield return null;
            }

            // If we've reached here via timeout, force end the episode
            if (Time.time - episodeStartTime >= episodeTimeout)
            {
                Debug.Log("Episode timed out");
                currentAgent.OnGameOver(); // This will end the episode
            }

            Debug.Log($"Episode {completedEpisodes + 1} completed");
            yield return new WaitForSeconds(1f); // Short delay between episodes
        }

        Debug.Log("Training complete!");
    }

    private bool IsEpisodeActive()
    {
        // Check if agent exists and if we've reached timeout
        if (currentAgent == null || Time.time - episodeStartTime >= episodeTimeout)
        {
            return false;
        }

        // Otherwise episode is still active
        return true;
    }

    // Note: Saving models is handled by the ML-Agents training process
    // Models are automatically saved during training in the results directory
    // This is just a notification method
    public void NotifyTrainingComplete()
    {
        if (Academy.IsInitialized)
        {
            Debug.Log("Training completed. The model will be saved by the ML-Agents training process.");
            Debug.Log("Check the 'results' directory in your project root for the saved model.");
            Debug.Log("You can then copy the .onnx file to your Assets/Models directory for use in the game.");
        }
        else
        {
            Debug.LogError("Academy not initialized");
        }
    }
}