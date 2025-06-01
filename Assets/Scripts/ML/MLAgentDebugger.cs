using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Policies;
using System.Collections;

[RequireComponent(typeof(TetrisMLAgent))]
public class MLAgentDebugger : MonoBehaviour
{
    [SerializeField] private bool enableDebugging = true;
    [SerializeField] private float debugUpdateInterval = 1.0f;

    private TetrisMLAgent mlAgent;
    private BehaviorParameters behaviorParams;
    private int episodeCount = 0;
    private int stepCount = 0;
    private float lastRewardValue = 0f;
    private float currentRewardAccumulated = 0f;

    // Updated: Track actions for multi-discrete action space
    private int[,] actionCounts = new int[2, 10]; // [branch, action] - branch 0: 10 columns, branch 1: 4 rotations (but sized to 10 for simplicity)
    private int[] lastActions = new int[2]; // Store last action for each branch

    private void Start()
    {
        mlAgent = GetComponent<TetrisMLAgent>();
        behaviorParams = GetComponent<BehaviorParameters>();

        if (enableDebugging)
        {
            StartCoroutine(DebugRoutine());

            // Hook into agent events
            Academy.Instance.AgentPreStep += RecordStep;

            Debug.Log($"ML Agent Debugger attached to {gameObject.name}");
            LogAgentConfiguration();
        }
    }

    private void OnDestroy()
    {
        if (Academy.IsInitialized)
        {
            Academy.Instance.AgentPreStep -= RecordStep;
        }
    }

    private void RecordStep(int academyStepCount)
    {
        stepCount++;

        // Record actions for multi-discrete action space
        if (mlAgent != null)
        {
            // Track column action (branch 0)
            int columnAction = GetLastColumnAction();
            if (columnAction >= 0 && columnAction < 10)
            {
                actionCounts[0, columnAction]++;
            }

            // Track rotation action (branch 1)
            int rotationAction = GetLastRotationAction();
            if (rotationAction >= 0 && rotationAction < 4)
            {
                actionCounts[1, rotationAction]++;
            }

            // Track reward changes
            float currentReward = mlAgent.GetCumulativeReward();
            float rewardDelta = currentReward - lastRewardValue;
            currentRewardAccumulated += rewardDelta;
            lastRewardValue = currentReward;
        }
    }

    // Updated: Store actions for both branches
    public void SetLastActions(int columnAction, int rotationAction)
    {
        lastActions[0] = columnAction;
        lastActions[1] = rotationAction;
    }

    private int GetLastColumnAction()
    {
        return lastActions[0];
    }

    private int GetLastRotationAction()
    {
        return lastActions[1];
    }

    private IEnumerator DebugRoutine()
    {
        while (true)
        {
            yield return new WaitForSeconds(debugUpdateInterval);

            if (mlAgent != null)
            {
                LogAgentStatus();
            }
        }
    }

    private void LogAgentConfiguration()
    {
        if (behaviorParams != null)
        {
            Debug.Log($"Agent Configuration:");
            Debug.Log($"- Name: {behaviorParams.BehaviorName}");
            Debug.Log($"- Type: {behaviorParams.BehaviorType}");
            Debug.Log($"- Observation Size: {behaviorParams.BrainParameters.VectorObservationSize}");

            // Updated: Log multi-discrete action space
            var actionSpec = behaviorParams.BrainParameters.ActionSpec;
            Debug.Log($"- Action Space: Multi-Discrete with {actionSpec.NumDiscreteActions} branches");
            for (int i = 0; i < actionSpec.BranchSizes.Length; i++)
            {
                string branchName = i == 0 ? "Column" : "Rotation";
                Debug.Log($"  - Branch {i} ({branchName}): {actionSpec.BranchSizes[i]} actions");
            }

            if (behaviorParams.Model != null)
            {
                Debug.Log($"- Model: {behaviorParams.Model.name}");
            }
            else
            {
                Debug.Log("- No model assigned");
            }
        }
    }

    private void LogAgentStatus()
    {
        // Updated: Display action distribution for multi-discrete actions
        string columnDistribution = "Column Actions: [";
        for (int i = 0; i < 10; i++)
        {
            columnDistribution += $"{i}:{actionCounts[0, i]}";
            if (i < 9)
                columnDistribution += ", ";
        }
        columnDistribution += "]";

        string rotationDistribution = "Rotation Actions: [";
        for (int i = 0; i < 4; i++)
        {
            rotationDistribution += $"{i}:{actionCounts[1, i]}";
            if (i < 3)
                rotationDistribution += ", ";
        }
        rotationDistribution += "]";

        Debug.Log($"Agent Status Update:");
        Debug.Log($"- Steps: {stepCount}");
        Debug.Log($"- Episodes: {episodeCount}");
        Debug.Log($"- Current Cumulative Reward: {mlAgent.GetCumulativeReward()}");
        Debug.Log($"- Reward since last update: {currentRewardAccumulated}");
        Debug.Log(columnDistribution);
        Debug.Log(rotationDistribution);

        // Reset the accumulated reward for the next interval
        currentRewardAccumulated = 0f;
    }

    // Call this when an episode ends
    public void OnEpisodeEnd()
    {
        episodeCount++;
        Debug.Log($"Episode {episodeCount} ended with reward: {mlAgent.GetCumulativeReward()}");

        // Reset action counts for both branches
        for (int branch = 0; branch < 2; branch++)
        {
            for (int action = 0; action < (branch == 0 ? 10 : 4); action++)
            {
                actionCounts[branch, action] = 0;
            }
        }
    }

    // Helper method to get action distribution statistics
    public void LogActionStatistics()
    {
        Debug.Log("=== Action Statistics ===");

        // Column action statistics
        int totalColumnActions = 0;
        for (int i = 0; i < 10; i++)
        {
            totalColumnActions += actionCounts[0, i];
        }

        Debug.Log("Column Action Percentages:");
        for (int i = 0; i < 10; i++)
        {
            float percentage = totalColumnActions > 0 ? (actionCounts[0, i] / (float)totalColumnActions) * 100f : 0f;
            Debug.Log($"  Column {i}: {percentage:F1}% ({actionCounts[0, i]} times)");
        }

        // Rotation action statistics
        int totalRotationActions = 0;
        for (int i = 0; i < 4; i++)
        {
            totalRotationActions += actionCounts[1, i];
        }

        Debug.Log("Rotation Action Percentages:");
        string[] rotationNames = { "0°", "90°", "180°", "270°" };
        for (int i = 0; i < 4; i++)
        {
            float percentage = totalRotationActions > 0 ? (actionCounts[1, i] / (float)totalRotationActions) * 100f : 0f;
            Debug.Log($"  {rotationNames[i]}: {percentage:F1}% ({actionCounts[1, i]} times)");
        }
    }
}