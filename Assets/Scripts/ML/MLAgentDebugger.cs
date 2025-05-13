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
    private int[] actionCounts = new int[7]; // Count for each possible action

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

        // Get the action taken by the agent (you'll need to access this from the agent)
        // This is a simplification - you'll need to adapt based on how you can access the action
        if (mlAgent != null)
        {
            // Example: if you can get the last action taken
            int lastActionTaken = GetLastActionTaken();
            if (lastActionTaken >= 0 && lastActionTaken < actionCounts.Length)
            {
                actionCounts[lastActionTaken]++;
            }

            // Track reward changes
            float currentReward = mlAgent.GetCumulativeReward();
            float rewardDelta = currentReward - lastRewardValue;
            currentRewardAccumulated += rewardDelta;
            lastRewardValue = currentReward;
        }
    }

    private int lastAction = 0;

    public void SetLastAction(int action)
    {
        lastAction = action;
    }

    private int GetLastActionTaken()
    {
        return lastAction;
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
            Debug.Log($"- Action Space: Discrete({behaviorParams.BrainParameters.ActionSpec.NumDiscreteActions} branches)");

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
        string actionDistribution = "Action Distribution: [";
        for (int i = 0; i < actionCounts.Length; i++)
        {
            actionDistribution += $"{i}:{actionCounts[i]}";
            if (i < actionCounts.Length - 1)
                actionDistribution += ", ";
        }
        actionDistribution += "]";

        Debug.Log($"Agent Status Update:");
        Debug.Log($"- Steps: {stepCount}");
        Debug.Log($"- Episodes: {episodeCount}");
        Debug.Log($"- Current Cumulative Reward: {mlAgent.GetCumulativeReward()}");
        Debug.Log($"- Reward since last update: {currentRewardAccumulated}");
        Debug.Log(actionDistribution);

        // Reset the accumulated reward for the next interval
        currentRewardAccumulated = 0f;
    }

    // Call this when an episode ends
    public void OnEpisodeEnd()
    {
        episodeCount++;
        Debug.Log($"Episode {episodeCount} ended with reward: {mlAgent.GetCumulativeReward()}");

        // Reset action counts
        for (int i = 0; i < actionCounts.Length; i++)
        {
            actionCounts[i] = 0;
        }
    }
}