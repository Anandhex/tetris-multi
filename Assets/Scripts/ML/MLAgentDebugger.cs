using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Policies;
using System.Collections;
using System.Collections.Generic;

[RequireComponent(typeof(TetrisMLAgent))]
public class MLAgentDebugger : MonoBehaviour
{
    [SerializeField] private bool enableDebugging = true;
    [SerializeField] private float debugUpdateInterval = 1.0f;
    [SerializeField] private bool logPlacementDetails = true;

    private TetrisMLAgent mlAgent;
    private BehaviorParameters behaviorParams;
    private int episodeCount = 0;
    private int stepCount = 0;
    private float lastRewardValue = 0f;
    private float currentRewardAccumulated = 0f;

    // Updated: Track placement actions (single discrete action space)
    private Dictionary<int, int> placementCounts = new Dictionary<int, int>();
    private int lastPlacementAction = -1;

    // Track placement quality metrics
    private List<float> recentLineClears = new List<float>();
    private List<float> recentHeights = new List<float>();
    private List<float> recentHoles = new List<float>();
    private int maxRecentMetrics = 100; // Keep last 100 placements

    // Track curriculum progress
    private float lastBoardHeight = 0f;
    private int lastTetrominoTypes = 0;
    private int lastBoardPreset = 0;


    private void Start()
    {
        mlAgent = GetComponent<TetrisMLAgent>();
        behaviorParams = GetComponent<BehaviorParameters>();

        if (enableDebugging && mlAgent != null)
        {
            StartCoroutine(DebugRoutine());
            Debug.Log($"ML Agent Debugger attached to {gameObject.name}");
            LogAgentConfiguration();

            Academy.Instance.AgentPreStep += RecordStep;
        }
        else
        {
            Debug.LogError("MLAgentDebugger: Could not find TetrisMLAgent component!");
        }
    }

    // Fix the stats recording method
    private void Update()
    {
        if (mlAgent != null && enableDebugging)
        {
            // Track reward changes every frame
            float currentReward = mlAgent.GetCumulativeReward();
            if (currentReward != lastRewardValue)
            {
                float rewardDelta = currentReward - lastRewardValue;
                currentRewardAccumulated += rewardDelta;
                lastRewardValue = currentReward;
            }
        }
    }

    private IEnumerator DebugRoutine()
    {
        while (true)
        {
            yield return new WaitForSeconds(debugUpdateInterval);

            if (mlAgent != null && enableDebugging)
            {
                LogAgentStatus();
                if (logPlacementDetails)
                {
                    LogPlacementQuality();
                }
            }
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

        if (mlAgent != null)
        {
            // Track placement action
            int placementAction = GetLastPlacementAction();
            if (placementAction >= 0)
            {
                if (!placementCounts.ContainsKey(placementAction))
                {
                    placementCounts[placementAction] = 0;
                }
                placementCounts[placementAction]++;
            }

            // Track reward changes
            float currentReward = mlAgent.GetCumulativeReward();
            float rewardDelta = currentReward - lastRewardValue;
            currentRewardAccumulated += rewardDelta;
            lastRewardValue = currentReward;

            // Track curriculum changes
            CheckCurriculumChanges();
        }
    }

    private void CheckCurriculumChanges()
    {
        if (mlAgent.curriculumBoardHeight != lastBoardHeight)
        {
            Debug.Log($"[Curriculum] Board height changed: {lastBoardHeight} -> {mlAgent.curriculumBoardHeight}");
            lastBoardHeight = mlAgent.curriculumBoardHeight;
        }

        if (mlAgent.allowedTetrominoTypes != lastTetrominoTypes)
        {
            Debug.Log($"[Curriculum] Tetromino types changed: {lastTetrominoTypes} -> {mlAgent.allowedTetrominoTypes}");
            lastTetrominoTypes = mlAgent.allowedTetrominoTypes;
        }

        if (mlAgent.curriculumBoardPreset != lastBoardPreset)
        {
            Debug.Log($"[Curriculum] Board preset changed: {lastBoardPreset} -> {mlAgent.curriculumBoardPreset}");
            lastBoardPreset = mlAgent.curriculumBoardPreset;
        }
    }

    // Updated: Store single placement action
    public void SetLastPlacementAction(int placementIndex)
    {
        lastPlacementAction = placementIndex;
    }

    // Record placement quality metrics
    public void RecordPlacementMetrics(float linesCleared, float maxHeight, float holes)
    {
        recentLineClears.Add(linesCleared);
        recentHeights.Add(maxHeight);
        recentHoles.Add(holes);

        // Keep only recent metrics
        if (recentLineClears.Count > maxRecentMetrics)
        {
            recentLineClears.RemoveAt(0);
            recentHeights.RemoveAt(0);
            recentHoles.RemoveAt(0);
        }
    }

    private int GetLastPlacementAction()
    {
        return lastPlacementAction;
    }



    private void LogAgentConfiguration()
    {
        if (behaviorParams != null)
        {
            Debug.Log($"Agent Configuration:");
            Debug.Log($"- Name: {behaviorParams.BehaviorName}");
            Debug.Log($"- Type: {behaviorParams.BehaviorType}");
            Debug.Log($"- Observation Size: {behaviorParams.BrainParameters.VectorObservationSize}");

            // Updated: Log single discrete action space for placements
            var actionSpec = behaviorParams.BrainParameters.ActionSpec;
            Debug.Log($"- Action Space: Single Discrete with {actionSpec.NumDiscreteActions} total actions");

            if (actionSpec.BranchSizes.Length > 0)
            {
                Debug.Log($"  - Placement Selection: {actionSpec.BranchSizes[0]} possible placements");
            }

            if (behaviorParams.Model != null)
            {
                Debug.Log($"- Model: {behaviorParams.Model.name}");
            }
            else
            {
                Debug.Log("- No model assigned");
            }

            // Log curriculum settings
            Debug.Log($"- Initial Curriculum Settings:");
            Debug.Log($"  - Board Height: {mlAgent.curriculumBoardHeight}");
            Debug.Log($"  - Tetromino Types: {mlAgent.allowedTetrominoTypes}");
            Debug.Log($"  - Board Preset: {mlAgent.curriculumBoardPreset}");
        }
    }

    private void LogAgentStatus()
    {
        // Display placement distribution
        string placementDistribution = "Placement Actions: [";
        int totalPlacements = 0;

        foreach (var kvp in placementCounts)
        {
            totalPlacements += kvp.Value;
        }

        int displayCount = 0;
        foreach (var kvp in placementCounts)
        {
            if (displayCount < 10) // Show top 10 most used placements
            {
                float percentage = totalPlacements > 0 ? (kvp.Value / (float)totalPlacements) * 100f : 0f;
                placementDistribution += $"{kvp.Key}:{percentage:F1}%";
                if (displayCount < 9 && displayCount < placementCounts.Count - 1)
                    placementDistribution += ", ";
            }
            displayCount++;
        }
        placementDistribution += "]";

        Debug.Log($"Agent Status Update:");
        Debug.Log($"- Steps: {stepCount}");
        Debug.Log($"- Episodes: {episodeCount}");
        Debug.Log($"- Current Cumulative Reward: {mlAgent.GetCumulativeReward():F2}");
        Debug.Log($"- Reward since last update: {currentRewardAccumulated:F2}");
        Debug.Log($"- Total unique placements used: {placementCounts.Count}");
        Debug.Log(placementDistribution);

        // Current curriculum status
        Debug.Log($"- Curriculum Status: Height={mlAgent.curriculumBoardHeight}, Types={mlAgent.allowedTetrominoTypes}, Preset={mlAgent.curriculumBoardPreset}");

        // Reset the accumulated reward for the next interval
        currentRewardAccumulated = 0f;
    }

    private void LogPlacementQuality()
    {
        if (recentLineClears.Count > 0)
        {
            float avgLineClears = CalculateAverage(recentLineClears);
            float avgHeight = CalculateAverage(recentHeights);
            float avgHoles = CalculateAverage(recentHoles);

            Debug.Log($"Placement Quality (last {recentLineClears.Count} placements):");
            Debug.Log($"- Avg Lines Cleared: {avgLineClears:F2}");
            Debug.Log($"- Avg Max Height: {avgHeight:F2}");
            Debug.Log($"- Avg Holes Created: {avgHoles:F2}");

            // Calculate efficiency metrics
            float tetrisRate = CalculateTetrisRate();
            if (tetrisRate > 0)
            {
                Debug.Log($"- Tetris Rate: {tetrisRate:F1}%");
            }
        }
    }

    private float CalculateAverage(List<float> values)
    {
        float sum = 0f;
        foreach (float value in values)
        {
            sum += value;
        }
        return values.Count > 0 ? sum / values.Count : 0f;
    }

    private float CalculateTetrisRate()
    {
        int tetrisCount = 0;
        foreach (float lineClears in recentLineClears)
        {
            if (lineClears >= 4f)
            {
                tetrisCount++;
            }
        }
        return recentLineClears.Count > 0 ? (tetrisCount / (float)recentLineClears.Count) * 100f : 0f;
    }

    // Call this when an episode ends
    public void OnEpisodeEnd()
    {
        episodeCount++;
        Debug.Log($"Episode {episodeCount} ended with reward: {mlAgent.GetCumulativeReward():F2}");

        // Log final placement statistics for this episode
        LogActionStatistics();

        // Reset placement counts
        placementCounts.Clear();

        // Clear recent metrics for new episode
        recentLineClears.Clear();
        recentHeights.Clear();
        recentHoles.Clear();
    }

    // Helper method to get placement distribution statistics
    public void LogActionStatistics()
    {
        if (placementCounts.Count == 0)
        {
            Debug.Log("=== No placement actions recorded this episode ===");
            return;
        }

        Debug.Log("=== Placement Action Statistics ===");

        int totalActions = 0;
        foreach (var kvp in placementCounts)
        {
            totalActions += kvp.Value;
        }

        // Sort by usage count
        var sortedPlacements = new List<KeyValuePair<int, int>>(placementCounts);
        sortedPlacements.Sort((x, y) => y.Value.CompareTo(x.Value));

        Debug.Log("Most Used Placements:");
        for (int i = 0; i < Mathf.Min(10, sortedPlacements.Count); i++)
        {
            var placement = sortedPlacements[i];
            float percentage = totalActions > 0 ? (placement.Value / (float)totalActions) * 100f : 0f;
            Debug.Log($"  Placement {placement.Key}: {percentage:F1}% ({placement.Value} times)");
        }

        // Diversity metric
        float diversity = placementCounts.Count / (float)totalActions;
        Debug.Log($"Placement Diversity: {diversity:F3} (unique placements per action)");
    }

    // Method for ML Agent to call when making a placement
    public void OnPlacementMade(int placementIndex, PlacementInfo placementInfo)
    {
        SetLastPlacementAction(placementIndex);
        RecordPlacementMetrics(
            placementInfo.linesCleared,
            placementInfo.maxHeight,
            placementInfo.holes
        );

        if (logPlacementDetails)
        {
            Debug.Log($"Placement {placementIndex}: Lines={placementInfo.linesCleared}, Height={placementInfo.maxHeight:F1}, Holes={placementInfo.holes}");
        }
    }

    internal void OnPlacementMade(object lastPlacementIndex, PlacementInfo placement)
    {
        throw new System.NotImplementedException();
    }
}
