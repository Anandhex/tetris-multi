using UnityEngine;
using Unity.MLAgents.Policies;
using Unity.Sentis;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;

public class TetrisMLAgentSetup : MonoBehaviour
{
    [SerializeField] private ModelAsset modelAsset;
    [SerializeField] private string behaviorName = "TetrisAgent";
    [SerializeField] private BehaviorType behaviorType = BehaviorType.Default;
    [SerializeField] private int observationSize = 217;  // Should match actual observation size
    [SerializeField] private bool addDebuggingComponents = true;
    [SerializeField] private bool enableModelLoggingOnStart = true;

    private void Awake()
    {

        BehaviorParameters behaviorParams = GetComponent<BehaviorParameters>();
        if (behaviorParams == null)
            behaviorParams = gameObject.AddComponent<BehaviorParameters>();

        behaviorParams.BehaviorName = behaviorName;
        behaviorParams.BehaviorType = behaviorType;

        // Set the model and verify it's assigned correctly
        if (modelAsset != null && behaviorType == BehaviorType.InferenceOnly)
        {
            behaviorParams.Model = modelAsset;
            Debug.Log($"Model '{modelAsset.name}' assigned to agent on {gameObject.name}");
        }
        else if (behaviorType == BehaviorType.InferenceOnly && modelAsset == null)
        {
            Debug.LogError($"BehaviorType is set to InferenceOnly but no model is assigned! Agent will use default actions.");
        }
        else
        {
            Debug.Log($"Agent will be in learning mode (BehaviorType: {behaviorType})");
        }

        // Set observation and action space
        behaviorParams.BrainParameters.VectorObservationSize = observationSize;
        behaviorParams.BrainParameters.NumStackedVectorObservations = 1;
        behaviorParams.BehaviorType = BehaviorType.HeuristicOnly;

        // Create ActionSpec for discrete actions only with a single branch of 7 actions
        // Fixed: Use the ActionSpec constructor with branch sizes
        ActionSpec actionSpec = ActionSpec.MakeDiscrete(new int[] { 7 });
        behaviorParams.BrainParameters.ActionSpec = actionSpec;

        Debug.Log($"Agent setup with observation size: {observationSize}, Action space: Discrete(7)");
        DecisionRequester decisionRequester = gameObject.AddComponent<DecisionRequester>();
        decisionRequester.DecisionPeriod = 5;



        Debug.Log($"TetrisMLAgent setup complete on {gameObject.name}");
    }

    private void Start()
    {
        if (enableModelLoggingOnStart)
        {
            LogMLAgentsInfo();
        }
    }

    private void LogMLAgentsInfo()
    {
        // Log information about ML-Agents setup
        if (Academy.IsInitialized)
        {
            Debug.Log($"ML-Agents Academy initialized: {Academy.Instance}");

            var behaviorParams = GetComponent<BehaviorParameters>();
            if (behaviorParams != null)
            {
                Debug.Log($"Behavior name: {behaviorParams.BehaviorName}");
                Debug.Log($"Behavior type: {behaviorParams.BehaviorType}");
                Debug.Log($"Action spec: Discrete({behaviorParams.BrainParameters.ActionSpec.BranchSizes[0]})");

                if (behaviorParams.Model != null)
                {
                    Debug.Log($"Model assigned: {behaviorParams.Model.name}");
                }
                else if (behaviorParams.BehaviorType == BehaviorType.InferenceOnly)
                {
                    Debug.LogError("MODEL IS MISSING but behavior type is InferenceOnly!");
                }
            }
        }
        else
        {
            Debug.LogError("ML-Agents Academy not initialized!");
        }
    }
}