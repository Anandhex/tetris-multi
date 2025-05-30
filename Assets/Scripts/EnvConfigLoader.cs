using System.IO;
using UnityEngine;

public class EnvConfigLoader : MonoBehaviour
{
    public string outputFolder;

    void Awake()
    {
        string configPath = Path.Combine(Application.persistentDataPath, "env_config.json");
        // Debug.Log($"Looking for config file at: {configPath}");

        if (File.Exists(configPath))
        {
            try
            {
                string jsonText = File.ReadAllText(configPath);
                EnvConfig config = JsonUtility.FromJson<EnvConfig>(jsonText);
                outputFolder = config.outputFolder;
                // Debug.Log($"Loaded outputFolder from config: {outputFolder}");
            }
            catch (System.Exception e)
            {
                // Debug.LogWarning("Failed to load env_config.json: " + e.Message);
                outputFolder = Path.Combine(Application.persistentDataPath, "results");
            }
        }
        else
        {
            // Debug.LogWarning("env_config.json not found, using default path.");
            outputFolder = Path.Combine(Application.persistentDataPath, "results");
        }
    }

    [System.Serializable]
    private class EnvConfig
    {
        public string outputFolder;
    }
}
