using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using Newtonsoft.Json;
public class SocketManager : MonoBehaviour
{
    [Header("Socket Settings")]
    public int port = 12345;
    public string host = "127.0.0.1";

    private TcpListener tcpListener;
    private Thread tcpListenerThread;
    private TcpClient connectedTcpClient;
    private bool isListening = false;

    public static SocketManager Instance { get; private set; }

    // Events for communication
    public event System.Action<GameCommand> OnCommandReceived;
    public event System.Action OnPythonConnected;
    public event System.Action OnPythonDisconnected;

    void Awake()
    {
        var args = Environment.GetCommandLineArgs();
        port = int.TryParse(GetArg(args, "-port", port.ToString()), out var p) ? p : port;
        host = GetArg(args, "-host", host);

        Debug.Log($"[SocketManager] Listening on {host}:{port}");
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else
        {
            Destroy(gameObject);
        }
    }

    void Start()
    {
        StartServer();
    }

    void StartServer()
    {
        try
        {
            tcpListener = new TcpListener(IPAddress.Any, port);
            tcpListenerThread = new Thread(new ThreadStart(ListenForTcpClients));
            tcpListenerThread.IsBackground = true;
            tcpListenerThread.Start();
            isListening = true;

            Debug.Log($"Unity Socket Server started on port {port}");
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to start server: {e.Message}");
        }
    }

    void ListenForTcpClients()
    {
        tcpListener.Start();

        while (isListening)
        {
            try
            {
                connectedTcpClient = tcpListener.AcceptTcpClient();
                Debug.Log("Python client connected!");

                // Notify connection on main thread
                UnityMainThreadDispatcher.Instance.Enqueue(() => OnPythonConnected?.Invoke());

                Thread clientThread = new Thread(new ParameterizedThreadStart(HandleTcpClient));
                clientThread.IsBackground = true;
                clientThread.Start(connectedTcpClient);
            }
            catch (Exception e)
            {
                if (isListening)
                {
                    Debug.LogError($"TCP Listener error: {e.Message}");
                }
            }
        }
    }

    void HandleTcpClient(object client)
    {
        TcpClient tcpClient = (TcpClient)client;
        NetworkStream clientStream = tcpClient.GetStream();

        byte[] message = new byte[4096];
        int bytesRead;

        while (true)
        {
            bytesRead = 0;

            try
            {
                bytesRead = clientStream.Read(message, 0, 4096);
            }
            catch (Exception e)
            {
                Debug.LogError($"Client read error: {e.Message}");
                break;
            }

            if (bytesRead == 0)
            {
                Debug.Log("Python client disconnected");
                UnityMainThreadDispatcher.Instance.Enqueue(() => OnPythonDisconnected?.Invoke());
                break;
            }

            string jsonMessage = Encoding.UTF8.GetString(message, 0, bytesRead);

            try
            {
                GameCommand command = JsonConvert.DeserializeObject<GameCommand>(jsonMessage);
                // Execute on main thread
                UnityMainThreadDispatcher.Instance.Enqueue(() => OnCommandReceived?.Invoke(command));
            }
            catch (Exception e)
            {
                Debug.LogError($"Failed to parse command: {e.Message}");
            }
        }

        tcpClient.Close();
    }

    public void SendGameState(GameState gameState)
    {
        if (connectedTcpClient == null || !connectedTcpClient.Connected)
            return;
        try
        {
            string json = JsonConvert.SerializeObject(gameState);
            byte[] data = Encoding.UTF8.GetBytes(json + "\n");

            NetworkStream stream = connectedTcpClient.GetStream();
            stream.Write(data, 0, data.Length);
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to send game state: {e.Message}");
        }
    }

    void OnApplicationQuit()
    {
        StopServer();
    }

    void OnDestroy()
    {
        StopServer();
    }

    void StopServer()
    {
        isListening = false;

        if (tcpListener != null)
        {
            tcpListener.Stop();
        }

        if (connectedTcpClient != null)
        {
            connectedTcpClient.Close();
        }

        if (tcpListenerThread != null)
        {
            tcpListenerThread.Abort();
        }


    }
    public static string GetArg(string[] args, string name, string defaultVal = null)
    {
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == name && i + 1 < args.Length)
                return args[i + 1];
        }
        return defaultVal;
    }
}
