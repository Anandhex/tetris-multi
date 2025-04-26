using UnityEngine;

public class AIController : IPlayerInputController
{

    public bool GetLeft() => Input.GetKeyDown(KeyCode.A);
    public bool GetRight() => Input.GetKeyDown(KeyCode.D);
    public bool GetRotateLeft() => Input.GetKeyDown(KeyCode.Q);
    public bool GetRotateRight() => Input.GetKeyDown(KeyCode.E);
    public bool GetDown() => Input.GetKeyDown(KeyCode.S);
    public bool GetHardDrop() => Input.GetKeyDown(KeyCode.Space);
}



