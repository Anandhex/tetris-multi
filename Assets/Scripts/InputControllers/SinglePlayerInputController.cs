using UnityEngine;

public class SinglePlayerInputController : IPlayerInputController
{
    public bool GetLeft() => Input.GetKey(KeyCode.A) || Input.GetKey(KeyCode.LeftArrow);
    public bool GetRight() => Input.GetKey(KeyCode.D) || Input.GetKey(KeyCode.RightArrow);
    public bool GetRotateLeft() => Input.GetKeyDown(KeyCode.Q);
    public bool GetRotateRight() => Input.GetKeyDown(KeyCode.E);
    public bool GetDown() => Input.GetKeyDown(KeyCode.S) || Input.GetKey(KeyCode.DownArrow);
    public bool GetHardDrop() => Input.GetKeyDown(KeyCode.Space);
}
