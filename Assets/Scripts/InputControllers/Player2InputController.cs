using UnityEngine;

public class Player2InputController : IPlayerInputController
{
    public bool GetLeft() => Input.GetKeyDown(KeyCode.LeftArrow);
    public bool GetRight() => Input.GetKeyDown(KeyCode.RightArrow);
    public bool GetRotateLeft() => Input.GetKeyDown(KeyCode.LeftBracket);
    public bool GetRotateRight() => Input.GetKeyDown(KeyCode.RightBracket);
    public bool GetDown() => Input.GetKeyDown(KeyCode.DownArrow);
    public bool GetHardDrop() => Input.GetKeyDown(KeyCode.Return);
}
