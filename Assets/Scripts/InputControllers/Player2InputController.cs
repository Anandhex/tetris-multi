using UnityEngine;

public class Player2InputController : BaseInputController
{
    public override bool GetLeft() => !isExecutingQueue && Input.GetKeyDown(KeyCode.LeftArrow);
    public override bool GetRight() => !isExecutingQueue && Input.GetKeyDown(KeyCode.RightArrow);
    public override bool GetRotateLeft() => !isExecutingQueue && Input.GetKeyDown(KeyCode.LeftBracket);
    public override bool GetRotateRight() => !isExecutingQueue && Input.GetKeyDown(KeyCode.RightBracket);
    public override bool GetDown() => !isExecutingQueue && Input.GetKeyDown(KeyCode.DownArrow);
    public override bool GetHardDrop() => !isExecutingQueue && Input.GetKeyDown(KeyCode.Return);
}
