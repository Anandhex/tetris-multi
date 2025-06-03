using UnityEngine;

public class Player1InputController : BaseInputController
{
    public override bool GetLeft() => !isExecutingQueue && Input.GetKeyDown(KeyCode.A);
    public override bool GetRight() => !isExecutingQueue && Input.GetKeyDown(KeyCode.D);
    public override bool GetRotateLeft() => !isExecutingQueue && Input.GetKeyDown(KeyCode.Q);
    public override bool GetRotateRight() => !isExecutingQueue && Input.GetKeyDown(KeyCode.E);
    public override bool GetDown() => !isExecutingQueue && Input.GetKeyDown(KeyCode.S);
    public override bool GetHardDrop() => !isExecutingQueue && Input.GetKeyDown(KeyCode.Space);
}
