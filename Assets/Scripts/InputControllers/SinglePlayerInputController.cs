using UnityEngine;

public class SinglePlayerInputController : BaseInputController
{
    public override bool GetLeft() => !isExecutingQueue && (Input.GetKey(KeyCode.A) || Input.GetKey(KeyCode.LeftArrow));
    public override bool GetRight() => !isExecutingQueue && (Input.GetKey(KeyCode.D) || Input.GetKey(KeyCode.RightArrow));
    public override bool GetRotateLeft() => !isExecutingQueue && Input.GetKeyDown(KeyCode.Q);
    public override bool GetRotateRight() => !isExecutingQueue && Input.GetKeyDown(KeyCode.E);
    public override bool GetDown() => !isExecutingQueue && (Input.GetKeyDown(KeyCode.S) || Input.GetKey(KeyCode.DownArrow));
    public override bool GetHardDrop() => !isExecutingQueue && Input.GetKeyDown(KeyCode.Space);
}

