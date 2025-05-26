using UnityEngine;

public class Player1InputController : IPlayerInputController
{
    private bool invertedControls = false;
    public bool GetRotateLeft() => Input.GetKeyDown(KeyCode.Q);
    public bool GetRotateRight() => Input.GetKeyDown(KeyCode.E);
    public bool GetDown() => Input.GetKeyDown(KeyCode.S);
    public bool GetHardDrop() => Input.GetKeyDown(KeyCode.Space);
    public void InvertControls(bool invert)
    {
        invertedControls = invert;
    }

    // Modify your existing input methods to respect the inverted flag
    public bool GetLeft()
    {
        return invertedControls ?
            Input.GetKey(KeyCode.D) :
            Input.GetKey(KeyCode.A);
    }

    public bool GetRight()
    {
        return invertedControls ?
            Input.GetKey(KeyCode.A) :
            Input.GetKey(KeyCode.D);
    }

}
