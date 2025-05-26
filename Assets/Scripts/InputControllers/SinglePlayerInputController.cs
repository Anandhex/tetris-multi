using UnityEngine;

public class SinglePlayerInputController : IPlayerInputController
{
    public bool GetRotateLeft() => Input.GetKeyDown(KeyCode.Q);
    public bool GetRotateRight() => Input.GetKeyDown(KeyCode.E);
    public bool GetDown() => Input.GetKeyDown(KeyCode.S) || Input.GetKey(KeyCode.DownArrow);
    public bool GetHardDrop() => Input.GetKeyDown(KeyCode.Space);

    private bool invertedControls = false;

    // Existing methods...

    public void InvertControls(bool invert)
    {
        invertedControls = invert;
    }

    // Modify your existing input methods to respect the inverted flag
    public bool GetLeft()
    {
        return invertedControls ?
            Input.GetKey(KeyCode.D) || Input.GetKey(KeyCode.RightArrow) :
            Input.GetKey(KeyCode.A) || Input.GetKey(KeyCode.LeftArrow);
    }

    public bool GetRight()
    {
        return invertedControls ?
            Input.GetKey(KeyCode.A) || Input.GetKey(KeyCode.LeftArrow) :
            Input.GetKey(KeyCode.D) || Input.GetKey(KeyCode.RightArrow);
    }

}
