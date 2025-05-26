using UnityEngine;

public class Player2InputController : IPlayerInputController
{

    private bool invertedControls = false;
    public bool GetRotateLeft() => Input.GetKeyDown(KeyCode.LeftBracket);
    public bool GetRotateRight() => Input.GetKeyDown(KeyCode.RightBracket);
    public bool GetDown() => Input.GetKeyDown(KeyCode.DownArrow);
    public bool GetHardDrop() => Input.GetKeyDown(KeyCode.Return);

    public void InvertControls(bool invert)
    {
        invertedControls = invert;
    }

    // Modify your existing input methods to respect the inverted flag
    public bool GetLeft()
    {
        return invertedControls ?
            Input.GetKey(KeyCode.RightArrow) :
            Input.GetKeyDown(KeyCode.LeftArrow);
    }

    public bool GetRight()
    {
        return invertedControls ?
            Input.GetKey(KeyCode.LeftArrow) :
            Input.GetKey(KeyCode.RightArrow);
    }
}
