using UnityEngine;

public class PlayerInputController : MonoBehaviour
{
    public KeyCode moveLeft;
    public KeyCode moveRight;
    public KeyCode softDrop;
    public KeyCode hardDrop;
    public KeyCode rotateLeft;
    public KeyCode rotateRight;

    public bool GetLeft() => Input.GetKeyDown(moveLeft);
    public bool GetRight() => Input.GetKeyDown(moveRight);
    public bool GetDown() => Input.GetKey(softDrop);
    public bool GetHardDrop() => Input.GetKeyDown(hardDrop);
    public bool GetRotateLeft() => Input.GetKeyDown(rotateLeft);
    public bool GetRotateRight() => Input.GetKeyDown(rotateRight);
}
