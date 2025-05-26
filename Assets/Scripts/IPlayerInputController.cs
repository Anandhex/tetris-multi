public interface IPlayerInputController
{
    bool GetLeft();
    bool GetRight();
    bool GetDown();
    bool GetRotateLeft();
    bool GetRotateRight();
    bool GetHardDrop();
    void InvertControls(bool invert);
}