public interface IPlayerInputController
{
    bool GetLeft();
    bool GetRight();
    bool GetDown();
    bool GetRotateLeft();
    bool GetRotateRight();
    bool GetHardDrop();

    void QueueActions(ActionSequence sequence);
    bool HasQueuedActions();
    void ClearQueue();
}