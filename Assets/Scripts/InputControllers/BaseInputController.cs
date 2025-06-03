public abstract class BaseInputController : IPlayerInputController
{
    protected ActionSequence? queuedAction = null;
    protected bool isExecutingQueue = false;

    // Traditional input methods (to be overridden by specific controllers)
    public abstract bool GetLeft();
    public abstract bool GetRight();
    public abstract bool GetDown();
    public abstract bool GetRotateLeft();
    public abstract bool GetRotateRight();
    public abstract bool GetHardDrop();

    // Queue-based methods
    public virtual void QueueActions(ActionSequence sequence)
    {
        queuedAction = sequence;
        isExecutingQueue = true;
    }

    public virtual bool HasQueuedActions()
    {
        return queuedAction.HasValue;
    }

    public virtual void ClearQueue()
    {
        queuedAction = null;
        isExecutingQueue = false;
    }
}
