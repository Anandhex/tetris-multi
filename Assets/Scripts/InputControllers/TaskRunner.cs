using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public interface ITetrisTask
{
    IEnumerator Execute();
    Board GetBoard();
    string Description { get; }
}

public class TaskQueueRunner : MonoBehaviour
{
    private Queue<ITetrisTask> taskQueue = new Queue<ITetrisTask>();
    private bool isRunning = false;

    public void EnqueueTask(ITetrisTask task)
    {
        taskQueue.Enqueue(task);
        if (!isRunning)
        {
            StartCoroutine(RunQueue());
        }
    }

    private IEnumerator RunQueue()
    {
        isRunning = true;

        while (taskQueue.Count > 0)
        {
            var task = taskQueue.Peek();
            Board boardProperty = task.GetBoard();
            while (boardProperty.isLocked)
            {
                yield return null;
            }
            boardProperty.Lock();
            task = taskQueue.Dequeue();
            yield return StartCoroutine(task.Execute());
            boardProperty.Unlock();
        }

        isRunning = false;
    }
}
