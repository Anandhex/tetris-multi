[System.Serializable]
public struct ActionSequence
{
    public int targetColumn;
    public int targetRotation;
    public bool useHardDrop;

    public ActionSequence(int column, int rotation, bool hardDrop = true)
    {
        targetColumn = column;
        targetRotation = rotation;
        useHardDrop = hardDrop;
    }
}