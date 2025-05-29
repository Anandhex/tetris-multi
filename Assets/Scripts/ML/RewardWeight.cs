public class RewardWeights
{
    public float clearReward;
    public float comboMultiplier;
    public float perfectClearBonus;
    public float stagnationPenaltyFactor;
    public float roughnessRewardMultiplier;
    public float roughnessPenaltyMultiplier;
    public float holeFillReward;
    public float holeCreationPenalty;
    public float wellRewardMultiplier;
    public float iPieceInWellBonus;
    public float stackHeightPenalty;
    public float uselessRotationPenalty;
    public float tSpinReward;
    public float iPieceGapFillBonus;
    public float accessibilityRewardMultiplier;
    public float accessibilityPenaltyMultiplier;
    public float deathPenalty;

    public float idleActionPenalty; // For 'Do nothing' action
    public float moveDownActionReward; // For 'moveDown' action
    public float hardDropActionReward; // For 'hardDrop' action
    public float doubleLineClearRewardMultiplier; // Multiplier for 2 lines
    public float tripleLineClearRewardMultiplier; // Multiplier for 3 lines
    public float tetrisClearRewardMultiplier;     // Multiplier for 4 lines (Tetris)
    public float maxWellRewardCap; // Maximum reward for well formation
}