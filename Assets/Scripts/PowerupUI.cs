using UnityEngine;
using UnityEngine.UI;
using TMPro;
using UnityEngine.EventSystems;

public class PowerUpUI : MonoBehaviour, IPointerClickHandler
{
    public Image iconImage;
    public TextMeshProUGUI nameText;
    public Image frameImage;
    public int slotNumber; // Add this field to track which slot number (1-3) this PowerUpUI represents

    private PowerUp powerUp;
    private int slotIndex;

    public void SetPowerUp(PowerUp powerUp, int index)
    {
        this.powerUp = powerUp;
        this.slotIndex = index;
        this.slotNumber = index + 1; // Convert 0-based index to 1-based slot number

        if (powerUp == null)
        {
            // Empty slot
            iconImage.enabled = false;
            nameText.text = "Empty";
            frameImage.color = Color.gray;
        }
        else
        {
            // Populated slot
            iconImage.enabled = true;
            iconImage.sprite = powerUp.icon;
            nameText.text = powerUp.name;

            // Set frame color based on type
            bool isSelfBenefit = powerUp.type == PowerUpType.ClearRow ||
                                powerUp.type == PowerUpType.SlowDown ||
                                powerUp.type == PowerUpType.BonusPoints ||
                                powerUp.type == PowerUpType.BlockFreeze ||
                                powerUp.type == PowerUpType.ExtraPiece;

            frameImage.color = isSelfBenefit ? Color.green : Color.red;
        }
    }

    public void OnPointerClick(PointerEventData eventData)
    {
        // We'll handle this differently - without requiring key press during click
        Debug.Log($"PowerUp slot {slotNumber} clicked");
        PowerUpManager.Instance.PowerUpSlotClicked(slotIndex);
    }
}