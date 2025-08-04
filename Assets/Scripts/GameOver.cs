using UnityEngine;
using UnityEngine.SceneManagement;
using TMPro;

public class GameOver : MonoBehaviour
{
  public TMP_Text scoreText;

  public void OnRetryClick()
  {
    SceneManager.LoadScene(1);
  }

  public void Start()
  {
    switch (Data.gameMode)
    {
      case BoardManager.GameMode.SinglePlayer:
        scoreText.text = $"Score: {Data.PlayerScore}";
        break;

      case BoardManager.GameMode.AIVsAI:
      case BoardManager.GameMode.VsAI:
      case BoardManager.GameMode.TwoPlayer:
        if (Data.WinnerName == "Draw")
        {
          scoreText.text = $"It's a Draw!\n\n" +
                           $"Player1: {Data.WinnerScore}\n" +
                           $"Player2: {Data.LoserScore}";
        }
        else
        {
          scoreText.text =
              $"{Data.WinnerName} Wins!\n\n" +
              $"{Data.WinnerName}: {Data.WinnerScore}\n" +
              $"{Data.LoserName}: {Data.LoserScore}";
        }
        break;

      default:
        scoreText.text = "Game Over";
        break;
    }
  }

  public void OnMenuClick()
  {
    SceneManager.LoadScene(0);
  }
}
