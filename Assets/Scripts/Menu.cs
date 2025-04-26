using UnityEngine;
using UnityEngine.SceneManagement;

public class Menu : MonoBehaviour
{
     public void OnSinglePlayerMode()
     {
          Data.gameMode = BoardManager.GameMode.SinglePlayer;
          SceneManager.LoadScene(1);
     }
     public void OnTwoPlayerMode()
     {
          Data.gameMode = BoardManager.GameMode.TwoPlayer;
          SceneManager.LoadScene(1);
     }
     public void OnAIvsAI()
     {
          Data.gameMode = BoardManager.GameMode.AIVsAI;
          SceneManager.LoadScene(1);
     }
     public void OnAI()
     {
          Data.gameMode = BoardManager.GameMode.AI;
          SceneManager.LoadScene(1);
     }
     public void HumanVsAI()
     {
          Data.gameMode = BoardManager.GameMode.VsAI;
          SceneManager.LoadScene(1);
     }

     public void OnQuitButton()
     {
          Application.Quit();
     }
}
