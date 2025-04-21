using UnityEngine;
using UnityEngine.SceneManagement;
using TMPro;

public class GameOver : MonoBehaviour
{
   public TMP_Text scoreText;

   public void OnRetryClick(){
    SceneManager.LoadScene(1);
   }

   public void Start(){
     this.scoreText.text = "Player Score " + Data.PlayerScore;
   }

   public void OnMenuClick(){
    SceneManager.LoadScene(0);
   }
}
