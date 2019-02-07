---
title: Game Recommender Design Sketch
tags: Projects
published: true
---
<H3>
Design Sketch
</H3>
<img src="https://raw.githubusercontent.com/youngtakcho/recommender/master/Sketch.png"/>

<H3>
  User usage scenario <br>
</H3>
1. User starts application. <br>
2. User inputs game which have played or wish to play on input1. <br>
3. User does step 2 on input 2 and input 3. <br>
4. User selects Pg-level.<br>
5. User selects Popularity level.<br>
6. User selects Minimum rating level.<br>
7. User selects range of Release Dates.<br>
8. User touches or pushes Search+ button.<br>
9. User writes text about a game to play or played.<br>
10. User pushes save button.<br>
11. User pushes search button.<br>
12. User pushes item which contains a game to get more information on the Listview.<br>
13. User pushes the back button to move to a previous view.<br>
14. User pushes the back button to move to start point.<br>
15. User pushes the back button to finish program.<br>


<H3>
  Detail usage scenario <br>
</H3>
1. User starts application.<br>
2. User inputs game which have played or wish to play on input1.<br>
3. User does step 2 on input 2 and input 3.<br>
    a. each input methods show the candidates(Game titles) by using dropdown list and data.<br>
4. User selects Pg-level.<br>
5. User selects Popularity level.<br>
6. User selects Minimum rating level.<br>
7. User selects range of Release Dates.<br>
    b. Date Select dialog is opened and when user selects dates it is closed automatically.<br>
8. User touches or pushes Search+ button.<br>
9. User writes text about a game to play or played.<br>
10. User pushes save button.<br>
    c. Previous view automatically shows.<br>
11. User pushes search button.<br>
    d. Classifier predict the user's class and pass it's result to filter.<br>
    e. Filter search datas from Database by using the result and select data which fit to user's criterion. <br>
    f. Recommender calculate similarity between the user's writing and other users reviews.<br>
    g. UI builder build listview and it's components.<br>
12. User pushes item which contains a game to get more information on the Listview.<br>
    h. UI shows selected game's information with several reviews.<br>
13. User pushes the back button to move to a previous view.<br>
14. User pushes the back button to move to start point.<br>
    i. Listview contents are removed and free.<br>
15. User pushes the back button to finish program.<br>

<br>
To See the Presentation <a href="https://1drv.ms/p/s!AgLMBfyzYYtXvS3woDkaWs4DhYt_">click</a>


<!--more-->



