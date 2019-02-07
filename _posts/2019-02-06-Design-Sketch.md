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
1. User starts application
2. User inputs game which have played or wish to play on input1.
3. User does step 2 on input 2 and input 3.
4. User selects Pg-level.
5. User selects Popularity level.
6. User selects Minimum rating level
7. User selects range of Release Dates.
8. User touches or pushes Search+ button.
9. User writes text about a game to play or played.
10. User pushes save button.
11. User pushes search button.
12. User pushes item which contains a game to get more information on the Listview.
13. User pushes the back button to move to a previous view.
14. User pushes the back button to move to start point.
15. User pushes the back button to finish program.


<H3>
  Detail usage scenario <br>
</H3>
1. User starts application
2. User inputs game which have played or wish to play on input1.
3. User does step 2 on input 2 and input 3.
    a. each input methods show the candidates(Game titles) by using dropdown list and data.
4. User selects Pg-level.
5. User selects Popularity level.
6. User selects Minimum rating level
7. User selects range of Release Dates.
    b. Date Select dialog is opened and when user selects dates it is closed automatically.
8. User touches or pushes Search+ button.
9. User writes text about a game to play or played.
10. User pushes save button.
    c. Previous view automatically shows.
11. User pushes search button.
    d. Classifier predict the user's class and pass it's result to filter.
    e. Filter search datas from Database by using the result and select data which fit to user's criterion. 
    f. Recommender calculate similarity between the user's writing and other users reviews.
    g. UI builder build listview and it's components.
12. User pushes item which contains a game to get more information on the Listview.
    h. UI shows selected game's information with several reviews.
13. User pushes the back button to move to a previous view.
14. User pushes the back button to move to start point.
    i. Listview contents are removed and free.
15. User pushes the back button to finish program.

<br>
To See the Presentation <a href="https://1drv.ms/p/s!AgLMBfyzYYtXvS3woDkaWs4DhYt_">click</a>


<!--more-->



