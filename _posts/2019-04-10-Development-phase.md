---
title: Development phase
tags: Projects
published: true
---

## Development phase 1
The data is collected from stream by using [Steam scraper](https://github.com/prncc/steam-scraper). This project collects reviews and game products data on [Scrapy](https://github.com/scrapy/scrapy) project. 

Collected data is JSON formatted file. Loading entire data to memory is vary heavy. This overhead can be solved with SQLite DB. SQLite DB is simple relational database. We can load data partially from a disk.

To store data to SQLite DB, the table structure is required. Here is my table structure.
For game products data table

    CREATE TABLE "products" (
	"id"	INTEGER,
	"title"	TEXT,
	"app_name"	TEXT,
	"metascore"	TEXT,
	"specs"	TEXT,
	"developer"	TEXT,
	"tags"	TEXT,
	"url"	TEXT,
	"genres"	TEXT,
	"release_date"	TEXT,
	"publisher"	TEXT,
	"price"	TEXT,
	"early_access"	TEXT,
	"reviews_url"	TEXT,
	"discount_price"	TEXT,
	"n_reviews"	TEXT,
	"sentiment"	TEXT,
	"p_genre"	TEXT,
	PRIMARY KEY("id"));

 For reviews data table 

    CREATE TABLE "reviews" (
	"r_id"	INTEGER PRIMARY KEY AUTOINCREMENT,
	"product_id"	INTEGER,
	"user_id"	INTEGER,
	"recommended"	TEXT,
	"hours"	REAL,
	"text"	TEXT,
	"found_funny"	TEXT,
	"compensation"	TEXT,
	"page_order"	TEXT,
	"products"	TEXT,
	"early_access"	TEXT,
	"username"	TEXT,
	"page"	TEXT,
	"date"	TEXT);

To store data to SQLite,  **data_to_db.py** can be used. This python script read Json file and store data to SQLite DB. In this step, the review texts have a lot of noise like non English texts, emoticons and special characters. the code in the python script remove these noise.

    " ".join(re.findall("[a-zA-Z]+", data_string))

After storing data, we have to build inverted index to make search process faster. 
In this project, inverted index is a python dictionary which has **a word as a key** and **reviews' id as a value**. **i_index.py** python script makes inverted index dictionary and stores it as a binary file.
With the Inverted index dictionary, search query can be select reviews which contains words in query **in one second**. However, without the inverted index dictionary, it takes unacceptable time (over 5 ~ 60 minutes depends on the number of words in a query)

The code which build the dictionary is quite simple.

    for (idx , text) in rows:  
    id_list = []  
    pre_processed = preprocess(text)  
    for i in pre_processed:  
        if i not in dictionary:  
            dictionary[i] = set()  
        dictionary[i].add(idx)
the preprocess function do lemmatization and stemming with Gensim. 
First for loop iterates words in a pre_process list and check whether it is in the dictionary or not. 
if not, add it to dictionary as a key and make set as it's value. python set do not store duplicated data which is reviews' id. After doing this for all review texts, we can get reviews' id by using a word as a key of the dictionary.



## Development phase 2

Genre can be one of features for classifying game reviews. Moreover, reviews from users contains keywords which can be query words for searching a game. In this reason, we can improve user search results by using classifier. the review texts and a game genre can be a train data set. A classifier train reviews and what genre is for reviewed game.
However, there are many genres in one game. one genre have to selected to a game.
Here is two possible simple solution for this problem.
1. Duplicate text for each genre
2. Select a genre from genres list randomly.

First one makes data size double or more and makes expected train time much longer. Second one make more simple and data size and train time is same to original data.
For select a genre from genres,  **select_genre.py** script file iterates all data rows in the database and select primary genre from a game's genres column.

After this, Naive Bayes classifier can be used. Here is mean time table which is calculated with varied option and 3 different type of Naive Bayes classifier.

<table>
  <tr>
    <th>Vectorize methods and Type of Naive Bayes</th>
    <th>Precision</th>
  </tr>
  <tr>
    <td>CountVectorizer multinormialNB</td>
    <td> 0.7603673613564111 </td>
  </tr>
  <tr>
    <td>CountVectorizer GaussianNB</td>
    <td>0.48851995761215117 </td>
  </tr>
  <tr>
    <td>CountVectorizer BernoulliNB </td>
    <td>0.5824090427410809</td>
  </tr>
  <tr>
    <td>tf-idf multinormialNB </td>
    <td>0.6729070999646768</td>
	</tr>
  <tr>
    <td>tf-idf GaussianNB  </td>
    <td>0.4487460261391734</td>
  </tr>
<tr>
    <td>tf-idf BernoulliNB  </td>
    <td>0.6478460391863427</td>
  </tr>
</table>

for this experiment, the number of sampled data of each genres is  <br>
'Sports': 2093, <br>
'Casual': 8107, <br>
'Racing': 4847, <br>
'Strategy': 5253, <br>
'Action': 6000, <br>
'Simulation': 6000, <br>
'Indie': 6000, <br>
'Adventure': 5965, <br>
'RPG': 3071<br>
 total: 41336<br>

When I get sample data, I only select reviews from a user who played review's game over 100 hours and reviews length over 100 characters. Since I assume that there is no person who write inappropriate words with effort to type over 100 characters and a plyer who played game over 100 hours is good at writing good reviews for game, I select reviews.

We can see **CountVectorizer multinormialNB** makes best result. Now the server use **CountVectorizer multinormialNB** to predict a class of user's query.

CounterVectorrizer and Tf-idf Vectorizer I made use python dictionary and words are key of the dictionary. This makes processing time longer. Due to string comparisons takes a lot of overheads, every serch operation in dictionary takes time. To get a idea to speed up process, now I study sklearn's source code. They don't use string comparisons. Instead of using String, they use a number comparison by mapping a word to a number. When training process run, there are a lot of getting word-value operation in dictionary. By using a number not string for storing words, they can redusse a lot of time to training process. In addtion, they also use numpy and Compressed Sparse 
Row Matrix which does not store all matrix elemetns but store non zero values and index of them. Numpy use a low level C impliaments which is very fast and CSR Matrix helps same memory space. 
Now I take this idea and re-build all my model codes.

## Ongoing tasks
1. Build a new counter vectorizer and tf-idf vecotrizer and naive bayes classifier without string comparisons.
2. Build a Recommender module.
3. Make documents for this projects.
