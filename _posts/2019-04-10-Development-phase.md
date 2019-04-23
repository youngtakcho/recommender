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

For vectorizing and calculating Tf-IDF, Please read phase 2.

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
<tr>
    <td>My CountVectorizer and Naive Beyse  </td>
    <td> calculating... </td>
  </tr></table>
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

CounterVectorrizer and Tf-idf Vectorizer I made use python dictionary and words are key of the dictionary. This makes processing time longer. Due to string comparisons takes a lot of overheads, every serch operation in dictionary takes time. To get a idea to speed up process, now I study sklearn's source code. They don't use string comparisons. Instead of using String, they use a number comparison by mapping a word to a number. When training process run, there are a lot of getting word-value operation in dictionary. By using a number not string for storing words, they can redusse a lot of time to training process. In addtion, they also use numpy and Compressed Sparse Row Matrix which does not store all matrix elemetns but store non zero values and index of them.
<br>Numpy use a low level C impliaments which is very fast and CSR Matrix helps to save memory space.

Now I re-make my inverted index, CountVectorizer and TfIdfVectorizer after I studied sklearn implimentation.

before and after change in processing time to calculate all rows in data set like below.

<table>
  <tr>
    <th>adopting numpy and CSR matrix</th>
    <th>processing time in sec</th>
  </tr>
  <tr>
    <td>before</td>
    <td>1824</td>
  </tr>
  <tr>
    <td>after</td>
    <td>869</td>
  </tr>
</table>

Now, I get inverted index dictionary, trained CountVectorizer and TfIdfVectorizer in half the time compared to before.

As I mentioned earlier, the factor which make the code fast is avoiding string mathing. 

Code 1

```python
if word not in i_dict:
	self.word_dictionary[word] = self.idx
	self.idx +=1
	self.i_dict[word] = [0, dict()]
	if d_id not in i_dict[word][1]:
		self.i_dict[word][1][d_id] = 0
		self.i_dict[word][1][d_id] += 1
		self.i_dict[word][0] += 1
```

Code 2

```python
indptr.append(0)
index = 0
for raw_doc in raw_docs:
  feature_counter = {}
  doc = self.preprocess(raw_doc)
  for feature in doc:
    if y is not None:
      if feature not in inverted_index_dict:
        inverted_index_dict[feature] = set()
        inverted_index_dict[feature].add(y[index])
			try:
        feature_idx = vocab[feature]
        if feature_idx not in feature_counter:
          feature_counter[feature_idx]=0
          feature_counter[feature_idx] +=1
			except KeyError:
              continue
		j_indices.extend(feature_counter.keys())
    values.extend((feature_counter.values()))
    indptr.append(len(j_indices))
    index += 1
```

Code 1 is old code and code 2 is new code. In code 1, I use a word as a key of dictionary for storing inverted index and doc-term occruence. In the code 2, it use the dictionary but key is a number of id for word. Removing stirng comparison and using doc-tarm matrix instead of using doc-tarm dictionary, these make the difference.

### Detail of CountVecotizer and TfidfVectorizer

To convert review documents to vectors, we can count word by word and store to a data structure. at the start of thid project, dictionary with string as a key was used to store vectorized documents like below.
```python
vectorized = { word : ( totoal count, { doc_id:count } ) }
```
This structure, as I mentioned, has a disadvantage about processing speed. Now, the used structure is this.

```python
# X = scipy.sparse.csr_matrix
# data = the list of numbers of counted words
# IA = a list of the cumulated number of data. It shows how many data are in a row. The number of rows in matrix is len(IA) - 1.
# JA = a list of the column index of data. each column index is parsed to a term in the dataset.
# Vocabulary = a dictionary which contains the id number of each tarms in documents (dataset).
X = scipy.sparse.csr_matrix((data,JA,IA),shape=(len(IA)-1,len(vocabulary)),dytpe=numpy.int64)

```

the Dictionary based representation of vector is simple and good to understend directly but it is slow. However, the Compressed Sparse Row Matrix method seems to bed to understend directly but it is fast. 

Now we have doc-term occurrence table as a CSR Matrix. Each rows in the table shows how the terms are occurred in the documents. 

```python
documents = [
	"apple apple iphone macos",
	"android google samsung motorola",
	"windows phone apple android samsung"
]
```

then the doc-term occurrence table is like below.

| doc id | apple | iphone | macos | android | google | samsung | motorola | windws | phone |
| ------ | ----- | ------ | ----- | ------- | ------ | ------- | -------- | ------ | ----- |
| 0      | 2     | 1      | 1     | 0       | 0      | 0       | 0        | 0      | 0     |
| 1      | 0     | 0      | 0     | 1       | 1      | 1       | 1        | 0      | 0     |
| 2      | 1     | 0      | 0     | 1       | 0      | 1       | 0        | 1      | 1     |

In the code 2, there are values, j_indices, indptr arrays. they are formed in the rule of CSR matrix ( data = values, j_array = JA , indptr = IA).

And Now How to calculate Tf-idf?, the doc-term occurrence table is Term Frequency table. Tf calculation is already done. How about the idf value? Idf calculation is done with this formula.
$$
idf_t = \log({N = The~number~of~Total~Documents \over df_t =  The~number~of~Documents~which~contains~a~term~t})
$$


The number of Total Documents in dataset is the number of rows in the table and the $df_t$ can be calculated by count the number of non zero values in the term column.

Here is the code which calculate idf of all terms in documents.

```python
n_samples , n_feature = X.shape
df = self._document_frequency(X).astype(dtype)
df += int(self.smooth_idf)
n_samples += int(self.smooth_idf)
idf = np.log(n_samples/df) +1
self._idf_diag = sp.diags(idf,offsets=0,
                          shape=(n_feature,n_feature),
                          format='csr',
                          dtype=dtype)
```

df is a document frquency vector which contains the number of tarm occurrence in each rows in the data. the value "smooth_idf" is smoothing value which protects idf result from over/underflow and divied by Zero Error. 

In this code, idf is a vector of numbers and it's dimention is ( 1, the number of terms). To make calculation faster, we can use the metrix operation for Tf-idf calculation. Making idf vector to a diagonal matrix and Doing vector - matrix multply operation are the way to speed up and easy to readable for us.

code line  `scipy.sparse.diags` method makes a diagonal matrix and the result of thid method will use in runtime like this.

`X = X * self._idf_diag`

X is term frequency matrix and _idf_diag is a digonal matrix. We can calculate Tf-Idf value by multipying them.



### Detail of Naive Bayes

The Naive Bayes classifier predicts a class of query by calculating probabilities for each class. For this project, it predicts a class of game from user's query.

The probability of a class over query X is here.  X is a vector to which the vectorizer change query. 
$$
Classes = \{Sports, Casual, Racing, Strategy, Action, Simulation, Indie, Adventure RPG\} \\
Class_k \in Classes\\
X = (x_1,x_2,...,x_n)\\
P(Class_k|X) = { {P(X|Class_k) * P(Class_k)} \over P(X) }
$$
## evaluation
Now updating

## Ongoing tasks
1. Build a new counter vectorizer and tf-idf vectorizer and naive bayes classifier without string comparisons.
2. Build a Recommender module.
3. Make documents for this projects.

## References

1. Scikit-Learn. <em>Sklearn text module</em>.<br><https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_extraction/text.py>
2. Wikipedia. <em>Sparse Matrix</em> <br><https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)>