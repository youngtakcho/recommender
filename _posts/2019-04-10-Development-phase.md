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

```python
for (idx , text) in rows:  
id_list = []  
pre_processed = preprocess(text)  
for i in pre_processed:  
    if i not in dictionary:  
        dictionary[i] = set()  
    dictionary[i].add(idx)
```

The preprocess function do lemmatization and stemming with Gensim. 
First for loop iterates words in a pre_process list and check whether it is in the dictionary or not. 
if not, add it to dictionary as a key and make set as it's value. python set do not store duplicated data which is reviews' id. After doing this for all review texts, we can get reviews' id by using a word as a key of the dictionary.

Genre can be one of features for classifying game reviews. Moreover, reviews from users contains keywords which can be query words for searching a game. In this reason, we can improve user search results by using classifier. the review texts and a game genre can be a train data set. A classifier train reviews and what genre is for reviewed game.
However, there are many genres in one game. one genre have to selected to a game.
Here is two possible simple solution for this problem.

1. Duplicate text for each genre
2. Select a genre from genres list randomly.

First one makes data size double or more and makes expected train time much longer. Second one make more simple and data size and train time is same to original data.
For select a genre from genres,  **select_genre.py** script file iterates all data rows in the database and select primary genre from a game's genres column.

CounterVectorrizer and Tf-idf Vectorizer I made before use the python dictionary. A word in data is a key of the  dictionary. This makes processing time longer. Due to string comparisons takes a lot of overheads, every serch operation in dictionary takes time. To get a idea to speed up process, now I study sklearn's source code. They don't use string comparisons. Instead of using String, they use a number comparison by mapping a word to a number. When training process run, there are a lot of getting word-value operation in dictionary. By using a number not string for storing words, they can redusse a lot of time to training process. In addtion, they also use numpy and Compressed Sparse Row Matrix which does not store all matrix elemetns but store non zero values and index of them.
<br>Numpy use a low level C impliaments which is very fast and CSR Matrix helps to save memory space.

Now I re-make my inverted index, CountVectorizer and TfIdfVectorizer after I studied sklearn implimentation.

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

Code 1 is old code and code 2 is new code. In code 1, I use a word as a key of dictionary for storing inverted index and doc-term occurrence. In the code 2, it use the dictionary but key is a number of id for word. Removing string comparison and using doc-term matrix instead of using doc-term dictionary, these make the difference.

By using CounterVectorrizer and TfidfVectorizer, reviews and a query from user can be transformed into vectors. After those process, similarity can be calculated with the formula below.
$$
v_{d1} = Vectorized~document~1\\
v_{d2} = Vectorized~document~2\\
Cosin~Similarity(v_{d1},v_{d2}) = cos(\theta) = {~{\sum(v_{ds1}~}_{i}~{v_{d2}~ }_{i})\over\sqrt\sum{~{v_{d1}~}_{i}~}\sqrt\sum{~{v_{d2}~}_{i}~}~}
$$
In the formula, denominator is multiplying the length of the vector 1 and the length of the vector 2. In the cosine similarity calculation, the distance between two vectors is not a matter but angle between two vectors. Moreover, the function sqrt  is expensive. If vectors are normalized before, denominator is 1. So now the formula can be represented.
$$
v_{d1} = Vectorized~document~1\\
v_{d2} = Vectorized~document~2\\
Cosin~Similarity(v_{d1},v_{d2}) =cos(\theta)= {~{~{\sum^{n}_{i=0}~}({v_{d1}~}_{i}~{v_{d2}~}_{i}~})}
$$



To convert review documents to vectors, we can count word by word and store to a data structure. at the start of the project, dictionary with string as a key was used to store vectorized documents like below.

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

the Dictionary based representation of vector is simple and easy to be implemented but slow. However, Compressed Sparse Row matrix based method is good for understanding and fast.

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

And Now How to calculate Tfidf?, the doc-term occurrence table is basically Term Frequency table. Tf calculation is already done. How about the idf value? Idf calculation is done with this formula.
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

df is a document frequency vector which contains the number of term occurrence in each rows in the data. the value "smooth_idf" is smoothing value which protects idf result from over/underflow and divided by Zero Error. 

In this code, idf is a vector of numbers and it's dimention is ( 1, the number of terms). To make calculation faster, we can use the metrix operation for Tf-idf calculation. Making idf vector to a diagonal matrix and Doing vector - matrix multply operation are the way to speed up and easy to readable for us.

code line  `scipy.sparse.diags` method makes a diagonal matrix and the result of thid method will use in runtime like this.

`X = X * self._idf_diag`

X is term frequency matrix and _idf_diag is a diagonal matrix. We can calculate Tfidf value by multipying them.

### The challenging things for me in this step

First, for counting words for transforming documents into vectors, the processing time is too long. to solve this, I apply matrix methods.

Second, the size of the inverted index is too big to load a light server. this is important to real time service on a server. Loading it from a disk is slow and it makes service quality poor. In this time, I cannot solve this problem in softwear manner. However, I move a server which has capability.

## Development phase 2

For classifying a game genre, We can use the Naive Bayes classifier. The Naive Bayes classifier predicts a class of query by calculating probabilities for each class. the Naive Bayes classifier based on CounterVectorizer.

The probability of a class over query X is here.  X is a vector to which the vectorizer change a query. 
$$
Classes = \{Sports, Casual, Racing, Strategy, Action, Simulation, Indie, Adventure RPG\} \\
Class_k \in Classes\\
a~Query~X = (x_1,x_2,...,x_n)\\
P(Class_k|X) = { {P(X|Class_k) * P(Class_k)} \over P(X) }
$$
And by the conditional independency, $P(X|Class_k)$ can be represented like this.

$$
P(x_1|Class_x) * P(x_2|Class_x) * P(x_3|Class_x)*... * P(x_n|Class_x)
$$

the formula can be transformed like this.
$$
P(Class_k|X) = { {P(x_1|Class_x) * P(x_2|Class_x) * P(x_3|Class_x)*... * P(x_n|Class_x) * P(Class_k)} \over P(X) }
$$
For Naive Bayes classifier, we do not need denominator of the formula since we just find Class_k which makes P(Class_k|X) to be highest values among them by comparing between them.
$$
P(Class_k|X) = { {P(x_1|Class_x) * P(x_2|Class_x) * P(x_3|Class_x)*... * P(x_n|Class_x) * P(Class_k)}}
$$
So how to calculate? $P(x_n|Class_k) n=1...n , Class_k \in Class $ can be pre-calculated by using CountVectorizer.

1. Add up all term occurrences in each classes. simply add up all rows in doc-term matrix class by class.
   1. add 1 to zero element in the row for smoothing.
2. divide the vector by total number of word occurrences in in the class + total number of terms in the class (for smoothing)
3. Once the calculation of the probabilities of all terms in class for all classes, make each of them into diagonal matrices.

Here is the code to calculating the probabilities. We already count word occurrences in data, class by class, when we calculate CountVectorizer processing.

```python
    def fit(self,X,y):
        r_index = 0
        # sum all number of words for each classes.
        classes = defaultdict()
        classes.default_factory = classes.__len__
        for cls in y:
            class_id = classes[cls]
            if class_id not in self.class_dictionary:
                self.class_dictionary[class_id] = X.getrow(r_index)
                self.n_doc_in_classes[class_id] = 1
            else:
                self.class_dictionary[class_id] = self.class_dictionary[class_id] + X.getrow(r_index)
                self.n_doc_in_classes[class_id] += 1
            r_index += 1
        self.total_doc = r_index
        
        # calculate probability of words in classes
        for cls in classes:
            self.total_words_in_class[cls] = self.class_dictionary[classes[cls]].sum()
            d_smoothing_factor = self.class_dictionary[classes[cls]].shape[1]
            Tf_vec = self.class_dictionary[classes[cls]].getrow(0).toarray() + 1 # for smoothing
            total_number_of_term_occurrence = self.total_words_in_class[cls]
            p_words_g_class = Tf_vec/(total_number_of_term_occurrence+d_smoothing_factor)  # for smoothing
            p_d = sp.lil_matrix((p_words_g_class.shape[1], p_words_g_class.shape[1]),dtype=np.float64)
            np.log(p_words_g_class)
            for i in range(0, ar.shape[1]):
                p_d[i, i] = ar[0][i]
            self.class_word_probability[cls] = p_d

        self.classes = dict(classes)
```

X is a doc-term matrix from CountVecotrizer, y is a clsses list of each documents ( list of game genre ). the first for loop of this code sum up all rows in X, class by class.

For each classes, term probability vector is transformed into a diagonal matrix form. What a diagonal matrix form  is for?

X is a matrix which is made by the CountVectorizer. when user queries come into the system, them will be transformed into a matrix as same as documents. if CountVectorizer transforms a user query into a boolean vector which contains only 0 or 1 (existence of a word in query), the result of multiplication of the vector and the diagonal matrix is  $P(Class_k\| X)$  in log scale form.

Here is the code.

```python
    def predict(self,X):
        predicted_results = []
        for i in range(0,X.shape[0]):
            target = X.getrow(i)
            results = []
            for cls in self.classes:
                cp_target = target.getrow(0)
                boolen_target = cp_target.getrow(0)
                boolen_target.data.fill(1) # change all value into 0 or 1
                proba_in_class = self.class_word_probability[cls]
                proba_in_query = (boolen_target) * proba_in_class
                for i in range(0,len(proba_in_query.data)):
                    proba_in_query.data[i] *=  target.data[i]
                class_proba = self.n_doc_in_classes[self.classes[cls]] / self.total_doc
                results.append((proba_in_query.sum() + np.log1p(class_proba),cls))
            predicted_results.append(results)
        return predicted_results
```

In this code, a user query is transformed into a boolean vector and multiplied with the diagonal matrix. And finally a log formed class probabilities and a probability of terms given class are added. By doing this process for each processes, the list of probability is calculated and resulted. We can sort the list to get the class which has the highest probability.

### The challenging things for me in this step

First, for calculation Naive Bayes, I choose to use a diagonal matrix. But expected matrix size is over 300k. it is almost impossible load it to the system memory. So I change it into link list implemented sparse matrix. It does not contain zero elements to save memory space.

Second, handing with multiple genre is hard to calculate probabilities. In the game dataset, each game has multiple genres. concatenating all genres of each game makes the process time longer. Moreover, the prediction rate is also bad. In this time, I solve this problem just choose one genre as a primary one and use it as a class in Naive Bayes Classifier processing.

## Development phase 3

In this phase, Recommend process is applied. Unfortunately, the user information is not collected.


##  evaluation

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

## Ongoing tasks
1. Build a new counter vectorizer and tf-idf vectorizer and naive bayes classifier without string comparisons.
2. Build a Recommender module.
3. Make documents for this projects.

## References

1. Scikit-Learn. <em>Sklearn text module</em>.<br><https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_extraction/text.py>
2. Wikipedia. <em>Sparse Matrix</em> <br><https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)>