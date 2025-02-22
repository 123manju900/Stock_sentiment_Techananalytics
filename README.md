# Sentiment Analysis and Stock Prediction Pipeline - Techanalytics ( IIT Banaras)
This markdown is the documentation for Techanalytics Round 2 case. The dataset contains the tweets text and sentiment as `binary (0 or 1 )`. We were suppose to model the tweets using NLP techniques and predict the model for sentiments 


## **1. Data Preparation and Cleaning**
- **Objective**: Prepare the dataset for analysis by removing noise and ensuring consistency.
- **Steps**:
  - Loaded the dataset from Kaggle input path.
  - Ensured all values in the `Sentence` column were strings.
  - Applied a cleaning function to remove:
    - URLs using regex (`re.sub(r"http\S+", "", text)`).
    - Special characters and extra spaces.
    - Converted text to lowercase.
    - converting `mn` to `millions`
  - Removed stopwords using NLTK's `stopwords.words('english')`.
 
  Below is a basic code outline on how we removed them as a part of data cleaning process 
    
- **Code**:

```python
def clean_text(text):
text = re.sub(r"http\S+", "", text) # Remove URLs
text = re.sub(r"[^a-zA-Z0-9\s]", "", text) # Remove special characters
text = re.sub(r"\s+", " ", text) # Remove extra spaces
return text.lower()
data['cleaned_text'] = data['Sentence'].apply(clean_text)
stop_words = set(stopwords.words('english'))
data['cleaned_text'] = data['cleaned_text'].apply(
lambda x: " ".join([word for word in word_tokenize(x) if word not in stop_words])
)
```

- The **output** is stored in a new column **`cleaned_text`** containing processed text.

---
# **Extract Relevant Information**

As a part of information extraction we have applied techniques like **`NER`** and **`SVO`** and **`regular expressions`** to effectively extract information and use dependency parsing to aggragate sentiments and used sentiment propogation to links sentiments to extracts from tweets

## **2. Named Entity Recognition (NER)**
- **Goal**: Identify key entities such as company names, financial metrics, and product mentions.
**Steps**:
- Used **SpaCy's** `en_core_web_sm` model to extract entities like `organizations, dates, monetary values, and percentages`.
- Stored extracted entities in the `entities` column.
Below is code excerpt from text extraction process

- **Code**:

```python
import spacy
nlp = spacy.load("en_core_web_sm")
def extract_entities(text):
doc = nlp(text)
return [(ent.text, ent.label_) for ent in doc.ents]
data['entities'] = data['cleaned_text'].apply(extract_entities)
```

- The **output** is a  new column `entities` containing lists of identified entities.

---

## **3. Dependency Parsing for SVO Extraction**
- **Objective**: Capture contextual relationships between entities using Subject-Verb-Object (SVO) triples.
- **Steps**:
- Applied SpaCy's dependency parser to extract grammatical relationships within sentences.
- Stored extracted SVO triples in the `svo_relationships` column.
- **Code**:

  
```python
def extract_svo(text):
doc = nlp(text)
svo_triples = []
for token in doc:
if token.dep_ == "ROOT":
subject = [child.text for child in token.children if child.dep_ == "nsubj"]
obj = [child.text for child in token.children if child.dep_ == "dobj"]
if subject and obj:
svo_triples.append((subject, token.text, obj))
return svo_triples
data['svo_relationships'] = data['cleaned_text'].apply(extract_svo)
```

- **Output**: A new column `svo_relationships` containing lists of extracted triples.

---

## **4. Sentiment Analysis- Initial plan**
Initially we tried using **FinBERT** to analyze sentiment associated with entities and their context.
-But we reverted to dataset-provided binary sentiment labels due to low accuracy (~58%).
- Linked sentiments to specific entities using dependency parsing results.


## **5. Regex-Based Financial Metric Extraction**
- **Goal** : From the `SVO` we couldn't extract all the metrics, so along with that we extracted structured financial metrics such as amounts and percentages from text using regex. 
- Applied regex patterns to identify monetary values (e.g., "EUR 13.1 mn") and percentages (e.g., "5%").

- Below is the code to extract financial metrics 

- **Code**:
```python
def extract_financial_metrics(text):
amounts = re.findall(r"\b\d+(.\d+)?\b", text)
percentages = re.findall(r"\b\d+(.\d+)?%\b", text)
return {"amounts": amounts, "percentages": percentages}
data['regex_extraction'] = data['cleaned_text'].apply(extract_financial_metrics)
```

- The **output** is a new column `regex_extraction` containing extracted financial metrics.

---
# **Analyse Sentiment and Semantic Relationships**
## **6. Semantic Graph Construction**
- We built semantic graphs to visualize relationships between entities, sentiments, and financial metrics.
- **Steps**:
- Nodes: Represented entities, financial metrics, and sentiments.
- Edges: Captured relationships based on co-occurrence or SVO triples.
- Enhanced graph connectivity by adding edges based on regex-extracted metrics.
- **Code**:

```python
import networkx as nx
def build_graph(row):
G = nx.Graph()
for entity in row['entities']:
G.add_node(entity, type=entity1)
for svo in row['svo_relationships']:
G.add_edge(svo, svo, relationship=svo1)
return G
data['semantic_graph'] = data.apply(build_graph, axis=1)
```

- **Output**: A new column `semantic_graph` containing semantic graphs.

---

## **7. Sentiment Propagation**
- To propagate sentiment across connected nodes in the graph.
- **Steps**:
- Nodes without sentiment inherited an average sentiment from their neighbors.
- **Code**:

```python
def propagate_sentiment(graph, entity_sentiments):
for node in graph.nodes():
sentiments = [sent for ent, _, sent in entity_sentiments if ent == node]
if not sentiments:
neighbors = list(graph.neighbors(node))
neighbor_sentiments = [
sent for neighbor in neighbors
for ent, _, sent in entity_sentiments if ent == neighbor
]
avg_sentiment = sum(neighbor_sentiments) / len(neighbor_sentiments) if neighbor_sentiments else None
graph.nodes[node]['sentiment'] = avg_sentiment
else:
graph.nodes[node]['sentiment'] = sentiments
return graph
data['propagated_graph'] = data.apply(
lambda row: propagate_sentiment(row['semantic_graph'], row['entity_sentiments']), axis=1
)
```
- **Output**: A new column `propagated_graph` with updated graphs.

---

## **8. Stock Category Prediction**
- Predict stock categories based on sentiment scores and financial metrics.

- Based on sentiment scores and financial metrics, we categorized stocks into:

- **High Growth Potential:** Strong positive sentiment & financials

- **Neutral:** Moderate sentiment or missing financial indicators

- **High Risk:** Negative sentiment & weak financials

- **Code**:

```python
##Example classification code 
def categorize_stock(row):
avg_sentiment = sum([sent for _, _, sent in row['entity_sentiments']]) / len(row['entity_sentiments'])
if avg_sentiment > 0.5 and len(row['regex_extraction']['amounts']) > threshold:
return "High Growth Potential"
elif avg_sentiment > threshold_low:
return "Stable"
else:
return "High Risk"
data['stock_category'] = data.apply(categorize_stock, axis=1)
```

- A new column `stock_category` with predicted categories.

---
## 9. Fueling the Machine: Feature Engineering

To train a predictive model, we needed to convert our data into numerical features. We engineered features from three primary sources:

**Textual Features**: Converted the cleaned text data into numerical vectors using CountVectorizer, capturing word frequencies.

**Financial Metrics:** Extracted and counted the amounts and percentages identified by our regex patterns.

**Graph Properties:** Calculated centrality scores (e.g., degree centrality, betweenness centrality) for each node in the semantic graphs. Centrality measures identify influential nodes within the network. 
![image](https://github.com/123manju900/Stock_sentiment_Techananalytics/blob/05287df00c437ec40897264b8d171ba0a332e060/Graphs%20and%20code/Top_entities.png)

---

# **Predict Stock Category Impacts**
## Using the modeled data for ML prediction 
We trained a **Random Forest Classifier** to predict stock categories using:

- Textual Features (Vectorized text from CountVectorizer)

- Financial Metrics (Counts of extracted amounts/percentages)

- Graph Properties (Node centrality scores)

**Model Training**
  ```python
  from sklearn.ensemble import RandomForestClassifier
  clf = RandomForestClassifier()
  clf.fit(X_train, y_train)
  ```

---

## **Model Metrics** 
Evaluation Metrics: We evaluated the model's performance using accuracy, precision, recall, and F1-score.

And Below are the metrics for the model 

```
   Metric     Score
0  Accuracy  0.917857
1  Precision  0.954193
2  Recall  0.956420
3  F1-Score  0.955305
```
![image](https://github.com/123manju900/Stock_sentiment_Techananalytics/blob/05287df00c437ec40897264b8d171ba0a332e060/Graphs%20and%20code/model_performance.png)
## Conclusion
This pipeline effectively transforms raw tweet data into actionable insights by integrating NLP techniques (NER, dependency parsing), semantic graph analysis, and machine learning-based predictions. The results provide a comprehensive framework for analyzing financial sentiment and predicting stock market impacts.

