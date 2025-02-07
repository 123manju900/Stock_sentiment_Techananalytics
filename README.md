# Technanalytics---R2
# Sentiment Analysis and Stock Prediction Pipeline

## **1. Data Preparation and Cleaning**
- **Objective**: Prepare the dataset for analysis by removing noise and ensuring consistency.
- **Steps**:
  - Loaded the dataset from Kaggle input path.
  - Ensured all values in the `Sentence` column were strings.
  - Applied a cleaning function to remove:
    - URLs using regex (`re.sub(r"http\S+", "", text)`).
    - Special characters and extra spaces.
    - Converted text to lowercase.
  - Removed stopwords using NLTK's `stopwords.words('english')`.
    
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

- **Output**: A new column `cleaned_text` containing processed text.

---

## **2. Named Entity Recognition (NER)**
- **Objective**: Identify key entities such as company names, financial metrics, and product mentions.
- **Steps**:
- Used SpaCy's `en_core_web_sm` model to extract entities like organizations, dates, monetary values, and percentages.
- Stored extracted entities in the `entities` column.
- **Code**:

```python
import spacy
nlp = spacy.load("en_core_web_sm")
def extract_entities(text):
doc = nlp(text)
return [(ent.text, ent.label_) for ent in doc.ents]
data['entities'] = data['cleaned_text'].apply(extract_entities)
```

- **Output**: A new column `entities` containing lists of identified entities.

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

## **4. Sentiment Analysis**
- **Objective**: Analyze sentiment associated with entities and their context.
- **Steps**:
- Initially used FinBERT but reverted to dataset-provided binary sentiment labels due to low accuracy (~58%).
- Linked sentiments to specific entities using dependency parsing results.
- **Code**:


```python
def map_entity_sentiments(row):
entity_sentiments = []
for entity in row['entities']:
entity_sentiments.append((entity, entity1, row['Sentiment']))
return entity_sentiments
data['entity_sentiments'] = data.apply(map_entity_sentiments, axis=1)
```
- **Output**: A new column `entity_sentiments` linking entities with sentiment scores.

---

## **5. Regex-Based Financial Metric Extraction**
- **Objective**: Extract structured financial metrics such as amounts and percentages from text.
- **Steps**:
- Applied regex patterns to identify monetary values (e.g., "EUR 13.1 mn") and percentages (e.g., "5%").
- **Code**:
```python
def extract_financial_metrics(text):
amounts = re.findall(r"\b\d+(.\d+)?\b", text)
percentages = re.findall(r"\b\d+(.\d+)?%\b", text)
return {"amounts": amounts, "percentages": percentages}
data['regex_extraction'] = data['cleaned_text'].apply(extract_financial_metrics)
```

- **Output**: A new column `regex_extraction` containing extracted financial metrics.

---

## **6. Semantic Graph Construction**
- **Objective**: Build semantic graphs to visualize relationships between entities, sentiments, and financial metrics.
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
- **Objective**: Propagate sentiment across connected nodes in the graph.
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
- **Objective**: Predict stock categories based on sentiment scores and financial metrics.
- **Steps**:
- Defined stock categories using rules based on sentiment and financial metrics (e.g., High Growth Potential).
- **Code**:

```python
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

- **Output**: A new column `stock_category` with predicted categories.

---

## Conclusion
This pipeline effectively transforms raw tweet data into actionable insights by integrating NLP techniques (NER, dependency parsing), semantic graph analysis, and machine learning-based predictions. The results provide a comprehensive framework for analyzing financial sentiment and predicting stock market impacts.

