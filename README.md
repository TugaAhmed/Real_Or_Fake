# 📰 Real Or Fake?! 🕵️‍♂️
## GNN-based Fake News Detection Challenge

Welcome to the **GNN-based Fake News Detection Challenge**! This competition focuses on detecting fake news propagation on Twitter using Graph Neural Networks (GNNs). 


**[Live Leaderboard](https://tugaahmed.github.io/Real_Or_Fake/leaderboard.html)**

Participants are asked to improve the baseline GNN model by including **user profile features** in addition to existing text embeddings.

---



## Repository Structure

```text
Real_Or_Fake/
├── data/
│   ├── public/
│   │   ├── A.txt
│   │   ├── new_bert_feature.npz
│   │   ├── new_spacy_feature.npz
│   │   ├── new_profile_feature.npz
│   │   ├── node_graph_id.npy
│   │   ├── train_idx.npy
│   │   ├── train_labels.csv
│   │   ├── val_idx.npy
│   │   ├── val_labels.csv
│   │   └── test_idx.npy
│   └── private/   #hidden
│       └── test_labels.csv  
├── submissions/
│   └── sample_submission/
│       └── predictions.csv
├── models/
│   ├── model.py
│   └── saved_model.model
├── dataloader.py
├── evaluate.py
├── test.py
├── train.py
├── requirements.txt
├── README.md
└── LICENSE
```


---

## 🗂 Dataset

We use the **GossipCop** and **Politifact** datasets, which contain Twitter news propagation graphs. Each graph represents a news article as the **root node**, and the users who retweeted the news as **child nodes**.
You can download the data from this link : https://drive.google.com/drive/folders/1OslTX91kLEYIi2WBnwuFtXsVz5SS_XeR

### Node Features

- **Text embeddings** of the news (root node) and historical tweets of users  
  - Pretrained **spaCy word2vec** (300-dim) or **BERT embeddings** (768-dim)
- **Task extension:** include **user profile features** (10-dim):
  - Account age
  - Verified status
  - Number of followers/friends
  - Number of tweets
  - Geolocation enabled
  - Description length
  - etc.

### Graph Labels

- `0`: Real news  
- `1`: Fake news  

---

## 📝 Problem Statement

**Task:** Classify each news propagation graph as real or fake.

- **Baseline:** Uses only text embeddings of the news and historical tweets of users.
- **Challenge:** Improve the baseline by including **user profile features** in the final node embeddings.

---

## ⚡ Baseline Model

The baseline GNN is implemented in `model.py`. It supports:

- Graph Convolutional Network (GCN)
- Graph Attention Network (GAT)
- GraphSAGE

**Features used in baseline:**  
- Text embeddings of news and historical user tweets  
**Features to add for challenge:**  
- User profile features (10-dimensional)

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/TugaAhmed/Open-GNN-Mini-Competition-.git
cd gnn-challenge
```
### 2. Install dependencies
``` bash
pip install -r requirements.txt
```
### 3. Download dataset from the link above and place the unzpied data inside `data` folder
### 4. After creating your model, run `test.py` that will generate `sumbission.csv` 

### Submission Workflow
* Fork the repo and add your submission CSV under submissions/
* Create a pull request
* GitHub Actions will automatically run scoring_script.py to evaluate your submission and update the leaderboard
