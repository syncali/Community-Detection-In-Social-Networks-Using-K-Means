# 🧠 Community Detection in Social Networks Using <br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;DeepWalk and K-Means

This project demonstrates how to detect communities in an undirected graph using **DeepWalk embeddings** and **K-Means clustering**.

The pipeline involves:
- Random walk generation on the graph
- Embedding nodes via **Word2Vec** (DeepWalk)
- Clustering the node embeddings using **K-Means**
- Evaluating clusters using the **Silhouette Score**
- Visualizing graph communities

---

## 📂 Project Structure

```
commmunity-detection.py
sample-graph (50).txt
community_detection.ipynb
GT Project Report.pdf
README.md

```

---

## 📦 Requirements

Install dependencies using:

```bash
pip install networkx matplotlib numpy scikit-learn gensim
```

---

## 📈 How It Works

1. **Load Graph**  
   Reads an edge list from a text file.

2. **Random Walk Generation**  
   Simulates sequences of node traversals to mimic graph context (like sentences in NLP).

3. **Train DeepWalk (Word2Vec)**  
   Applies the skip-gram model from `gensim` to learn embeddings.

4. **Cluster Embeddings**  
   Uses `KMeans` to detect communities in embedding space.

5. **Optimize Cluster Count**  
   Picks the best number of clusters using **Silhouette Score**.

6. **Visualize Results**  
   Uses `matplotlib` to show community structure with different colors.

---

## 📌 How to Run

### 🐍 Using Python Script

```bash
python commmunity-detection.py
```

> Make sure to update the file path in the script if your edge list is named differently.

### 📒 Using Jupyter Notebook

Open `community_detection.ipynb` to view and interactively run the entire process with detailed explanations.

---

## 📁 Input Format

Your graph input file (`sample-graph (50).txt`) should contain one edge per line:

```
1 2
2 3
3 4
...
```

---

## 📊 Output

- **Console output** with number of nodes, edges, silhouette scores.
- **Graph plot** showing communities in different colors.

---

## 🧪 Evaluation Metric

We use the **Silhouette Score** to measure the quality of clustering:

- Score ranges from **-1 to 1**
- Higher values indicate better-defined clusters

---

## ✍️ Authors

**Shahzaib Ahmed**  
📧 shahzaib3769@gmail.com  
🌐 [LinkedIn](https://www.linkedin.com/in/shahzaib3769) | [GitHub](https://github.com/Shahzaib3769)

**Ali Wasif**  
🌐 [LinkedIn](https://www.linkedin.com/in/ali-wasif/) | [GitHub](https://github.com/syncali)

---

## 📜 License

This project is open-source and free to use under the MIT License.
