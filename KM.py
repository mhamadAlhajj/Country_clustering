from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

app = Flask(__name__)

# ====== Helper Functions ======

def reduce_outliers(data):
    columns = data.select_dtypes(include="number").columns
    for col in columns:
        zero_count = (data[col] <= 0).sum()
        if zero_count > 0:
            data[col] = np.where(data[col] > 0, np.log1p(data[col]), data[col])
        else:
            data[col] = np.log1p(data[col])
    return data.fillna(0)

def preprocess_data(data):
    data = data.drop(columns=["country"], errors="ignore")
    minMaxScaler = MinMaxScaler()
    data_scaled = minMaxScaler.fit_transform(data)
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)
    return data_pca

def perform_clustering():
    url = "https://raw.githubusercontent.com/mhamadAlhajj/Country_clustering/main/Country-data.csv"
    k = 5

    df = pd.read_csv(url)
    df_clean = reduce_outliers(df.copy())
    data_pca = preprocess_data(df_clean)

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_pca)

    labels = kmeans.labels_
    df['cluster'] = labels
    return df[['country', 'cluster']].to_dict(orient='records'), k

# ========== API Route ==========

@app.route('/', methods=['GET'])
def home():
    results, num_clusters = perform_clustering()
    return render_template('results.html', results=results, num_clusters=num_clusters)

@app.route('/cluster', methods=['POST'])
def cluster_data():
    results, num_clusters = perform_clustering()
    return render_template('results.html', results=results, num_clusters=num_clusters)

# ========== Run App ==========
if __name__ == '__main__':
    app.run(debug=True)
