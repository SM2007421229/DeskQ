import os
import json
import pickle
import numpy as np
from datetime import datetime
try:
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.preprocessing import normalize
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

class SimpleVectorDB:
    def __init__(self, collection_name="file_index", persistence_dir="vector_store"):
        self.collection_name = collection_name
        self.persistence_dir = os.path.join(os.getcwd(), persistence_dir)
        self.index_path = os.path.join(self.persistence_dir, f"{collection_name}.pkl")
        
        if not os.path.exists(self.persistence_dir):
            os.makedirs(self.persistence_dir)
            
        # In-memory storage
        self.ids = []
        self.documents = []
        self.metadatas = []
        self.embeddings = None # numpy array

        self.vectorizer = HashingVectorizer(
            n_features=1024, # Higher dim for better collision avoidance
            analyzer='char_wb',
            ngram_range=(2, 4),
            norm=None,
            alternate_sign=False
        )
        
        self.load()

    def load(self):
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, 'rb') as f:
                    data = pickle.load(f)
                    self.ids = data.get('ids', [])
                    self.documents = data.get('documents', [])
                    self.metadatas = data.get('metadatas', [])
                    self._rebuild_index()
                print(f"Loaded vector index with {len(self.ids)} entries.")
            except Exception as e:
                print(f"Failed to load vector index: {e}")

    def save(self):
        try:
            data = {
                'ids': self.ids,
                'documents': self.documents,
                'metadatas': self.metadatas
            }
            with open(self.index_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Failed to save vector index: {e}")

    def _rebuild_index(self):
        if not self.documents:
            self.embeddings = None
            return

        if HAS_SKLEARN:
            X = self.vectorizer.transform(self.documents)
            self.embeddings = normalize(X.toarray(), norm='l2')
        else:
            self.embeddings = None

    def add(self, ids, documents, metadatas):
        if not ids:
            return

        # Update existing or add new
        for i, id in enumerate(ids):
            if id in self.ids:
                idx = self.ids.index(id)
                self.documents[idx] = documents[i]
                self.metadatas[idx] = metadatas[i]
            else:
                self.ids.append(id)
                self.documents.append(documents[i])
                self.metadatas.append(metadatas[i])
        
        self._rebuild_index()
        self.save()

    def delete(self, ids):
        if not ids:
            return
            
        for id in ids:
            if id in self.ids:
                idx = self.ids.index(id)
                self.ids.pop(idx)
                self.documents.pop(idx)
                self.metadatas.pop(idx)
        
        self._rebuild_index()
        self.save()

    def get(self, include=None):
        return {
            "ids": self.ids,
            "metadatas": self.metadatas,
            "documents": self.documents
        }
        
    def count(self):
        return len(self.ids)

    def query(self, query_texts, n_results=5):
        if not HAS_SKLEARN or self.embeddings is None or len(self.ids) == 0:
            return {'ids': [[]], 'metadatas': [[]], 'distances': [[]]}

        # Vectorize query
        query_vec = self.vectorizer.transform(query_texts)
        query_vec = normalize(query_vec.toarray(), norm='l2')
        
        # Calculate cosine similarity
        # similarity = query . doc_T (since normalized)
        # We want distance = 1 - similarity
        similarities = cosine_similarity(query_vec, self.embeddings)
        
        results = {'ids': [], 'metadatas': [], 'distances': []}
        
        for i in range(len(query_texts)):
            # Get top N
            sims = similarities[i]
            # argsort returns indices of sorted elements (ascending)
            # we want descending
            sorted_indices = np.argsort(sims)[::-1][:n_results]
            
            row_ids = []
            row_metas = []
            row_dists = []
            
            for idx in sorted_indices:
                if sims[idx] > 0.0: # Only include relevant matches
                    row_ids.append(self.ids[idx])
                    row_metas.append(self.metadatas[idx])
                    row_dists.append(1 - sims[idx]) # Convert sim to dist
            
            results['ids'].append(row_ids)
            results['metadatas'].append(row_metas)
            results['distances'].append(row_dists)
            
        return results
