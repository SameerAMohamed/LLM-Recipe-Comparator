import pinecone
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

PINECONE_KEY = os.environ.get('PINECONE_KEY')
# Initialize Pinecone
pinecone.init(api_key=PINECONE_KEY, environment='gcp-starter')

# Use an existing Pinecone index
index_name = "recipe-index"

# Connect to the existing index
index = pinecone.Index(index_name=index_name)

# Read recipes_df from the pkl file
recipes_df = pd.read_pickle('all_text_with_bert_embeddings.pkl')

# Prepare vectors and metadata for uploading
vectors = recipes_df["BERT_Embeddings"].tolist()
metadata = recipes_df["all_text_full"].tolist()
ids = recipes_df.index.astype(str).tolist()  # Use DataFrame index as ID

# Create a tqdm progress bar
with tqdm(total=len(ids), desc="Uploading", unit="item") as pbar:
    # Batch upsert vectors and metadata to Pinecone
    batch_size = 100  # Adjust the batch size as needed
    for i in range(0, len(ids), batch_size):
        batch_items = [
            {
                'id': id,
                'values': vector.flatten().tolist(),  # Ensure each vector is a flat list of floats
                'metadata': {'all_text': text}
            }
            for id, vector, text in zip(ids[i:i+batch_size], vectors[i:i+batch_size], metadata[i:i+batch_size])
        ]

        index.upsert(vectors=batch_items)
        pbar.update(len(batch_items))  # Update the progress bar

# You don't need to delete the index if you want to use it again later
# pinecone.delete_index(index_name)