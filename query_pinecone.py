import pinecone
import torch
from transformers import BertTokenizer, BertModel
import os

PINECONE_KEY = os.environ.get('PINECONE_KEY')
# Initialize Pinecone
pinecone.init(api_key=PINECONE_KEY, environment='gcp-starter')
# Get index recipe-index
index = pinecone.Index("recipe-index")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.to(device)  # Move model to GPU if available
model.eval()  # Set model to evaluation mode

def get_embeddings(text):
    # Tokenize input text
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(device)
    # Get the embeddings
    with torch.no_grad():
        outputs = model(input_ids)
        embeddings = outputs[0][0, 1:-1, :]
    # Get the average embedding
    return embeddings.mean(dim=0).cpu().numpy().tolist()

def query(query_text, k=5):
    results = index.query(top_k=k, include_values=True, include_metadata=True, vector=get_embeddings(query_text))
    return results