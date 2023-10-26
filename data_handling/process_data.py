from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
from tqdm import tqdm

# Use GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
print(f'Using {torch.cuda.device_count()} GPUs.')

# Load pre-trained model tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.to(device)  # Move model to GPU if available
model.eval()  # Set model to evaluation mode

# Load the filtered recipes data
recipes_df = pd.read_parquet('all_text.parquet', engine='pyarrow')

# Tokenize the recipes
def get_bert_embeddings(text):
    if text is None:
        return None

    if isinstance(text, list) and len(text) == 1:
        text = text[0]

    if not isinstance(text, str):
        return None

    # Tokenize the text and handle texts longer than 512 tokens
    tokens = tokenizer.tokenize(text)
    chunks = [tokens[i:i + 512] for i in range(0, len(tokens), 512)]  # Divide tokens into chunks of 512

    embeddings_list = []
    for chunk in chunks:
        inputs = tokenizer.convert_tokens_to_ids(chunk)
        inputs = torch.tensor([inputs]).to(device)  # Move to GPU if available

        with torch.no_grad():
            outputs = model(inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings_list.append(embeddings)

    # Average the embeddings of all chunks to get a single embedding for the entire text
    avg_embedding = torch.mean(torch.stack(embeddings_list), dim=0)
    return avg_embedding.cpu().numpy()

# Create a tqdm progress bar
progress_bar = tqdm(total=len(recipes_df['all_text']), position=0, leave=True)

# Apply the function to get BERT embeddings for each recipe
def apply_with_progress_update(text):
    progress_bar.update(1)
    return get_bert_embeddings(text)
recipes_df['title'] = recipes_df['all_text'].apply(lambda x: (x.split('Description:')[0]).split('Name: ')[1])
recipes_df['BERT_Embeddings'] = recipes_df['title'].apply(apply_with_progress_update)

# Print the result
print(recipes_df['BERT_Embeddings'].head())

# Close the progress bar
progress_bar.close()

# Save the result
print("Saving embeddings dataframe to 'all_text_with_bert_embeddings.pkl'")
recipes_df.to_pickle('all_text_with_bert_embeddings.pkl')