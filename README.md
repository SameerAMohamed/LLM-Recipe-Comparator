# LLM Comparator
## Overview
This is a Streamlit application that compares two LLMs. One is a regular LLM, and the other is a LLM with context. The context is derived from a database of recipes. The application is hosted on OpenRouter and uses Pinecone to store the embeddings.

## Purpose
The purpose of this application is to provide a side-by-side comparison of two LLM approaches. One uses a regular LLM, and one pre-fixes the query by giving context derived from a database to hopefully provide insight about the query. The application is written in Python using Streamlit and uses models hosted on OpenRouter.

The reason recipes are used is that they are a good example of a domain-specific language. Language models are likely already familiar with basics such as how to make a chocolate cake, but is it also familiar with how to make something more specific like Emily Mariko's Salmon and Rice Bowl?

This is meant to be a proof of concept to show the capabilities that RAG systems have to bring the power of foundational models to more narrow use cases. More practical applications that could be useful is filling the database with company documentation, regulatory information, educational information (such as for a specific class), and more!
## How to run
simply run `streamlit run app.py` in the root directory of the project. This will open a browser window with the application running.

## How to use
The application is split into two sections: the left side is the regular LLM, and the right side is the LLM with context. The user can input a query into the text box and press the "Run" button to run the query. The results will be displayed below the text box. The user can also press the "Clear" button to clear the results.

## Future Work
The following is a list of potential next steps I've been considering taking this project:
- [ ] Add a third column that shows the difference between the 2 results and a fine tuned LLM (would like a hosting service like OpenRouter for self-uploaded models)
- [ ] Allow the user to select which model to use
- [ ] Allow the user to select which database to use
- [ ] Allow the user to select the k for the RAG system
- [ ] Try out different databases such as ChromaDB, Elasticsearch, Postgres, etc.

## Files and Directories
- `app.py`: The main file that runs the application using Streamlit
- `environment.yml`: The environment file used to create the conda environment
- `README.md`: This file
- `prompts.py`: The file that contains the prompts and templates used for the LLMs
- `./data_handling`: The directory that contains the files used to handle the data and upload it to Pinecone
  - `get_data.py`: The file that contains the functions used to get the data from the source
  - `process_data.py`: The file that contains the functions used to process the data and create embeddings
  - `put_on_pinecone.py`: The file that contains the functions used to upload the data to Pinecone
  - `recipe_explorer.ipynb`: notebook used to modify format for storage to Pinecone