import streamlit as st
import time
from query_pinecone import query  # Import the query function
import openai
from prompts import PROMPT, PROMPT_RAG
import os

OPENROUTER_KEY = os.environ.get('OPENROUTER_KEY')

model_name = "mistralai/mistral-7b-instruct"
openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = OPENROUTER_KEY

def get_response_from_openrouter(content_to_pass):
    response = openai.ChatCompletion.create(
      model=model_name, # Optional (user controls the default)
      messages=[{"role": "user", "content": content_to_pass}],
      headers={
        "HTTP-Referer": 'http://localhost:3000', # To identify your app. Can be set to e.g. http://localhost:3000 for testing
    #    "X-Title": $YOUR_APP_NAME, # Optional. Shows on openrouter.ai
      },
    )
    return response


st.title(f"Query Pinecone and Huggingface LLM Using {model_name}")
user_input = st.text_area("Enter the name of a dish you want a recipe for:")

if st.button("Query"):
    if user_input:
        # Create columns for side-by-side display
        col1, col2 = st.columns(2)

        # Query Pinecone
        pinecone_results = query(user_input)
        recipes = pinecone_results['matches'][0]['metadata']['all_text']

        recipe_list = [pinecone_results['matches'][i]['metadata']['all_text'] for i in range(5)]
        recipes = "\n".join(recipe_list)

        # Query RAG LLM with a delay to avoid API rate limit
        with col1:
            st.subheader("Querying RAG LLM...")
            time.sleep(.25)  # Delay to avoid hitting the API rate limit
            print("Printing what's given to openrouter.ai for RAG:")
            print(PROMPT_RAG.format(recipe_name=user_input, recipes=recipes))
            rag_results = get_response_from_openrouter(PROMPT_RAG.format(recipe_name=user_input, recipes=recipes))
            st.write(rag_results.choices[0].message["content"])

            if st.button("Show Pinecone Results"):
                st.subheader("Pinecone Results")
                recipe_results = "\n".join(
                    f"{match['metadata']['all_text']}" for match in pinecone_results['matches']
                )
                st.write(recipe_results)
                st.write("---" * 10)

        # Query Local LLM with a delay to avoid API rate limit
        with col2:
            st.subheader("Vanilla LLM Results")
            time.sleep(.25)  # Delay to avoid hitting the API rate limit
            llm_response = get_response_from_openrouter(PROMPT.format(recipe_name=user_input))
            st.write(llm_response.choices[0].message["content"])
    else:
        st.warning("Please enter a query.")
