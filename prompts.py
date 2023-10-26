from langchain.prompts import PromptTemplate

PROMPT = PromptTemplate(
    input_variables=["recipe_name"],
    template=(
        "Question: What are the instructions and ingredients for a recipe of {recipe_name}? "
        "Answer: "
    )
)

PROMPT_RAG = PromptTemplate(
    input_variables=["recipes", "recipe_name"],
    template=(
        "The following are a series of possibly relevant recipes. They may or may not be relevant. {recipes}. "
        "Question: What are the instructions and ingredients for a recipe of {recipe_name}? "
        "Answer: "
    )
)