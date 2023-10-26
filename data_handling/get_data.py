import os
import zipfile
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import pyarrow.parquet as pq

# Step 1: Download the dataset from Kaggle if not already downloaded
dataset_name = "irkaal/foodcom-recipes-and-reviews"
zip_file_name = dataset_name.split('/')[-1] + '.zip'

if not os.path.exists(zip_file_name):
    # Make sure to place your kaggle.json file in the location "~/.kaggle/kaggle.json"
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_name, path='../', unzip=False)

# Step 2: Unzip the downloaded file if not already unzipped
if not os.path.exists('recipes.parquet') or not os.path.exists('reviews.parquet'):
    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        zip_ref.extractall()

# Step 3: Read the Parquet files
recipes_df = pd.read_parquet('recipes.parquet', engine='pyarrow')
reviews_df = pd.read_parquet('reviews.parquet', engine='pyarrow')

# Step 4: Calculate the average rating for each recipe
average_ratings = reviews_df.groupby('RecipeId')['Rating'].mean().reset_index()
average_ratings.columns = ['RecipeId', 'AverageRating']

# Step 5: Merge the average ratings with the recipes data
recipes_with_ratings = pd.merge(recipes_df, average_ratings, on='RecipeId', how='left')

# Step 6: Filter the recipes with 3.5+ star average reviews and more than 30 reviews
filtered_recipes = recipes_with_ratings[(recipes_with_ratings['AverageRating'] >= 4) & (recipes_with_ratings['ReviewCount'] >= 5)]
print(filtered_recipes.shape)

# Step 7: Save the filtered recipes to a new Parquet file
filtered_recipes.to_parquet('filtered_recipes_4.parquet')
print("Filtered recipes with 3.5+ star average reviews are saved to 'filtered_recipes_4+.parquet'")
