import pandas as pd
# USED TO IMPORT DATA AND HANDLE DATAFRAMES
# CONVERTS THE EXCEL TO ROWS AND COLUMNS SO THE PYTHON CAN UNDERSTAND IT
#  



#  Recommendation Logic: Muhammad Saad Zia 
DATA_REQUIRED_COLUMNS = ["user_id", "product_name", "rating"]


#  Data Validation: IT VALIDES THE COLUMNS IN DATAFRAME EITHER THEY ARE PRESENT OR NOT??
def validate_columns(df: pd.DataFrame, required_cols: list[str], csv_name: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_name} missing required columns: {missing}")


#  Load Recommender Data: HELPS IN TYPECASTING AND CLEANING THE DATA AND DROPS THE RATINGS WHICH ARE NOT NUMERIC LIKE FIVE FOR EXAMPLE 
def load_recommender_data(csv_path: str = "data.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    validate_columns(df, DATA_REQUIRED_COLUMNS, csv_path)
    df["user_id"] = df["user_id"].astype(str)
    df["product_name"] = df["product_name"].astype(str)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    return df


# Product Statistics:GROUP BY PRODUCT NAME AND GET AVERAGE RATING AND REVIEW COUNT USED FOR A SPECIFIC PRODUCT
def build_product_stats(df_data: pd.DataFrame) -> pd.DataFrame:
    return (
        df_data.groupby("product_name")
        .agg(avg_rating=("rating", "mean"), review_count=("rating", "count"))
        .reset_index()
    )


#  Item-Item Similarity (Pearson): HELPS IN BUILDING THE SIMILARITY MATRIX BASED ON PEARSON CORRELATION 
def build_item_similarity(df_data: pd.DataFrame) -> pd.DataFrame:
    rating_matrix = (
        df_data.pivot_table(index="user_id", columns="product_name", values="rating")
        .fillna(0)
    )
    return rating_matrix.corr()


# Top Similar Items:suggests TOP K SIMILAR ITEMS BASED ON THE SIMILARITY MATRIX for alternative recommendations
def get_top_similar_items(item_similarity: pd.DataFrame, product_name: str, top_k: int = 3):
    if product_name not in item_similarity.columns:
        raise KeyError(f"Product '{product_name}' not found in similarity matrix.")
    similar = item_similarity[product_name].sort_values(ascending=False)
    similar = similar.drop(labels=[product_name], errors="ignore")
    top = similar.head(top_k)
    return list(top.items())


# 
def recommend_same_product_by_rating(df_data: pd.DataFrame, product_name: str) -> pd.DataFrame:
    same_products = df_data[df_data["product_name"] == product_name].copy()
    if same_products.empty:
        raise ValueError(f"Product '{product_name}' not found in data.")
    product_ratings = (
        same_products.groupby(["product_id", "product_name"])
        .agg(avg_rating=("rating", "mean"), review_count=("rating", "count"))
        .reset_index()
    )
    product_ratings = product_ratings.sort_values("avg_rating", ascending=False)
    return product_ratings


# ================== Comprehensive Recommendations: Muhammad Saad Zia ==================
def get_comprehensive_recommendations(df_data: pd.DataFrame, item_similarity: pd.DataFrame, 
                                     product_name: str, top_k_alternatives: int = 3):
    same_products = recommend_same_product_by_rating(df_data, product_name)
    try:
        alternatives = get_top_similar_items(item_similarity, product_name, top_k=top_k_alternatives)
    except KeyError:
        alternatives = []
    return same_products, alternatives