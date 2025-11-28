from typing import Tuple

import dspy
import numpy as np
import polars as pl
import torch
from bertopic import BERTopic
from hdbscan import HDBSCAN

# NLP & ML Libraries
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# BAAI/bge-m3 is a state-of-the-art model for retrieval and clustering.
# It supports Multi-Linguality, Dense Retrieval, and Long Context (up to 8192 tokens).
EMBEDDING_MODEL_ID = "BAAI/bge-m3"

# ==============================================================================
# 1. DATA LOADING
# ==============================================================================


def load_data() -> Tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """
    Parses the Excel file provided by Andy.
    """

    df_core = pl.read_excel("andy_data.xlsx", sheet_name="data")
    df_types = pl.read_excel("andy_data.xlsx", sheet_name="types")
    keywords = pl.read_excel("andy_data.xlsx", sheet_name="search_terms")[
        "term"
    ].to_list()

    return df_core, df_types, keywords


def consolidate_data(df_core: pl.DataFrame, df_types: pl.DataFrame) -> pl.DataFrame:
    """
    Joins type labels and prepares text for embedding.
    Handles missing Titles/Abstracts to prevent tokenizer crashes.
    """
    # Join types (Mapping 1 -> Journal, 7 -> MSc Thesis)
    df_clean = df_core.join(
        df_types.select(["Type code", "Label"]),
        left_on="Type group",
        right_on="Type code",
        how="left",
    )
    df_clean = df_clean.with_columns(pl.col("Title").fill_null("Untitled Document"))

    # 2. Create combined text field safely
    if "Abstract" in df_clean.columns:
        df_clean = df_clean.with_columns(
            (pl.col("Title") + ". " + pl.col("Abstract").fill_null("")).alias(
                "text_for_embedding"
            )
        )
    else:
        df_clean = df_clean.with_columns(pl.col("Title").alias("text_for_embedding"))
    return df_clean


# ==============================================================================
# 2. EMBEDDING & FILTERING (STRICT ALIGNMENT)
# ==============================================================================


def generate_embeddings_and_filter(
    df: pl.DataFrame, legacy_keywords: list[str]
) -> Tuple[pl.DataFrame, np.ndarray, SentenceTransformer]:
    """
    1. Generates BGE-M3 embeddings for the FULL dataset.
    2. Runs the 'Bake-off' comparison (Keyword vs Semantic).
    3. Returns the FILTERED dataframe and FILTERED embeddings perfectly aligned.
    """
    print("\n" + "=" * 60)
    print(f"PHASE 1: EMBEDDING WITH {EMBEDDING_MODEL_ID}")
    print("=" * 60)


def generate_embeddings_and_filter(
    df: pl.DataFrame, legacy_keywords: list[str]
) -> Tuple[pl.DataFrame, np.ndarray, SentenceTransformer]:
    """
    1. Generates BGE-M3 embeddings for the FULL dataset.
    2. Runs the 'Bake-off' comparison (Keyword vs Semantic).
    3. Returns the FILTERED dataframe and FILTERED embeddings perfectly aligned.
    """
    print("\n" + "=" * 60)
    print(f"PHASE 1: EMBEDDING WITH {EMBEDDING_MODEL_ID}")
    print("=" * 60)

    # 1. Load Model with GPU Support
    print(f"Loading {EMBEDDING_MODEL_ID} on GPU...")

    # device='cuda' forces it to GPU.
    # trust_remote_code=True is usually safe for BAAI models,
    # but strictly speaking BGE-M3 works with standard SentenceTransformer now.
    model = SentenceTransformer(EMBEDDING_MODEL_ID, device="cuda")

    # OPTIONAL: Compile model for speed (works on PyTorch 2.0+ / Windows)
    # model = torch.compile(model)

    # 2. Embed EVERYTHING
    texts = df["text_for_embedding"].to_list()
    print(f"Generating embeddings for {len(texts)} items...")

    # BATCH SIZE IS CRITICAL ON 8GB VRAM
    # BGE-M3 is large. If you get "CUDA Out of Memory", lower batch_size to 2 or 1.
    # Using float16 (half precision) keeps quality high but uses 50% less VRAM.
    with torch.autocast("cuda"):
        full_embeddings = model.encode(
            texts,
            batch_size=4,  # Keep small for 8GB VRAM
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,  # Ensure we get numpy back for Polars/BERTopic
        )

    # 3. Semantic Search Logic
    target_concept = (
        "Food security, agriculture, crop yield, and sustainable farming systems"
    )

    # Encode query on GPU
    query_embedding = model.encode(
        target_concept, normalize_embeddings=True, device="cuda"
    )

    # Calculate Similarity Scores
    # We used numpy=True above, so we can use dot product or util.cos_sim
    # util.cos_sim handles numpy arrays or tensors fine.
    scores = util.cos_sim(query_embedding, full_embeddings)[0].cpu().numpy()
    # Attach scores to DF
    df_scored = df.with_columns(pl.Series("semantic_score", scores))

    # --- BAKE OFF DISPLAY ---
    # Show Andy the difference before we filter
    pattern = "|".join([k for k in legacy_keywords if isinstance(k, str) and k.strip()])
    df_legacy_check = df_scored.with_columns(
        pl.col("Title").str.to_lowercase().str.contains(pattern).alias("keyword_match")
    )

    print(f"\nTarget Concept: '{target_concept}'")
    print("-" * 30)

    # Show items that have high semantic relevance but NO keyword match
    missed_by_keywords = (
        df_legacy_check.filter(
            (pl.col("semantic_score") > 0.45) & (pl.col("keyword_match") == False)
        )
        .sort("semantic_score", descending=True)
        .head(5)
    )

    if len(missed_by_keywords) > 0:
        print("👀 Items found by AI but missed by Keywords:")
        for row in missed_by_keywords.iter_rows(named=True):
            print(f"[{row['semantic_score']:.2f}] {row['Title'][:80]}...")
    else:
        print("No obvious keyword misses in this subset.")

    # 4. STRICT FILTERING & ALIGNMENT
    # This is the crucial part to fix the "safety/simplicity" comment.
    # We define a boolean mask based on the score.
    threshold = 0.40
    mask = df_scored["semantic_score"] > threshold

    # Filter DataFrame
    df_filtered = df_scored.filter(mask)

    # Filter Embeddings using the SAME boolean mask
    # We convert Polars boolean Series to a Numpy boolean array
    numpy_mask = mask.to_numpy()
    filtered_embeddings = full_embeddings[numpy_mask]

    print(
        f"\nFiltered {len(df)} items down to {len(df_filtered)} relevant items using threshold {threshold}."
    )

    return df_filtered, filtered_embeddings, model


# ==============================================================================
# 3. CLUSTERING & LABELING (BERTOPIC + DSPY)
# ==============================================================================


class ExpertiseLabel(dspy.Signature):
    """
    Generates a high-level expertise label and strategic summary.
    """

    keywords: str = dspy.InputField(desc="Keywords from the cluster")
    titles: str = dspy.InputField(desc="Representative paper titles")

    label: str = dspy.OutputField(desc="Short, professional label (max 5 words)")
    summary: str = dspy.OutputField(desc="One sentence summary of the expertise.")
    strategic_fit: str = dspy.OutputField(
        desc="Relevance to Digital Twins or Climate Finance."
    )


def run_clustering_pipeline(
    df: pl.DataFrame, embeddings: np.ndarray, embedding_model: SentenceTransformer
):
    print("\n" + "=" * 60)
    print("PHASE 2: CLUSTERING & AUTO-LABELING")
    print("=" * 60)

    # Docs must match the embeddings. Since we passed filtered_embeddings
    # and df_filtered together, they are aligned.
    docs = df["Title"].to_list()

    # 1. BERTopic Setup
    # BGE-M3 embeddings are high quality, so we can use standard UMAP/HDBSCAN settings.
    # min_cluster_size set to 30 for broader topics as Andy requested.
    topic_model = BERTopic(
        embedding_model=embedding_model,  # Reuse the loaded BGE-M3 model
        umap_model=UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        ),
        hdbscan_model=HDBSCAN(
            min_cluster_size=30,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        ),
        vectorizer_model=CountVectorizer(stop_words="english"),
        verbose=True,
    )

    print("Fitting BERTopic with BGE-M3 embeddings...")
    # CRITICAL: We pass the pre-calculated embeddings here.
    # This prevents BERTopic from re-calculating them (saving massive time)
    # and ensures alignment.
    topics, probs = topic_model.fit_transform(docs, embeddings)
    print(f"Identified {len(set(topics)) - (1 if -1 in topics else 0)} topics.")
    # 2. DSPy Labeling
    try:
        # Configuration as requested
        lm = dspy.LM(
            "gemini/gemini-2.5-flash",
            api_key="AIzaSyD6GJlKQJT3yo3Xsx3mLbwodDaFn3P3uM8",  # NOTE: This key looks invalid/example-only.
        )
        dspy.settings.configure(lm=lm)
        has_api = True
    except Exception as e:
        print(f"⚠️ DSPy Config Error: {e}. Using mock labels.")
        has_api = False

    topic_info = topic_model.get_topic_info()
    label_generator = dspy.Predict(ExpertiseLabel)

    print("\n--- Generated Clusters & Labels ---")

    new_labels = {}

    for index, row in topic_info.iterrows():
        print(f"row {index} of {len(topic_info)}")
        t_id = row["Topic"]
        if t_id == -1:
            continue

        # Context for LLM
        keywords = ", ".join(row["Representation"][:5])
        rep_docs = topic_model.get_representative_docs(t_id)
        # Use first 3 representative docs, truncate to 150 chars to save tokens
        titles_str = " | ".join([t[:150] for t in rep_docs[:3]])

        if has_api:
            try:
                pred = label_generator(keywords=keywords, titles=titles_str)
                label = pred.label
                desc = pred.summary
            except:
                # retry at least once
                try:
                    pred = label_generator(keywords=keywords, titles=titles_str)
                    label = pred.label
                    desc = pred.summary
                except:
                    label = f"Cluster {t_id}"
                    desc = "API Error during generation"
        else:
            # Fallback
            label = f"{keywords.split(',')[0].capitalize()} & {keywords.split(',')[1].capitalize()}"
            desc = "Automated summary placeholder."

        print(f"\nTopic {t_id}: {label}")
        print(f"   Keywords: {keywords}")
        print(f"   Summary: {desc}")

        new_labels[t_id] = label

    topic_model.set_topic_labels(new_labels)

    # 3. Visualization
    # Using custom_labels=True to show the DSPy/GenAI labels on the plot
    fig = topic_model.visualize_documents(
        docs,
        embeddings=embeddings,
        custom_labels=True,
        title="ITC Expertise Map (BGE-M3 Embeddings)",
    )
    fig.write_html("itc_expertise_map.html")
    print("\n✅ Interactive map saved to 'itc_expertise_map.html'")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    # 1. Load & Clean
    df_core, df_types, keywords = load_data()
    df_clean = consolidate_data(df_core, df_types)
    print(f"Loaded {len(df_clean)} raw items.")

    # 2. Embed, Score & Filter (The "Bake-Off")
    # This returns the filtered DF and the aligned embeddings
    df_filtered, filtered_embeddings, model = generate_embeddings_and_filter(
        df_clean, keywords
    )

    # 3. Run Clustering
    if len(df_filtered) > 5:
        run_clustering_pipeline(df_filtered, filtered_embeddings, model)
    else:
        print("Not enough data to cluster (check your filter threshold).")
