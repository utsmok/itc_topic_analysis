import marimo

__generated_with = "0.18.0"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import os

    import dspy
    import numpy as np
    import polars as pl
    import torch
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer, util
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    # Configuration
    EMBEDDING_MODEL_ID = "BAAI/bge-m3"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@app.cell
def load_functions():
    def load_data_from_excel(path="andy_data.xlsx"):
        """Parses the Excel file provided by Andy."""
        if not os.path.exists(path):
            # Fallback for demo if file missing
            return (
                pl.DataFrame({"Title": ["Demo Title"]}),
                pl.DataFrame({"Type code": [1], "Label": ["Journal"]}),
                ["maize"],
            )

        df_core = pl.read_excel(path, sheet_name="data")
        df_types = pl.read_excel(path, sheet_name="types")
        keywords = pl.read_excel(path, sheet_name="search_terms")["term"].to_list()
        return df_core, df_types, keywords

    def consolidate_data(df_core, df_types):
        """Joins type labels and prepares text for embedding."""
        df_clean = df_core.join(
            df_types.select(["Type code", "Label"]),
            left_on="Type group",
            right_on="Type code",
            how="left",
        )

        # Safe string concatenation handling nulls
        df_clean = df_clean.with_columns(pl.col("Title").fill_null("Untitled Document"))

        if "Abstract" in df_clean.columns:
            df_clean = df_clean.with_columns(
                (pl.col("Title") + ". " + pl.col("Abstract").fill_null("")).alias(
                    "text_for_embedding"
                )
            )
        else:
            df_clean = df_clean.with_columns(
                pl.col("Title").alias("text_for_embedding")
            )
        return df_clean
    return consolidate_data, load_data_from_excel


@app.cell
def cached_embedding_state(consolidate_data, load_data_from_excel):
    @mo.persistent_cache
    def get_initial_state():
        print(f"Loading Model: {EMBEDDING_MODEL_ID} on {DEVICE}...")
        model = SentenceTransformer(EMBEDDING_MODEL_ID, device=DEVICE)

        print("Loading Data...")
        df_core, df_types, keywords = load_data_from_excel()
        df_full = consolidate_data(df_core, df_types)

        texts = df_full["text_for_embedding"].fill_null("").to_list()
        print(f"Embedding {len(texts)} items (this may take a moment)...")

        # GPU Encoding
        # Note: BGE-M3 handles batching, but we ensure it fits in VRAM
        if DEVICE == "cuda":
            with torch.cuda.amp.autocast():
                embeddings = model.encode(
                    texts,
                    batch_size=4,
                    show_progress_bar=True,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
        else:
            embeddings = model.encode(
                texts,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

        return df_full, embeddings, model, keywords

    with mo.status.spinner(subtitle="Loading Data & AI Models..."):
        df_full, full_embeddings, model, legacy_keywords = get_initial_state()
    return df_full, full_embeddings, legacy_keywords, model


@app.cell
def sidebar_ui(df_full):
    # --- UI COMPONENTS ---

    mo.md("# 🔬 ITC Research Explorer")

    # 1. Concept Input
    default_concept = (
        "Food security, agriculture, crop yield, and sustainable farming systems"
    )
    text_concept = mo.ui.text_area(
        value=default_concept,
        label="🔍 Target Research Concept",
        rows=2,
        full_width=True,
    )

    # 2. Threshold Slider
    slider_threshold = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=0.40,
        label="Semantic Relevance Threshold",
        show_value=True,
    )

    # 3. Clustering Params (Hidden in Accordion)
    slider_cluster_size = mo.ui.slider(
        start=5, stop=100, step=5, value=30, label="Min Cluster Size"
    )
    slider_neighbors = mo.ui.slider(start=2, stop=50, value=15, label="UMAP Neighbors")

    settings = mo.accordion(
        {
            "⚙️ Advanced Clustering Settings": mo.vstack(
                [slider_cluster_size, slider_neighbors]
            )
        }
    )

    # 4. Legacy Keyword Toggle
    check_legacy = mo.ui.checkbox(label="Compare with Andy's Keyword List?")

    # Sidebar Layout
    sidebar = mo.sidebar(
        mo.vstack(
            [
                mo.md("## Filter Settings"),
                text_concept,
                slider_threshold,
                mo.md("---"),
                settings,
                mo.md("---"),
                check_legacy,
                mo.md(f"**Total Database:** {len(df_full)} items"),
            ]
        )
    )
    sidebar
    return (
        check_legacy,
        slider_cluster_size,
        slider_neighbors,
        slider_threshold,
        text_concept,
    )


@app.cell
def semantic_filtering(
    df_full,
    full_embeddings,
    legacy_keywords,
    model,
    slider_threshold,
    text_concept,
):
    # This cell REACTS to the text_concept and slider_threshold

    # 1. Encode the Query
    query_text = text_concept.value
    query_emb = model.encode(query_text, normalize_embeddings=True)

    # 2. Compute Scores (Fast, using pre-calculated embeddings)
    scores = util.cos_sim(query_emb, full_embeddings)[0].cpu().numpy()

    # 3. Create Scored DataFrame
    df_scored = df_full.with_columns(pl.Series("semantic_score", scores))

    # 4. Check Legacy Keywords (for comparison)
    pattern = "|".join([str(k) for k in legacy_keywords if str(k).strip()])
    df_scored = df_scored.with_columns(
        pl.col("Title").str.to_lowercase().str.contains(pattern).alias("keyword_match")
    )

    # 5. Apply Filter
    mask = df_scored["semantic_score"] >= slider_threshold.value
    df_filtered = df_scored.filter(mask)

    # 6. Align Embeddings
    filtered_embeddings = full_embeddings[mask.to_numpy()]
    return df_filtered, df_scored, filtered_embeddings


@app.cell
def comparison_stats(check_legacy, df_scored):
    # Only calculate if the checkbox is checked to save visual noise
    if not check_legacy.value:
        legacy_view = mo.md("")
    else:
        # Find "Missed Opportunities"
        missed = (
            df_scored.filter(
                (pl.col("semantic_score") > 0.45) & (pl.col("keyword_match") == False)
            )
            .sort("semantic_score", descending=True)
            .head(5)
        )

        missed_table = mo.ui.table(
            missed.select(["Title", "semantic_score", "Type group"]),
            label="High Relevance but No Keyword Match",
        )

        legacy_view = mo.vstack(
            [
                mo.md("### 🆚 Methodology Bake-Off"),
                mo.md("These items were **found by AI** but **missed by keywords**:"),
                missed_table,
            ]
        )
    return (legacy_view,)


@app.cell
def clustering_process(
    df_filtered,
    filtered_embeddings,
    model,
    slider_cluster_size,
    slider_neighbors,
):
    # This cell runs BERTopic. It runs when df_filtered changes OR clustering sliders change.

    if len(df_filtered) < 10:
        topic_model = None
        topics = None
        fig = None
    else:
        # Initialize BERTopic
        topic_model = BERTopic(
            embedding_model=model,  # Reuse loaded model
            umap_model=UMAP(
                n_neighbors=slider_neighbors.value,
                n_components=5,
                min_dist=0.0,
                metric="cosine",
                random_state=42,
            ),
            hdbscan_model=HDBSCAN(
                min_cluster_size=slider_cluster_size.value,
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=True,
            ),
            vectorizer_model=CountVectorizer(stop_words="english"),
            verbose=False,
        )

        # Fit
        topics, probs = topic_model.fit_transform(
            df_filtered["Title"].to_list(), filtered_embeddings
        )

        # Visualize
        fig = topic_model.visualize_documents(
            df_filtered["Title"].to_list(),
            embeddings=filtered_embeddings,
            custom_labels=True,
            title=f"<b>Expertise Map</b> ({len(df_filtered)} papers)",
        )
    return fig, topic_model


@app.cell
def main_display(df_filtered, fig, legacy_view, slider_threshold, topic_model):
    # --- OUTPUT LAYOUT ---

    # 1. Top Metrics
    metric_count = mo.stat(
        value=len(df_filtered),
        label="Relevant Papers",
        caption=f"Score > {slider_threshold.value}",
    )

    if topic_model:
        n_clusters = len(topic_model.get_topic_info()) - 1  # exclude noise
        metric_clusters = mo.stat(value=n_clusters, label="Identified Clusters")
    else:
        metric_clusters = mo.stat(value=0, label="Clusters")

    metrics = mo.hstack([metric_count, metric_clusters], gap="lg")

    # 2. Main Visualization (Tabs)
    if topic_model:
        # Create Topic Table
        topic_info = topic_model.get_topic_info()
        topic_table = mo.ui.table(
            topic_info, selection=None, label="Cluster Details"
        )

        # Create Data Table (Interactive)
        # We select useful columns
        data_table = mo.ui.table(
            df_filtered.select(["Title", "semantic_score", "Type group", "Label"]),
            selection="multi",
            pagination=True,
            page_size=10,
            label="Filtered Dataset",
        )

        tabs = mo.ui.tabs(
            {
                "🗺️ Cluster Map": mo.ui.plotly(fig),
                "📋 Cluster List": topic_table,
                "📄 Papers Data": data_table,
            }
        )
    else:
        tabs = mo.md("⚠️ **Not enough data.** Lower the threshold to see clusters.")

    # 3. Assemble Main View
    main_view = mo.vstack([metrics, mo.md("---"), tabs, mo.md("---"), legacy_view])
    return (main_view,)


@app.cell
def display_root(main_view):
    # This renders the app
    # Sidebar is implicitly rendered by being defined in mo.sidebar() earlier,
    # but we return main_view here to show the content.
    main_view
    return


if __name__ == "__main__":
    app.run()
