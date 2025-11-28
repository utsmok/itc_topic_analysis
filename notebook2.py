import marimo

__generated_with = "0.18.0"
app = marimo.App(width="full")

with app.setup:
    import polars as pl
    import pandas as pd
    import numpy as np
    import altair as alt
    import torch
    from sentence_transformers import SentenceTransformer, util
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    import dspy
    import os
    import marimo as mo
    import pyarrow

    # Configuration
    EMBEDDING_MODEL_ID = "BAAI/bge-m3"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Configure Altair to handle large datasets if needed
    alt.data_transformers.disable_max_rows()


@app.cell
def data_logic():
    def load_data_from_excel(path="andy_data.xlsx"):
        """Parses the Excel file provided by Andy."""
        if not os.path.exists(path):
            return pl.DataFrame({"Title": ["Demo Title"]}), pl.DataFrame({"Type code": [1], "Label": ["Journal"]}), ["maize"]

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
            how="left"
        )

        # Safe string concatenation handling nulls
        df_clean = df_clean.with_columns(
            pl.col("Title").fill_null("Untitled Document")
        )

        if "Abstract" in df_clean.columns:
            df_clean = df_clean.with_columns(
                (pl.col("Title") + ". " + pl.col("Abstract").fill_null("")).alias("text_for_embedding")
            )
        else:
            df_clean = df_clean.with_columns(
                pl.col("Title").alias("text_for_embedding")
            )
        return df_clean
    return consolidate_data, load_data_from_excel


@app.cell
def cached_embedding_state(consolidate_data, load_data_from_excel):
    # Using persistent_cache as requested for reuse across sessions
    @mo.persistent_cache
    def get_initial_state():
        print(f"Loading Model: {EMBEDDING_MODEL_ID} on {DEVICE}...")
        model = SentenceTransformer(EMBEDDING_MODEL_ID, device=DEVICE)

        print("Loading Data...")
        df_core, df_types, keywords = load_data_from_excel()
        df_full = consolidate_data(df_core, df_types)

        texts = df_full["text_for_embedding"].fill_null("").to_list()
        print(f"Embedding {len(texts)} items...")

        if DEVICE == "cuda":
            with torch.cuda.amp.autocast():
                embeddings = model.encode(
                    texts,
                    batch_size=4,
                    show_progress_bar=True,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                )
        else:
            embeddings = model.encode(
                texts,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True
            )

        return df_full, embeddings, keywords

    # We load the model separately so it doesn't need to be pickled in the persistent cache
    # (Pickling models can be finicky)
    model = SentenceTransformer(EMBEDDING_MODEL_ID, device=DEVICE)

    with mo.status.spinner(subtitle="Loading Cached Data..."):
        df_full, full_embeddings, legacy_keywords = get_initial_state()
    return df_full, full_embeddings, legacy_keywords, model


@app.cell
def sidebar_ui(df_full):
    # --- SIDEBAR INPUTS ---
    mo.md("# 🔬 ITC Research Explorer")

    # 1. Concept Input
    default_concept = "Food security, agriculture, crop yield, and sustainable farming systems"
    text_concept = mo.ui.text_area(
        value=default_concept,
        label="🔍 Target Research Concept",
        rows=2,
        full_width=True
    )

    # 2. Threshold Slider
    slider_threshold = mo.ui.slider(
        start=0.30,
        stop=0.80,
        step=0.01,
        value=0.45,
        label="Semantic Relevance Threshold",
        show_value=True
    )

    # 3. Clustering Params (Algorithm Control)
    # Changing these forces a re-cluster (calculation heavy)
    mo.md("### 🧠 Clustering Algorithms")
    slider_cluster_size = mo.ui.slider(start=5, stop=100, step=5, value=20, label="Min Cluster Size")
    slider_neighbors = mo.ui.slider(start=2, stop=50, value=15, label="UMAP Neighbors (Global vs Local)")

    # Sidebar Layout
    sidebar = mo.sidebar(
        mo.vstack([
            mo.md("## 1. Filter Data"),
            text_concept,
            slider_threshold,
            mo.md("---"),
            mo.md("## 2. Cluster Settings"),
            mo.md("_Changes here trigger re-calculation_"),
            slider_cluster_size,
            slider_neighbors,
            mo.md("---"),
            mo.md(f"**Total Database:** {len(df_full)} items")
        ])
    )
    sidebar
    return (
        slider_cluster_size,
        slider_neighbors,
        slider_threshold,
        text_concept,
    )


@app.cell
def filtering_logic(
    df_full,
    full_embeddings,
    legacy_keywords,
    model,
    slider_threshold,
    text_concept,
):
    # This cell calculates the relevant subset of data

    query_text = text_concept.value
    query_emb = model.encode(query_text, normalize_embeddings=True)

    # Compute Scores
    scores = util.cos_sim(query_emb, full_embeddings)[0].cpu().numpy()

    # Create Scored DataFrame
    df_scored = df_full.with_columns(pl.Series("semantic_score", scores))

    # Legacy check
    pattern = "|".join([str(k) for k in legacy_keywords if str(k).strip()])
    df_scored = df_scored.with_columns(
        pl.col("Title").str.to_lowercase().str.contains(pattern).alias("keyword_match")
    )

    # Apply Filter
    mask = df_scored["semantic_score"] >= slider_threshold.value
    df_filtered = df_scored.filter(mask)

    # Align Embeddings
    filtered_embeddings = full_embeddings[mask.to_numpy()]
    return df_filtered, df_scored, filtered_embeddings


@app.cell
def clustering_calculation(
    df_filtered,
    filtered_embeddings,
    model,
    slider_cluster_size,
    slider_neighbors,
):
    if len(df_filtered) < 10:
        coord_df = pl.DataFrame()
        topic_info_df = pl.DataFrame()
    else:
        # Initialize BERTopic with user params
        topic_model = BERTopic(
            embedding_model=model, 
            umap_model=UMAP(
                n_neighbors=slider_neighbors.value, 
                n_components=2, # Force 2D for plotting coordinates
                min_dist=0.0, 
                metric='cosine', 
                random_state=42
            ),
            hdbscan_model=HDBSCAN(
                min_cluster_size=slider_cluster_size.value, 
                metric='euclidean', 
                cluster_selection_method='eom', 
                prediction_data=True
            ),
            vectorizer_model=CountVectorizer(stop_words="english"),
            verbose=False
        )

        # Fit & Transform
        topics, probs = topic_model.fit_transform(
            df_filtered["Title"].to_list(), 
            filtered_embeddings
        )

        # --- Extract Coordinates & Info ---

        # 1. Get Topic Info (BERTopic returns Pandas, so we cast to Polars immediately)
        topic_info_pd = topic_model.get_topic_info()
        topic_info_df = pl.from_pandas(topic_info_pd)

        # 2. Extract UMAP Coordinates
        umap_coords = topic_model.umap_model.embedding_

        # 3. Construct Polars DataFrame
        # We start with the filtered data
        coord_df = df_filtered.select(["Title", "Label", "semantic_score"])

        # Add the computed columns
        coord_df = coord_df.with_columns([
            pl.Series("x", umap_coords[:, 0]),
            pl.Series("y", umap_coords[:, 1]),
            pl.Series("topic_id", topics)
        ])

        # 4. Map Topic Labels using a Join (Idiomatic Polars)
        # We create a mapping dataframe from the topic info
        label_map = topic_info_df.select(
            pl.col("Topic").alias("topic_id"),
            # Clean up the name (remove the ID prefix BERTopic adds)
            pl.col("Name").str.split("_").list.get(1).alias("topic_short_name")
        )

        # Join to get the labels
        coord_df = coord_df.join(label_map, on="topic_id", how="left")

        # Create final label column handling noise
        coord_df = coord_df.with_columns(
            pl.when(pl.col("topic_id") == -1)
            .then(pl.lit("Noise"))
            .otherwise(
               pl.col("topic_id").cast(pl.String) + ": " + pl.col("topic_short_name")
            ).alias("topic_label")
        )
    return coord_df, topic_info_df


@app.cell
def viz_controls():
    # Controls that ONLY affect the visual rendering, not the clustering calculation

    mo.md("### 🎨 Visual Settings")

    viz_opacity = mo.ui.slider(start=0.1, stop=1.0, step=0.1, value=0.6, label="Point Opacity")
    viz_size = mo.ui.slider(start=10, stop=200, step=10, value=60, label="Point Size")
    viz_show_clouds = mo.ui.checkbox(value=True, label="Show Density Clouds")

    viz_row = mo.hstack([viz_opacity, viz_size, viz_show_clouds], justify="center")
    return viz_opacity, viz_row, viz_show_clouds, viz_size


@app.cell
def altair_plot(coord_df, viz_opacity, viz_show_clouds, viz_size):
    # This cell generates the Altair chart using the calculated coord_df
    # It updates instantly when viz sliders change, without re-clustering.

    if coord_df.is_empty():
        chart = mo.md("⚠️ Not enough data to plot.")
        final_chart = None
    else:
        # 1. Base Chart (Altair works with Polars natively)
        base = alt.Chart(coord_df).encode(
            x=alt.X('x', axis=None),
            y=alt.Y('y', axis=None),
        )

        # 2. The "Cloud" (Density Effect)
        if viz_show_clouds.value:
            # Filter noise for the clouds so we don't blur the background
            cloud_base = base.transform_filter(
                alt.datum.topic_id != -1 
            )
        
            # METHOD: "The Splatter"
            # Instead of complex math, we plot very large, very transparent circles.
            # Where points cluster, the opacity adds up (0.05 + 0.05 + ...), creating a "dense" core.
            bg_layer = cloud_base.mark_circle(
                size=5000,    # Very large markers to merge together
                opacity=0.03  # Extremely low opacity so only overlaps show up
            ).encode(
                color=alt.Color('topic_label', legend=None),
                tooltip=alt.value(None) # Disable tooltips on the cloud background
            )
        else:
            bg_layer = base.mark_circle(opacity=0).encode()

        # 3. The Selection Mechanism
        brush = alt.selection_interval(name="brush") 
        click = alt.selection_point(fields=['topic_label'], bind='legend') 

        # 4. The Scatter Points (Foreground)
        points = base.mark_circle(
            size=viz_size.value,
            opacity=viz_opacity.value
        ).encode(
            color=alt.Color(
                'topic_label', 
                legend=alt.Legend(title="Identified Clusters", columns=1, symbolLimit=0)
            ),
            tooltip=['Title', 'topic_label', 'Label', 'semantic_score'],
            opacity=alt.condition(brush | click, alt.value(viz_opacity.value), alt.value(0.05))
        ).add_params(
            brush,
            click
        )

        final_chart = (bg_layer + points).properties(
            width=800,
            height=600,
            title="Interactive Expertise Map (Drag to Select)"
        ).interactive()

        # Marimo UI Wrapper
        chart = mo.ui.altair_chart(final_chart)


    return (chart,)


@app.cell
def comparison_logic(df_scored):
    # Logic to show what Andy missed
    # We recalculate this here purely for display purposes in the tabs
    missed = df_scored.filter(
        (pl.col("semantic_score") > 0.45) & (pl.col("keyword_match") == False)
    ).sort("semantic_score", descending=True).head(10)

    missed_view = mo.ui.table(
        missed.select(["Title", "semantic_score", "Type group"]),
        label="Found by AI, Missed by Keywords"
    )
    return (missed_view,)


@app.cell
def main_layout(chart, coord_df, missed_view, topic_info_df, viz_row):
    # --- RESULT TABLE LOGIC ---

    # 1. Handle Selection
    # Marimo returns the filtered data in chart.value. 
    # If the input was Polars, chart.value is usually Polars (or list of dicts).
    # We cast to ensure consistency.

    selection_df = pl.DataFrame()
    table_title = "No Data"

    if hasattr(chart, "value") and not chart.value.is_empty():
        # User selected points
        selection_df = chart.value
        table_title = f"Selected Items ({len(selection_df)})"
    elif not coord_df.is_empty():
        # Default: Show everything
        selection_df = coord_df
        table_title = f"All Filtered Items ({len(selection_df)})"

    # 2. Convert to Marimo Table
    result_table = mo.ui.table(
        selection_df.select(['Title', 'topic_label', 'semantic_score', 'Label']),
        label=table_title,
        selection=None,
        page_size=10
    )

    # --- LAYOUT ASSEMBLY ---

    # Left Column: Plot + Visual Controls
    left_col = mo.vstack([
        viz_row,
        chart
    ])

    # Right Column: Data Tables (Tabs)
    right_col = mo.ui.tabs({
        "📄 Selected Papers": result_table,
        "📊 Cluster Stats": mo.ui.table(topic_info_df, selection=None),
        "🔍 AI vs Keywords": missed_view
    })

    layout = mo.hstack([left_col, right_col], widths=[6, 4], gap="md")
    return (layout,)


@app.cell
def render(layout):
    # Render the App
    layout
    return


if __name__ == "__main__":
    app.run()
