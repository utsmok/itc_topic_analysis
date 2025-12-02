import marimo

__generated_with = "0.18.0"
app = marimo.App(width="full")

with app.setup:
    import os

    import altair as alt
    import dspy
    import marimo as mo
    import polars as pl
    import torch
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer, util
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    import httpx
    from time import sleep

    # Configuration
    # bge-m3 is a relatively heavy embedding model, takes about ~5 minutes to embed all 16k items on my nvidia gpu, and 1 hour+ on cpu
    # for lighter models, you can use e.g. "sentence-transformers/all-MiniLM-L6-v2" , "google/embeddinggemma-300m", or "nomic-ai/nomic-embed-text-v1.5"

    EMBEDDING_MODEL_ID = "BAAI/bge-m3"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Configure Altair
    alt.data_transformers.disable_max_rows()


@app.cell
def data_functions():
    def load_data_from_excel(path="andy_data.xlsx"):
        if not os.path.exists(path):
            # Fallback dummy data
            return (
                pl.DataFrame({"Title": ["Demo Title"]}),
                pl.DataFrame({"Type code": [1], "Label": ["Journal"]}),
                ["maize"],
            )

        df_core = pl.read_excel(path, sheet_name="data")
        df_types = pl.read_excel(path, sheet_name="types")
        keywords = pl.read_excel(path, sheet_name="search_terms")["term"].to_list()
        df_core, oa_data = enrich_data_by_doi(df_core, True)
        return df_core, df_types, keywords, oa_data


    def consolidate_data(df_core, df_types):
        """Joins type labels and prepares text for embedding.
        For now, just merges Andy's title+abstract (if available) for the embedding text.
        Could also add openalex data to the embeddings for example.
        """
        df_clean = df_core.join(
            df_types.select(["Type code", "Label"]),
            left_on="Type group",
            right_on="Type code",
            how="left",
        )
        # Handle missing titles/abstracts safely
        df_clean = df_clean.with_columns(pl.col("Title").fill_null("Untitled Document"))
        if "Abstract" in df_clean.columns:
            df_clean = df_clean.with_columns(
                (pl.col("Title") + ". " + pl.col("Abstract").fill_null("")).alias("text_for_embedding")
            )
        else:
            df_clean = df_clean.with_columns(pl.col("Title").alias("text_for_embedding"))
        return df_clean
    return consolidate_data, load_data_from_excel


@app.function
def get_openalex_works_bulk_by_id(ids: list[str], id_type: str = "doi", per_page: int = 50) -> pl.DataFrame:
    """For a given list of DOIs retrieve the corresponding works from OpenAlex API."""
    openalex_works_api_url = "https://api.openalex.org/works"
    headers = {"User-Agent": "mailto:s.mok@utwente.nl"}
    ids_chunks = [ids[i : i + 50] for i in range(0, len(ids), 50)]
    all_works = []
    print(f"retrieving {len(ids)} items from OpenAlex using {id_type} identifiers")
    with httpx.Client(headers=headers) as client:
        for chunk in mo.status.progress_bar(
            collection=ids_chunks,
            show_eta=True,
            show_rate=True,
            title="Retrieving works from OpenAlex",
        ):
            filter_query = "|".join([str(x).replace("doi: ", "") for x in chunk])
            params = {"filter": f"{id_type}:{filter_query}", "per-page": per_page}
            response = client.get(openalex_works_api_url, params=params)
            try:
                response.raise_for_status()
                data = response.json()
            except Exception:
                # wait 4 seconds and retry
                sleep(4)
                response = client.get(openalex_works_api_url, params=params)
                try:
                    response.raise_for_status()
                    data = response.json()
                except Exception:
                    # reduce amount of results per page, split into multiple queries, combine results
                    temp_per_page = 5
                    for start in range(0, len(chunk), temp_per_page):
                        try:
                            sub_chunk = chunk[start : start + temp_per_page]
                            sub_filter_query = "|".join([str(x).replace("doi: ", "") for x in sub_chunk])
                            sub_params = {
                                "filter": f"{id_type}:{sub_filter_query}",
                                "per-page": temp_per_page,
                            }
                            sub_response = client.get(openalex_works_api_url, params=sub_params)
                            sub_response.raise_for_status()
                            sub_data = sub_response.json()
                            all_works.extend(sub_data["results"])
                            sleep(1)
                        except Exception as e:
                            print(
                                f"error while trying to retrieve data from OA api for sub-chunk with ids {sub_filter_query}: {e}. Skipping."
                            )
                            continue
                    continue

            all_works.extend(data["results"])
            print(f"retrieved {len(all_works)} items. (+{len(data['results'])})")
    results = pl.from_dicts(all_works)
    return results


@app.function
def enrich_data_by_doi(
    input_df: pl.DataFrame, store: bool = True, refresh_data: bool = False
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Enrich the input DataFrame with metadata fetched via DOIs. Returns the merged dataframe and the OpenAlex works dataframe."""
    input_df = input_df.with_columns(
        pl.col("DOI").str.strip_chars().str.to_lowercase().str.replace_all(",", "").alias("DOI_clean")
    )
    dois = input_df.select("DOI_clean").filter(pl.col("DOI_clean").is_not_null()).unique().to_series().to_list()
    if ("oa_results.xlsx" or "oa_results.parquet" in os.listdir()) and not refresh_data:
        # load data from parquet instead of fetching again
        filename = "oa_results.parquet" if "oa_results.parquet" in os.listdir() else "oa_results.xlsx"
        oa_works_df = pl.read_parquet(filename)
    else:
        print(f"Fetching OpenAlex metadata for {len(dois)} DOIs...")

        oa_works_df = get_openalex_works_bulk_by_id(dois, id_type="doi")
        if store:
            oa_works_df.write_parquet("oa_results.parquet")
    oa_works_df = oa_works_df.with_columns(
        pl.col("doi").str.replace("https://doi.org/", "").str.to_lowercase().alias("DOI_clean")
    )
    # merge back to input_df
    input_df = input_df.join(oa_works_df, on="DOI_clean", how="left", suffix="_oa")

    return input_df, oa_works_df


@app.cell
def _(df_full):
    df_full
    return


@app.cell
def cached_state(consolidate_data, load_data_from_excel):
    # Persistent cache for the heavy embedding step

    def get_initial_state():
        print(f"Loading Model: {EMBEDDING_MODEL_ID} on {DEVICE}...")
        model = SentenceTransformer(EMBEDDING_MODEL_ID, device=DEVICE)

        print("Loading Data...")
        df_core, df_types, keywords, oa_data = load_data_from_excel()
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
                    convert_to_numpy=True,
                )
        else:
            embeddings = model.encode(
                texts,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

        return df_full, embeddings, keywords, oa_data


    # Load model separately (avoid pickling issues)
    model = SentenceTransformer(EMBEDDING_MODEL_ID, device=DEVICE)

    with mo.status.spinner(subtitle="Loading Data & Embeddings..."):
        df_full, full_embeddings, legacy_keywords, oa_data = get_initial_state()
    return df_full, full_embeddings, legacy_keywords, model


@app.cell
def ui_controls(df_full):
    # --- UNIFIED CONTROL PANEL ---

    # Group 1: Data & Filters
    check_recalc = mo.ui.switch(value=True, label="🟢 Live Calc")
    default_concept = "Food security, agriculture, crop yield, and sustainable farming systems"
    text_concept = mo.ui.text_area(value=default_concept, label="Target Concept", rows=2)
    slider_threshold = mo.ui.slider(start=0.3, stop=0.8, step=0.01, value=0.45, label="Relevance")

    col_data = mo.vstack(
        [
            mo.md("**Data & Filters**"),
            mo.hstack(
                [check_recalc, mo.md(f"_{len(df_full)} Items_")],
                justify="space-between",
            ),
            text_concept,
            slider_threshold,
        ],
        gap="sm",
    )

    # Group 2: Clustering Logic
    ignore_top_percent = mo.ui.slider(
        start=0.65, stop=1.0, step=0.05, value=0.9, label="Ignore words that appear more than this % of documents"
    )
    slider_cluster_min = mo.ui.slider(start=5, stop=100, step=5, value=20, label="Min Size")
    slider_neighbors = mo.ui.slider(start=2, stop=50, value=15, label="Neighbors")
    check_dspy = mo.ui.checkbox(value=False, label="AI Labeling (DSPy)")

    col_algo = mo.vstack(
        [
            mo.md("**Clustering Algorithm**"),
            ignore_top_percent,
            slider_cluster_min,
            slider_neighbors,
            check_dspy,
        ],
        gap="sm",
    )

    # Group 3: Visuals & Interactions
    slider_opacity = mo.ui.slider(start=0.1, stop=1.0, step=0.1, value=0.6, label="Dots opacity")
    slider_size = mo.ui.slider(start=10, stop=200, value=60, label="Dot size")
    # add sliders for cloud opacity and spread
    slider_cloud_opacity = mo.ui.slider(start=0.01, stop=0.2, step=0.01, value=0.03, label="Cloud Opacity")
    slider_cloud_spread = mo.ui.slider(start=1000, stop=10000, step=500, value=5000, label="Cloud Spread")

    check_clouds = mo.ui.checkbox(value=True, label="Show Density Clouds")

    # Interaction settings
    check_pan_zoom = mo.ui.checkbox(value=True, label="Enable Pan & Zoom")
    check_selection = mo.ui.checkbox(value=True, label="Enable Selection Brush")

    col_viz = mo.vstack(
        [
            mo.md("**Visuals & Interaction**"),
            mo.hstack([slider_cloud_opacity, slider_cloud_spread]),
            mo.hstack([slider_opacity, slider_size]),
            check_clouds,
            mo.hstack([check_pan_zoom, check_selection]),
        ],
        gap="sm",
    )

    # Layout: 3 Columns
    controls_layout = mo.hstack([col_data, col_algo, col_viz], align="start")

    # Wrap in a nice callout
    control_panel = mo.callout(controls_layout, kind="neutral")
    return (
        check_clouds,
        check_dspy,
        check_pan_zoom,
        check_recalc,
        check_selection,
        control_panel,
        ignore_top_percent,
        slider_cloud_opacity,
        slider_cloud_spread,
        slider_cluster_min,
        slider_neighbors,
        slider_opacity,
        slider_size,
        slider_threshold,
        text_concept,
    )


@app.cell
def logic_state_management():
    # This cell holds the "Result State" to allow pausing updates
    get_results, set_results = mo.state(None)
    return get_results, set_results


@app.cell
def logic_calculation(
    check_dspy,
    check_recalc,
    df_full,
    full_embeddings,
    ignore_top_percent,
    legacy_keywords,
    model,
    set_results,
    slider_cluster_min,
    slider_neighbors,
    slider_threshold,
    text_concept,
):
    # This cell orchestrates the filtering AND clustering.
    class TopicSig(dspy.Signature):
        """Name a research topic."""

        keywords = dspy.InputField()
        titles = dspy.InputField()
        short_name = dspy.OutputField(desc="A 3-5 word descriptive label")


    if not check_recalc.value:
        pass
    else:
        # --- 1. FILTERING ---
        query_emb = model.encode(text_concept.value, normalize_embeddings=True)
        scores = util.cos_sim(query_emb, full_embeddings)[0].cpu().numpy()

        # are we certain that this will match up the score with the correct document?
        df_scored = df_full.with_columns(pl.Series("semantic_score", scores))

        # Legacy check
        pattern = "|".join([str(k) for k in legacy_keywords if str(k).strip()])
        df_scored = df_scored.with_columns(pl.col("Title").str.to_lowercase().str.contains(pattern).alias("keyword_match"))

        mask = df_scored["semantic_score"] >= slider_threshold.value
        df_filtered = df_scored.filter(mask)
        filtered_embs = full_embeddings[mask.to_numpy()]

        # --- 2. CLUSTERING ---
        if len(df_filtered) < 10:
            result_data = None
        else:
            topic_model = BERTopic(
                embedding_model=model,
                umap_model=UMAP(
                    n_neighbors=slider_neighbors.value,
                    n_components=2,
                    min_dist=0.0,
                    metric="cosine",
                    random_state=2445,
                ),
                hdbscan_model=HDBSCAN(
                    min_cluster_size=slider_cluster_min.value,
                    max_cluster_size=50,
                    metric="euclidean",
                    cluster_selection_method="eom",
                    prediction_data=True,
                ),
                vectorizer_model=CountVectorizer(max_df=ignore_top_percent.value),
                verbose=False,
            )

            topics, probs = topic_model.fit_transform(df_filtered["Title"].to_list(), filtered_embs)

            # --- 3. COORDS & LABELS ---
            umap_coords = topic_model.umap_model.embedding_

            topic_info_pd = topic_model.get_topic_info()
            doc_info = topic_model.get_document_info(df_filtered["Title"].to_list())
            # merge doc_info back into the main documents data frame df_filtered
            # first turn into polars df
            doc_info_pl = pl.from_pandas(doc_info)
            df_docs_with_topics = df_filtered.join(
                doc_info_pl,
                left_on="Title",
                right_on="Document",
                how="left",
            )

            # Label Mapping
            label_map_dict = {}

            # Setup DSPy if requested
            if check_dspy.value:
                try:
                    # Configure API (Ensure API key is available in env or inserted here)
                    lm = dspy.LM(
                        "gemini/gemini-2.5-flash",
                    )
                    dspy.settings.configure(lm=lm)

                    generator = dspy.Predict(TopicSig)
                except Exception as e:
                    print("⚠️ DSPy Config Failed (No Key?). Falling back to keywords.")
                    print(f"Error: {e}")

            for i, row in topic_info_pd.iterrows():
                tid = row["Topic"]
                if tid == -1:
                    continue

                if check_dspy.value:
                    # AI Labeling
                    kws = ", ".join(row["Representation"][:5])
                    # Get docs
                    rep_docs = topic_model.get_representative_docs(tid)
                    titles = " | ".join([d[:100] for d in rep_docs[:3]])
                    try:
                        pred = generator(keywords=kws, titles=titles)
                        label = pred.short_name
                    except:
                        label = row["Name"]  # Fallback
                else:
                    # Keyword Labeling
                    name_parts = row["Name"].split("_")
                    label = " ".join(name_parts[1:3]) if len(name_parts) > 2 else row["Name"]

                label_map_dict[tid] = label

            # Update topic_info_pd with the refined labels
            topic_info_pd["Refined Label"] = topic_info_pd["Topic"].map(label_map_dict).fillna("Noise")

            # Convert Topic Info to Polars for result view
            topic_info_pl = pl.from_pandas(topic_info_pd)

            # Build Coordinate DataFrame
            _coord_df = df_docs_with_topics
            _coord_df = _coord_df.with_columns(
                [
                    pl.Series("x", umap_coords[:, 0]),
                    pl.Series("y", umap_coords[:, 1]),
                    pl.Series("topic_id", topics),
                ]
            )

            # Apply labels via map
            labels_df = pl.DataFrame(
                {
                    "topic_id": list(label_map_dict.keys()),
                    "topic_name": list(label_map_dict.values()),
                }
            ).with_columns(pl.col("topic_id").cast(pl.Int64))

            _coord_df = _coord_df.join(labels_df, on="topic_id", how="left")
            # also do this for the df_docs_with_topics

            # Final Formatting
            _coord_df = _coord_df.with_columns(
                pl.when(pl.col("topic_id") == -1)
                .then(pl.lit("Noise"))
                .otherwise(pl.col("topic_id").cast(pl.String) + ": " + pl.col("topic_name").fill_null("Unknown"))
                .alias("topic_label")
            )

            result_data = {
                "coord_df": _coord_df,
                "topic_info": topic_info_pl,
                "missed": df_scored.filter((pl.col("semantic_score") > 0.45) & (pl.col("keyword_match") == False))
                .sort("semantic_score", descending=True)
                .head(10),
            }

        # Update State
        set_results(result_data)
    return (topic_info_pl,)


@app.cell
def _(topic_info_pl):
    topic_info_pl
    return


@app.cell
def visualization(
    check_clouds,
    check_pan_zoom,
    check_selection,
    get_results,
    slider_cloud_opacity,
    slider_cloud_spread,
    slider_opacity,
    slider_size,
):
    # This cell reads the state and renders the plot.
    results = get_results()


    def create_chart(openalex=False, openalex_colors=False) -> mo.ui.altair_chart:
        """
        Creates the Altair chart based on the current results and settings.
        Used to visualize the results of clustering and compare them with the openalex model.
        Shows the top 5 topics and OpenAlex subfields with density clouds and interactive points.
        x-y coordinates are taken from UMAP embeddings in the BERTopic model.
        """
        coord_df = results["coord_df"]
        coord_df = coord_df.with_columns(
            pl.col("primary_topic").struct.field("display_name").alias("oa_topic"),
            pl.col("primary_topic").struct.field("subfield").struct.field("display_name").alias("oa_subfield"),
            pl.col("primary_topic").struct.field("field").struct.field("display_name").alias("oa_field"),
        )

        if openalex:
            # filter to only show items with openalex data
            coord_df = coord_df.filter(pl.col("oa_subfield").is_not_null())

        if openalex:
            #  instead of our topic labels we use the openalex subfield labels
            top_topics = (
                coord_df.group_by("oa_subfield")
                .agg([pl.col("x").mean(), pl.col("y").mean(), pl.len().alias("count")])
                .sort("count", descending=True)
                .head(5)
            )
        else:
            top_topics = (
                coord_df.filter(pl.col("topic_id") != -1)
                .group_by("topic_label")
                .agg([pl.col("x").mean(), pl.col("y").mean(), pl.len().alias("count")])
                .sort("count", descending=True)
                .head(5)
            )

        # filter points to only show those that have one of the top topics
        # we do this because otherwise it's a mess

        coord_df = coord_df.join(
            top_topics.select([pl.col("topic_label" if not openalex else "oa_subfield").alias("top_topic_label")]),
            left_on=("topic_label" if not openalex else "oa_subfield"),
            right_on="top_topic_label",
            how="inner",
        )

        # --- ALTAIR PLOT ---
        base = alt.Chart(coord_df).encode(x=alt.X("x", axis=None), y=alt.Y("y", axis=None))

        # set up specific colors for top 5 topics
        # if openalex and openalex_colors, also set up colors for openalex subfields
        # ?? not sure how / if we need to do this

        # 1. Clouds
        # grab value of sliders for opacity and spread
        # then draw clouds indicating the density of points per topic
        # if openalex_colors is true, ALSO draw clouds that represent the openalex subfields

        if check_clouds.value:
            if openalex and openalex_colors:
                # use oa_colors
                bg_layer = (
                    base.transform_filter(alt.datum.oa_subfield != None)
                    .mark_circle(size=slider_cloud_spread.value, opacity=slider_cloud_opacity.value)
                    .encode(color=alt.Color("topic_label", legend=None), tooltip=alt.value(None))
                )
            bg_layer = (
                base.transform_filter(alt.datum.topic_id != -1)
                .mark_circle(size=slider_cloud_spread.value, opacity=slider_cloud_opacity.value)
                .encode(color=alt.Color("topic_label", legend=None), tooltip=alt.value(None))
            )
        else:
            bg_layer = base.mark_circle(opacity=0).encode()

        # 2. Points
        # Handle Selection vs Pan/Zoom
        # If selection is enabled, we add the selection param.
        points_base = base.mark_circle(size=slider_size.value, opacity=slider_opacity.value)
        tooltips = ["Title", "topic_label", "semantic_score"]
        if openalex:
            tooltips.extend(["oa_subfield", "oa_field"])
        if check_selection.value:
            brush = alt.selection_interval(name="brush")
            points = points_base.encode(
                color=alt.Color("topic_label", legend=None),
                tooltip=tooltips,
                opacity=alt.condition(brush, alt.value(slider_opacity.value), alt.value(0.05)),
            ).add_params(brush)
        else:
            points = points_base.encode(
                color=alt.Color("topic_label", legend=None),
                tooltip=tooltips,
            )

        # 3. Text Labels (Centroids)

        # Create text with Halo (Stroke) by layering
        # Bottom Layer: Black stroke
        text_outline = (
            alt.Chart(top_topics)
            .mark_text(
                align="center",
                baseline="middle",
                fontSize=12,
                fontWeight="bold",
                color="black",
                stroke="black",
                strokeWidth=3,
            )
            .encode(x="x", y="y", text=("topic_label" if not openalex else "oa_subfield"))
        )
        # On top: slightly smaller white stroke (to get a black/white stroke so it's visible on both light and dark areas)
        text_outline2 = (
            alt.Chart(top_topics)
            .mark_text(
                align="center",
                baseline="middle",
                fontSize=12,
                fontWeight="bold",
                color="white",
                stroke="white",
                strokeWidth=1.5,
            )
            .encode(x="x", y="y", text=("topic_label" if not openalex else "oa_subfield"))
        )

        # Top Layer: Fill with the color of the topic / subfield (align with the point/cloud colors)

        text_fill = (
            alt.Chart(top_topics)
            .mark_text(
                align="center",
                baseline="middle",
                fontSize=12,
                fontWeight="bold",
            )
            .encode(
                x="x",
                y="y",
                text=("topic_label" if not openalex else "oa_subfield"),
                color=alt.Color("topic_label" if not openalex else "oa_subfield", legend=None),
            )
        )

        final_chart = (bg_layer + points + text_outline + text_fill).properties(
            width=800,
            height=600,
            title=("Topic Clusters" if not openalex else "Topic Clusters (+OpenAlex Subfield labels)"),
        )

        # Apply Pan/Zoom if requested
        if check_pan_zoom.value:
            final_chart = final_chart.interactive()

        chart = mo.ui.altair_chart(final_chart)
        return base, chart, coord_df


    if results is None:
        chart = mo.md("Waiting for data... (Check thresholds)")
    else:
        base, chart, coord_df = create_chart(openalex=False)
        _, chart2, coord_df = create_chart(openalex=True, openalex_colors=True)
        final_chart_overlap = create_overlap_chart(
            results, top_n=100, opacity=slider_opacity.value, dot_size=slider_size.value
        )
    return chart, coord_df, final_chart_overlap, results


@app.function
def create_overlap_chart(results: dict, top_n: int = 5, opacity: float = 0.6, dot_size: int = 10) -> mo.ui.altair_chart:
    """
    Creates a Linked View chart to compare Topic Clusters vs OpenAlex Subfields.

    - Left Chart: Colored by Topic Label (Method A)
    - Right Chart: Colored by OpenAlex Subfield (Method B)
    - Interaction: Dragging/Selecting on one filters the other.
    """

    # --- 1. Efficient Data Prep (Polars) ---
    coord_df = results["coord_df"]

    # Unpack OpenAlex structs  & filter
    df = (
        coord_df.lazy()
        .with_columns(
            [
            pl.col("primary_topic").struct.field("display_name").alias("oa_topic"),
            pl.col("primary_topic").struct.field("subfield").struct.field("display_name").alias("oa_subfield"),
            pl.col("primary_topic").struct.field("field").struct.field("display_name").alias("oa_field"),
            ]
        )
        .filter(pl.col("oa_subfield").is_not_null())
        .filter(pl.col("topic_label") != "Noise")
        .collect()
    )

    # Calculate Top N for BOTH methods independently
    def get_top_labels(col_name, n):
        return df.group_by(col_name).len().sort("len", descending=True).head(n).select(col_name).to_series().to_list()

    top_topics = get_top_labels("topic_label", top_n)
    top_oa = get_top_labels("oa_subfield", top_n)
    top_oa_topic = get_top_labels("oa_topic", top_n)
    print(f"got {len(top_topics)} top topics, {len(top_oa)} top oa subfields, {len(top_oa_topic)} top oa topics")
    # Filter dataset to only relevant items (union of both top sets)
    # This keeps the data lighter for the browser
    df_vis = df.filter((pl.col("topic_label").is_in(top_topics)) | (pl.col("oa_subfield").is_in(top_oa)))

    # --- 2. Chart Construction (Altair) ---

    # A shared selection brush
    brush = alt.selection_interval(resolve="global")

    # Base chart definition
    base = (
        alt.Chart(df_vis)
        .mark_circle(size=dot_size)
        .encode(x=alt.X("x", axis=None), y=alt.Y("y", axis=None), tooltip=["topic_label", "oa_subfield", "oa_topic"])
        .properties(width=400, height=400)
    )

    # --- Left Panel: Your Topics ---
    # We use a condition to gray out points not selected by the brush OR not in the top topics
    chart_topics = (
        base.encode(
            color=alt.condition(
                brush,
                alt.Color(
                    "topic_label",
                    scale=alt.Scale(domain=top_topics),  # Fix colors to top topics
                    legend=alt.Legend(title="Topic Clusters"),
                    type="nominal",
                ),
                alt.value("lightgray"),
            ),
            opacity=alt.condition(brush, alt.value(opacity), alt.value(0.05)),
        )
        .add_params(brush)
        .properties(title="Method A: Topic Clusters")
    )

    # --- Right Panel: OpenAlex ---
    chart_oa = (
        base.encode(
            color=alt.condition(
                brush,
                alt.Color(
                    "oa_subfield",
                    scale=alt.Scale(domain=top_oa, scheme="tableau10"),  # Different scheme
                    legend=alt.Legend(title="OpenAlex Subfields"),
                    type="nominal",
                ),
                alt.value("lightgray"),
            ),
            opacity=alt.condition(brush, alt.value(opacity), alt.value(0.05)),
        )
        .add_params(brush)
        .properties(title="Method B: OpenAlex")
    )

    # additional plot for topics

    chart_oa_topics = (
        base.encode(
            color=alt.condition(
                brush,
                alt.Color(
                    "oa_topic",
                    scale=alt.Scale(domain=top_oa_topic, scheme="category10"),  # Different scheme
                    legend=alt.Legend(title="OpenAlex Topics"),
                    type="nominal",
                ),
                alt.value("lightgray"),
            ),
            opacity=alt.condition(brush, alt.value(opacity), alt.value(0.05)),
        )
        .add_params(brush)
        .properties(title="Method C: OpenAlex Topics")
    )
    # --- 3. Text Labels (Centroids) ---
    # We calculate centroids for the labels so they sit in the middle of the cluster

    def create_labels(group_col, color_domain):
        # Calculate centroids in Polars
        centroids = (
            df_vis.filter(pl.col(group_col).is_in(color_domain))
            .group_by(group_col)
            .agg([pl.col("x").mean(), pl.col("y").mean()])
        )

        text_layer = (
            alt.Chart(centroids)
            .mark_text(align="center", baseline="middle", fontWeight="bold", fontSize=5, dy=-10)
            .encode(
                x="x",
                y="y",
                text=group_col,
                color=alt.value("black"),  # Keep text black for readability
                stroke=alt.value("white"),
                strokeWidth=alt.value(0.2),  
            )
        )

        text_front = text_layer.mark_text(
            align="center", baseline="middle", fontWeight="bold", fontSize=5, dy=-10
        ).encode(x="x", y="y", text=group_col, color=alt.value("black"))
        return text_layer + text_front

    # Add labels to respective charts
    final_topics = chart_topics + create_labels("topic_label", top_topics)
    final_oa = chart_oa 
    final_oa_topics = chart_oa_topics 
    
    # Combine side-by-side
    final_chart = alt.hconcat(final_topics, final_oa, final_oa_topics).resolve_scale(
        color="independent"  # Important: Keeps the two color legends separate
    )
    final_chart.save("overlap_chart.html")
    return mo.ui.altair_chart(final_chart)


@app.cell
def _(chart, coord_df, results):
    # --- RESULT TABLES ---

    # 1. Selection Table

    if hasattr(chart, "value") and not chart.value.is_empty():
        display_df = chart.value
        title = f"Selected ({len(display_df)})"
    else:
        display_df = coord_df
        title = f"All Items ({len(display_df)})"

    table_selection = mo.ui.table(
        display_df.select(["Title", "topic_label", "semantic_score"]),
        label=title,
        selection=None,
    )

    # 2. Topic Details (New Request)
    # We clean up the topic info for better display
    if results and "topic_info" in results:
        topics_df = (
            results["topic_info"]
            .select(
                [
                    pl.col("Refined Label").alias("Label"),
                    pl.col("Count"),
                    pl.col("Representation").list.slice(0, 5).list.join(", ").alias("Top Keywords"),
                ]
            )
            .filter(pl.col("Label") != "Noise")
            .sort("Count", descending=True)
        )

        table_topics = mo.ui.table(topics_df, label=f"Topics ({len(topics_df)})")
    else:
        table_topics = mo.md("No topics generated yet.")

    # 3. Missed Table
    missed_table = mo.ui.table(
        results["missed"].select(["Title", "semantic_score"]),
        label="Missed by Keywords",
    )

    result_view = mo.ui.tabs(
        {
            "Selection": table_selection,
            "Topic List": table_topics,
            "AI vs Keywords": missed_table,
        }
    )
    return (result_view,)


@app.cell
def main_layout(control_panel, final_chart_overlap, result_view):
    mo.vstack([control_panel, mo.hstack([final_chart_overlap, result_view], widths=[7, 5])])
    return


if __name__ == "__main__":
    app.run()
