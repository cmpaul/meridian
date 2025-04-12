#!/usr/bin/env python3
"""
Meridian Brief Generator
------------------------
This script automates the generation of news digests by:
1. Fetching processed articles from the database
2. Clustering related articles using embeddings and HDBSCAN
3. Analyzing clusters with LLM to identify stories
4. Generating a final markdown brief

Usage:
    python brief_generator.py [--date YYYY-MM-DD]
"""

import argparse
import json
import logging
import os
from datetime import datetime, timedelta
from typing import List, Literal, Optional

import hdbscan
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import umap
from hdbscan.validity import validity_index
from json_repair import repair_json
from pydantic import BaseModel, Field, model_validator
from retry import retry
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("brief_generator.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Import local modules
try:
    from src.events import get_events
    from src.llm import call_llm
except ImportError:
    logger.error(
        "Could not import local modules. Make sure you're running from the correct directory."
    )
    raise


# Define Pydantic models for validation
class Story(BaseModel):
    title: str = Field(description="title of the story")
    importance: int = Field(
        ge=1,
        le=10,
        description="global significance (1=minor local event, 10=major global impact)",
    )
    articles: List[int] = Field(description="list of article ids in the story")


class StoryValidation(BaseModel):
    answer: Literal["single_story", "collection_of_stories", "pure_noise", "no_stories"]

    # optional fields that depend on the answer type
    title: Optional[str] = None
    importance: Optional[int] = Field(None, ge=1, le=10)
    outliers: List[int] = Field(default_factory=list)
    stories: Optional[List[Story]] = None

    @model_validator(mode="after")
    def validate_structure(self):
        if self.answer == "single_story":
            if self.title is None or self.importance is None:
                raise ValueError(
                    "'title' and 'importance' are required for 'single_story'"
                )
            if self.stories is not None:
                raise ValueError("'stories' should not be present for 'single_story'")

        elif self.answer == "collection_of_stories":
            if not self.stories:
                raise ValueError("'stories' is required for 'collection_of_stories'")
            if self.title is not None or self.importance is not None or self.outliers:
                raise ValueError(
                    "'title', 'importance', and 'outliers' should not be present for 'collection_of_stories'"
                )

        elif self.answer == "pure_noise" or self.answer == "no_stories":
            if (
                self.title is not None
                or self.importance is not None
                or self.outliers
                or self.stories is not None
            ):
                raise ValueError(
                    "no additional fields should be present for 'pure_noise'"
                )

        return self


# Helper function for embeddings
def average_pool(last_hidden_states, attention_mask):
    """Average pooling function for embeddings"""
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def generate_embeddings(articles_df):
    """Generate embeddings for article summaries"""
    logger.info("Generating embeddings for %d articles", len(articles_df))

    # Load the multilingual model
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
    model = AutoModel.from_pretrained("intfloat/multilingual-e5-small")

    # Batch processing to avoid memory issues
    batch_size = 64
    all_embeddings = []

    # Process in batches with progress bar
    for i in tqdm(range(0, len(articles_df), batch_size)):
        batch_texts = articles_df["text_to_embed"].iloc[i : i + batch_size].tolist()

        # Tokenize
        batch_dict = tokenizer(
            batch_texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # Generate embeddings
        with torch.no_grad():
            outputs = model(**batch_dict)

        # Pool and normalize
        embeddings = average_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Convert to numpy and add to list
        all_embeddings.extend(embeddings.numpy())

    return all_embeddings


def optimize_clusters(embeddings, umap_params, hdbscan_params):
    """Find optimal clustering parameters"""
    logger.info("Optimizing clustering parameters")
    best_score = -1
    best_params = None

    # Grid search both umap and hdbscan params
    for n_neighbors in umap_params["n_neighbors"]:
        # Fit umap once per n_neighbors config
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=10,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )
        reduced_data = reducer.fit_transform(embeddings)

        for min_cluster_size in hdbscan_params["min_cluster_size"]:
            for min_samples in hdbscan_params["min_samples"]:
                for epsilon in hdbscan_params["epsilon"]:
                    # Cluster with hdbscan
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        cluster_selection_epsilon=epsilon,
                        metric="euclidean",
                        prediction_data=True,
                    )

                    cluster_labels = clusterer.fit_predict(reduced_data)

                    # Skip if all noise
                    if np.all(cluster_labels == -1):
                        continue

                    # Evaluate with dbcv (better for density clusters)
                    valid_points = cluster_labels != -1
                    if (
                        valid_points.sum() > 1
                        and len(set(cluster_labels[valid_points])) > 1
                    ):
                        try:
                            reduced_data_64 = reduced_data[valid_points].astype(
                                np.float64
                            )
                            score = validity_index(
                                reduced_data_64, cluster_labels[valid_points]
                            )

                            if score > best_score:
                                best_score = score
                                best_params = {
                                    "umap": {"n_neighbors": n_neighbors},
                                    "hdbscan": {
                                        "min_cluster_size": min_cluster_size,
                                        "min_samples": min_samples,
                                        "epsilon": epsilon,
                                    },
                                }
                                logger.info(
                                    f"New best: {best_score:.4f} with {best_params}"
                                )
                        except Exception as e:
                            # Sometimes dbcv can fail on weird cluster shapes
                            logger.warning(f"Failed with {e}")
                            continue

    return best_params, best_score


def cluster_articles(embeddings, params=None):
    """Cluster articles using UMAP and HDBSCAN"""
    logger.info("Clustering articles")

    if params is None:
        # Default parameters if optimization wasn't run
        umap_embeddings = umap.UMAP(
            n_neighbors=15,
            n_components=10,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        ).fit_transform(embeddings)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=3,
            min_samples=None,
            cluster_selection_epsilon=0.0,
            metric="euclidean",
            prediction_data=True,
        )
    else:
        # Use optimized parameters
        umap_embeddings = umap.UMAP(
            n_neighbors=params["umap"]["n_neighbors"],
            n_components=10,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        ).fit_transform(embeddings)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=params["hdbscan"]["min_cluster_size"],
            min_samples=params["hdbscan"]["min_samples"],
            cluster_selection_epsilon=params["hdbscan"]["epsilon"],
            metric="euclidean",
            prediction_data=True,
        )

    cluster_labels = clusterer.fit_predict(umap_embeddings)

    # Log clustering stats
    logger.info(
        f"Found {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)} clusters"
    )
    logger.info(f"Noise points: {sum(cluster_labels == -1)}")

    return cluster_labels


@retry(tries=3, delay=2, backoff=2, jitter=2, max_delay=20)
def process_story(cluster, events):
    """Process a cluster of articles to determine if it's a story"""
    logger.info(
        f"Processing cluster {cluster['cluster_id']} with {len(cluster['articles_ids'])} articles"
    )

    story_articles_ids = cluster["articles_ids"]

    story_article_md = ""
    for article_id in story_articles_ids:
        article = next((e for e in events if e.id == article_id), None)
        if article is None:
            continue
        story_article_md += f"- (#{article.id}) [{article.title}]({article.url})\n"

    story_article_md = story_article_md.strip()

    prompt = f"""
# Task
Determine if the following collection of news articles is:
1) A single story - A cohesive narrative where all articles relate to the same central event/situation and its direct consequences
2) A collection of stories - Distinct narratives that should be analyzed separately
3) Pure noise - Random articles with no meaningful pattern
4) No stories - Distinct narratives but none of them have more than 3 articles

# Important clarification
A "single story" can still have multiple aspects or angles. What matters is whether the articles collectively tell one broader narrative where understanding each part enhances understanding of the whole.

# Handling outliers
- For single stories: You can exclude true outliers in an "outliers" array
- For collections: Focus **only** on substantive stories (3+ articles). Ignore one-off articles or noise.

# Title guidelines
- Titles should be purely factual, descriptive and neutral
- Include necessary context (region, countries, institutions involved)
- No editorialization, opinion, or emotional language
- Format: "[Subject] [action/event] in/with [location/context]"

# Input data
Articles (format is (#id) [title](url)):
{story_article_md}

# Output format
Start by reasoning step by step. Consider:
- Central themes and events
- Temporal relationships (are events happening in the same timeframe?)
- Causal relationships (do events influence each other?)
- Whether splitting the narrative would lose important context

Return your final answer in JSON format:
```json
{{
    "answer": "single_story" | "collection_of_stories" | "pure_noise" | "no_stories",
    // single_story_start: if answer is "single_story", include the following fields:
    "title": "title of the story",
    "importance": 1-10, // global significance (1=minor local event, 10=major global impact)
    "outliers": [] // array of article ids to exclude as unrelated
    // single_story_end
    // collection_of_stories_start: if answer is "collection_of_stories", include the following fields:
    "stories": [
        {{
            "title": "title of the story",
            "importance": 1-10, // global significance scale
            "articles": [] // list of article ids in the story (**only** include substantial stories with **3+ articles**)
        }},
        ...
    ]
    // collection_of_stories_end
}}
```
"""

    # Call LLM to analyze the cluster
    response = call_llm(prompt)

    # Extract and parse JSON from response
    try:
        # Find JSON block in response
        json_start = response.find("```json")
        json_end = response.rfind("```")

        if json_start != -1 and json_end != -1:
            json_text = response[json_start + 7 : json_end].strip()
        else:
            json_text = response

        # Try to repair and parse JSON
        repaired_json = repair_json(json_text)
        result = json.loads(repaired_json)

        # Validate with Pydantic
        validated = StoryValidation(**result)
        return validated

    except Exception as e:
        logger.error(f"Error processing story: {e}")
        logger.error(f"Response: {response}")
        return None


def main():
    """Main function to generate a brief"""
    parser = argparse.ArgumentParser(description="Generate a news brief")
    parser.add_argument("--date", type=str, help="Date in YYYY-MM-DD format")
    args = parser.parse_args()

    # Set date
    if args.date:
        date = args.date
    else:
        # Default to yesterday
        yesterday = datetime.now() - timedelta(days=1)
        date = yesterday.strftime("%Y-%m-%d")

    logger.info(f"Generating brief for date: {date}")

    # Fetch events
    try:
        sources, events = get_events(date=date)
        logger.info(f"Fetched {len(events)} events from {len(sources)} sources")
    except Exception as e:
        logger.error(f"Error fetching events: {e}")
        return

    # Create DataFrame
    articles_df = pd.DataFrame(events)

    # Clean up tuples
    for col in articles_df.columns:
        articles_df[col] = articles_df[col].apply(
            lambda x: x[1] if isinstance(x, tuple) else x
        )

    # Rename columns
    articles_df.columns = [
        "id",
        "sourceId",
        "url",
        "title",
        "publishDate",
        "content",
        "location",
        "relevance",
        "completeness",
        "summary",
    ]

    # Extract summary
    articles_df["summary"] = (
        articles_df["summary"]
        .str.split("EVENT:")
        .str[1]
        .str.split("CONTEXT:")
        .str[0]
        .str.strip()
    )

    # Prepare text for embedding
    articles_df["text_to_embed"] = "query: " + articles_df["summary"]

    # Generate embeddings
    all_embeddings = generate_embeddings(articles_df)
    articles_df["embedding"] = all_embeddings

    # Optimize clustering parameters
    umap_params = {"n_neighbors": [10, 15, 20]}
    hdbscan_params = {
        "min_cluster_size": [5, 8, 10, 15],
        "min_samples": [2, 3, 5],
        "epsilon": [0.1, 0.2, 0.3],
    }

    best_params, best_score = optimize_clusters(
        all_embeddings, umap_params, hdbscan_params
    )
    logger.info(
        f"Best clustering parameters: {best_params} with score {best_score:.4f}"
    )

    # Cluster articles
    cluster_labels = cluster_articles(all_embeddings, best_params)
    articles_df["cluster"] = cluster_labels

    # Prepare clusters for processing
    clusters_ids = list(set(cluster_labels) - {-1})
    clusters_with_articles = []

    for cluster_id in clusters_ids:
        cluster_df = articles_df[articles_df["cluster"] == cluster_id]
        articles_ids = cluster_df["id"].tolist()
        clusters_with_articles.append(
            {"cluster_id": cluster_id, "articles_ids": articles_ids}
        )

    # Sort clusters by size (most articles to least)
    clusters_with_articles = sorted(
        clusters_with_articles, key=lambda x: len(x["articles_ids"]), reverse=True
    )
    logger.info(f"Found {len(clusters_with_articles)} clusters to process")

    # Process each cluster
    processed_stories = []

    for cluster in clusters_with_articles:
        result = process_story(cluster, events)
        if result:
            processed_stories.append(result)

    # Save results
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/brief_{date}.json", "w") as f:
        json.dump([story.model_dump() for story in processed_stories], f, indent=2)

    logger.info(
        f"Brief generation complete. Results saved to {output_dir}/brief_{date}.json"
    )

    # TODO: Generate final markdown brief from processed stories
    # This would involve calling another LLM function to synthesize the stories into a cohesive brief


if __name__ == "__main__":
    main()
