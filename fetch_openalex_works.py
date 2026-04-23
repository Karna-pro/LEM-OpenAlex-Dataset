# -*- coding: utf-8 -*-
"""
OpenAlex data collection pipeline
================================

A lightweight and reproducible script for:
1. Retrieving works from the OpenAlex API
2. Reconstructing abstracts from abstract_inverted_index
3. Extracting field/subfield labels
4. Exporting data to CSV for downstream multi-label classification research

"""

import argparse
import json
import time
from typing import Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm


BASE_URL = "https://api.openalex.org/works"
DEFAULT_PER_PAGE = 200
DEFAULT_SLEEP = 0.2
DEFAULT_TIMEOUT = 30
MAX_PER_PAGE = 200


def reconstruct_abstract(abstract_inverted_index: Optional[Dict]) -> str:
    """
    Reconstruct abstract text from OpenAlex abstract_inverted_index.

    Parameters
    ----------
    abstract_inverted_index : dict or None
        OpenAlex inverted abstract index:
        {
            "word1": [0, 3],
            "word2": [1],
            ...
        }

    Returns
    -------
    str
        Reconstructed abstract text. Returns empty string if unavailable.
    """
    if not abstract_inverted_index or not isinstance(abstract_inverted_index, dict):
        return ""

    index_to_word = {}

    for word, positions in abstract_inverted_index.items():
        if not isinstance(positions, list):
            continue
        for pos in positions:
            if isinstance(pos, int):
                index_to_word[pos] = word

    if not index_to_word:
        return ""

    abstract_words = [index_to_word[idx] for idx in sorted(index_to_word.keys())]
    return " ".join(abstract_words)


def extract_labels(topics: Optional[List[Dict]]) -> (List[str], List[str]):
    """
    Extract L1 (field) and L2 (subfield) labels from OpenAlex topics.

    Parameters
    ----------
    topics : list of dict or None

    Returns
    -------
    tuple[list[str], list[str]]
        Sorted field labels and sorted subfield labels.
    """
    l1_labels = set()
    l2_labels = set()

    if not topics or not isinstance(topics, list):
        return [], []

    for topic in topics:
        if not isinstance(topic, dict):
            continue

        field_info = topic.get("field")
        subfield_info = topic.get("subfield")

        if isinstance(field_info, dict):
            field_name = field_info.get("display_name")
            if isinstance(field_name, str) and field_name.strip():
                l1_labels.add(field_name.strip())

        if isinstance(subfield_info, dict):
            subfield_name = subfield_info.get("display_name")
            if isinstance(subfield_name, str) and subfield_name.strip():
                l2_labels.add(subfield_name.strip())

    return sorted(l1_labels), sorted(l2_labels)


def build_filter_string(
    from_year: int,
    require_abstract: bool = True,
    require_field: bool = True,
    require_subfield: bool = True,
) -> str:
    """
    Build OpenAlex API filter string.
    """
    filters = []

    if require_abstract:
        filters.append("has_abstract:true")
    if require_field:
        filters.append("topics.field.id:!null")
    if require_subfield:
        filters.append("topics.subfield.id:!null")

    filters.append(f"from_publication_date:{from_year}-01-01")

    return ",".join(filters)


def request_page(
    session: requests.Session,
    email: str,
    cursor: str,
    per_page: int,
    filter_str: str,
    select_fields: str,
    timeout: int,
) -> Dict:
    """
    Request one page from OpenAlex API.
    """
    params = {
        "filter": filter_str,
        "select": select_fields,
        "per_page": per_page,
        "cursor": cursor,
        "mailto": email,
    }

    response = session.get(BASE_URL, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_openalex_works(
    email: str,
    output_file: str,
    target_size: int,
    from_year: int,
    per_page: int = DEFAULT_PER_PAGE,
    sleep_seconds: float = DEFAULT_SLEEP,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Fetch OpenAlex works and export to CSV.

    Parameters
    ----------
    email : str
        Contact email for OpenAlex polite pool.
    output_file : str
        Output CSV file path.
    target_size : int
        Number of records to collect.
    from_year : int
        Lower bound publication year.
    per_page : int
        OpenAlex page size (max 200).
    sleep_seconds : float
        Sleep time between requests.
    timeout : int
        Request timeout in seconds.
    max_retries : int
        Number of retries per failed request.

    Returns
    -------
    pandas.DataFrame
        Collected dataframe.
    """
    if per_page > MAX_PER_PAGE:
        raise ValueError(f"per_page must be <= {MAX_PER_PAGE}")
    if target_size <= 0:
        raise ValueError("target_size must be a positive integer")

    filter_str = build_filter_string(from_year=from_year)
    select_fields = "id,title,abstract_inverted_index,publication_year,topics"

    data_list = []
    cursor = "*"
    processed_count = 0

    print("=" * 80)
    print("OpenAlex data collection pipeline")
    print(f"Target size      : {target_size}")
    print(f"From year        : {from_year}")
    print(f"Per page         : {per_page}")
    print(f"Output file      : {output_file}")
    print("=" * 80)

    session = requests.Session()
    progress_bar = tqdm(total=target_size, desc="Collecting works", ncols=100)

    while processed_count < target_size:
        page_data = None

        for attempt in range(1, max_retries + 1):
            try:
                page_data = request_page(
                    session=session,
                    email=email,
                    cursor=cursor,
                    per_page=per_page,
                    filter_str=filter_str,
                    select_fields=select_fields,
                    timeout=timeout,
                )
                break
            except requests.RequestException as e:
                print(f"[Warning] Request failed (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    time.sleep(min(5 * attempt, 15))
                else:
                    print("[Error] Max retries reached. Stopping early.")
                    page_data = None

        if page_data is None:
            break

        results = page_data.get("results", [])
        meta = page_data.get("meta", {})

        if not results:
            print("[Info] No more results returned by OpenAlex.")
            break

        for work in results:
            if processed_count >= target_size:
                break

            if not isinstance(work, dict):
                continue

            title = work.get("title") or ""
            abstract_inv = work.get("abstract_inverted_index")
            publication_year = work.get("publication_year")
            topics = work.get("topics", [])

            l1_labels, l2_labels = extract_labels(topics)
            if not l1_labels or not l2_labels:
                continue

            abstract_text = reconstruct_abstract(abstract_inv)

            record = {
                "openalex_id": work.get("id", ""),
                "title": title,
                "abstract": abstract_text,
                "l1_fields": json.dumps(l1_labels, ensure_ascii=False),
                "l2_subfields": json.dumps(l2_labels, ensure_ascii=False),
                "publication_year": publication_year,
            }

            data_list.append(record)
            processed_count += 1
            progress_bar.update(1)

        cursor = meta.get("next_cursor")
        if not cursor:
            print("[Info] No next_cursor found. Stopping.")
            break

        time.sleep(sleep_seconds)

    progress_bar.close()

    df = pd.DataFrame(data_list)
    df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"\nDone. Saved {len(df)} records to: {output_file}")
    return df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch OpenAlex works and export a CSV for downstream text classification research."
    )
    parser.add_argument(
        "--email",
        type=str,
        required=True,
        help="Your contact email for OpenAlex API polite pool.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="openalex_works.csv",
        help="Output CSV filename.",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=1000,
        help="Number of works to collect.",
    )
    parser.add_argument(
        "--from_year",
        type=int,
        default=2018,
        help="Collect works published from this year onward.",
    )
    parser.add_argument(
        "--per_page",
        type=int,
        default=DEFAULT_PER_PAGE,
        help="OpenAlex page size (maximum 200).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP,
        help="Sleep time between API requests.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Timeout (seconds) for each request.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum retry attempts for failed requests.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    fetch_openalex_works(
        email=args.email,
        output_file=args.output,
        target_size=args.target_size,
        from_year=args.from_year,
        per_page=args.per_page,
        sleep_seconds=args.sleep,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()