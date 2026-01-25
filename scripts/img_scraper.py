#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# GNU General Public License v3.0
# @knedl1k 2026

import json
import time
import argparse
import re
import sys
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Scryfall requests 50-100ms delay between requests.
# Being slightly slower avoids IP bans.
DELAY_PER_REQUEST = 0.1
TWO_SIDED = ["transform", "modal_dfc", "reversible_card", "double_faced_token", "meld"]


def create_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def download_image(session, url, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        response = session.get(url, stream=True, timeout=10)
        response.raise_for_status()

        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Respect Scryfall API limits
        time.sleep(DELAY_PER_REQUEST)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def get_downloaded_ids(directory):
    existing_ids = set()
    uuid_regex = re.compile(r"([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})")

    dir_path = Path(directory)
    if not dir_path.exists():
        dir_path.mkdir(parents=True)
        return existing_ids

    print(f"Scanning {directory} for existing files...")
    count = 0
    for file in dir_path.iterdir():
        if file.is_file():
            match = uuid_regex.search(file.name)
            if match:
                existing_ids.add(match.group(1))
                count += 1

    print(f"Found {count} already downloaded IDs.")
    return existing_ids


def parse_args():
    parser = argparse.ArgumentParser(description="download MTG card images from Scryfall bulk JSON.")
    parser.add_argument("-f", "--file", required=True, help="path to .json file from Scryfall", type=str)
    parser.add_argument("-s", "--store", required=True, help="directory where to store downloaded images", type=str)
    return parser.parse_args()


def clean_name(name_uri):
    return name_uri.replace("https://scryfall.com/card/", "").replace("?utm_source=api", "").replace("/", "_")


def main():
    args = parse_args()
    source_json = Path(args.file)
    store_dir = Path(args.store)

    if not source_json.exists():
        print(f"Error: JSON file '{source_json}' not found.")
        sys.exit(1)

    existing_ids = get_downloaded_ids(store_dir)
    session = create_session()

    print("Loading JSON data...")
    try:
        with open(source_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON. Check if the file is valid.")
        sys.exit(1)

    total_cards = len(data)
    print(f"Starting process for {total_cards} cards...")

    for i, item in enumerate(data):
        if i % 100 == 0:
            print(f"Progress: {i}/{total_cards} ({(i / total_cards) * 100:.1f}%)")

        card_id = item.get("id")
        if not card_id or card_id in existing_ids:
            continue

        # Skip if no image is available (e.g. textless placeholders, very new cards)
        if item.get("image_status") == "missing":
            continue

        name_prefix = clean_name(item.get("scryfall_uri", "unknown"))
        layout = item.get("layout")

        if layout in TWO_SIDED:
            if "card_faces" not in item:
                print(f"Warning: {card_id} is {layout} but missing 'card_faces'. Skipping.")
                continue

            faces = item["card_faces"]
            # Some double faced cards (meld) might not have images on faces in JSON, but on main object
            # Use logic: if faces have image_uris, use them.
            try:
                for j, face in enumerate(faces):
                    if "image_uris" in face:
                        face_suffix = "back" if j else "front"
                        face_id = f"{card_id}_{face_suffix}"
                        target_file = store_dir / f"{name_prefix}-{face_id}.png"
                        download_image(session, face["image_uris"]["png"], target_file)
                    else:
                        # Fallback for some weird layouts where image is on parent
                        if "image_uris" in item:
                            target_file = store_dir / f"{name_prefix}-{card_id}.png"
                            download_image(session, item["image_uris"]["png"], target_file)
                            break
                existing_ids.add(card_id)
            except Exception as e:
                print(f"Error processing faces for {card_id}: {e}")

        elif layout == "art_series":
            # print(f"Skipping art_series: {card_id}")
            continue

        else:
            if "image_uris" in item and "png" in item["image_uris"]:
                target_file = store_dir / f"{name_prefix}-{card_id}.png"
                download_image(session, item["image_uris"]["png"], target_file)
                existing_ids.add(card_id)
            else:
                # Some cards don't have images (tokens sometimes, or placeholders)
                pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting.")
        sys.exit(0)
