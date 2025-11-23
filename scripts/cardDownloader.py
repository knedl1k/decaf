#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# GNU General Public License v3.0
# @knedl1k 2025

import json
import logging
import subprocess
import time
import random
from pathlib import Path
import argparse
import re

logging.basicConfig(format='LOG: %(message)s')
log = logging.getLogger(__name__)

SLEEP_MIN = 10 # sec
SLEEP_MAX = 30 # sec
TWO_SIDED = ["transform", "modal_dfc", "reversible_card", "double_faced_token", "meld"] # "art_series"

dir_name = ""
source_json = ""

def wgetDownload(
        url,
        output_path=None,
        retries=1,
        timeout=30,
        show_progress=False
):
    cmd = ['wget', url]
    if output_path:
        cmd.extend(['-O', output_path])
    if retries:
        cmd.extend(['--tries', str(retries)])
    if timeout:
        cmd.extend(['--timeout', str(timeout)])
    if not show_progress:
        cmd.append('--quiet')

    try:
        subprocess.run(cmd, check=True)
        return {"success": True, "message": "Download completed successfully."}
    except subprocess.CalledProcessError as e:
        return {"success": False, "message": f"Download failed: {e}"}

def parse():
    global source_json, dir_name
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True, help="path to .csv file from Scryfall", type=str)
    parser.add_argument("-s", "--store", required=True, help="path where to store downloaded images", type=str)
    args = parser.parse_args()
    source_json = args.file
    dir_name = args.store

def downloadImage(image, name, id):
    try:
        wgetDownload(image, output_path=f"{dir_name}/{name}-{id}.png")
    except Exception as e:
        log.warning(id+": is sus-"+str(e))

def getDownloadedIDs(directory):
    existing_ids = set()
    uuid_regex = re.compile(r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})')

    try:
        for file in Path(directory).iterdir():
            if file.is_file():
                match = uuid_regex.search(file.name)
                if match:
                    existing_ids.add(match.group(1))
        print(f"Found {len(existing_ids)} downloaded IDs!")
        return existing_ids
    except FileNotFoundError:
        log.error(f"Directory '{directory}' not found.")
        return None

def main():
    existing_ids = getDownloadedIDs(dir_name)
    if existing_ids is None:
        return
        
    # https://scryfall.com/docs/api/bulk-data
    with open(source_json) as f:
        d = json.load(f)
        n = len(d)
        for i, item in enumerate(d):
            id = item['id']

            if id in existing_ids:
                continue

            if item['image_status'] not in ['lowres', 'highres_scan']: # https://scryfall.com/docs/api/images
                log.warning(f"{id}: exists but has no image available.")
                continue

            if i%250 == 0 and i>0: # naively try not to get banned
                t = random.randint(SLEEP_MIN, SLEEP_MAX)
                print(f"Processed {i}/{n} cards. Sleeping for {t} s.")
                time.sleep(t)
                print("Continuing...")

            name = item['scryfall_uri'] \
            .replace('https://scryfall.com/card/', '').replace('?utm_source=api', '').replace("/", "_")

            # https://scryfall.com/docs/api/layouts
            if item['layout'] and item['layout'] in TWO_SIDED:
                try:
                    faces = item['card_faces']
                    for j in range(2):
                        try:
                            face_id = id + "_" +("back" if j else "front")
                            downloadImage(faces[j]['image_uris']['png'], name, face_id)
                            existing_ids.add(id)
                        except Exception as e:
                            log.warning(f"{id}: should have 2 faces but some problem occured - {str(e)}")
                except Exception as e:
                    log.warning(f"{id}: should be double faced, but has no 'card_faces' attr.")
            if item['layout'] and item['layout'] == "art_series":
                log.warning(f"{id}: art_series cards are being omitted.")
            else:
                try:
                    downloadImage(item['image_uris']['png'], name, id)
                    existing_ids.add(id)
                except Exception as e:
                    log.warning(f"{id}: is sus- {str(e)}")

if __name__ == "__main__":
    parse()
    main()