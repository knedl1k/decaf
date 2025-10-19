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

logging.basicConfig(format='LOG: %(message)s')
log = logging.getLogger(__name__)

SLEEP_MIN = 10 # sec
SLEEP_MAX = 30 # sec
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

def main():
    try:
        existing_files = {file.name for file in Path(dir_name).iterdir() if file.is_file()}
        print(f"Found {len(existing_files)} existing cards.")
    except FileNotFoundError:
        log.error(f"Directory '{dir_name}' not found.")
        return None

    with open(source_json) as f:
        d = json.load(f)
        # d = ijson.items(f, 'item')
        n = len(d)
        for i, item in enumerate(d):
            id = item['id']

            if any(id in file_name for file_name in existing_files):
                # print(f"{id} already present!")
                continue

            if item['image_status'] not in ['lowres', 'highres_scan']: # https://scryfall.com/docs/api/images
                log.warning(f"{id}: exists and has no image.")
                continue

            if i%250 == 0 and i>0: # naively try not to get banned
                t = random.randint(SLEEP_MIN, SLEEP_MAX)
                print(f"Processed {i} cards. Sleeping for {t}s.")
                time.sleep(t)
                print("Continuing...")

            name = item['scryfall_uri'] \
            .replace('https://scryfall.com/card/', '').replace('?utm_source=api', '').replace("/", "_")

            # https://scryfall.com/docs/api/layouts
            if item['layout'] in ["transform", "modal_dfc", "reversible_card", "double_faced_token", "meld"]: # has two sides # "art_series"
                faces = item['card_faces']
                for j in range(2):
                    side = faces[j]
                    try:
                        face_id = id + "_" +("back" if j else "front")
                        downloadImage(side['image_uris']['png'], name, face_id)
                        existing_files.add(f"{name}-{face_id}.png")
                    except Exception as e:
                        log.warning(f"{id}: is sus- {str(e)}")
            else:
                try:
                    downloadImage(item['image_uris']['png'], name, id)
                    existing_files.add(f"{name}-{id}.png")
                except Exception as e:
                    log.warning(f"{id}: is sus- {str(e)}")

if __name__ == "__main__":
    parse()
    main()