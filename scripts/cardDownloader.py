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
    return None

def parse():
    global source_json, dir_name
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True, help="path to .csv file from Scryfall", type=str)
    parser.add_argument("-s", "--store", required=True, help="path where to store downloaded images", type=str)
    args = parser.parse_args()
    source_json = args.file
    dir_name = args.store
    return None

def downloadImage(image, name, id):
    try:
        wgetDownload(image, output_path=f"{dir_name}/{name}-{id}.png")
    except Exception as e:
        log.warning(id+": is sus-"+str(e))
    return None

def main():
    try:
        existing_files = {file.name for file in Path(dir_name).iterdir() if file.is_file()}
        print(f"Found {len(existing_files)} existing cards.")
    except FileNotFoundError:
        log.error(f"Directory '{dir_name}' not found.")
        return None

    with open(source_json) as f:
        d = json.load(f)
        n = len(d)
        for i in range(n):
            id = d[i]['id']

            if any(id in file.name for file in existing_files):
                print(f"{id} already present!")
                continue

            if i%250 == 0: # naively try not to get banned
                time.sleep(random.randint(SLEEP_MIN, SLEEP_MAX))

            name = d[i]['scryfall_uri'] \
            .replace('https://scryfall.com/card/', '').replace('?utm_source=api', '').replace("/", "_")

            if d[i]['layout'] in ["transform", "modal_dfc"]: # has two sides, we need both # "art_series",
                faces = d[i]['card_faces']
                for j in range(0, 2, 1):
                    side = faces[j]
                    try:
                        downloadImage(side['image_uris']['png'], name, id + "_" +("back" if j else "front"))
                    except Exception as e:
                        log.warning(id+": is sus- "+str(e))
            else:
                try:
                    downloadImage(d[i]['image_uris']['png'], name, id)
                except Exception as e:
                    log.warning(id+": is sus- "+str(e))
    return None

if __name__ == "__main__":
    parse()
    main()