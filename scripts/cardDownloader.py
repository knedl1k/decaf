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
DIR_NAME = "images"
source_json = "unique-artwork.json"

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
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True, help="Path to .csv file from Scryfall", type=str)
    args = parser.parse_args()
    source_json = args.file
    return None

def downloadImage(image, name, id):
    try:
        wgetDownload(image, output_path=f"{DIR_NAME}/{name}-{id}.png")
    except Exception as e:
        log.warning(id+": is sus-"+str(e))
    return None          

def main():
    with open(source_json) as f:
        d = json.load(f)
        n = len(d)
        for i in range(500): # TODO: change range
            id = d[i]['id']

            if any(id in file.name for file in Path(DIR_NAME).iterdir() if file.is_file()): # already downloaded
                print(f"{id} already present!")
                continue

            if i%50 == 0: # naively try not to get banned
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
