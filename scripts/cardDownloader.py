#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# GNU General Public License v3.0
# @knedl1k 2025

import json
import logging
import subprocess
import time

logging.basicConfig(format='LOG: %(message)s')
log = logging.getLogger(__name__)

SLEEP_MIN = 10 # sec
SLEEP_MAX = 30 # sec

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

def help():
    pass

def main():
    with open('unique-artwork.json') as f:
        d = json.load(f)
        n = len(d)
        for i in range(12):
            id = d[i]['id']
            if i%50 == 0: # naivly try to not get banned
                time.sleep(random.randint(SLEEP_MIN, SLEEP_MAX))

            name = d[i]['scryfall_uri'] \
            .replace('https://scryfall.com/card/', '').replace('?utm_source=api', '').replace("/", "_")
            try:
                image = d[i]['image_uris']['png']
                wgetDownload(image, output_path=f"images/{name}-{id}.png")
            except:
                log.warning(id+": is sus")

if __name__ == "__main__":
    main()
