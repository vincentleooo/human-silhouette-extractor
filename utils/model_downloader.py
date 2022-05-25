import os
import requests
import logging
import sys
from tqdm import tqdm

def download(url: str, dest_folder: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist

    filename = url.split('/')[-1].replace(" ", "_")  # be careful with file names
    file_path = os.path.join(dest_folder, filename)

    r = requests.get(url, stream=True)
    if r.ok:
        logging.info(f"Saving to {os.path.abspath(file_path)}")
        with open(file_path, 'wb') as f:
            pbar = tqdm(unit='B', desc='Downloading the checkpoint', unit_scale=True)
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
                pbar.update(1024 * 1024)
    else:  # HTTP status code 4XX/5XX
        logging.error("Download failed: status code {}\n{}".format(r.status_code, r.text))

if __name__ == "__main__":
    raise NotImplementedError