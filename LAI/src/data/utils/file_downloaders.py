"""Module with downloaders of file from web."""
import json
import os
from urllib.parse import quote

import requests

from logger import log


class FileDownloader:
    """Class to download files."""

    @staticmethod
    def download_file(url: str, destination_path: str) -> None:
        """Download file with url."""
        response = requests.get(url)

        if response.status_code == 200:
            log.info(f"Saving file to {destination_path}")
            with open(destination_path, 'wb') as f:
                f.write(response.content)
        else:
            log.warning(f"URL {url} was not downloaded, status {response.status_code}")

    @staticmethod
    def download_pdf_from_orzeczenia_ms(url: str, destination_path: str) -> None:
        encoded_url = quote(url, safe=':/?&=')

        # Send a GET request to the URL with headers
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/58.0.3029.110 Safari/537.3"
        }
        response = requests.get(encoded_url, headers=headers)

        if response.status_code == 200:
            with open(destination_path, "wb") as file:
                file.write(response.content)
            log.info("File downloaded successfully.")
        else:
            log.warning("Failed to download the file. Status code:", response.status_code)


class OrzeczeniaDataset:
    """Class to download files from Orzeczenia dataset."""

    def __init__(self, dataset_path: str) -> None:
        """Initialize Orzeczenia dataset."""
        self.dataset_path = dataset_path

    def download_dataset(self, set_name: str, output_directory: str) -> None:
        """Parse and download .pdf files from dataset."""
        dataset = json.load(open(self.dataset_path, "r", encoding="utf8"))
        os.makedirs(output_directory, exist_ok=True)
        for orzeczenie in dataset[set_name]['orzeczenia']:
            url = orzeczenie["url"]
            output_file_name = f"{orzeczenie['id']}.pdf"
            destination_path = os.path.join(output_directory, output_file_name)
            FileDownloader.download_pdf_from_orzeczenia_ms(url, destination_path)
