"""Module with downloaders of file from web."""

from urllib.parse import quote
from venv import logger

import requests


class FileDownloader:
    """Class to download files."""

    @staticmethod
    def download_file(url: str, destination_path: str) -> None:
        """Download file with url."""
        response = requests.get(url)

        if response.status_code == 200:
            logger.info(f"Saving file to {destination_path}")
            with open(destination_path, 'wb') as f:
                f.write(response.content)
        else:
            logger.warning(f"URL {url} was not downloaded, status {response.status_code}")

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
            logger.info("File downloaded successfully.")
            print(1)
        else:
            logger.warning("Failed to download the file. Status code:", response.status_code)
