import asyncio
import os
import logging
import time
import zipfile

from urllib import parse, request


class Asset:
    assets_path = "assets"

    def __init__(self, asset_id: str, asset_path: str, storage_root: str):
        self.asset_id = asset_id

        self.asset_name = asset_path.split("/")[2]
        self.asset_url = f"{storage_root}{parse.quote(asset_path)}"

        self.asset_path = os.path.join(self.assets_path, self.asset_id)
        self.zip_path = f"{self.asset_path}.zip"
        self.dir_path = f"{self.asset_path}/input"

        os.makedirs(self.dir_path, exist_ok=True)

    async def download(self):
        logging.info(f"Downloading asset {self.asset_id}...")
        start_time = time.time()

        await asyncio.get_event_loop().run_in_executor(None, self.__download)

        end_time = time.time()
        logging.info(
            f"Asset downloaded successfully in {end_time - start_time:.2f} seconds"
        )

    async def unzip(self):
        logging.info(f"Extracting asset {self.asset_id}...")
        start_time = time.time()

        await asyncio.get_event_loop().run_in_executor(None, self.__unzip)

        end_time = time.time()
        logging.info(
            f"Asset extracted successfully in {end_time - start_time:.2f} seconds"
        )

    def __download(self):
        response = request.urlopen(self.asset_url)
        with open(self.zip_path, "wb") as f:
            f.write(response.read())

    def __unzip(self):
        with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
            zip_ref.extractall(self.dir_path)
        os.remove(self.zip_path)
