import asyncio
import os
import logging
import requests
import shutil
import time
import zipfile

from pathlib import Path
from urllib import parse, request


class AssetUploadError(Exception):
    pass


class Asset:
    assets_path = "assets"

    def __init__(
        self, asset_id: str, images_path: str, pcl_path: str, storage_root: str
    ):
        self.storage_root = storage_root
        self.asset_id = asset_id
        self.asset_path = os.path.join(self.assets_path, self.asset_id)

        self.images_url = f"{storage_root}{parse.quote(images_path)}"
        self.zip_path = f"{self.asset_path}.zip"
        self.dir_path = f"{self.asset_path}/input"

        os.makedirs(self.dir_path, exist_ok=True)

        if pcl_path is not None:
            self.pcl_url = f"{storage_root}{parse.quote(pcl_path)}"
            self.pcl_path = f"{self.asset_path}/input/lidar.ply"

    def exists(self, path: str):
        return Path(self.asset_path, path).exists()

    async def download(self):
        logging.info(f"Downloading asset {self.asset_id}...")
        start_time = time.time()

        await asyncio.get_event_loop().run_in_executor(None, self.__download_images)

        if self.pcl_url:
            await asyncio.get_event_loop().run_in_executor(None, self.__download_pcl)

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

    async def upload(self, source_path: str, target_path: str):
        logging.info(f"Uploading asset {self.asset_id}/{target_path}...")
        start_time = time.time()

        response = await asyncio.get_event_loop().run_in_executor(
            None, self.__upload, source_path, target_path
        )

        if response.status_code != 200:
            raise AssetUploadError(response.reason)

        end_time = time.time()
        logging.info(
            f"Asset uploaded successfully in {end_time - start_time:.2f} seconds"
        )

        return response.json()["url"][0]

    def clear(self):
        shutil.rmtree(self.asset_path)

    def __download_images(self):
        response = request.urlopen(self.images_url)
        with open(self.zip_path, "wb") as f:
            f.write(response.read())

    def __download_pcl(self):
        response = request.urlopen(self.pcl_url)
        with open(self.pcl_path, "wb") as f:
            f.write(response.read())

    def __unzip(self):
        with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
            zip_ref.extractall(self.dir_path)
        os.remove(self.zip_path)

    def __upload(self, source_path: str, target_path: str):
        source = os.path.join("assets", self.asset_id, source_path)
        files = {"file": (target_path, open(source, "rb"))}
        data = {"folder": self.asset_id}
        return requests.post(f"{self.storage_root}/upload", data=data, files=files)
