import asyncio
import json
import os
import requests
import shutil
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

        if images_path is not None:
            self.images_url = (
                f"{storage_root}{parse.quote(images_path)}?isDownload=true"
            )
            self.zip_path = f"{self.asset_path}.zip"
            self.dir_path = f"{self.asset_path}/input"

            os.makedirs(self.dir_path, exist_ok=True)

        if pcl_path is not None:
            self.pcl_url = f"{storage_root}{parse.quote(pcl_path)}?isDownload=true"
            self.pcl_path = f"{self.asset_path}/input/lidar.ply"
        else:
            self.pcl_url = None

    def exists(self, path: str):
        return Path(self.asset_path, path).exists()
    
    def read_json(self, path: str):
        with open(os.path.join(self.asset_path, path), 'r') as file:
            data = json.load(file)
            return data

    async def download(self):
        await asyncio.get_event_loop().run_in_executor(None, self.__download_images)
        if self.pcl_url:
            await asyncio.get_event_loop().run_in_executor(None, self.__download_pcl)

    async def unzip(self):
        await asyncio.get_event_loop().run_in_executor(None, self.__unzip)

    async def upload(self, source_path: str, target_path: str):
        response = await asyncio.get_event_loop().run_in_executor(
            None, self.__upload, source_path, target_path
        )

        if response.status_code != 200:
            raise AssetUploadError(response.reason)

        return response.json()["url"][0]

    async def upload_folder(self, source_folder: str, target_folder: str):
        source_folder_path = os.path.join(self.asset_path, source_folder)
        for _, _, files in os.walk(source_folder_path):
            for file in files:
                source_path = os.path.join(source_folder, file)

                response = await asyncio.get_event_loop().run_in_executor(
                    None, self.__upload, source_path, file, "saga"
                )

                if response.status_code != 200:
                    raise AssetUploadError(response.reason)

        return f"files/{self.asset_id}/{target_folder}"

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

    def __upload(self, source_path: str, target_path: str, subfolder_path: str = None):
        source = os.path.join("assets", self.asset_id, source_path)

        target = self.asset_id
        if subfolder_path is not None:
            target += "/" + subfolder_path

        with open(source, "rb") as file:
            files = {"file": (target_path, file)}
            data = {"folder": target}
            return requests.post(f"{self.storage_root}/upload", data=data, files=files)
