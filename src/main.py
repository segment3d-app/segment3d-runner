import asyncio
import json
import logging
import os
import requests
import time

from dotenv import load_dotenv

from aio_pika import connect_robust
from aio_pika.abc import AbstractIncomingMessage

from assets import Asset, AssetUploadError
from models import (
    ColmapError,
    GaussianSplatting,
    GaussianSplattingError,
    PTv3,
    PTv3ConvertError,
    PTv3InferenceError,
    PTv3PreprocessError,
    PTv3ReconstructionError,
)


class PatchError(Exception):
    pass


async def process_task(message: AbstractIncomingMessage):
    logging.info("Received message:")

    # ==== Parse message and create model instances

    data = json.loads(message.body.decode())

    logging.info(f"└- Asset ID: {data['asset_id']}")
    logging.info(f"└- Asset type: {data['type']}")
    logging.info(f"└- Photos URL: {data['photo_dir_url']}")
    if "point_cloud_url" in data:
        logging.info(f"└- Point cloud URL: {data['point_cloud_url']}")

    asset_type = data["type"]
    asset = Asset(
        storage_root=storage_root,
        asset_id=data["asset_id"],
        images_path=data["photo_dir_url"],
        pcl_path=(None if asset_type != "lidar" else data["point_cloud_url"]),
    )

    ptv3 = PTv3(asset_id=asset.asset_id, asset_type=asset_type)
    gaussian_splatting = GaussianSplatting(
        asset_id=asset.asset_id, asset_type=asset_type
    )

    try:
        # Download and unzip raw data from user
        await download_asset(asset)
        await unzip_asset(asset)

        # Generate point cloud for asset
        await generate_pointcloud(asset, gaussian_splatting)

        # Generate gaussian splatting for asset
        await generate_gaussian(asset, gaussian_splatting)

        # Process PTv3
        await process_ptv3(asset, ptv3)

    except:
        logging.error("")
        await message.nack()

    # asset.clear()
    # await message.ack()


async def download_asset(asset: Asset):
    logging.info(f"Downloading asset {asset.asset_id}...")
    start_time = time.time()

    try:
        await asset.download()

    except Exception as e:
        logging.error(f"└- Error downloading asset:")
        logging.error(str(e))
        raise Exception()

    duration = time.time() - start_time
    logging.info(f"└- Asset downloaded successfully in {duration:.2f} seconds")


async def unzip_asset(asset: Asset):
    logging.info(f"Extracting asset {asset.asset_id}...")
    start_time = time.time()

    try:
        await asset.unzip()

    except Exception as e:
        logging.error(f"└- Error extracting asset:")
        logging.error(str(e))
        raise Exception()

    duration = time.time() - start_time
    logging.info(f"└- Asset extracted successfully in {duration:.2f} seconds")


async def generate_pointcloud(asset: Asset, gaussian_splatting: GaussianSplatting):
    if asset.exists("sparse/0/pointcloud.ply"):
        logging.info(f"[SKIPPED] Generating pointcloud for asset {asset.asset_id}")
        return

    logging.info(f"Generating pointcloud for asset {asset.asset_id}...")
    start_time = time.time()

    try:
        await gaussian_splatting.generate_pointcloud()

        if not asset.exists("sparse/0/pointcloud.ply"):
            raise ColmapError("pointcloud.ply not found")

        colmap_url = await asset.upload("sparse/0/pointcloud.ply", "pointcloud.ply")
        response = requests.patch(
            f"{api_root}/assets/pointcloud/{asset.asset_id}",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"url": colmap_url}),
        )

        if response.status_code != 200:
            raise PatchError(response.reason)

    except ColmapError as e:
        logging.error(f"└- Failed generating pointcloud:")
        logging.error(e.args[0])
        raise Exception()

    except AssetUploadError as e:
        logging.error(f"└- Failed uploading pointcloud:")
        logging.error(e.args[0])
        raise Exception()

    except PatchError as e:
        logging.error(f"└- Failed patching pointcloud:")
        logging.error(e.args[0])
        raise Exception()

    except Exception as e:
        logging.error(f"└- Unknown error when generating pointcloud:")
        logging.error(str(e))
        raise Exception()

    duration = time.time() - start_time
    logging.info(f"└- Pointcloud generated successfully in {duration:.2f} seconds")


async def generate_gaussian(asset: Asset, gaussian_splatting: GaussianSplatting):
    if asset.exists("output/point_cloud/iteration_7000/scene_point_cloud.ply"):
        logging.info(f"[SKIPPED] Generating gaussian for asset {asset.asset_id}")
        return

    logging.info(f"Generating gaussian for asset {asset.asset_id}...")
    start_time = time.time()

    try:
        await gaussian_splatting.generate_gaussian()

        if not asset.exists("output/point_cloud/iteration_7000/scene_point_cloud.ply"):
            raise GaussianSplattingError("scene_point_cloud.ply not found")

        gaussian_url = await asset.upload(
            "output/point_cloud/iteration_7000/scene_point_cloud.ply",
            "3dgs.ply",
        )

        response = requests.patch(
            f"{api_root}/assets/gaussian/{asset.asset_id}",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"url": gaussian_url}),
        )

        if response.status_code != 200:
            raise PatchError(response.reason)

    except GaussianSplattingError as e:
        logging.error(f"└- Failed generating gaussian:")
        logging.error(e.args[0])
        raise Exception

    except AssetUploadError as e:
        logging.error(f"└- Failed uploading gaussian:")
        logging.error(e.args[0])
        raise Exception()

    except PatchError as e:
        logging.error(f"└- Failed patching gaussian:")
        logging.error(e.args[0])
        raise Exception()

    except Exception as e:
        logging.error(f"└- Unknown error when generating gaussian:")
        logging.error(str(e))
        raise Exception()

    duration = time.time() - start_time
    logging.info(f"└- Gaussian generated successfully in {duration:.2f} seconds")


async def process_ptv3(asset: Asset, ptv3: PTv3):
    if asset.exists("segmentation/ptv3.ply"):
        logging.info(f"[SKIPPED] Processing PTv3 for asset {asset.asset_id}")
        return

    logging.info(f"Processing PTv3 for asset {asset.asset_id}...")

    try:
        # Convert
        logging.info(f"└- Converting PTv3...")
        start_time = time.time()
        await ptv3.convert()

        if not asset.exists("data/scene/scene/scene_alignmentAngle.txt"):
            raise PTv3ConvertError("scene_alignmentAngle.txt not found")

        duration = time.time() - start_time
        logging.info(f"└--- PTv3 converted successfully in {duration:.2f} seconds")

        # Preprocess
        logging.info(f"└- Preprocessing PTv3...")
        start_time = time.time()
        await ptv3.preprocess()

        if not asset.exists("data/scene/scene/scene.pth"):
            raise PTv3PreprocessError("scene.pth not found")

        duration = time.time() - start_time
        logging.info(f"└--- PTv3 preprocessed successfully in {duration:.2f} seconds")

        # Inferrence
        logging.info(f"└- Inferring PTv3...")
        start_time = time.time()
        await ptv3.infer()

        if not asset.exists("data/result/scene.npy"):
            raise PTv3InferenceError("scene.npy not found")

        duration = time.time() - start_time
        logging.info(f"└--- PTv3 inferred successfully in {duration:.2f} seconds")

        # Reconstruction
        logging.info(f"└- Reconstructing PTv3...")
        start_time = time.time()
        await ptv3.reconstruct()

        if not asset.exists("segmentation/ptv3.ply"):
            raise PTv3ReconstructionError("ptv3.ply not found")

        duration = time.time() - start_time
        logging.info(f"└--- PTv3 reconstructed successfully in {duration:.2f} seconds")

        # Upload and patch result
        ptv3_url = await asset.upload("segmentation/ptv3.ply", "ptv3.ply")
        response = requests.patch(
            f"{api_root}/assets/ptv3/{asset.asset_id}",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"url": ptv3_url}),
        )

        if response.status_code != 200:
            raise PatchError(response.reason)

    except PTv3ConvertError as e:
        logging.error(f"└- Failed converting PTv3:")
        logging.error(e.args[0])
        raise Exception()

    except PTv3PreprocessError as e:
        logging.error(f"└- Failed preprocessing PTv3:")
        logging.error(e.args[0])
        raise Exception()

    except PTv3InferenceError as e:
        logging.error(f"└- Failed inferring PTv3:")
        logging.error(e.args[0])
        raise Exception()

    except PTv3ReconstructionError as e:
        logging.error(f"└- Failed reconstructing PTv3:")
        logging.error(e.args[0])
        raise Exception()

    except AssetUploadError as e:
        logging.error(f"└- Failed uploading PTv3:")
        logging.error(e.args[0])
        raise Exception()

    except PatchError as e:
        logging.error(f"└- Failed patching PTv3:")
        logging.error(e.args[0])
        raise Exception()

    except Exception as e:
        logging.error(f"└- Unknown error when processing PTv3:")
        logging.error(str(e))
        raise Exception()

    duration = time.time() - start_time
    logging.info(f"└- PTv3 processed successfully in {duration:.2f} seconds")


async def main():
    connection = await connect_robust(
        host=os.getenv("RABBITMQ_HOST"),
        port=int(os.getenv("RABBITMQ_PORT")),
        login=os.getenv("RABBITMQ_USER"),
        password=os.getenv("RABBITMQ_PASSWORD"),
    )

    channel = await connection.channel()
    await channel.set_qos(prefetch_count=1)

    queue_name = os.getenv("RABBITMQ_QUEUE")
    queue = await channel.declare_queue(queue_name, durable=True)

    await queue.consume(process_task)

    try:
        logging.info("Listening for messages. Press CTRL+C to exit.")
        await asyncio.Future()
    finally:
        await connection.close()


if __name__ == "__main__":
    load_dotenv()

    api_root = os.getenv("API_ROOT") + "/api"
    storage_root = os.getenv("STORAGE_ROOT")

    log_format = "%(asctime)s [%(levelname)s]: (%(name)s) %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

    asyncio.run(main())
