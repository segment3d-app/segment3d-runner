import asyncio
import json
import logging
import os
import requests

from dotenv import load_dotenv

from aio_pika import connect_robust
from aio_pika.abc import AbstractIncomingMessage

from assets import Asset, AssetUploadError
from .models import (
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


async def task(message: AbstractIncomingMessage):
    logging.info("Received message: %s", message.body)

    # ==== Parse message and create model instances

    data = json.loads(message.body.decode())

    asset_type = data["type"]
    asset = Asset(
        storage_root=storage_root,
        asset_id=data["asset_id"],
        images_path=data["photo_dir_url"],
        pcl_path=(None if asset_type != "lidar" else data["point_cloud_url"]),
    )

    gaussian_splatting = GaussianSplatting(asset_id=asset.asset_id)
    ptv3 = PTv3(asset_id=asset.asset_id, asset_type=asset_type)

    # ==== Download and unzip raw data from user

    try:
        await asset.download()
        await asset.unzip()

    except Exception as e:
        logging.error(f"Error processing asset {asset.asset_id}:")
        logging.error(str(e))
        return message.nack()

    # ==== Generate colmap for asset

    try:
        await gaussian_splatting.generate_colmap()

        if asset.exists("sparse/0/colmap.ply"):
            colmap_url = await asset.upload("sparse/0/colmap.ply", "pointcloud.ply")
            response = requests.patch(
                f"{api_root}/assets/pointcloud/{asset.asset_id}",
                headers={"Content-Type": "application/json"},
                data=json.dumps({"url": colmap_url}),
            )

            if response.status_code != 200:
                raise PatchError(response.reason)

    except ColmapError as e:
        logging.error(f"Failed generating colmap for asset {asset.asset_id}:")
        logging.error(e.args[0])
        return message.nack()

    except AssetUploadError as e:
        logging.error(f"Failed uploading colmap for asset {asset.asset_id}:")
        logging.error(e.args[0])
        return message.nack()

    except PatchError as e:
        logging.error(f"Failed patching colmap for asset {asset.asset_id}:")
        logging.error(e.args[0])
        return message.nack()

    except Exception as e:
        logging.error(
            f"Unknown error when processing colmap for asset {asset.asset_id}:"
        )
        logging.error(str(e))
        return message.nack()

    # ==== Generate gaussian splatting for asset

    try:
        await gaussian_splatting.generate_gaussian_splatting()

        if asset.exists("output/point_cloud/iteration_7000/scene_point_cloud.ply"):
            gaussian_url = await asset.upload(
                "output/point_cloud/iteration_7000/scene_point_cloud.ply", "3dgs.ply"
            )
            response = requests.patch(
                f"{api_root}/assets/gaussian/{asset.asset_id}",
                headers={"Content-Type": "application/json"},
                data=json.dumps({"url": gaussian_url}),
            )

            if response.status_code != 200:
                raise PatchError(response.reason)

    except GaussianSplattingError as e:
        logging.error(
            f"Failed generating gaussian splatting for asset {asset.asset_id}:"
        )
        logging.error(e.args[0])
        return message.nack()

    except AssetUploadError as e:
        logging.error(
            f"Failed uploading gaussian splatting for asset {asset.asset_id}:"
        )
        logging.error(e.args[0])
        return message.nack()

    except PatchError as e:
        logging.error(f"Failed patching gaussian splatting for asset {asset.asset_id}:")
        logging.error(e.args[0])
        return message.nack()

    except Exception as e:
        logging.error(
            f"Unknown error when processing gaussian splatting for asset {asset.asset_id}:"
        )
        logging.error(str(e))
        return message.nack()

    # ==== Process object segmentation with PTv3

    try:
        await ptv3.process()

        if asset.exists("segmentation/ptv3.ply"):
            ptv3_url = await asset.upload("segmentation/ptv3.ply", "ptv3.ply")
            response = requests.patch(
                f"{api_root}/assets/ptv3/{asset.asset_id}",
                headers={"Content-Type": "application/json"},
                data=json.dumps({"url": ptv3_url}),
            )

            if response.status_code != 200:
                raise PatchError(response.reason)

    except PTv3ConvertError as e:
        logging.error(f"Failed converting ptv3 dataset for asset {asset.asset_id}:")
        logging.error(e.args[0])
        return message.nack()

    except PTv3PreprocessError as e:
        logging.error(f"Failed preprocessing ptv3 dataset for asset {asset.asset_id}:")
        logging.error(e.args[0])
        return message.nack()

    except PTv3InferenceError as e:
        logging.error(f"Failed inferring ptv3 dataset for asset {asset.asset_id}:")
        logging.error(e.args[0])
        return message.nack()

    except PTv3ReconstructionError as e:
        logging.error(f"Failed reconstructing ptv3 dataset for asset {asset.asset_id}:")
        logging.error(e.args[0])
        return message.nack()

    except AssetUploadError as e:
        logging.error(f"Failed uploading ptv3 segmentation for asset {asset.asset_id}:")
        logging.error(e.args[0])
        return message.nack()

    except PatchError as e:
        logging.error(f"Failed patching ptv3 segmentation for asset {asset.asset_id}:")
        logging.error(e.args[0])
        return message.nack()

    except Exception as e:
        logging.error(
            f"Unknown error when processing ptv3 segmentation for asset {asset.asset_id}:"
        )
        logging.error(str(e))
        return message.nack()

    # asset.clear()
    # await message.ack()


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

    await queue.consume(task)

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
