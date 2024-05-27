import asyncio
import json
import logging
import os
import requests

from dotenv import load_dotenv

from aio_pika import connect_robust
from aio_pika.abc import AbstractIncomingMessage

from assets import Asset, AssetUploadError
from models import ColmapError, GaussianSplatting, GaussianSplattingError


async def task(message: AbstractIncomingMessage):
    logging.info("Received message: %s", message.body)

    data = json.loads(message.body.decode())
    asset = Asset(
        asset_id=data["asset_id"],
        asset_path=data["photo_dir_url"],
        storage_root=storage_root,
    )

    try:
        await asset.download()
        await asset.unzip()

    except Exception as e:
        logging.error(f"Error processing asset {asset.asset_id}:")
        logging.error(str(e))
        return message.nack()

    # ==========

    try:
        gaussian_splatting = GaussianSplatting(asset_id=asset.asset_id)
        await gaussian_splatting.generate()

    except ColmapError as e:
        logging.error(f"Failed generating colmap for asset {asset.asset_id}:")
        logging.error(e.args[0])
        return message.nack()

    except GaussianSplattingError as e:
        logging.error(
            f"Failed generating gaussian splatting for asset {asset.asset_id}:"
        )
        logging.error(e.args[0])
        return message.nack()

    except Exception as e:
        logging.error(
            f"Error processing gaussian splatting for asset {asset.asset_id}:"
        )
        logging.error(str(e))
        return message.nack()

    # ==========

    try:
        pointcloud_url = await asset.upload("sparse/0/points3D.ply", "pointcloud.ply")

        response = requests.patch(
            f"{api_root}/assets/pointcloud/{asset.asset_id}",
            headers={"Content-Type": "application/json"},
            data={"url": pointcloud_url},
        )

        if response.status_code != 200:
            raise Exception(response.reason)

        gaussian_url = await asset.upload(
            "output/point_cloud/iteration_7000/point_cloud.ply", "3dgs.ply"
        )

        response = requests.patch(
            f"{api_root}/assets/gaussian/{asset.asset_id}",
            headers={"Content-Type": "application/json"},
            data={"url": gaussian_url},
        )

        if response.status_code != 200:
            raise Exception(response.reason)

    except AssetUploadError as e:
        logging.error(f"Failed uploading asset {asset.asset_id}:")
        logging.error(e.args[0])
        return await message.nack()

    except Exception as e:
        logging.error(f"Error uploading asset {asset.asset_id}:")
        logging.error(str(e))
        return await message.nack()

    # ==========

    asset.clear()
    await message.ack()


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
