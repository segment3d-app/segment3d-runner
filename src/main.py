import asyncio
import json
import logging
import os

from dotenv import load_dotenv

from aio_pika import connect_robust
from aio_pika.abc import AbstractIncomingMessage

from assets import Asset
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

    # ==========

    try:
        gaussian_splatting = GaussianSplatting(asset_id=asset.asset_id)
        await gaussian_splatting.generate()

    except ColmapError as e:
        logging.error(f"Failed generating colmap for asset {asset.asset_id}:")
        logging.error(e.args[0])

    except GaussianSplattingError as e:
        logging.error(
            f"Failed generating gaussian splatting for asset {asset.asset_id}:"
        )
        logging.error(e.args[0])

    except Exception as e:
        logging.error(
            f"Error processing gaussian splatting for asset {asset.asset_id}:"
        )
        logging.error(str(e))

    # ==========

    # logging.info(f"Uploading colmap result for asset {asset_id}...")
    # response = await asyncio.get_event_loop().run_in_executor(
    #     None,
    #     upload,
    #     os.path.join("assets", f"{asset_id}/sparse/0/points3D.ply"),
    #     asset_id,
    #     "colmap.py",
    # )

    # if response.status_code == 200:
    #     url = f"{storage_root}/files/{asset_id}/colmap.ply"
    #     requests.patch(
    #         f"{api_root}/assets/pointcloud/{asset_id}",
    #         headers={"Content-Type": "application/json"},
    #         data=json.dumps({"url": url}),
    #     )
    #     logging.info(f"Colmap result uploaded successfully!")
    # else:
    #     logging.error(f"Error uploading colmap result:")
    #     logging.error(response.reason)

    # logging.info(f"Uploading gaussian splatting result for asset {asset_id}...")
    # response = await asyncio.get_event_loop().run_in_executor(
    #     None,
    #     upload,
    #     os.path.join(
    #         "assets",
    #         f"{asset_id}/output/point_cloud/iteration_7000/point_cloud.ply",
    #     ),
    #     asset_id,
    #     "3dgs.ply",
    # )

    # if response.status_code == 200:
    #     url = f"{storage_root}/files/{asset_id}/3dgs.ply"
    #     requests.patch(
    #         f"{api_root}/assets/gaussian/{asset_id}",
    #         headers={"Content-Type": "application/json"},
    #         data=json.dumps({"url": url}),
    #     )
    #     logging.info(f"Gaussian splatting result uploaded successfully!")
    # else:
    #     logging.error(f"Error uploading gaussian splatting result:")
    #     logging.error(response.reason)

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

    api_root = os.getenv("API_ROOT")
    storage_root = os.getenv("STORAGE_ROOT")

    log_format = "%(asctime)s [%(levelname)s]: (%(name)s) %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

    asyncio.run(main())
