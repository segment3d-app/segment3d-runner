import asyncio
import json
import logging
import os
import subprocess
import zipfile

from urllib import parse, request
from dotenv import load_dotenv

from aio_pika import connect_robust
from aio_pika.abc import AbstractIncomingMessage


async def task(message: AbstractIncomingMessage):
    logging.info("Received message: %s", message.body)
    data = json.loads(message.body.decode())

    asset_id = data["asset_id"]
    asset_path = parse.quote(data["photo_dir_url"])
    asset_url = f"{storage_root}{asset_path}"

    try:
        zip_path = os.path.join("assets", f"{asset_id}.zip")
        dir_path = os.path.join("assets", f"{asset_id}/input")
        os.makedirs(dir_path, exist_ok=True)

        await asyncio.get_event_loop().run_in_executor(
            None, download, asset_url, zip_path
        )
        await asyncio.get_event_loop().run_in_executor(None, unzip, zip_path, dir_path)

        logging.info(
            f"Photos for asset {asset_id} downloaded and extracted successfully!"
        )

        await asyncio.get_event_loop().run_in_executor(
            None, generate_gaussian_splatting, asset_id
        )

    except Exception as e:
        logging.error(f"Error processing asset {asset_id}: {str(e)}")


def download(source: str, destination: str):
    response = request.urlopen(source)
    with open(destination, "wb") as f:
        f.write(response.read())


def unzip(source: str, destination: str):
    with zipfile.ZipFile(source, "r") as zip_ref:
        zip_ref.extractall(destination)
    os.remove(source)


def generate_gaussian_splatting(asset_id: str):
    # convert_command = f"python ./models/gaussian-splatting/convert.py -s ./assets/{asset_id}"
    # train_command = f"python ./models/gaussian-splatting/train.py -s ./assets/{asset_id}"

    process = subprocess.run(
        f"""
        bash -c "source activate gaussian_splatting && python ./models/gaussian-splatting/convert.py -s ./assets/{asset_id} && conda deactivate"
        """,
        text=True,
        shell=True,
        capture_output=True,
    )

    if process.returncode == 0:
        logging.info("Gaussian Splatting completed successfully!")
        logging.info(process.stdout)

    else:
        logging.error("Gaussian Splatting failed:")
        logging.error(process.stderr)


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
