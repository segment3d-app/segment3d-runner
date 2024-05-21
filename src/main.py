import asyncio
import json
import logging
import os
import requests
import subprocess
import zipfile

from urllib import parse, request
from dotenv import load_dotenv

from aio_pika import connect_robust
from aio_pika.abc import AbstractIncomingMessage


async def task(message: AbstractIncomingMessage):
    logging.info("Received message: %s", message.body)

    try:
        data = json.loads(message.body.decode())

        asset_id = data["asset_id"]
        asset_path = parse.quote(data["photo_dir_url"])
        asset_url = f"{storage_root}{asset_path}"

        zip_path = os.path.join("assets", f"{asset_id}.zip")
        dir_path = os.path.join("assets", f"{asset_id}/input")
        os.makedirs(dir_path, exist_ok=True)

        logging.info(f"Downloading asset {asset_id}...")
        await asyncio.get_event_loop().run_in_executor(
            None, download, asset_url, zip_path
        )
        logging.info(f"Asset downloaded successfully!")

        logging.info(f"Extracting asset {asset_id}...")
        await asyncio.get_event_loop().run_in_executor(None, unzip, zip_path, dir_path)
        logging.info(f"Asset extracted successfully!")

        logging.info(f"Generating 3D gaussian splatting for asset {asset_id}...")
        await asyncio.get_event_loop().run_in_executor(
            None, generate_gaussian_splatting, asset_id
        )

        logging.info(f"Uploading colmap result for asset {asset_id}...")
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            upload,
            os.path.join("assets", f"{asset_id}/sparse/0/points3D.ply"),
            f"/files/{asset_id}/colmap.ply",
        )

        if response.status_code == 200:
            url = f"{storage_root}/files/{asset_id}/colmap.ply"
            requests.patch(
                f"{api_root}/assets/pointcloud/{asset_id}",
                headers={"Content-Type": "application/json"},
                data=json.dumps({"url": url}),
            )
            logging.info(f"Colmap result uploaded successfully!")
        else:
            logging.error(f"Error uploading colmap result:")
            logging.error(response.reason)

        logging.info(f"Uploading gaussian splatting result for asset {asset_id}...")
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            upload,
            os.path.join(
                "assets",
                f"{asset_id}/output/point_cloud/iteration_30000/point_cloud.ply",
            ),
            f"/files/{asset_id}/3dgs.ply",
        )

        if response.status_code == 200:
            url = f"{storage_root}/files/{asset_id}/3dgs.ply"
            requests.patch(
                f"{api_root}/assets/gaussian/{asset_id}",
                headers={"Content-Type": "application/json"},
                data=json.dumps({"url": url}),
            )
            logging.info(f"Gaussian splatting result uploaded successfully!")
        else:
            logging.error(f"Error uploading gaussian splatting result:")
            logging.error(response.reason)

    except Exception as e:
        logging.error(f"Error processing asset {asset_id}:")
        logging.error(str(e))

    # await message.ack()


def download(source: str, destination: str):
    response = request.urlopen(source)
    with open(destination, "wb") as f:
        f.write(response.read())


def unzip(source: str, destination: str):
    with zipfile.ZipFile(source, "r") as zip_ref:
        zip_ref.extractall(destination)
    os.remove(source)


def upload(source: str, target: str):
    data = {"folder": target}
    files = {"file": (source.split("/")[-1], open(source, "rb"))}
    return requests.post(f"{storage_root}/upload", data=data, files=files)


def generate_gaussian_splatting(asset_id: str):
    convert_command = (
        f"python ./models/gaussian-splatting/convert.py -s ./assets/{asset_id}"
    )

    output_path = os.path.join("assets", f"{asset_id}/output")
    train_command = f"python ./models/gaussian-splatting/train.py -s ./assets/{asset_id} --model_path {output_path} --test_iterations 7000"

    conda_setup_script = "/opt/conda/etc/profile.d/conda.sh"
    process = subprocess.run(
        f"""
        bash -c "source {conda_setup_script} && conda activate gaussian_splatting && {convert_command} && {train_command} && conda deactivate"
        """,
        text=True,
        shell=True,
        capture_output=True,
    )

    if process.returncode == 0:
        logging.info("Gaussian splatting generated successfully!")
        logging.info(process.stdout)
    else:
        logging.error("Error generating gaussian splatting:")
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
