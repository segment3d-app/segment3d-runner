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
    Saga,
    SagaExtractFeaturesError,
    SagaExtractMasksError,
    SagaRenderError,
    SagaSegmentError,
    SagaTrainFeaturesError,
    SagaTrainSceneError,
)


class PatchError(Exception):
    pass


async def process_task(message: AbstractIncomingMessage):
    logging.info("Received process message:")

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
    saga = Saga(asset_id=asset.asset_id, asset_type=asset_type)
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
        # await process_ptv3(asset, ptv3)

        # Process SAGA
        await process_saga(asset, saga)

        await message.ack()

    except:
        logging.error("")
        await message.nack()


async def process_query(message: AbstractIncomingMessage):
    logging.info("Received SAGA message:")

    # ==== Parse message and create model instances

    data = json.loads(message.body.decode())

    logging.info(f"└- Asset ID: {data['asset_id']}")
    logging.info(f"└- Segment ID: {data['unique_identifier']}")
    logging.info(f"└- Photo URL: {data['url']}")
    logging.info(f"└- X coordinate: {data['x']}")
    logging.info(f"└- Y coordinate: {data['y']}")

    asset = Asset(
        storage_root=storage_root,
        asset_id=data["asset_id"],
        images_path=None,
        pcl_path=None,
    )

    saga = Saga(asset_id=asset.asset_id, asset_type="lidar")

    segment_id = data["unique_identifier"]
    image_url = data["url"]
    x = data["x"]
    y = data["y"]

    image_name = image_url.split("/")[-1].split(".")[0]

    try:
        # Segment SAGA
        await segment_saga(asset, saga, segment_id, image_name, x, y)

    except:
        logging.error("")
        await message.nack()


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
    start_start_time = time.time()

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

    duration = time.time() - start_start_time
    logging.info(f"└- PTv3 processed successfully in {duration:.2f} seconds")


async def process_saga(asset: Asset, saga: Saga):
    logging.info(f"Processing SAGA for asset {asset.asset_id}...")
    start_start_time = time.time()

    try:
        # Extract features
        logging.info(f"└- Extracting features...")
        start_time = time.time()
        await saga.extract_features()

        if not asset.exists("features"):
            raise SagaExtractFeaturesError("features/ not found")

        duration = time.time() - start_time
        logging.info(f"└--- Features extracted successfully in {duration:.2f} seconds")

        # Extract masks
        logging.info(f"└- Extracting masks...")
        start_time = time.time()
        await saga.extract_masks()

        if not asset.exists("sam_masks"):
            raise SagaExtractMasksError("sam_masks/ not found")

        duration = time.time() - start_time
        logging.info(f"└--- Masks extracted successfully in {duration:.2f} seconds")

        # Train scene
        logging.info(f"└- Training scene...")
        start_time = time.time()
        await saga.train_scene()

        if not asset.exists("saga"):
            raise SagaTrainSceneError("saga/ not found")

        duration = time.time() - start_time
        logging.info(f"└--- Scene trained successfully in {duration:.2f} seconds")

        # Train features
        logging.info(f"└- Training features...")
        start_time = time.time()
        await saga.train_features()

        duration = time.time() - start_time
        logging.info(f"└--- Features trained successfully in {duration:.2f} seconds")

        # Upload result
        folder_url = await asset.upload_folder("images", "saga")
        response = requests.patch(
            f"{api_root}/assets/saga/{asset.asset_id}",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"url": "/" + folder_url}),
        )

        if response.status_code != 200:
            raise PatchError(response.reason)

    except SagaExtractFeaturesError as e:
        logging.error(f"└- Failed extracting features:")
        logging.error(e.args[0])
        raise Exception()

    except SagaExtractMasksError as e:
        logging.error(f"└- Failed extracting masks:")
        logging.error(e.args[0])
        raise Exception()

    except SagaTrainSceneError as e:
        logging.error(f"└- Failed training scene:")
        logging.error(e.args[0])
        raise Exception()

    except SagaTrainFeaturesError as e:
        logging.error(f"└- Failed training features:")
        logging.error(e.args[0])
        raise Exception()

    except AssetUploadError as e:
        logging.error(f"└- Failed uploading SAGA:")
        logging.error(e.args[0])
        raise Exception()

    except Exception as e:
        logging.error(f"└- Unknown error when processing SAGA:")
        logging.error(str(e))
        raise Exception()

    duration = time.time() - start_start_time
    logging.info(f"└- SAGA processed successfully in {duration:.2f} seconds")


async def segment_saga(
    asset: Asset, saga: Saga, segment_id: str, image_name: str, x: int, y: int
):
    logging.info(f"Segmenting SAGA for asset {asset.asset_id}...")
    start_start_time = time.time()

    try:
        if not asset.exists("saga/cameras.json"):
            raise Exception("saga/cameras.json not found")

        cameras = asset.read_json("saga/cameras.json")
        for camera in cameras:
            if camera["img_name"] == image_name:
                image_index = camera["id"]
                break
        else:
            raise Exception("Image name not found")

        # Extract features
        logging.info(f"└- Segmenting...")
        start_time = time.time()
        await saga.segment(segment_id, image_index, 1, x, y)

        if not asset.exists(f"saga/segmentation/{segment_id}"):
            raise SagaSegmentError(f"saga/segmentation/{segment_id} not found")

        duration = time.time() - start_time
        logging.info(f"└--- Segmented successfully in {duration:.2f} seconds")

        # Render masks
        logging.info(f"└- Rendering...")
        start_time = time.time()
        await saga.render(segment_id)

        if not asset.exists(
            "saga/point_cloud/iteration_7000/segmentation/segmentation_seg_no_mask_point_cloud.ply"
        ):
            raise SagaRenderError("segmentation_seg_no_mask_point_cloud.ply not found")

        duration = time.time() - start_time
        logging.info(f"└--- Rendered successfully in {duration:.2f} seconds")

        # Upload result
        await asset.upload(
            "saga/point_cloud/iteration_7000/segmentation/segmentation_seg_no_mask_point_cloud.ply",
            f"{segment_id}.ply",
        )

    except SagaSegmentError as e:
        logging.error(f"└- Failed segmenting:")
        logging.error(e.args[0])
        raise Exception()

    except SagaRenderError as e:
        logging.error(f"└- Failed rendering:")
        logging.error(e.args[0])
        raise Exception()

    except AssetUploadError as e:
        logging.error(f"└- Failed uploading SAGA:")
        logging.error(e.args[0])
        raise Exception()

    except Exception as e:
        logging.error(f"└- Unknown error when processing SAGA:")
        logging.error(str(e))
        raise Exception()

    duration = time.time() - start_start_time
    logging.info(f"└- SAGA processed successfully in {duration:.2f} seconds")


async def main():
    connection = await connect_robust(
        host=os.getenv("RABBITMQ_HOST"),
        port=int(os.getenv("RABBITMQ_PORT")),
        login=os.getenv("RABBITMQ_USER"),
        password=os.getenv("RABBITMQ_PASSWORD"),
    )

    channel = await connection.channel()
    await channel.set_qos(prefetch_count=1)

    process_queue_name = os.getenv("RABBITMQ_QUEUE_PROCESS")
    process_queue = await channel.declare_queue(process_queue_name, durable=True)

    query_queue_name = os.getenv("RABBITMQ_QUEUE_SAGA")
    query_queue = await channel.declare_queue(query_queue_name, durable=True)

    await process_queue.consume(process_task)
    await query_queue.consume(process_query)

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
