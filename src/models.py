import asyncio
import logging
import os
import subprocess
import time

from utils import pick_available_gpus

assets_path = "assets"
conda_source = "/opt/conda/etc/profile.d/conda.sh"


class ColmapError(Exception):
    pass


class GaussianSplattingError(Exception):
    pass


class GaussianSplatting:
    assets_path = "assets"
    model_path = "models/saga"

    conda_env = "saga"

    def __init__(self, asset_id: str):
        self.asset_id = asset_id
        self.asset_path = os.path.join(self.assets_path, asset_id)
        self.output_path = os.path.join(self.asset_path, "output")

    async def generate_colmap(self):
        logging.info(f"Generating colmap for asset {self.asset_id}...")

        start_time = time.time()
        await asyncio.get_event_loop().run_in_executor(None, self.__generate_colmap)
        await asyncio.get_event_loop().run_in_executor(None, self.__convert_colmap)

        end_time = time.time()
        logging.info(
            f"Colmap generated successfully in {end_time - start_time:.2f} seconds"
        )

    async def generate_gaussian_splatting(self):
        logging.info(f"Generating gaussian splatting for asset {self.asset_id}...")

        start_time = time.time()
        await asyncio.get_event_loop().run_in_executor(
            None, self.__generate_gaussian_splatting
        )

        end_time = time.time()
        logging.info(
            f"Gaussian splatting generated successfully in {end_time - start_time:.2f} seconds"
        )

    def __generate_colmap(self):
        colmap_command = [
            "python",
            os.path.join(self.model_path, "convert.py"),
            "-s",
            os.path.join(self.assets_path, self.asset_id),
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(pick_available_gpus())
        process = subprocess.run(
            f'bash -c "{self.__append_environment(" ".join(colmap_command))}"',
            text=True,
            shell=True,
            capture_output=True,
            env=env,
        )

        if process.returncode != 0:
            raise ColmapError(process.stderr)

    def __convert_colmap(self):
        colmap_command = [
            "colmap",
            "model_converter",
            "--input_path",
            os.path.join(self.asset_path, "sparse/0"),
            "--output_path",
            os.path.join(self.asset_path, "sparse/0/colmap.ply"),
            "--output_type",
            "PLY",
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(pick_available_gpus())
        process = subprocess.run(
            f'bash -c "{self.__append_environment(" ".join(colmap_command))}"',
            text=True,
            shell=True,
            capture_output=True,
            env=env,
        )

        if process.returncode != 0:
            raise ColmapError(process.stderr)

    def __generate_gaussian_splatting(self):
        train_command = [
            "python",
            os.path.join(self.model_path, "train_scene.py"),
            "-s",
            os.path.join(self.assets_path, self.asset_id),
            "--model_path",
            self.output_path,
            "--iterations",
            "7000",
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(pick_available_gpus())
        process = subprocess.run(
            f'bash -c "{self.__append_environment(" ".join(train_command))}"',
            text=True,
            shell=True,
            capture_output=True,
            env=env,
        )

        if process.returncode != 0:
            raise GaussianSplattingError(process.stderr)

    def __append_environment(self, command: str):
        return f"source {conda_source} && conda activate {self.conda_env} && {command} && conda deactivate"


class PTv3ConvertError(Exception):
    pass


class PTv3PreprocessError(Exception):
    pass


class PTv3InferenceError(Exception):
    pass


class PTv3ReconstructionError(Exception):
    pass


class PTv3:
    model_path = "models/pointcept"
    conda_env = "pointcept"

    def __init__(self, asset_id: str, asset_type: str):
        self.asset_type = asset_type
        self.asset_id = asset_id
        self.asset_path = os.path.join(assets_path, asset_id)

    async def process(self):
        logging.info(f"Converting PLY for asset {self.asset_id}...")

        start_time = time.time()
        await asyncio.get_event_loop().run_in_executor(None, self.__convert)

        end_time = time.time()
        logging.info(
            f"PLY converted successfully in {end_time - start_time:.2f} seconds"
        )

        # ==========

        logging.info(f"Preprocessing dataset for asset {self.asset_id}...")

        start_time = time.time()
        await asyncio.get_event_loop().run_in_executor(None, self.__preprocess)

        end_time = time.time()
        logging.info(
            f"Dataset preprocessed successfully in {end_time - start_time:.2f} seconds"
        )

        # ==========

        logging.info(f"Inferring segmentation for asset {self.asset_id}...")

        start_time = time.time()
        await asyncio.get_event_loop().run_in_executor(None, self.__infer)

        end_time = time.time()
        logging.info(
            f"Segmentation inferred successfully in {end_time - start_time:.2f} seconds"
        )

        # ==========

        logging.info(f"Reconstructing dataset for asset {self.asset_id}...")

        start_time = time.time()
        await asyncio.get_event_loop().run_in_executor(None, self.__reconstruct)

        end_time = time.time()
        logging.info(
            f"Dataset reconstructed successfully in {end_time - start_time:.2f} seconds"
        )

    def __convert(self):
        ply_path = (
            "input/lidar.ply"
            if self.asset_type == "lidar"
            else "output/point_cloud/iteration_7000/scene_point_cloud.ply"
        )
        command = f"""python {os.path.join(self.model_path, "convert_ply.py")} \
            -n scene \
            -p {os.path.join(self.asset_path, ply_path)} \
            -d {os.path.join(self.asset_path, "data")}
            """

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(pick_available_gpus())
        process = subprocess.run(
            f'bash -c "{self.__append_environment(command)}"',
            text=True,
            shell=True,
            capture_output=True,
            env=env,
        )

        if process.returncode != 0:
            raise PTv3ConvertError(process.stderr)

    def __preprocess(self):
        command = f"""python {os.path.join(self.model_path, "preprocess.py")} \
            --dataset_root {os.path.join(self.asset_path, "data/scene")}
            --output_root {os.path.join(self.asset_path, "data/scene")}
            """

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(pick_available_gpus())
        process = subprocess.run(
            f'bash -c "{self.__append_environment(command)}"',
            text=True,
            shell=True,
            capture_output=True,
            env=env,
        )

        if process.returncode != 0:
            raise PTv3PreprocessError(process.stderr)

    def __infer(self):
        options = " ".join(
            [
                f"""weight={os.path.join(self.model_path, "models/ptv3/model/model_best.pth")}""",
                f"""save_path={os.path.join(self.asset_path, "data")}""",
                f"""data_root={os.path.join(self.asset_path, "data/scene")}""",
                f"""data.test.data_root={os.path.join(self.asset_path, "data/scene")}""",
            ]
        )

        command = f"""python {os.path.join(self.model_path, "tools/pred.py")} \
            --config_file {os.path.join(self.model_path, "models/ptv3/config.py")} \
            --test_split scene \
            --num_gpus 2 \
            --options {options}
            """

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(pick_available_gpus())
        process = subprocess.run(
            f'export PYTHONPATH=models/pointcept && bash -c "{self.__append_environment(command)}"',
            text=True,
            shell=True,
            capture_output=True,
            env=env,
        )

        if process.returncode != 0:
            raise PTv3InferenceError(process.stderr)

    def __reconstruct(self):
        command = f"""python {os.path.join(self.model_path, "convert_npy.py")} \
            --gaussian {os.path.join(self.asset_path, "output/point_cloud/iteration_7000/scene_point_cloud.ply")} \
            --scene {os.path.join(self.asset_path, "scene/result/scene.npy")} \
            --destination {os.path.join(self.asset_path, "segmentation")} \
            --name ptv3
            """

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(pick_available_gpus())
        process = subprocess.run(
            f'export PYTHONPATH=models/pointcept && bash -c "{self.__append_environment(command)}"',
            text=True,
            shell=True,
            capture_output=True,
            env=env,
        )

        if process.returncode != 0:
            raise PTv3ReconstructionError(process.stderr)

    def __append_environment(self, command: str):
        command = " ".join(line.strip() for line in command.splitlines())
        return f"source {conda_source} && conda activate {self.conda_env} && {command} && conda deactivate"
