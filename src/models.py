import asyncio
import os
import subprocess

from typing import Dict
from utils import pick_available_gpus, parse_command

conda_source = "/opt/conda/etc/profile.d/conda.sh"


class Model:
    assets_path = "assets"
    model_path = ""
    conda_env = ""

    def __init__(self, asset_id: str, asset_type: str, conda_env: str, model_path: str):
        self.asset_type = asset_type
        self.asset_id = asset_id

        self.conda_env = conda_env
        self.model_path = model_path

    def run_command(self, command: str, environment: Dict[str, str] = dict()):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(pick_available_gpus())

        for key, value in environment.items():
            env[key] = value

        command = self.__append_environment(parse_command(command))
        process = subprocess.run(
            f'bash -c "{command}"',
            text=True,
            shell=True,
            capture_output=True,
            env=env,
        )

        return process

    def __append_environment(self, command: str):
        return f"source {conda_source} && conda activate {self.conda_env} && {command} && conda deactivate"


class ColmapError(Exception):
    pass


class GaussianSplattingError(Exception):
    pass


class GaussianSplatting(Model):
    def __init__(self, asset_id: str, asset_type: str):
        Model.__init__(
            self,
            asset_id=asset_id,
            asset_type=asset_type,
            conda_env="saga",
            model_path="models/saga",
        )

        self.asset_path = os.path.join(self.assets_path, asset_id)
        self.output_path = os.path.join(self.asset_path, "output")

    async def generate_pointcloud(self):
        await asyncio.get_event_loop().run_in_executor(None, self.__generate_pointcloud)
        await asyncio.get_event_loop().run_in_executor(None, self.__convert_pointcloud)

    async def generate_gaussian(self):
        await asyncio.get_event_loop().run_in_executor(None, self.__generate_gaussian)

    def __generate_pointcloud(self):
        command = f"""
            python {os.path.join(self.model_path, "convert.py")} 
            -s {os.path.join(self.assets_path, self.asset_id)}
        """

        process = self.run_command(command)
        if process.returncode != 0:
            raise ColmapError(process.stderr)

    def __convert_pointcloud(self):
        command = f"""
            colmap model_converter
            --input_path {os.path.join(self.asset_path, "sparse/0")}
            --output_path {os.path.join(self.asset_path, "sparse/0/pointcloud.ply")}
            --output_type PLY
        """

        process = self.run_command(command)
        if process.returncode != 0:
            raise ColmapError(process.stderr)

    def __generate_gaussian(self):
        command = f"""
            python {os.path.join(self.model_path, "train_scene.py")}
            -s {os.path.join(self.assets_path, self.asset_id)}
            --model_path {self.output_path}
            --iterations 7000
        """

        process = self.run_command(command)
        if process.returncode != 0:
            raise GaussianSplattingError(process.stderr)


class PTv3ConvertError(Exception):
    pass


class PTv3PreprocessError(Exception):
    pass


class PTv3InferenceError(Exception):
    pass


class PTv3ReconstructionError(Exception):
    pass


class PTv3(Model):
    def __init__(self, asset_id: str, asset_type: str):
        Model.__init__(
            self,
            asset_id=asset_id,
            asset_type=asset_type,
            conda_env="pointcept",
            model_path="models/pointcept",
        )

        self.asset_path = os.path.join(self.assets_path, asset_id)

        if self.asset_type == "lidar":
            self.input_path = "input/lidar.ply"
        else:
            self.input_path = "output/point_cloud/iteration_7000/scene_point_cloud.ply"

    async def convert(self):
        await asyncio.get_event_loop().run_in_executor(None, self.__convert)

    async def preprocess(self):
        await asyncio.get_event_loop().run_in_executor(None, self.__preprocess)

    async def infer(self):
        await asyncio.get_event_loop().run_in_executor(None, self.__infer)

    async def reconstruct(self):
        await asyncio.get_event_loop().run_in_executor(None, self.__reconstruct)

    def __convert(self):
        command = f"""python {os.path.join(self.model_path, "convert_ply.py")}
            -p {os.path.join(self.asset_path, self.input_path)}
            -d {os.path.join(self.asset_path, "data")}
            -n scene
        """

        process = self.run_command(command)
        if process.returncode != 0:
            raise PTv3ConvertError(process.stderr)

    def __preprocess(self):
        command = f"""python {os.path.join(self.model_path, "preprocess.py")}
            --dataset_root {os.path.join(self.asset_path, "data/scene")}
            --output_root {os.path.join(self.asset_path, "data/scene")}
        """

        process = self.run_command(command)
        if process.returncode != 0:
            raise PTv3PreprocessError(process.stderr)

    def __infer(self):
        options = {
            "weight": os.path.join(self.model_path, "models/ptv3/model/model_best.pth"),
            "save_path": os.path.join(self.asset_path, "data"),
            "data_root": os.path.join(self.asset_path, "data/scene"),
            "data.test.data_root": os.path.join(self.asset_path, "data/scene"),
            "data.test.test_cfg.voxelize.grid_size": 0.1,
            "num_worker": 2,
        }

        command = f"""python {os.path.join(self.model_path, "tools/pred.py")}
            --config-file {os.path.join(self.model_path, "models/ptv3/config.py")}
            --options {" ".join([f"{key}={value}" for key, value in options.items()])}
            --test_split scene
            --num-gpus 2
        """

        process = self.run_command(command, {"PYTHONPATH": "models/pointcept"})
        if process.returncode != 0:
            raise PTv3InferenceError(process.stderr)

    def __reconstruct(self):
        command = f"""python {os.path.join(self.model_path, "convert_npy.py")}
            --gaussian {os.path.join(self.asset_path, "output/point_cloud/iteration_7000/scene_point_cloud.ply")}
            --scene {os.path.join(self.asset_path, "scene/result/scene.npy")}
            --destination {os.path.join(self.asset_path, "segmentation")}
            --name ptv3
        """

        process = self.run_command(command, {"PYTHONPATH": "models/pointcept"})
        if process.returncode != 0:
            raise PTv3ReconstructionError(process.stderr)
