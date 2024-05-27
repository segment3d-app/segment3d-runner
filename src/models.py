import asyncio
import logging
import os
import subprocess
import time

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
        self.output_path = os.path.join(self.assets_path, f"{asset_id}/output")

    async def generate_colmap(self):
        logging.info(f"Generating colmap for asset {self.asset_id}...")

        start_time = time.time()
        await asyncio.get_event_loop().run_in_executor(None, self.__generate_colmap)

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
        convert_command = " ".join([
            "python",
            os.path.join(self.model_path, "convert.py"),
            "-s",
            os.path.join(self.assets_path, self.asset_id),
        ])

        process = subprocess.run(
            f'bash -c "{self.__append_environment(convert_command)}"',
            text=True,
            shell=True,
            capture_output=True,
        )

        if process.returncode != 0:
            raise ColmapError(process.stderr)

    def __generate_gaussian_splatting(self):
        train_command = " ".join([
            "python",
            os.path.join(self.model_path, "train_scene.py"),
            "-s",
            os.path.join(self.assets_path, self.asset_id),
            "--model_path",
            self.output_path,
            "--iterations",
            "7000",
        ])

        process = subprocess.run(
            f'bash -c "{self.__append_environment(train_command)}"',
            text=True,
            shell=True,
            capture_output=True,
        )

        if process.returncode != 0:
            raise GaussianSplattingError(process.stderr)

    def __append_environment(self, command: str):
        return f"source {conda_source} && conda activate {self.conda_env} && {command} && conda deactivate"
