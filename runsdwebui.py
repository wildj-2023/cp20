import os
import sys
import subprocess
import shlex

import modal


persist_dir = "/content"
webui_dir = persist_dir + "/stable-diffusion-webui"

stub = modal.Stub("stable-diffusion-webui")
volume = modal.NetworkFileSystem.new().persist("sdwebui-volume")


image = (
    modal.Image.from_registry("python:3.10-slim")
    .apt_install("git", "libgl1-mesa-dev", "libglib2.0-0", "libsm6", "libxrender1", "libxext6")
    .pip_install(
        "torch==2.0.1",
        "torchvision==0.15.2",
        extra_index_url="https://download.pytorch.org/whl/cu118"
    )
    .pip_install(
        "GitPython==3.1.30",
        "Pillow==9.5.0",
        "accelerate==0.18.0",
        "basicsr==1.4.2",
        "blendmodes==2022",
        "clean-fid==0.1.35",
        "einops==0.4.1",
        "fastapi==0.94.0",
        "gfpgan==1.3.8",
        "gradio==3.32.0",
        "httpcore==0.15",
        "inflection==0.5.1",
        "jsonmerge==1.8.0",
        "kornia==0.6.7",
        "lark==1.1.2",
        "numpy==1.23.5",
        "omegaconf==2.2.3",
        "open-clip-torch==2.20.0",
        "piexif==1.1.3",
        "psutil==5.9.5",
        "pytorch_lightning==1.9.4",
        "realesrgan==0.3.0",
        "resize-right==0.0.2",
        "safetensors==0.3.1",
        "scikit-image==0.20.0",
        "timm==0.6.7",
        "tomesd==0.1.2",
        "torchdiffeq==0.2.3",
        "torchsde==0.2.5",
        "transformers==4.25.1",
    )
    .pip_install(
        "git+https://github.com/mlfoundations/open_clip.git@bb6e834e9c70d9c27d0dc3ecedeebeaeb1ffad6b",
    )
    .run_commands(
        "python -m pip install --no-deps xformers==0.0.20"
    )
    .pip_install(
        "mediapipe",
        "svglib",
        "fvcore",
        "send2trash~=1.8",
        "dynamicprompts[attentiongrabber,magicprompt]~=0.29.0",
    )
)

@stub.function(image=image,
               network_file_systems={persist_dir: volume},
               gpu="T4",
               timeout=36000)
def run_webui():
    sys.path.append(webui_dir)
    os.chdir(webui_dir)

    subprocess.run("python launch.py --share --xformers --no-half-vae --opt-sdp-no-mem-attention --opt-channelslast --enable-insecure-extension-access", shell=True)


@stub.local_entrypoint()
def main():
    run_webui.remote()
