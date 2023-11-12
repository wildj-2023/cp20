# run_fkgw.py
# stable-diffusion-webui_controlNet-aria2.py
from colorama import Fore
from pathlib import Path

import modal
import shutil
import subprocess
import sys
import shlex
import os

# modal系の変数の定義
stub = modal.Stub("stable-diffusion-webui")
volume_main = modal.SharedVolume().persist("stable-diffusion-webui-main")

# パスの定義
webui_dir = "/content/stable-diffusion-webui"
webui_model_dir = webui_dir + "/models/Stable-diffusion/"

# ControlNetのパスを定義
webui_controlNet_dir = webui_dir + "/extensions/sd-webui-controlnet"
webui_controlNet_model_dir = webui_dir + "/extensions/sd-webui-controlnet/models/"

# モデルのID
model_ids = [
    {
        "repo_id": "hakurei/waifu-diffusion-v1-4",
        "model_path": "wd-1-4-anime_e1.ckpt",
        "config_file_path": "wd-1-4-anime_e1.yaml",
    },
 ]

@stub.function(

    image=modal.Image.from_dockerhub("python:3.10-slim")
    .apt_install(
        "git", "libgl1-mesa-dev", "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "gcc", "libcairo2-dev", "aria2",
    )
    .run_commands(
        "pip install -U -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers"
        
    )
    .pip_install(
        "blendmodes==2022",
        "transformers==4.25.1",
        "accelerate==0.12.0",
        "basicsr==1.4.2",
        "gfpgan==1.3.8",
        "gradio==3.16.2",
        "numpy==1.23.3",
        "Pillow==9.4.0",
        "realesrgan==0.3.0",
        "torch",
        "omegaconf==2.2.3",
        "pytorch_lightning==1.7.6",
        "scikit-image==0.19.2",
        "fonts",
        "font-roboto",
        "timm==0.6.7",
        "piexif==1.1.3",
        "einops==0.4.1",
        "jsonmerge==1.8.0",
        "clean-fid==0.1.29",
        "resize-right==0.0.2",
        "torchdiffeq==0.2.3",
        "kornia==0.6.7",
        "lark==1.1.2",
        "inflection==0.5.1",
        "GitPython==3.1.27",
        "torchsde==0.2.5",
        "safetensors==0.2.7",
        "httpcore<=0.15",
        "tensorboard==2.9.1",
        "taming-transformers==0.0.1",
        "clip",
        "xformers",
        "test-tube",
        "diffusers",
        "invisible-watermark",
        "pyngrok",
        "xformers==0.0.16rc425",
        "gdown",
        "huggingface_hub",
        "colorama",
    )
    .pip_install("git+https://github.com/mlfoundations/open_clip.git@bb6e834e9c70d9c27d0dc3ecedeebeaeb1ffad6b"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    shared_volumes={webui_dir: volume_main},
    
    #ここでGPUを指定
    gpu="a10g",
    # gpu=modal.gpu.A10G(count=2),
    # gpu=modal.gpu.T4(count=2),
    timeout=6000,
)
async def run_stable_diffusion_webui():
    print(Fore.CYAN + "\n---------- セットアップ開始 ----------\n")

    webui_dir_path = Path(webui_model_dir)
    if not webui_dir_path.exists():
        
        subprocess.run(f"git clone -b v2.1 https://github.com/camenduru/stable-diffusion-webui {webui_dir}", shell=True)

    #controlNetをgit clone
    controlNet_dir_path = Path(webui_controlNet_dir)
    if not controlNet_dir_path.exists():
        subprocess.run(f"git clone https://github.com/Mikubill/sd-webui-controlnet {webui_controlNet_dir}", shell=True)

    def download_hf_file(repo_id, filename):
        from huggingface_hub import hf_hub_download

        download_dir = hf_hub_download(repo_id=repo_id, filename=filename)
        return download_dir

    for model_id in model_ids:
        print(Fore.GREEN + model_id["repo_id"] + "のセットアップを開始します...")

        if not Path(webui_model_dir + model_id["model_path"]).exists():
            model_downloaded_dir = download_hf_file(
                model_id["repo_id"],
                model_id["model_path"],
            )
            shutil.copy(model_downloaded_dir, webui_model_dir + os.path.basename(model_id["model_path"]))

        if "config_file_path" not in model_id:
          continue

        if not Path(webui_model_dir + model_id["config_file_path"]).exists():
            config_downloaded_dir = download_hf_file(
                model_id["repo_id"], model_id["config_file_path"]
            )
            shutil.copy(config_downloaded_dir, webui_model_dir + os.path.basename(model_id["config_file_path"]))

        print(Fore.GREEN + model_id["repo_id"] + "のセットアップが完了しました！")

    controlNet_files = [
        "control_v11e_sd15_ip2p_fp16.safetensors",
        "control_v11e_sd15_shuffle_fp16.safetensors",
        "control_v11p_sd15_canny_fp16.safetensors",
        "control_v11f1p_sd15_depth_fp16.safetensors",
        "control_v11p_sd15_inpaint_fp16.safetensors",
        "control_v11p_sd15_lineart_fp16.safetensors",
        "control_v11p_sd15_mlsd_fp16.safetensors",
        "control_v11p_sd15_normalbae_fp16.safetensors",
        "control_v11p_sd15_openpose_fp16.safetensors",
        "control_v11p_sd15_scribble_fp16.safetensors",
        "control_v11p_sd15_seg_fp16.safetensors",
        "control_v11p_sd15_softedge_fp16.safetensors",
        "control_v11p_sd15s2_lineart_anime_fp16.safetensors",
        "control_v11f1e_sd15_tile_fp16.safetensors",
        "control_v11e_sd15_ip2p_fp16.yaml",
        "control_v11e_sd15_shuffle_fp16.yaml",
        "control_v11p_sd15_canny_fp16.yaml",
        "control_v11f1p_sd15_depth_fp16.yaml",
        "control_v11p_sd15_inpaint_fp16.yaml",
        "control_v11p_sd15_lineart_fp16.yaml",
        "control_v11p_sd15_mlsd_fp16.yaml",
        "control_v11p_sd15_normalbae_fp16.yaml",
        "control_v11p_sd15_openpose_fp16.yaml",
        "control_v11p_sd15_scribble_fp16.yaml",
        "control_v11p_sd15_seg_fp16.yaml",
        "control_v11p_sd15_softedge_fp16.yaml",
        "control_v11p_sd15s2_lineart_anime_fp16.yaml",
        "control_v11f1e_sd15_tile_fp16.yaml",
        "t2iadapter_style_sd14v1.pth",
        "t2iadapter_sketch_sd14v1.pth",
        "t2iadapter_seg_sd14v1.pth",
        "t2iadapter_openpose_sd14v1.pth",
        "t2iadapter_keypose_sd14v1.pth",
        "t2iadapter_depth_sd14v1.pth",
        "t2iadapter_color_sd14v1.pth",
        "t2iadapter_canny_sd14v1.pth",
        "t2iadapter_canny_sd15v2.pth",
        "t2iadapter_depth_sd15v2.pth",
        "t2iadapter_sketch_sd15v2.pth",
        "t2iadapter_zoedepth_sd15v1.pth"
    ]

    for file_name in controlNet_files:
        if not Path(webui_controlNet_model_dir + file_name).exists():
            if file_name.endswith('.pth') or file_name.endswith('.safetensors'):
                controlNet_url = "https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/" + file_name
            elif file_name.endswith('.yaml'):
                controlNet_url = "https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/" + file_name
            subprocess.run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", controlNet_url, "-d", webui_controlNet_model_dir, "-o", file_name])

        if not Path(webui_model_dir + "BloodNightOrangeMix.vae.pt").exists():
            vae_url = "https://huggingface.co/WarriorMama777/OrangeMixs/resolve/main/VAEs/orangemix.vae.pt"
            subprocess.run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", vae_url, "-d", webui_model_dir, "-o", "BloodNightOrangeMix.vae.pt"])

    print(Fore.CYAN + "\n---------- セットアップ完了 ----------\n")

    # WebUIを起動
    sys.path.append(webui_dir)
    sys.argv += shlex.split("--skip-install --xformers")
    os.chdir(webui_dir)
    from launch import start, prepare_environment

    prepare_environment()
    # 拡張機能が使えるように--enable-insecure-extension-accessを追加
    sys.argv = shlex.split("--a --gradio-debug --share --xformers --enable-insecure-extension-access")
    start()
    
    
@stub.local_entrypoint()
def main():
    run_stable_diffusion_webui.call()
