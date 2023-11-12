# run_bob.py
# stable-diffusion-webui_controlNet3.py
from colorama import Fore
from pathlib import Path

import modal
import shutil
import subprocess
import sys
import shlex
import os

# modal系の変数の定義 ※既存に被っちゃうとあれなので-3にしておきます。
stub = modal.Stub("stable-diffusion-webui-3")
volume_main = modal.NetworkFileSystem.persisted("stable-diffusion-webui-main-3")

# 色んなパスの定義
webui_dir = "/content/stable-diffusion-webui"
webui_model_dir = webui_dir + "/models/Stable-diffusion/"

webui_controlNet_dir = webui_dir + "/extensions/sd-webui-controlnet"
webui_controlNet_model_dir = webui_dir + "/extensions/sd-webui-controlnet/models"

# モデルのID　※model_pathがmodel名と違う場合にも再ダウンロードされないようにmodel_nameを追加しました。あとControlNetがsd15用のためBloodNightOrangeMixを追加
model_ids = [
    {
        "repo_id": "hakurei/waifu-diffusion-v1-4",
        "model_path": "wd-1-4-anime_e1.ckpt",
        "config_file_path": "wd-1-4-anime_e1.yaml",
        "model_name": "wd-1-4-anime_e1.ckpt",
    },
   {
       "repo_id": "WarriorMama777/OrangeMixs",
       "model_path": "Models/BloodOrangeMix/BloodNightOrangeMix.ckpt",
       "config_file_path": "",
       "model_name": "BloodNightOrangeMix.ckpt",
    },
]

# ※v2.6ブランチを使用するのでpythonのバージョン上げたのとrequirements_versions.txtになんとなく合わせました
@stub.function(
    image=modal.Image.from_registry("python:3.10.6-slim")
    .apt_install(
        "git", "libgl1-mesa-dev", "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "gcc", "libcairo2-dev", "aria2",
    )
    .run_commands(
        "pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers"
    )
    .pip_install(
        "blendmodes==2022",
        "transformers==4.30.2",
        "accelerate==0.21.0",
        "basicsr==1.4.2",
        "gfpgan==1.3.8",
        "gradio==3.41.2",
        "numpy==1.23.5",
        "Pillow==9.5.0",
        "realesrgan==0.3.0",
        "torch",
        "omegaconf==2.2.3",
        "pytorch_lightning==1.9.4",
        "scikit-image==0.21.0",
        "fonts",
        "font-roboto",
        "timm==0.9.2",
        "piexif==1.1.3",
        "einops==0.4.1",
        "jsonmerge==1.8.0",
        "clean-fid==0.1.35",
        "resize-right==0.0.2",
        "torchdiffeq==0.2.3",
        "kornia==0.6.7",
        "lark==1.1.2",
        "inflection==0.5.1",
        "GitPython==3.1.32",
        "torchsde==0.2.5",
        "safetensors==0.3.1",
        "httpcore==0.15",
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
        "torchmetrics==0.11.4",
        "fastapi==0.94.0",
        "open-clip-torch==2.20.0",
        "psutil==5.9.5",
        "tomesd==0.1.3",
        "httpx==0.24.1",
        
    )
    .pip_install("git+https://github.com/mlfoundations/open_clip.git@bb6e834e9c70d9c27d0dc3ecedeebeaeb1ffad6b"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    network_file_systems={webui_dir: volume_main},
    gpu="a10g",
    timeout=6000,
)
async def run_stable_diffusion_webui():
    print(Fore.CYAN + "\n---------- セットアップ開始 ----------\n")

    webui_dir_path = Path(webui_model_dir)
    if not webui_dir_path.exists():
        subprocess.run(f"git clone -b v2.6 https://github.com/camenduru/stable-diffusion-webui {webui_dir}", shell=True)

    controlNet_dir_path = Path(webui_controlNet_dir)
    if not controlNet_dir_path.exists():
        subprocess.run(f"git clone https://github.com/Mikubill/sd-webui-controlnet {webui_controlNet_dir}", shell=True)

    # Hugging faceからファイルをダウンロードしてくる関数
    def download_hf_file(repo_id, filename):
        from huggingface_hub import hf_hub_download

        download_dir = hf_hub_download(repo_id=repo_id, filename=filename)
        return download_dir

    # 一部model_nameに修正
    for model_id in model_ids:
        print(Fore.GREEN + model_id["repo_id"] + "のセットアップを開始します...")

        if not Path(webui_model_dir + model_id["model_name"]).exists():
            # モデルのダウンロード＆コピー
            model_downloaded_dir = download_hf_file(
                model_id["repo_id"],
                model_id["model_path"],
            )
            shutil.copy(model_downloaded_dir, webui_model_dir + os.path.basename(model_id["model_path"]))


        if "config_file_path" not in model_id:
          continue

        if not Path(webui_model_dir + model_id["config_file_path"]).exists():
            # コンフィグのダウンロード＆コピー
            config_downloaded_dir = download_hf_file(
                model_id["repo_id"], model_id["config_file_path"]
            )
            shutil.copy(config_downloaded_dir, webui_model_dir + os.path.basename(model_id["config_file_path"]))


        print(Fore.GREEN + model_id["repo_id"] + "のセットアップが完了しました！")

        # controlNetのモデルをダウンロード　※後から導入でも可(後述)。ダウンロードに時間がかかるので必要な場合のみコメント解除してください。
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11e_sd15_ip2p_fp16.safetensors", "-d", webui_controlNet_model_dir ,"-o", "control_v11e_sd15_ip2p_fp16.safetensors"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11e_sd15_shuffle_fp16.safetensors", "-d", webui_controlNet_model_dir ,"-o", "control_v11e_sd15_shuffle_fp16.safetensors"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny_fp16.safetensors", "-d", webui_controlNet_model_dir ,"-o", "control_v11p_sd15_canny_fp16.safetensors"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth_fp16.safetensors", "-d", webui_controlNet_model_dir ,"-o", "control_v11f1p_sd15_depth_fp16.safetensors"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_inpaint_fp16.safetensors", "-d", webui_controlNet_model_dir ,"-o", "control_v11p_sd15_inpaint_fp16.safetensors"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart_fp16.safetensors", "-d", webui_controlNet_model_dir ,"-o", "control_v11p_sd15_lineart_fp16.safetensors"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_mlsd_fp16.safetensors", "-d", webui_controlNet_model_dir ,"-o", "control_v11p_sd15_mlsd_fp16.safetensors"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_normalbae_fp16.safetensors", "-d", webui_controlNet_model_dir ,"-o", "control_v11p_sd15_normalbae_fp16.safetensors"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose_fp16.safetensors", "-d", webui_controlNet_model_dir ,"-o", "control_v11p_sd15_openpose_fp16.safetensors"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_scribble_fp16.safetensors", "-d", webui_controlNet_model_dir ,"-o", "control_v11p_sd15_scribble_fp16.safetensors"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_seg_fp16.safetensors", "-d", webui_controlNet_model_dir ,"-o", "control_v11p_sd15_seg_fp16.safetensors"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_softedge_fp16.safetensors", "-d", webui_controlNet_model_dir ,"-o", "control_v11p_sd15_softedge_fp16.safetensors"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15s2_lineart_anime_fp16.safetensors", "-d", webui_controlNet_model_dir ,"-o", "control_v11p_sd15s2_lineart_anime_fp16.safetensors"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile_fp16.safetensors", "-d", webui_controlNet_model_dir ,"-o", "control_v11f1e_sd15_tile_fp16.safetensors"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11e_sd15_ip2p_fp16.yaml", "-d", webui_controlNet_model_dir ,"-o", "control_v11e_sd15_ip2p_fp16.yaml"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11e_sd15_shuffle_fp16.yaml", "-d", webui_controlNet_model_dir ,"-o", "control_v11e_sd15_shuffle_fp16.yaml"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_canny_fp16.yaml", "-d", webui_controlNet_model_dir ,"-o", "control_v11p_sd15_canny_fp16.yaml"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11f1p_sd15_depth_fp16.yaml", "-d", webui_controlNet_model_dir ,"-o", "control_v11f1p_sd15_depth_fp16.yaml"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_inpaint_fp16.yaml", "-d", webui_controlNet_model_dir ,"-o", "control_v11p_sd15_inpaint_fp16.yaml"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_lineart_fp16.yaml", "-d", webui_controlNet_model_dir ,"-o", "control_v11p_sd15_lineart_fp16.yaml"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_mlsd_fp16.yaml", "-d", webui_controlNet_model_dir ,"-o", "control_v11p_sd15_mlsd_fp16.yaml"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_normalbae_fp16.yaml", "-d", webui_controlNet_model_dir ,"-o", "control_v11p_sd15_normalbae_fp16.yaml"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_openpose_fp16.yaml", "-d", webui_controlNet_model_dir ,"-o", "control_v11p_sd15_openpose_fp16.yaml"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_scribble_fp16.yaml", "-d", webui_controlNet_model_dir ,"-o", "control_v11p_sd15_scribble_fp16.yaml"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_seg_fp16.yaml", "-d", webui_controlNet_model_dir ,"-o", "control_v11p_sd15_seg_fp16.yaml"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_softedge_fp16.yaml", "-d", webui_controlNet_model_dir ,"-o", "control_v11p_sd15_softedge_fp16.yaml"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15s2_lineart_anime_fp16.yaml", "-d", webui_controlNet_model_dir ,"-o", "control_v11p_sd15s2_lineart_anime_fp16.yaml"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11f1e_sd15_tile_fp16.yaml", "-d", webui_controlNet_model_dir ,"-o", "control_v11f1e_sd15_tile_fp16.yaml"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_style_sd14v1.pth", "-d", webui_controlNet_model_dir ,"-o", "t2iadapter_style_sd14v1.pth"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_sketch_sd14v1.pth", "-d", webui_controlNet_model_dir ,"-o", "t2iadapter_sketch_sd14v1.pth"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_seg_sd14v1.pth", "-d", webui_controlNet_model_dir ,"-o", "t2iadapter_seg_sd14v1.pth"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_openpose_sd14v1.pth", "-d", webui_controlNet_model_dir ,"-o", "t2iadapter_openpose_sd14v1.pth"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_keypose_sd14v1.pth", "-d", webui_controlNet_model_dir ,"-o", "t2iadapter_keypose_sd14v1.pth"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_depth_sd14v1.pth", "-d", webui_controlNet_model_dir ,"-o", "t2iadapter_depth_sd14v1.pth"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_color_sd14v1.pth", "-d", webui_controlNet_model_dir ,"-o", "t2iadapter_color_sd14v1.pth"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_canny_sd14v1.pth", "-d", webui_controlNet_model_dir ,"-o", "t2iadapter_canny_sd14v1.pth"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_canny_sd15v2.pth", "-d", webui_controlNet_model_dir ,"-o", "t2iadapter_canny_sd15v2.pth"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_depth_sd15v2.pth", "-d", webui_controlNet_model_dir ,"-o", "t2iadapter_depth_sd15v2.pth"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_sketch_sd15v2.pth", "-d", webui_controlNet_model_dir ,"-o", "t2iadapter_sketch_sd15v2.pth"])
        #subprocess .run(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", "https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_zoedepth_sd15v1.pth", "-d", webui_controlNet_model_dir ,"-o", "t2iadapter_zoedepth_sd15v1.pth"])

    print(Fore.CYAN + "\n---------- セットアップ完了 ----------\n")

    # WebUIを起動
    sys.path.append(webui_dir)
    sys.argv += shlex.split("--skip-install --xformers")
    os.chdir(webui_dir)
    from launch import start, prepare_environment

    prepare_environment()
    # 最初のargumentは無視されるので注意 ※なんかPythonのライブラリを更新を促されますが調子悪くなりそうだったので一旦--skip-version-checkで飛ばしてます
    sys.argv = shlex.split("--a --gradio-debug --share --xformers --enable-insecure-extension-access --skip-version-check")
    start()


@stub.local_entrypoint()
def main():
    run_stable_diffusion_webui.remote()