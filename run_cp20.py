# run_cp20.py
# 2023/11/04 19:11 chilloutmix_NiPrunedFp32Fix.safetensors adde.
# 2023/11/04 19:48 –-enable-insecure-extension-access added.

from colorama import Fore
from pathlib import Path

import modal
import shutil
import subprocess
import sys
import shlex
import os

import requests

# modal系の変数の定義
stub = modal.Stub("stable-diffusion-webui-2")
volume_main = modal.NetworkFileSystem.persisted("stable-diffusion-webui-main-2")

# 色んなパスの定義
webui_dir = "/content/stable-diffusion-webui"
webui_model_dir = webui_dir + "/models/Stable-diffusion/"

# kanpiromix ##########
# CORRECTED.
target_string = '/content'
print(target_string + ': ' + str(os.path.exists(target_string)))

# CORRECTED.
# CORRECTED.
target_string = webui_dir
safe_path = Path(target_string)
print('safe_path: ' + str(safe_path))
print(target_string + ': ' + str(os.path.exists(safe_path)))

# CORRECTED.
# CORRECTED.
target_string = '/content/stable-diffusion-webui'
safe_path = Path(target_string)
print('safe_path: ' + str(safe_path))
print(target_string + ': ' + str(os.path.exists(safe_path)))

# CORRECTED.
# CORRECTED.
target_string = '/content/stable-diffusion-webui/models'
safe_path = Path(target_string)
print('safe_path: ' + str(safe_path))
print(target_string + ': ' + str(os.path.exists(safe_path)))

# CORRECTED.
# CORRECTED.
target_string = '/content/stable-diffusion-webui/models/Stable-diffusion'
safe_path = Path(target_string)
print('safe_path: ' + str(safe_path))
print(target_string + ': ' + str(os.path.exists(safe_path)))

# CORRECTED.
# CORRECTED.
target_string = '/models'
safe_path = Path(target_string)
print('safe_path: ' + str(safe_path))
print(target_string + ': ' + str(os.path.exists(safe_path)))

# chdir ##########
#os.chdir(webui_dir)
#print('chdir: OK.')

# requirement.txt TEST ##########
print('*** requirement.txt START ***')
#os.chdir(webui_dir)
import pip, site, importlib
#pip.main(['install', '--user', 'modelx'])  # pip install --user modelx を実行
pip.main(['freeze'])  # pip freezeを実行？？？
importlib.reload(site)  
print('*** requirement.txt END ***')


# モデルのID
# coffee0412/KanPiroMix
model_ids = [
    {
        "repo_id": "92john/chilloutmix_NiPrunedFp32Fix.safetensors",
        "model_path": "chilloutmix_NiPrunedFp32Fix.safetensors",
#        "config_file_path": "stable-diffusion/configs/stable-diffusion/v2-inference-v.yaml",
    },
    {
        "repo_id": "hakurei/waifu-diffusion-v1-4",
        "model_path": "wd-1-4-anime_e1.ckpt",
        "config_file_path": "wd-1-4-anime_e1.yaml",
    },
    {
        "repo_id": "stabilityai/stable-diffusion-2-1",
        "model_path": "v2-1_768-ema-pruned.safetensors",
#        "config_file_path": "stable-diffusion/configs/stable-diffusion/v2-inference-v.yaml",
    },
    {
        "repo_id": "coffee0412/KanPiroMix",
        "model_path": "kanpiromix_v10.safetensors",
#        "config_file_path": "stable-diffusion/configs/stable-diffusion/kanpiromix_v10.yaml",
    },
]


@stub.function(
    image=modal.Image.from_registry("python:3.8-slim")
    .apt_install(
        "git", "libgl1-mesa-dev", "libglib2.0-0", "libsm6", "libxrender1", "libxext6"
    )
    .run_commands(
        "pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers"
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
        "torchmetrics==0.11.4",
    )
    .pip_install("git+https://github.com/mlfoundations/open_clip.git@bb6e834e9c70d9c27d0dc3ecedeebeaeb1ffad6b"),
    secret=modal.Secret.from_name("my-huggingface-secret"),
    network_file_systems={webui_dir: volume_main},
    gpu="T4",
    timeout=36000,
)
async def run_stable_diffusion_webui():
    print(Fore.CYAN + "\n---------- セットアップ開始 ----------\n")

    webui_dir_path = Path(webui_model_dir)
    if not webui_dir_path.exists():
        subprocess.run(f"git clone -b v2.0 https://github.com/camenduru/stable-diffusion-webui {webui_dir}", shell=True)

    # Hugging faceからファイルをダウンロードしてくる関数
    def download_hf_file(repo_id, filename):
        from huggingface_hub import hf_hub_download

        download_dir = hf_hub_download(repo_id=repo_id, filename=filename)
        return download_dir


    for model_id in model_ids:
        print(Fore.GREEN + model_id["repo_id"] + "のセットアップを開始します...")

        if not Path(webui_model_dir + model_id["model_path"]).exists():
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

    print(Fore.CYAN + "\n---------- セットアップ完了 ----------\n")
    print(Fore.BLACK + "\n---------- ***** ----------\n")
    
    # DOWNLOAD ##########
    check_target = 'kanpiromix_v20.safetensors'

    check_target_full = '/content/stable-diffusion-webui/models/Stable-diffusion/' + check_target
    result = Path(check_target_full).exists()
    print(check_target_full + ": " + str(result))

    # /
    check_target_full = '/' + check_target
    result = Path(check_target_full).exists()
    print(check_target_full + ": " + str(result))
    
    # /content/
    check_target_full = '/content/' + check_target
    result = Path(check_target_full).exists()
    print(check_target_full + ": " + str(result))
    
    # /content/stable-diffusion-webui/
    check_target_full = '/content/stable-diffusion-webui/' + check_target
    result = Path(check_target_full).exists()
    print(check_target_full + ": " + str(result))
    
    # /content/stable-diffusion-webui/models/
    check_target_full = '/content/stable-diffusion-webui/models/' + check_target
    result = Path(check_target_full).exists()
    print(check_target_full + ": " + str(result))
    
    # /content/stable-diffusion-webui/models/Stable-diffusion/
    check_target_full = '/content/stable-diffusion-webui/models/Stable-diffusion/' + check_target
    result = Path(check_target_full).exists()
    print(check_target_full + ": " + str(result))
    
    # sys.exit('END!!!')
    if not Path(check_target_full).exists():
        url='https://civitai.com/api/download/models/64558'
        filename = check_target_full
        r = requests.get(url, stream=True)
        with open(filename, 'wb') as f:
          # CORRECTED.
          counter = 0
          for chunk in r.iter_content(chunk_size=1024):
            counter += 1
          if counter % 1000 == 0:
            print(Fore.CYAN + " (downloading..." + str(counter) + ") ", end=', ')
    
            if chunk:
              f.write(chunk)
              f.flush()
    # yaml
    url='https://huggingface.co/spaces/stabilityai/stable-diffusion/raw/b028a73583ec2096d4fc3c7e95c9e06a24a5e92b/configs/stable-diffusion/v2-inference-v.yaml'
    filename = '/content/stable-diffusion-webui/models/Stable-diffusion/kanpiromix_v20.yaml'
    r = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
      # CORRECTED.
      counter = 0
      for chunk in r.iter_content(chunk_size=1024):
        counter += 1
      if counter % 1000 == 0:
        print(Fore.CYAN + " (downloading..." + str(counter) + ") ", end=', ')

        if chunk:
          f.write(chunk)
          f.flush()
    

    # WebUIを起動
    sys.path.append(webui_dir)
    sys.argv += shlex.split("--skip-install --xformers")
    os.chdir(webui_dir)
    from launch import start, prepare_environment

    prepare_environment()
    # 最初のargumentは無視されるので注意
    # enable-insecure-extension-access
    sys.argv = shlex.split("--a --gradio-debug --share --xformers --enable-insecure-extension-access")
    start()


@stub.local_entrypoint()
def main():
    run_stable_diffusion_webui.remote()
  
