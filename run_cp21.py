# run_cp21.py
# 2023/11/04 19:11 chilloutmix_NiPrunedFp32Fix.safetensors adde.
# 2023/11/04 19:48 –-enable-insecure-extension-access added.
# 2023/11/12 17:07 "wheel" added.

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
        "wheel",
        "absl-py==2.0.0",
        "accelerate==0.12.0",
        "addict==2.4.0",
        "aenum==3.1.15",
        "aiofiles==23.2.1",
        "aiohttp==3.8.3",
        "aiosignal==1.3.1",
        "aiostream==0.4.4",
        "altair==5.1.2",
        "antlr4-python3-runtime==4.9.3",
        "anyio==3.7.1",
        "asgiref==3.5.2",
        "asttokens==2.4.0",
        "async-timeout==4.0.3",
        "attrs==23.1.0",
        "backcall==0.2.0",
        "basicsr==1.4.2",
        "beautifulsoup4==4.12.2",
        "blendmodes==2022",
        "boltons==23.0.0",
        "bytecode==0.15.1",
        "cachetools==5.3.1",
        "cattrs==23.1.2",
        "certifi==2023.7.22",
        "chardet==4.0.0",
        "charset-normalizer==2.1.1",
        "clean-fid==0.1.29",
        "click==8.1.7",
        "clip==0.2.0",
        "cloudpickle==2.0.0",
        "colorama==0.4.6",
        "commonmark==0.9.1",
        "contourpy==1.1.1",
        "cycler==0.12.1",
        "ddsketch==2.0.4",
        "ddtrace==1.5.2",
        "decorator==5.1.1",
        "deprecation==2.1.0",
        "diffusers==0.18.2",
        "einops==0.4.1",
        "envier==0.4.0",
        "exceptiongroup==1.1.3",
        "executing==2.0.0",
        "facexlib==0.3.0",
        "fastapi==0.88.0",
        "fastprogress==1.0.0",
        "ffmpy==0.3.1",
        "filelock==3.12.4",
        "filterpy==1.4.5",
        "font-roboto==0.0.1",
        "fonts==0.0.3",
        "fonttools==4.43.1",
        "frozenlist==1.4.0",
        "fsspec==2023.10.0",
        "ftfy==6.1.1",
        "future==0.18.3",
        "gdown==4.7.1",
        "gfpgan==1.3.8",
        "gitdb==4.0.11",
        "GitPython==3.1.27",
        "google-auth==2.23.3",
        "google-auth-oauthlib==0.4.6",
        "gradio==3.16.2",
        "grpcio==1.59.0",
        "grpclib==0.4.3",
        "h11==0.12.0",
        "h2==4.1.0",
        "hpack==4.0.0",
        "httpcore==0.15.0",
        "httpx==0.24.1",
        "huggingface-hub==0.18.0",
        "hyperframe==6.0.1",
        "idna==2.10",
        "imageio==2.31.5",
        "importlib-metadata==6.8.0",
        "importlib-resources==6.1.0",
        "inflection==0.5.1",
        "invisible-watermark==0.2.0",
        "ipython==8.12.3",
        "jedi==0.19.1",
        "Jinja2==3.1.2",
        "jsonmerge==1.8.0",
        "jsonschema==4.19.1",
        "jsonschema-specifications==2023.7.1",
        "kiwisolver==1.4.5",
        "kornia==0.6.7",
        "lark==1.1.2",
        "linkify-it-py==2.0.2",
        "llvmlite==0.41.1",
        "lmdb==1.4.1",
        "Markdown==3.5",
        "markdown-it-py==3.0.0",
        "MarkupSafe==2.1.3",
        "matplotlib==3.7.3",
        "matplotlib-inline==0.1.6",
        "mdit-py-plugins==0.4.0",
        "mdurl==0.1.2",
        "modal==0.55.4091",
        "mpmath==1.3.0",
        "multidict==6.0.4",
        "mypy-extensions==1.0.0",
        "networkx==3.1",
        "numba==0.58.1",
        "numpy==1.23.3",
        "nvidia-cublas-cu11==11.10.3.66",
        "nvidia-cublas-cu12==12.1.3.1",
        "nvidia-cuda-cupti-cu12==12.1.105",
        "nvidia-cuda-nvrtc-cu11==11.7.99",
        "nvidia-cuda-nvrtc-cu12==12.1.105",
        "nvidia-cuda-runtime-cu11==11.7.99",
        "nvidia-cuda-runtime-cu12==12.1.105",
        "nvidia-cudnn-cu11==8.5.0.96",
        "nvidia-cudnn-cu12==8.9.2.26",
        "nvidia-cufft-cu12==11.0.2.54",
        "nvidia-curand-cu12==10.3.2.106",
        "nvidia-cusolver-cu12==11.4.5.107",
        "nvidia-cusparse-cu12==12.1.0.106",
        "nvidia-nccl-cu12==2.18.1",
        "nvidia-nvjitlink-cu12==12.3.52",
        "nvidia-nvtx-cu12==12.1.105",
        "oauthlib==3.2.2",
        "omegaconf==2.2.3",
        "open-clip-torch @ git+https://github.com/mlfoundations/open_clip.git@bb6e834e9c70d9c27d0dc3ecedeebeaeb1ffad6b",
        "opencv-python==4.8.1.78",
        "orjson==3.9.9",
        "packaging==23.2",
        "pandas==2.0.3",
        "parso==0.8.3",
        "pexpect==4.8.0",
        "pickleshare==0.7.5",
        "piexif==1.1.3",
        "Pillow==9.4.0",
        "pkgutil_resolve_name==1.3.10",
        "platformdirs==3.11.0",
        "prompt-toolkit==3.0.39",
        "protobuf==3.20.0",
        "psutil==5.9.6",
        "ptyprocess==0.7.0",
        "pure-eval==0.2.2",
        "pyasn1==0.5.0",
        "pyasn1-modules==0.3.0",
        "pycryptodome==3.19.0",
        "pydantic==1.10.13",
        "pyDeprecate==0.3.2",
        "pydub==0.25.1",
        "Pygments==2.16.1",
        "pyngrok==7.0.0",
        "pyparsing==3.1.1",
        "pyre-extensions==0.0.23",
        "PySocks==1.7.1",
        "python-dateutil==2.8.2",
        "python-multipart==0.0.6",
        "pytorch-lightning==1.7.6",
        "pytz==2023.3.post1",
        "PyWavelets==1.4.1",
        "PyYAML==6.0.1",
        "realesrgan==0.3.0",
        "referencing==0.30.2",
        "regex==2023.10.3",
        "requests==2.25.1",
        "requests-oauthlib==1.3.1",
        "resize-right==0.0.2",
        "rich==12.3.0",
        "rpds-py==0.10.6",
        "rsa==4.9",
        "safetensors==0.2.7",
        "scikit-image==0.19.2",
        "scipy==1.10.1",
        "sentencepiece==0.1.99",
        "six==1.16.0",
        "smmap==5.0.1",
        "sniffio==1.3.0",
        "soupsieve==2.5",
        "stack-data==0.6.3",
        "starlette==0.22.0",
        "sympy==1.12",
        "-e git+https://github.com/CompVis/taming-transformers.git@3ba01b241669f5ade541ce990f7650a3b8f65318#egg=taming_transformers",
        "tb-nightly==2.12.0a20230126",
        "tblib==1.7.0",
        "tenacity==8.2.3",
        "tensorboard==2.9.1",
        "tensorboard-data-server==0.6.1",
        "tensorboard-plugin-wit==1.8.1",
        "test-tube==0.7.5",
        "tifffile==2023.7.10",
        "timm==0.6.7",
        "tokenizers==0.13.3",
        "toml==0.10.2",
        "tomli==2.0.1",
        "toolz==0.12.0",
        "torch==1.13.1",
        "torchdiffeq==0.2.3",
        "torchmetrics==0.11.4",
        "torchsde==0.2.5",
        "torchvision==0.14.1",
        "tqdm==4.66.1",
        "traitlets==5.11.2",
        "trampoline==0.1.2",
        "transformers==4.25.1",
        "triton==2.1.0",
        "typeguard==4.1.5",
        "typer==0.6.1",
        "types-certifi==2021.10.8.3",
        "types-toml==0.10.4",
        "typing-inspect==0.9.0",
        "typing_extensions==4.8.0",
        "tzdata==2023.3",
        "uc-micro-py==1.0.2",
        "urllib3==1.26.18",
        "uvicorn==0.23.2",
        "wcwidth==0.2.8",
        "websockets==12.0",
        "Werkzeug==3.0.0",
        "xformers==0.0.16rc425",
        "xmltodict==0.13.0",
        "yapf==0.40.2",
        "yarl==1.9.2",
        "zipp==3.17.0",
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
  
