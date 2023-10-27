# cloningsdwebui.py
# Replaced old style of modal.com.
# !!!!!!!!!!!!!!!!!!
import os
import subprocess

import modal


persist_dir = "/content"

stub = modal.Stub("stable-diffusion-webui")
volume = modal.NetworkFileSystem.new().persist("sdwebui-volume")

image = modal.Image.debian_slim().apt_install("git")

@stub.function(image=image,
               network_file_systems={persist_dir: volume},
               timeout=1800)
def cloning():
    os.chdir(persist_dir)
    # subprocess.run("git config --global http.postBuffer 200M", shell=True)
    subprocess.run("git clone -b v2.5 https://github.com/camenduru/stable-diffusion-webui", shell=True)

    # download some extensions
    os.chdir(persist_dir + "/stable-diffusion-webui/extensions")
    subprocess.run("git clone https://github.com/kohya-ss/sd-webui-additional-networks", shell=True)
    subprocess.run("git clone https://github.com/Mikubill/sd-webui-controlnet", shell=True)
    subprocess.run("git clone https://github.com/hako-mikan/sd-webui-lora-block-weight", shell=True)
    subprocess.run("git clone https://github.com/hako-mikan/sd-webui-regional-prompter", shell=True)


@stub.local_entrypoint()
def main():
    cloning.remote() # NEW SYNTAX!!!
  
