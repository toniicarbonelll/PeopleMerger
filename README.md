# People Merger
This project aims at showing a web application I developped to merge fashion ad fotos. It does so by:
1. Using LLAVA to analye the person, clothes and scenario of every foto 
2. Using GEMMA to summarize the combined prompt (ex: "*foto_5*"'s person, "*foto_1*"'s clothes and "*foto_3*"'s scenario) into a maximum of 55 words (approx 77 tokens)
3. Inputing the summarized prompt into FLUX and generating an image

<br>
I did not publish my angular code. I did however commit all the python notebooks and APIs for anyone that whishes to copy or use them. All the notebooks are correctly commented and very straightforward to use.  

<br>
🎬 Here you can access a demo of the angular app (total of **3mins**, but recommended to play it a 1.5 speed):

<a href="https://youtu.be/LkixjV7Zq2M"><img src="assets/IOG.png" alt="HTML tutorial"></a>


## Previous requirements

### 1. Conda
We will use a conda environment called "pmerger".
- In order to create it:
```bash
conda create --name pmerger python=3.12.12
conda activate pmerger
conda install -n pmerger ipykernel --update-deps --force-reinstall
```
 - This environment will require the following installations:
    - ipywidgets
    - ninja
    - packaging
    - transformers
    - xformers 
    - accelerate
    - diffusers
    - triton-windows
    - huggingface_hub
    - protobuf
    - sentenpiece
    - safetensors
    - fastapi
    - uvicorn

- You will also need "flash-attention" installed, use a pre-built-wheel to install it (ex: *pip install ./downloaded_file_path*):
    - https://huggingface.co/Wildminder/AI-windows-whl/tree/mainhttps://huggingface.co/Wildminder/AI-windows-whl/tree/main

- You will also need to install NUNCHAKU in order to use flux:
    -  https://nunchaku.tech/docs/nunchaku/installation/installation.html
    - https://nunchaku.tech/docs/nunchaku/usage/basic_usage.html

### 2. CUDA
You will need to have an NVIDIA GPU in your system, and the corresponding driver also installed. Then, make sure you have a matching pyTorch version installed (*pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128*). You can verify if everything works with the following commands:
1. conda activate pmerger
2. python
3. > ... import torch
4. > ... torch.cuda.is_available()

In my case I had:
- CUDA: cuda_12.8.0_571.96_windows.exe
- torch: 2.9.0+cu128 

If you are not sure which versions to use:
1. Open cmd
2. Type nvidia-smi
3. Search for your corresponding NVIDIA DRIVER 
4. Download and install the CUDA TOOLKIT: (ex: https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)
5. Search for the corresponding pyTorch version and install it inside your conda environment

### 3. Paths and credentials
1. Some lines of code use paths, and will hence fail if directly executed. Beware of this and correct it using the adequate paths within your computer
2. A lot of functions use "credentials.py", mostly for proxy credentials setting. Create a file as follows and set it up with your credentials information:

    ```bash
    class Credentials:
        http_proxy:str = "http://..."
        https_proxy:str = "http://..."
        proxy_server: str = "..."
        hf_token: str = "..."

### 4. HuggingFace Hub
You will need a HuggingFace account with an access token set-up, which you will need to insert into credentials.py, in order to be able to use some of the HF models.

## Usage
1. Make sure everything is correctly set up. (See requirements section).
2. We recommend to use the notebooks directly in order to try the models (in the specified order).
3. If you want to see how you could "APIFY" those models, then check the "/api" folder and the "app.py" code. Use "uvicorn" to deploy it and "POSTMAN" to try it.