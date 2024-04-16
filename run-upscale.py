import torch
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image

from modules import platform as p

# ref
# https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/upscale

# torch
torch_device = torch.device("cpu")

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    torch_device = torch.device("mps")

if torch.cuda.is_available():
    torch_device = torch.device("cuda")

dtype = torch.bfloat16 if torch_device.type == "cpu" else torch.float
revision = "fp16"

# model
sd_model_path = "stabilityai/stable-diffusion-x4-upscaler"

pipe = StableDiffusionUpscalePipeline.from_pretrained(
    sd_model_path,
    torch_dtype=dtype,
    revision=revision,
)

# pipe
pipe = pipe.to(torch_device)

if p.memory_less_64gb():
    pipe.enable_attention_slicing()

# prompt
suffix = ", 8k, photography, cgi, unreal engine, octane render, best quality"
prompt = f"a white cat {suffix}"
negative_prompt = "jpeg artifacts, lowres, bad quality"

# generate
low_res_img = Image.open("extras/images/low-res-model.png").convert("RGB")

output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=low_res_img,
)

# save image
out_image = output.images[0]
out_image.save("output-upscale.png")
