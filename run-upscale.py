import torch
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image

from modules import platform as p

# ref
# https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/upscale

# torch
torch_device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

dtype = torch.bfloat16
revision = "fp16"

if p.is_mac_arm():
    # BFloat16 is not supported on MPS
    torch_device = "mps"
    dtype = None

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
