import torch
from diffusers import (
    EulerDiscreteScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from modules import platform as p

# ref
# https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img

# torch
torch_device = torch.device("cuda")
dtype = torch.bfloat16 if torch_device.type == "cpu" else torch.float
variant = "fp16"

# images - one or multiple
image_paths = ["extras/images/model.jpg"]
reference_images = [load_image(image_path) for image_path in image_paths]

# model
base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_4step_unet.safetensors"

unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(
    torch_device, torch.float16
)
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=torch_device))

# pipe
pipe = StableDiffusionXLPipeline.from_pretrained(
    base, unet=unet, torch_dtype=torch.float16, variant=variant
)
pipe = pipe.to(torch_device)

if p.memory_less_64gb():
    pipe.enable_attention_slicing()

# ensure sampler uses "trailing" timesteps
pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config, timestep_spacing="trailing"
)

# prompt
prompt = "a woman with red hair, realistic"
negative_prompt = "tattooing, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, braid hair"

# generate
output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=20,
    guidance_scale=7.5,
)

# save image
out_image = output.images[0]
out_image.save("output-sdxl-lighting.png")
