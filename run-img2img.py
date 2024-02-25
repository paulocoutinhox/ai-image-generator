import torch
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.utils import load_image

from modules import platform as p

# ref
# https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img

# torch
torch_device = torch.device("cpu")
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    torch_device = torch.device("mps")
if torch.cuda.is_available():
    torch_device = torch.device("cuda")

dtype = torch.bfloat16 if torch_device.type == "cpu" else torch.float
variant = "fp16"

# images - one or multiple
image_paths = ["extras/images/model.jpg"]
reference_images = [load_image(image_path) for image_path in image_paths]

# model
sd_model_path = "Lykon/dreamshaper-8"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    sd_model_path,
    use_safetensors=True,
    torch_dtype=dtype,
    variant=variant,
)

# pipe
pipe = pipe.to(torch_device)

if p.memory_less_64gb():
    pipe.enable_attention_slicing()

# prompt
prompt = "a man face with red hair"
negative_prompt = "tattooing, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, braid hair"

# generate
output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=reference_images,
    num_inference_steps=20,
    guidance_scale=8.0,
)

# save image
out_image = output.images[0]
out_image.save("output-img2img.png")
