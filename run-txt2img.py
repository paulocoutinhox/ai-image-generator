import torch
from diffusers import StableDiffusionPipeline

from modules import platform as p

# ref
# https://huggingface.co/docs/diffusers/api/pipelines/overview

# torch
torch_device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

dtype = torch.bfloat16
variant = "fp16"

if p.is_mac_arm():
    # BFloat16 is not supported on MPS
    torch_device = "mps"
    dtype = None

# model
sd_model_path = "Lykon/dreamshaper-8"

pipe = StableDiffusionPipeline.from_pretrained(
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
out_image.save("output-txt2img.png")
