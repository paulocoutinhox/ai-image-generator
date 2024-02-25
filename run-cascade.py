import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline

# ref
# https://huggingface.co/stabilityai/stable-cascade

# torch
torch_device = torch.device("cpu")
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    torch_device = torch.device("mps")
if torch.cuda.is_available():
    torch_device = torch.device("cuda")

dtype = torch.bfloat16 if torch_device.type == "cpu" else torch.float

# model
prior = StableCascadePriorPipeline.from_pretrained(
    "stabilityai/stable-cascade-prior",
    torch_dtype=dtype,
).to(torch_device)

decoder = StableCascadeDecoderPipeline.from_pretrained(
    "stabilityai/stable-cascade",
    use_safetensors=True,
    torch_dtype=torch.half,
).to(torch_device)

model_cpu_offload = False

if model_cpu_offload:
    prior.enable_model_cpu_offload()
    decoder.enable_model_cpu_offload()

# prompt
prompt = "a woman with red hair, realistic"
negative_prompt = "tattooing, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, braid hair"

# generate
num_images_per_prompt = 2

prior_output = prior(
    prompt=prompt,
    height=1024,
    width=1024,
    negative_prompt=negative_prompt,
    guidance_scale=4.0,
    num_images_per_prompt=num_images_per_prompt,
    num_inference_steps=20,
)

decoder_output = decoder(
    image_embeddings=prior_output.image_embeddings,
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=0.0,
    output_type="pil",
    num_inference_steps=10,
).images

# save image
out_image = decoder_output[0]
out_image.save("output-cascade.png")
