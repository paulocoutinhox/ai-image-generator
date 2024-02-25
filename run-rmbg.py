import numpy as np
import torch
import torch.nn.functional as F
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.transforms.functional import normalize

from modules import platform as p
from modules.briarmbg import BriaRMBG


def prepare_image(img_url):
    image = load_image(img_url)
    return image


def remove_background(image):
    # torch
    torch_device = torch.device("cpu")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        torch_device = torch.device("mps")
    if torch.cuda.is_available():
        torch_device = torch.device("cuda")

    net = BriaRMBG()
    model_path = hf_hub_download("briaai/RMBG-1.4", "model.pth")
    net.load_state_dict(torch.load(model_path, map_location=torch_device))
    net.to(torch_device).eval()

    # prepare image
    image = image.convert("RGB")
    orig_size = image.size

    # if need resize
    # image = image.resize((1024, 1024), Image.BILINEAR)

    im_tensor = (
        torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        / 255.0
    )

    im_tensor = normalize(im_tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]).to(torch_device)

    # inference
    with torch.no_grad():
        result = net(im_tensor)

    result = F.interpolate(result[0][0], size=orig_size, mode="bilinear").squeeze(0)
    mask = ((result - result.min()) / (result.max() - result.min()) * 255).byte()
    pil_mask = Image.fromarray(mask.cpu().numpy()[0], mode="L")

    # final image without bg
    final_image = Image.new("RGBA", pil_mask.size, (0, 0, 0, 0))
    final_image.paste(image.resize(orig_size), mask=pil_mask)
    return final_image


# execution
img_url = "extras/images/model.jpg"
original_image = prepare_image(img_url)
out_image = remove_background(original_image)

# save image
out_image.save("output-rmbg.png")
