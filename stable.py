from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.float16)
pipe = pipe.to("mps")

prompt = "Dhoni hitting a six"
image = pipe(prompt).images[0]
image.save("output.png")
