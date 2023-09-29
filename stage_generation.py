import torch
import numpy as np
from PIL import Image

from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL


class StageGenerator:

    def __init__(self, device):
        self.device = device
        self.depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(self.device)
        self.feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0",
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to(self.device)
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to(self.device)
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            vae=vae,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to(self.device)
        self.pipe.enable_model_cpu_offload()

    def get_depth_map(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        image = self.feature_extractor(images=image, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad(), torch.autocast('cuda'):
            depth_map = self.depth_estimator(image).predicted_depth

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(1024, 1024),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        image = torch.cat([depth_map] * 3, dim=1)

        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
        return image

    def __call__(self, prompt, depth_image: Image.Image, steps=30, controlnet_conditioning_scale=0.5) -> Image.Image:
        result = self.pipe(
            prompt, image=depth_image, num_inference_steps=30,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        ).images[0]
        return result


def main(prompt, input_image_path, device):
    stage_generator = StageGenerator(device)
    depth_map = stage_generator.get_depth_map(Image.open(input_image_path))
    depth_map.save('data/depth_map.png')
    generated = stage_generator(prompt, depth_map)
    generated.save('data/generated.png')


if __name__ == "__main__":
    # prompt = "Sidescroller game level of an 80's cyberpunk style scrap yard, arcade sidescroller, pixel art, arcade, low contrast, flat art, shadowrun, bladerunner, vladimir trechikoff, stylized comic, game art, hand painted, low detail"
    prompt = "a city street filled with lots of tall buildings, muted deep neon color, metal slug concept art, chasm, 8 k highly detailed ❤🔥 🔥 💀 🤖 🚀, lofi artstyle, background ( dark _ smokiness ), hand - drawn 2 d art, industrial space, 21:9, garage, by senior artist, unknown space, dark blurry background"
    main(prompt, 'data/3_1.jpg', 'cuda:0')
