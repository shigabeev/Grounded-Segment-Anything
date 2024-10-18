import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image
import cv2
from torchvision.ops import box_convert
from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionXLInpaintPipeline
# Remove PIL image size limit
Image.MAX_IMAGE_PIXELS = None
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict

# Segment Anything
from segment_anything import build_sam, SamPredictor

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class GroundedSAM:
    def __init__(self, grounding_dino_config, grounding_dino_checkpoint, sam_checkpoint):
        self.groundingdino_model = self.load_groundingdino_model(grounding_dino_config, grounding_dino_checkpoint)
        self.sam_predictor = self.load_sam_model(sam_checkpoint)
        self.pipe = self.load_inpainting_model()

    def load_groundingdino_model(self, config_path, checkpoint_path):
        args = SLConfig.fromfile(config_path)
        model = build_model(args)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        model.eval()
        return model

    def load_sam_model(self, checkpoint_path):
        sam = build_sam(checkpoint=checkpoint_path)
        sam.to(device=DEVICE)
        return SamPredictor(sam)

    def load_inpainting_model(self):
        float_type = torch.float16 if DEVICE.type != 'cpu' else torch.float32
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=float_type,
        )
        if DEVICE.type != 'cpu':
            pipe = pipe.to("cuda")
        return pipe

    def process_image(self, image_path, text_prompt, box_threshold, text_threshold):
        image_source, image = load_image(image_path)

        boxes, logits, phrases = predict(
            model=self.groundingdino_model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=DEVICE
        )

        # Check if any objects were detected
        if len(boxes) == 0:
            print(f"No objects detected in {image_path}. Returning original image.")
            return Image.fromarray(image_source)

        self.sam_predictor.set_image(image_source)

        H, W, _ = image_source.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(DEVICE)

        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        image_mask = masks[0][0].cpu().numpy()
        kernel = np.ones((5,5), np.uint8)
        dilated_mask = cv2.dilate(image_mask.astype(np.uint8), kernel, iterations=2)
        image_mask = dilated_mask.astype(bool)

        image_source_pil = Image.fromarray(image_source)
        image_mask_pil = Image.fromarray(image_mask)

        expanded_bbox = self.get_expanded_bbox(image_mask_pil, image_source_pil.size)
        image_source_for_inpaint, image_mask_for_inpaint = self.prepare_for_inpainting(image_source_pil, image_mask_pil, expanded_bbox)

        generator = torch.Generator(device="cuda").manual_seed(0)
        prompt = "seamless background, continuous texture"

        image_inpainting = self.inpaint_image(self.pipe, prompt, image_source_for_inpaint, image_mask_for_inpaint, generator)

        final_image = self.paste_inpainted_region(image_source_pil, image_inpainting, expanded_bbox)

        if max(final_image.size) > 2048:
            final_image.thumbnail((2048, 2048), Image.LANCZOS)

        return final_image

    @staticmethod
    def get_expanded_bbox(mask_pil, image_size, target_size=2048):
        mask_bbox = mask_pil.getbbox()
        W, H = image_size
        expand_factor = min(target_size / max(mask_bbox[2] - mask_bbox[0], mask_bbox[3] - mask_bbox[1]), 1)
        expanded_bbox = [
            int(mask_bbox[0] - (1024 / expand_factor - (mask_bbox[2] - mask_bbox[0])) / 2),
            int(mask_bbox[1] - (1024 / expand_factor - (mask_bbox[3] - mask_bbox[1])) / 2),
            int(mask_bbox[2] + (1024 / expand_factor - (mask_bbox[2] - mask_bbox[0])) / 2),
            int(mask_bbox[3] + (1024 / expand_factor - (mask_bbox[3] - mask_bbox[1])) / 2)
        ]
        return [
            max(0, expanded_bbox[0]),
            max(0, expanded_bbox[1]),
            min(W, expanded_bbox[2]),
            min(H, expanded_bbox[3])
        ]

    @staticmethod
    def prepare_for_inpainting(image_pil, mask_pil, bbox):
        image_for_inpaint = image_pil.crop(bbox)
        mask_for_inpaint = mask_pil.crop(bbox)
        image_for_inpaint = image_for_inpaint.resize((1024, 1024), Image.LANCZOS)
        mask_for_inpaint = mask_for_inpaint.resize((1024, 1024), Image.NEAREST)
        return image_for_inpaint, mask_for_inpaint

    @staticmethod
    def inpaint_image(pipe, prompt, image, mask, generator):
        return pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            guidance_scale=8.0,
            num_inference_steps=20,
            strength=0.99,
            generator=generator,
            added_cond_kwargs={"text_embeds":None}
        ).images[0]

    @staticmethod
    def paste_inpainted_region(original_image, inpainted_image, bbox):
        result = original_image.copy()
        inpainted_image = inpainted_image.resize((bbox[2] - bbox[0], bbox[3] - bbox[1]), Image.LANCZOS)
        result.paste(inpainted_image, (bbox[0], bbox[1]))
        return result

def process_folder(input_folder, output_folder, model):
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_inpainted.png")
            
            try:
                result_image = model.process_image(
                    input_path,
                    text_prompt="watermark, logo, brand, text",
                    box_threshold=0.3,
                    text_threshold=0.25
                )
                if result_image.size == Image.open(input_path).size:
                    print(f"Skipped {filename} - No objects detected")
                else:
                    result_image.save(output_path)
                    print(f"Processed {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Process images in a folder using Grounded-SAM")
    parser.add_argument("--input_folder", help="Path to the input folder containing images")
    parser.add_argument("--output_folder", help="Path to the output folder for processed images")
    args = parser.parse_args()

    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filename = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

    grounding_dino_config = hf_hub_download(repo_id=ckpt_repo_id, filename=ckpt_config_filename)
    grounding_dino_checkpoint = hf_hub_download(repo_id=ckpt_repo_id, filename=ckpt_filename)
    sam_checkpoint = 'sam_vit_h_4b8939.pth'

    model = GroundedSAM(grounding_dino_config, grounding_dino_checkpoint, sam_checkpoint)
    process_folder(args.input_folder, args.output_folder, model)

if __name__ == "__main__":
    main()
