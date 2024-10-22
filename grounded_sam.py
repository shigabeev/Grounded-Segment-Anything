import os
import sys
import argparse
import numpy as np
import torch
import shutil
from PIL import Image
from tqdm.auto import tqdm
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
        image_height, image_width, _ = image_source.shape

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
            print(f"No objects detected in {image_path}.")
            return False

        self.sam_predictor.set_image(image_source)

        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([image_width, image_height, image_width, image_height])
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(DEVICE)

        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        image_mask = masks[0][0].cpu().numpy()
        
        # Check if the mask is valid for a watermark
        if not self.is_valid_watermark_mask(image_mask, image_height, image_width):
            print(f"Invalid mask detected in {image_path}.")
            return False


        kernel = np.ones((15,15), np.uint8)
        dilated_mask = cv2.dilate(image_mask.astype(np.uint8), kernel, iterations=2)
        image_mask = dilated_mask.astype(bool)

        image_source_pil = Image.fromarray(image_source)
        image_mask_pil = Image.fromarray(image_mask)

        # Calculate the crop size based on the watermark proportion
        crop_size = self.calculate_crop_size(image_mask_pil, image_source_pil.size)
        
        # Crop and resize the image and mask only if necessary
        image_for_inpaint, mask_for_inpaint, crop_box = self.prepare_for_inpainting(image_source_pil, image_mask_pil, crop_size)

        generator = torch.Generator(device="cuda").manual_seed(0)
        prompt = "A portrait of a beautiful asian woman"
        negative_prompt = "watermark, logo, brand, text"

        image_inpainting = self.inpaint_image(self.pipe, prompt, negative_prompt, image_for_inpaint, mask_for_inpaint, generator)

        final_image = self.paste_inpainted_region(image_source_pil, image_inpainting, crop_box)

        return final_image

    @staticmethod
    def calculate_crop_size(mask_pil, image_size):
        mask_bbox = mask_pil.getbbox()
        W, H = image_size
        
        mask_width = mask_bbox[2] - mask_bbox[0]
        mask_height = mask_bbox[3] - mask_bbox[1]
        
        # Calculate the proportion of the image occupied by the watermark
        width_proportion = mask_width / W
        height_proportion = mask_height / H
        
        # Estimate crop size based on watermark proportion
        crop_width = min(int(mask_width / width_proportion), 2048)
        crop_height = min(int(mask_height / height_proportion), 2048)
        
        return (crop_width, crop_height)

    @staticmethod
    def get_expanded_bbox(mask_pil, image_size, max_size=2048*2048):
        mask_bbox = mask_pil.getbbox()
        W, H = image_size
        
        # Calculate the aspect ratio of the mask
        mask_width = mask_bbox[2] - mask_bbox[0]
        mask_height = mask_bbox[3] - mask_bbox[1]
        aspect_ratio = mask_width / mask_height

        # Determine the new dimensions while maintaining the aspect ratio
        if aspect_ratio > 1:  # Wider than tall
            new_width = min(int(np.sqrt(max_size * aspect_ratio)), W)
            new_height = int(new_width / aspect_ratio)
        else:  # Taller than wide
            new_height = min(int(np.sqrt(max_size / aspect_ratio)), H)
            new_width = int(new_height * aspect_ratio)

        # Calculate the expansion factors
        expand_x = (new_width - mask_width) / 2
        expand_y = (new_height - mask_height) / 2

        # Calculate the new bounding box
        expanded_bbox = [
            int(max(0, mask_bbox[0] - expand_x)),
            int(max(0, mask_bbox[1] - expand_y)),
            int(min(W, mask_bbox[2] + expand_x)),
            int(min(H, mask_bbox[3] + expand_y))
        ]

        return expanded_bbox

    @staticmethod
    def prepare_for_inpainting(image_pil, mask_pil, crop_size):
        W, H = image_pil.size
        crop_width, crop_height = crop_size
        
        # Calculate crop box
        mask_bbox = mask_pil.getbbox()
        center_x = (mask_bbox[0] + mask_bbox[2]) // 2
        center_y = (mask_bbox[1] + mask_bbox[3]) // 2
        
        left = max(0, center_x - crop_width // 2)
        top = max(0, center_y - crop_height // 2)
        right = min(W, left + crop_width)
        bottom = min(H, top + crop_height)
        
        crop_box = (left, top, right, bottom)
        
        # Crop image and mask
        image_for_inpaint = image_pil.crop(crop_box)
        mask_for_inpaint = mask_pil.crop(crop_box)
        
        # Resize only if necessary
        if crop_width > 2048 or crop_height > 2048:
            image_for_inpaint.thumbnail((2048, 2048), Image.LANCZOS)
            mask_for_inpaint = mask_for_inpaint.resize(image_for_inpaint.size, Image.NEAREST)
        
        # Ensure dimensions are divisible by 8
        width, height = image_for_inpaint.size
        new_width = (width // 8) * 8
        new_height = (height // 8) * 8
        
        if (width, height) != (new_width, new_height):
            image_for_inpaint = image_for_inpaint.resize((new_width, new_height), Image.LANCZOS)
            mask_for_inpaint = mask_for_inpaint.resize((new_width, new_height), Image.NEAREST)
        
        return image_for_inpaint, mask_for_inpaint, crop_box

    @staticmethod
    def inpaint_image(pipe, prompt, negative_prompt, image, mask, generator):
        return pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            guidance_scale=8.0,
            num_inference_steps=20,
            strength=0.99,
            generator=generator,
            height=image.size[1],
            width=image.size[0]
        ).images[0]

    @staticmethod
    def paste_inpainted_region(original_image, inpainted_image, crop_box):
        result = original_image.copy()
        result.paste(inpainted_image, (crop_box[0], crop_box[1]))
        return result

    @staticmethod
    def is_valid_watermark_mask(mask, height, width, max_mask_percentage = 0.10):
        mask_percentage = np.sum(mask) / (height * width)
        return mask_percentage <= max_mask_percentage


def process_folder(input_folder, output_folder, model):
    os.makedirs(output_folder, exist_ok=True)
    
    # Get a list of all files in the input folder
    all_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
    
    # Get a list of already processed files
    processed_files = [f.replace('_inpainted.png', '') for f in os.listdir(output_folder) if f.endswith('_inpainted.png')]
    
    # Filter out already processed files
    files_to_process = [f for f in all_files if os.path.splitext(f)[0] not in processed_files]
    
    for filename in tqdm(files_to_process):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_inpainted.png")
        
        try:
            result_image = model.process_image(
                input_path,
                text_prompt="watermark, logo, brand, text",
                box_threshold=0.3,
                text_threshold=0.25
            )
            if result_image is False:
                # Copy the original file with a different suffix
                original_output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_original{os.path.splitext(filename)[1]}")
                shutil.copy2(input_path, original_output_path)
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
