# VLM Utilities
# Simple functions for VLM data preprocessing

from qwen_vl_utils import process_vision_info
from typing import List, Dict, Any, Tuple, Union
import logging
from PIL import Image

logger = logging.getLogger(__name__)

def preprocess_vlm_messages(text_instruction: str, images: Union[Image.Image, List[Image.Image]], processor):
    """
    Complete VLM preprocessing - create messages, process vision, and get final inputs.
    
    Args:
        text_instruction: Robot task instruction
        image_pil: PIL Image object
        processor: VLM processor (AutoProcessor)
        
    Returns:
        VLM inputs ready for model forward
    """
    # Create VLM messages format
    # 1. Normalize images to a list.
    if not isinstance(images, list):
        images = [images]

    # 2. Build the content list.
    # Qwen2-VL supports multiple images in a single user message.
    content = []
    for img in images:
        content.append({"type": "image", "image": img})
    
    # Append the text instruction last.
    content.append({"type": "text", "text": text_instruction})

    # 3. Build the messages.
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Process vision info
    image_inputs, video_inputs = process_vision_info(messages)
    
    # Get final processor inputs
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    return inputs