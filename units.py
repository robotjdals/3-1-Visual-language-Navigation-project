import torch
import numpy as np
import os


from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor



def make_cfg(settings):
    # AI2-THOR adapter path keeps the old helper signature but no longer needs
    # a Habitat-specific configuration object.
    return settings
        
def last_non_space_char(s):
    for char in reversed(s):
        if char != ' ':
            return char
    return None 

def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    box_info_list = []

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)
        
        # 去除label中的括号及其内容
        clean_label = label.split('(')[0]
        
        # 将语义和 box 信息添加到 box_info_list 中
        box_info_list.append({
            "label": clean_label,
            "box": [x0, y0, x1, y1]  # 保存的 box 是在图像坐标系中的整数值
        })

    return image_pil, mask, box_info_list



def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None,text_prompt = None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = model["device"]
    processor = model["processor"]
    detector = model["model"]

    image_pil = image if isinstance(image, Image.Image) else None
    if image_pil is None:
        raise ValueError("HF Grounding DINO wrapper expects a PIL image input.")

    if token_spans is not None:
        # TODO: HF native Grounding DINO wrapper below currently supports the
        # prompt-based path used by EfficientNav. Token-span supervision is kept
        # as a compatibility hook but is not implemented.
        raise NotImplementedError("token_spans path is not implemented for HF-native Grounding DINO.")

    target_labels = [item.strip() for item in caption.split(".") if item.strip()]
    text = ". ".join(target_labels) + "."
    inputs = processor(images=image_pil, text=text, return_tensors="pt")
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.no_grad():
        outputs = detector(**inputs)

    processed = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image_pil.size[::-1]],
    )[0]

    boxes = processed["boxes"].detach().cpu()
    scores = processed["scores"].detach().cpu()
    labels = processed.get("text_labels", processed["labels"])
    width, height = image_pil.size
    boxes_filt = []
    pred_phrases = []
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.tolist()
        cx = ((x1 + x2) / 2.0) / width
        cy = ((y1 + y2) / 2.0) / height
        w = max(x2 - x1, 1e-6) / width
        h = max(y2 - y1, 1e-6) / height
        boxes_filt.append([cx, cy, w, h])
        if with_logits:
            pred_phrases.append(f"{label}({score.item():.2f})")
        else:
            pred_phrases.append(str(label))

    if len(boxes_filt) == 0:
        boxes_filt = torch.zeros((0, 4), dtype=torch.float32)
    else:
        boxes_filt = torch.tensor(boxes_filt, dtype=torch.float32)
    return boxes_filt, pred_phrases


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    model_id = os.environ.get("EFFICIENTNAV_GDINO_MODEL_ID", model_checkpoint_path)
    device = "cuda" if torch.cuda.is_available() and not cpu_only else "cpu"
    print(f"[EfficientNav units] HF-native Grounding DINO loader: file={__file__}, model_id={model_id}, device={device}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    return {
        "processor": processor,
        "model": model,
        "device": device,
    }

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image
    return image_pil, image_pil
