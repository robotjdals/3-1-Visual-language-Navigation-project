import os
import sys
import site
import gzip
import json
import re
import time

# Keep the planner env self-contained by preventing ~/.local packages from
# shadowing conda-installed torch/torchvision dependencies.
site.ENABLE_USER_SITE = False
user_site = site.getusersitepackages()
if user_site:
    sys.path = [path for path in sys.path if os.path.abspath(path) != os.path.abspath(user_site)]

from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    AutoProcessor,
)
try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    AutoModelForVision2Seq = None
try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None
import torch
import random
import numpy as np
import imageio
import matplotlib.pyplot as plt
from math import ceil
from PIL import Image as I, ImageDraw
import math
import gc
import copy
from collections import namedtuple
from transformers import CLIPTokenizer, CLIPTextModel
from scipy.spatial.distance import cosine,euclidean
import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
from navigation_map import Navigation_map
import units
from units import load_image,load_model,get_grounding_output,plot_boxes_to_image,last_non_space_char,make_cfg
from thor_adapter import ThorAgentState, ThorShortestPath, ThorSim, canonical_goal_name, load_procthor_houses, vector_to_yaw, yaw_to_vector
import rclpy
from rclpy.node import Node
try:
    from efficientnav_interfaces.srv import DetectObjects
except ImportError:
    DetectObjects = None

print(f"[EfficientNav] running file: {__file__}")
print(f"[EfficientNav] units module: {units.__file__}")

os.makedirs("navigation_images", exist_ok=True)
os.makedirs("tmp/navigation_images", exist_ok=True)

os.environ.setdefault("ROS_DOMAIN_ID", "30")
os.environ.setdefault("EFFICIENTNAV_USE_ROS2_DETECTION", "1")
os.environ.setdefault("EFFICIENTNAV_USE_KV_CACHE", "1")
os.environ.setdefault("EFFICIENTNAV_PLANNER_MODEL_PATH", "/home/min/test/models/SmolVLM-500M-Instruct")
os.environ.setdefault("EFFICIENTNAV_CLIP_PATH", "/home/min/models/clip-vit-base-patch32")
os.environ.setdefault("EFFICIENTNAV_RENDER_WIDTH", "512")
os.environ.setdefault("EFFICIENTNAV_RENDER_HEIGHT", "512")
os.environ.setdefault("EFFICIENTNAV_PLANNER_ATTN_IMPLEMENTATION", "eager")

cuda_available = torch.cuda.is_available()
cuda_device_count = torch.cuda.device_count() if cuda_available else 0
primary_device = "cuda:0" if cuda_available and cuda_device_count > 0 else "cpu"
max_memory = {idx: "47GiB" for idx in range(cuda_device_count)} if cuda_available else None
planner_device_map = "auto" if cuda_device_count > 0 else None
planner_model_path = os.environ.get("EFFICIENTNAV_PLANNER_MODEL_PATH", "/home/min/test/models/SmolVLM-500M-Instruct")
planner_processor = AutoProcessor.from_pretrained(planner_model_path, trust_remote_code=True)
planner_tokenizer = getattr(planner_processor, "tokenizer", None)
if planner_tokenizer is None:
    planner_tokenizer = AutoTokenizer.from_pretrained(planner_model_path, trust_remote_code=True)
if planner_tokenizer.pad_token is None:
    planner_tokenizer.pad_token = planner_tokenizer.eos_token
planner_model_kwargs = {
    "torch_dtype": torch.float16 if cuda_available else torch.float32,
    "low_cpu_mem_usage": True,
    "trust_remote_code": True,
}
planner_attn_implementation = os.environ.get("EFFICIENTNAV_PLANNER_ATTN_IMPLEMENTATION")
if planner_attn_implementation:
    planner_model_kwargs["attn_implementation"] = planner_attn_implementation
if planner_device_map is not None:
    planner_model_kwargs["device_map"] = planner_device_map
    if max_memory:
        planner_model_kwargs["max_memory"] = max_memory
if AutoModelForVision2Seq is not None:
    try:
        planner_model = AutoModelForVision2Seq.from_pretrained(planner_model_path, **planner_model_kwargs)
    except Exception:
        planner_model = AutoModelForImageTextToText.from_pretrained(planner_model_path, **planner_model_kwargs)
else:
    planner_model = AutoModelForImageTextToText.from_pretrained(planner_model_path, **planner_model_kwargs)
planner_supports_vision = hasattr(planner_processor, "image_processor")
use_ros2_detection = os.environ.get("EFFICIENTNAV_USE_ROS2_DETECTION", "1") == "1"
observation_rotation_pause = float(os.environ.get("EFFICIENTNAV_OBSERVATION_ROTATION_PAUSE", "0.25"))
ros2_detection_timeout_sec = float(os.environ.get("EFFICIENTNAV_ROS2_DETECTION_TIMEOUT", "30.0"))
render_width = int(os.environ.get("EFFICIENTNAV_RENDER_WIDTH", "512"))
render_height = int(os.environ.get("EFFICIENTNAV_RENDER_HEIGHT", "512"))


grounding_dino_config_path = os.environ.get(
    "EFFICIENTNAV_GDINO_CONFIG",
    "",
)
checkpoint_path = os.environ.get(
    "EFFICIENTNAV_GDINO_MODEL_ID",
    "IDEA-Research/grounding-dino-base",
)
output_dir = "images_output"
box_threshold = 0.5
text_threshold = 0.25

token_spans = None

os.makedirs(output_dir, exist_ok=True)

model_dino = None if use_ros2_detection else load_model(grounding_dino_config_path, checkpoint_path, cpu_only=not cuda_available)

device = "cuda" if cuda_available else "cpu"
device0 = primary_device
local_model_path = os.environ.get("EFFICIENTNAV_CLIP_PATH", "/home/min/models/clip-vit-base-patch32")
clip_tokenizer = CLIPTokenizer.from_pretrained(local_model_path)
model_clip = CLIPTextModel.from_pretrained(local_model_path).to(device0)
clip_model_max_length = getattr(clip_tokenizer, "model_max_length", 77)

group_node = True ##
delete_traj = True ##
depth_threshould = 0.25
hebing_threshould = 0.001
node_pruning_num = 4
object_describe_multi_time = False ##
through_door = True ##
use_traj = False #
pay_attention_to_door = True ##
use_real_semetic = True ##
early_stop  = True #
directly_find =True ##
use_kv_cache = os.environ.get("EFFICIENTNAV_USE_KV_CACHE", "1") == "1"
use_pruning = True

num_episode = 20
num_environment = 15
use_door_as_trajectory = False
final_goal_list = ['toilet','tv','chair','sofa','bed','plant']
trusted_planner_labels = {
    'chair', 'sofa', 'tv', 'bed', 'plant', 'laptop', 'table', 'desk',
    'window', 'wall', 'doorway', 'door frame', 'painting', 'cup',
    'floorlamp', 'armchair', 'cabinet', 'lamp', 'phone', 'cellphone',
    'tablet', 'statue'
}
low_value_planner_labels = {'wall', 'floor', 'ceiling'}
transition_planner_labels = {'doorway', 'door frame', 'door'}
text_embedding_cache = {}


def _pool_text_embedding(model_outputs, attention_mask=None):
    hidden_state = model_outputs.last_hidden_state
    if attention_mask is not None:
        mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
        pooled = (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    else:
        pooled = hidden_state.mean(dim=1)
    pooled = pooled / pooled.norm(dim=-1, keepdim=True)
    return pooled


def get_text_embedding_cached(text):
    normalized_text = canonical_goal_name(str(text or "").strip().lower())
    if not normalized_text:
        return None
    if normalized_text not in text_embedding_cache:
        tokenizer_kwargs = {"return_tensors": "pt", "truncation": True}
        if clip_model_max_length is not None:
            tokenizer_kwargs["max_length"] = clip_model_max_length
        inputs = clip_tokenizer(normalized_text, **tokenizer_kwargs).to(device0)
        with torch.no_grad():
            attention_mask = inputs.get("attention_mask")
            text_embedding = _pool_text_embedding(model_clip(**inputs), attention_mask)
        text_embedding_cache[normalized_text] = text_embedding[0].detach().cpu().numpy()
    return text_embedding_cache[normalized_text]


def get_text_similarity(text1, text2):
    vec1 = get_text_embedding_cached(text1)
    vec2 = get_text_embedding_cached(text2)
    if vec1 is None or vec2 is None:
        return -1.0
    similarity = 1 - cosine(vec1, vec2)
    if math.isnan(similarity):
        return -1.0
    return float(similarity)


def is_semantically_reasonable_planner_label(label):
    normalized_label = canonical_goal_name(label)
    if not normalized_label:
        return False
    if normalized_label in trusted_planner_labels:
        return True
    tokens = [token for token in re.split(r"[^a-z0-9]+", normalized_label) if token]
    if not tokens or len(tokens) > 3:
        return False
    if all(token.isdigit() for token in tokens):
        return False
    if any(token in {"object", "objects", "angle", "shadow"} for token in tokens):
        return False
    return True


def get_planner_label_priority(label, final_goal=None):
    normalized_label = canonical_goal_name(label)
    normalized_goal = canonical_goal_name(final_goal) if final_goal is not None else None
    score = get_text_similarity(normalized_goal, normalized_label) if normalized_goal else 0.0
    if normalized_goal and normalized_label == normalized_goal:
        score += 1.0
    if normalized_label in low_value_planner_labels:
        score -= 1.0
    return score


def is_preferred_planner_label(label, final_goal=None):
    normalized_label = canonical_goal_name(label)
    normalized_goal = canonical_goal_name(final_goal) if final_goal is not None else None
    if normalized_goal and normalized_label == normalized_goal:
        return True
    return normalized_label not in low_value_planner_labels


def ensure_goal_name_registered(goal_name):
    normalized_goal_name = canonical_goal_name(goal_name)
    if normalized_goal_name not in final_goal_list:
        final_goal_list.append(normalized_goal_name)
    return normalized_goal_name


def get_selectable_goal_names(scene):
    selectable_goal_names = set()
    for idx, obj in enumerate(scene.objects):
        if idx == 0:
            continue
        selectable_goal_names.add(canonical_goal_name(obj.category.name()))
    return sorted(selectable_goal_names)


def choose_goal_name_for_house(scene):
    selectable_goal_names = get_selectable_goal_names(scene)
    if not selectable_goal_names:
        return None

    requested_goal_name = os.environ.get("EFFICIENTNAV_TARGET_OBJECT")
    if requested_goal_name not in (None, ""):
        normalized_requested_goal = canonical_goal_name(requested_goal_name.strip())
        if normalized_requested_goal in selectable_goal_names:
            print(f"[Debug] using requested target object={normalized_requested_goal}")
            return ensure_goal_name_registered(normalized_requested_goal)
        print(
            f"[Debug] requested EFFICIENTNAV_TARGET_OBJECT={requested_goal_name!r} "
            f"not found in current house; falling back to interactive selection"
        )

    print("\nSelectable target objects in this house:")
    for idx, goal_name in enumerate(selectable_goal_names, start=1):
        print(f"  {idx}. {goal_name}")

    while True:
        selected_value = input("Choose target object by number or name: ").strip()
        if not selected_value:
            print("Please enter a number or object name.")
            continue
        if selected_value.isdigit():
            selected_index = int(selected_value)
            if 1 <= selected_index <= len(selectable_goal_names):
                chosen_goal_name = selectable_goal_names[selected_index - 1]
                print(f"[Debug] selected target object={chosen_goal_name}")
                return ensure_goal_name_registered(chosen_goal_name)
            print(f"Please choose a number between 1 and {len(selectable_goal_names)}.")
            continue

        normalized_selected_name = canonical_goal_name(selected_value)
        if normalized_selected_name in selectable_goal_names:
            print(f"[Debug] selected target object={normalized_selected_name}")
            return ensure_goal_name_registered(normalized_selected_name)
        print("That object is not in the current house list. Try again.")


def select_far_start_position(reachable_positions, goal_center):
    if len(reachable_positions) == 0:
        return None, None, None

    far_start_min_ratio = float(os.environ.get("EFFICIENTNAV_FAR_START_MIN_RATIO", "0.75"))
    far_start_top_ratio = float(os.environ.get("EFFICIENTNAV_FAR_START_TOP_RATIO", "0.20"))

    far_start_min_ratio = min(max(far_start_min_ratio, 0.0), 1.0)
    far_start_top_ratio = min(max(far_start_top_ratio, 0.0), 1.0)

    ranked_positions = []
    for idx, position in enumerate(reachable_positions):
        euclidean_distance = math.sqrt(
            (position[0] - goal_center[0]) ** 2
            + (position[2] - goal_center[2]) ** 2
        )
        ranked_positions.append((euclidean_distance, idx, position))

    ranked_positions.sort(key=lambda item: item[0], reverse=True)
    max_distance = ranked_positions[0][0]
    min_required_distance = max_distance * far_start_min_ratio

    candidate_positions = [
        item for item in ranked_positions
        if item[0] >= min_required_distance
    ]

    if not candidate_positions:
        candidate_count = max(1, int(math.ceil(len(ranked_positions) * far_start_top_ratio)))
        candidate_positions = ranked_positions[:candidate_count]

    selected_distance, selected_index, selected_position = random.choice(candidate_positions)
    selection_metadata = {
        "candidate_count": len(candidate_positions),
        "reachable_count": len(reachable_positions),
        "max_distance": max_distance,
        "min_required_distance": min_required_distance,
        "selected_distance": selected_distance,
    }
    return selected_index, copy.deepcopy(selected_position), selection_metadata


class DetectionROSClient(Node):
    def __init__(self):
        super().__init__("efficientnav_detection_client")
        if DetectObjects is None:
            raise RuntimeError(
                "efficientnav_interfaces.srv.DetectObjects could not be imported. "
                "source /home/min/DINO_ws/install/setup.bash before running EfficientNav."
            )
        self.detect_client = self.create_client(DetectObjects, "/detection/detect_objects")
        while not self.detect_client.wait_for_service(timeout_sec=1.0):
            print("[Debug] waiting for /detection/detect_objects service...")

    def detect_objects(self, position_id, angle, prompt, image_np, timeout_sec=None):
        if timeout_sec is None:
            timeout_sec = ros2_detection_timeout_sec
        request = DetectObjects.Request()
        request.position_id = str(position_id)
        request.angle = int(angle)
        request.prompt = str(prompt)
        request.height = int(image_np.shape[0])
        request.width = int(image_np.shape[1])
        request.encoding = "rgb8"
        request.data = bytearray(image_np.astype(np.uint8).tobytes())
        future = self.detect_client.call_async(request)
        print(
            f"[Debug] calling detection service: position_id={position_id} angle={int(angle)} "
            f"prompt={prompt!r}"
        )

        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if future.done():
                response = future.result()
                if response is None:
                    raise RuntimeError(
                        f"Detection service call failed for position_id={position_id} angle={angle}"
                    )
                print(
                    f"[Debug] received detection service result: position_id={position_id} "
                    f"angle={int(angle)}"
                )
                return json.loads(response.result_json)
        future.cancel()
        raise TimeoutError(f"Timed out waiting for detection service result for {position_id=} {angle=}")


_detection_ros_client = None


def get_detection_ros_client():
    global _detection_ros_client
    if _detection_ros_client is None:
        if not rclpy.ok():
            rclpy.init(args=None)
        _detection_ros_client = DetectionROSClient()
    return _detection_ros_client


def convert_ros_detection_payload_to_box_info_list(payload):
    box_info_list = []
    for detection in payload.get("detections", []):
        label = str(detection.get("label", "")).strip()
        box = detection.get("box", [])
        if not label or len(box) != 4:
            continue
        box_info_list.append(
            {
                "label": label,
                "box": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
            }
        )
    return box_info_list


def build_chat_prompt(user_text):
    messages = [{"role": "user", "content": user_text}]
    return planner_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _normalize_observation_label(raw_label):
    cleaned = re.sub(r"[^a-zA-Z0-9 ]+", " ", str(raw_label or "").strip().lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None
    cleaned = re.sub(r"^\d+\s+", "", cleaned).strip()
    if cleaned in {"none", "null", "n a", "na"}:
        return None
    if cleaned.isdigit():
        return None
    if cleaned in {"object", "objects", "angle"}:
        return None
    tokens = [token for token in cleaned.split() if token]
    if not tokens:
        return None
    sentence_markers = {
        "image", "shows", "show", "scene", "background", "sky", "clouds",
        "sidewalk", "street", "paved", "building", "mounted", "displays",
        "appears", "commercial", "office", "importance", "environment",
        "impact", "human", "activities", "author", "passage", "discusses",
        "begins", "highlights", "emphasizes", "sustainable", "planet",
        "responsibility", "choices", "carbon", "footprint", "reflected",
        "running", "inside", "outside", "center", "roof", "room",
    }
    if any(token in sentence_markers for token in tokens):
        return None
    if len(tokens) > 4:
        return None
    return cleaned


def parse_observation_response(raw_text, angle):
    raw_text = str(raw_text or "").strip()

    start = raw_text.find("{")
    end = raw_text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            data = json.loads(raw_text[start:end])
            objects = data.get("Objects", [])
            if isinstance(objects, str):
                objects = [objects]
            normalized_objects = []
            seen = set()
            for obj in objects:
                normalized = _normalize_observation_label(obj)
                if normalized and normalized not in seen:
                    normalized_objects.append(normalized)
                    seen.add(normalized)
            return {"Angle": angle, "Objects": normalized_objects[:4]}
        except Exception:
            pass

    extracted_objects = []
    seen = set()
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            _, value = line.split(":", 1)
            candidates = re.split(r",|\band\b", value)
        else:
            candidates = re.split(r",|\band\b", line)
        for candidate in candidates:
            normalized = _normalize_observation_label(candidate)
            if not normalized:
                continue
            if normalized.startswith("angle "):
                continue
            if normalized.startswith("objects "):
                normalized = _normalize_observation_label(normalized[len("objects "):])
            if normalized and normalized not in seen:
                extracted_objects.append(normalized)
                seen.add(normalized)

    if extracted_objects:
        return {"Angle": angle, "Objects": extracted_objects[:4]}

    return None


def parse_planner_response(raw_text, allowed_objects_by_place, final_goal):
    raw_text = str(raw_text or "").strip()
    start = raw_text.find("{")
    end = raw_text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            parsed = json.loads(raw_text[start:end])
            objects = parsed.get("Objects", [])
            if isinstance(objects, str):
                objects = [objects]
            parsed["Objects"] = [str(obj).strip().lower() for obj in objects if str(obj).strip()]
            return parsed
        except Exception:
            pass

    allowed_label_to_places = {}
    for place_idx, labels in allowed_objects_by_place.items():
        for label in labels:
            allowed_label_to_places.setdefault(label, []).append(place_idx)

    lowered_text = raw_text.lower()
    if canonical_goal_name(final_goal) in allowed_label_to_places and canonical_goal_name(final_goal) in lowered_text:
        goal_label = canonical_goal_name(final_goal)
        return {"Place": allowed_label_to_places[goal_label][0], "Angle": 0, "Objects": [goal_label]}

    best_label = None
    best_pos = None
    for label in allowed_label_to_places:
        pos = lowered_text.find(label.lower())
        if pos == -1:
            continue
        if best_pos is None or pos < best_pos:
            best_pos = pos
            best_label = label

    if best_label is not None:
        return {"Place": allowed_label_to_places[best_label][0], "Angle": 0, "Objects": [best_label]}

    for place_idx, labels in allowed_objects_by_place.items():
        if labels:
            return {"Place": place_idx, "Angle": 0, "Objects": [labels[0]]}

    return None


def save_goal_bbox_debug(color_image, semantic_frame, object_id, label, output_path):
    mask = semantic_frame == object_id
    if not np.any(mask):
        return False
    ys, xs = np.where(mask)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    image = I.fromarray(color_image.astype(np.uint8))
    draw = ImageDraw.Draw(image)
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=4)
    draw.text((x1, max(0, y1 - 18)), label, fill=(255, 0, 0))
    image.save(output_path)
    return True


def get_object_visibility_metrics(semantic_frame, object_id):
    mask = semantic_frame == object_id
    if not np.any(mask):
        return 0.0, 0, 0
    visible_ratio = float(np.mean(mask))
    ys, xs = np.where(mask)
    bbox_width = int(xs.max() - xs.min() + 1)
    bbox_height = int(ys.max() - ys.min() + 1)
    return visible_ratio, bbox_width, bbox_height


def is_object_clearly_visible(semantic_frame, object_id, visible_ratio_threshold, min_bbox_side_px):
    visible_ratio, bbox_width, bbox_height = get_object_visibility_metrics(semantic_frame, object_id)
    is_visible = (
        visible_ratio >= visible_ratio_threshold
        and bbox_width >= min_bbox_side_px
        and bbox_height >= min_bbox_side_px
    )
    return is_visible, visible_ratio, bbox_width, bbox_height


def is_goal_label_match(goal_name, detected_label):
    return canonical_goal_name(detected_label) == canonical_goal_name(goal_name)


def detect_goal_in_current_view(color_image, goal_name, request_id, angle):
    prompt = f"{goal_name.lower()} ."
    min_detected_goal_bbox_side_px = int(os.environ.get("EFFICIENTNAV_MIN_DETECTED_GOAL_BBOX_SIDE_PX", "24"))
    if use_ros2_detection:
        ros_client = get_detection_ros_client()
        payload = ros_client.detect_objects(request_id, angle, prompt, color_image.astype(np.uint8))
        box_info_list = convert_ros_detection_payload_to_box_info_list(payload)
    else:
        image_path = f"tmp/navigation_images/{request_id}_goal_check.png"
        imageio.imwrite(image_path, color_image)
        image_pil, image = load_image(image_path)
        boxes_filt, pred_phrases = get_grounding_output(
            model_dino,
            image,
            prompt,
            box_threshold,
            text_threshold,
            cpu_only=not cuda_available,
            token_spans=eval(f"{token_spans}") if token_spans is not None else None,
            text_prompt=goal_name,
        )
        size = image_pil.size
        pred_dict = {
            "boxes": boxes_filt,
            "size": [size[1], size[0]],
            "labels": pred_phrases,
        }
        _, _, box_info_list = plot_boxes_to_image(image_pil, pred_dict)

    best_detection = None
    best_area = 0
    for box_info in box_info_list:
        label = str(box_info.get("label", "")).strip().lower()
        if not is_goal_label_match(goal_name, label):
            continue
        box = box_info.get("box", [])
        if len(box) != 4:
            continue
        width = max(0, int(box[2]) - int(box[0]))
        height = max(0, int(box[3]) - int(box[1]))
        if width < min_detected_goal_bbox_side_px or height < min_detected_goal_bbox_side_px:
            continue
        area = width * height
        if area > best_area:
            best_area = area
            best_detection = {
                "label": label,
                "box": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                "width": width,
                "height": height,
            }
    return best_detection


def convert_legacy_kv_to_runtime_cache(cache_value):
    if cache_value is None or DynamicCache is None:
        return cache_value
    if not isinstance(cache_value, tuple):
        return cache_value
    runtime_cache = DynamicCache()
    for layer_idx, layer_cache in enumerate(cache_value):
        if not isinstance(layer_cache, tuple) or len(layer_cache) != 2:
            raise TypeError(f"invalid legacy cache entry at layer {layer_idx}")
        key_states, value_states = layer_cache
        runtime_cache.update(key_states, value_states, layer_idx)
    return runtime_cache


def get_observation(images,depth):
    if not planner_supports_vision:
        fallback_observation = []
        position_looked = []
        for i in range(0,4):
            if depth[i].mean() <= depth_threshould:
                print(f"[Debug] get_observation skip angle={i * 90} reason=depth")
                continue
            position_looked.append(i * 90)
            fallback_observation.append(json.dumps({"Angle": i * 90, "Objects": ["door frame"]}, indent=4))
        return fallback_observation, position_looked

    observation_instruction = '''You need to observe the image from the current perspective.
Output only a JSON object in this format:
{ "Angle": 0, "Objects": ["object name", "object name"] }
Here are some things you should be aware of:
1. Entrances or doorways to other spaces in the room count as objects, but do not describe doors.
2. Objects that are too small need no description.
3. Describe the same object only once. You can describe at most 4 objects.
4. Each object must be a short noun label of 1 to 3 words. Do not output sentences, explanations, or scene descriptions.
5. Do not describe objects in the mirror.
6. If you are unsure, output fewer objects rather than a sentence.'''

    llm_answer = []
    for image in images:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": observation_instruction},
                ],
            }
        ]
        prompt = planner_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = planner_processor(images=image, text=prompt, return_tensors="pt").to(device0)
        with torch.no_grad():
            output = planner_model.generate(**inputs, max_new_tokens=200, pad_token_id=planner_tokenizer.pad_token_id)
        generated = output[:, inputs["input_ids"].shape[1]:]
        real_output = planner_tokenizer.decode(generated[0], skip_special_tokens=True)
        llm_answer.append(real_output.strip())

    llm_answer1 = []
    json_data = llm_answer
    num_look = 4
    position_unlooked = []
    position_looked = []
    for i in range(0,4):
        if depth[i].mean() <= depth_threshould:
            num_look -= 1
            position_unlooked.append(i*90)
            print(f"[Debug] get_observation skip angle={i * 90} reason=depth")
            continue
        else:
            position_looked.append(i*90)
        data = parse_observation_response(json_data[i], i * 90)
        if data is None:
            print(f"[Debug] get_observation skip angle={i * 90} reason=json_parse raw={json_data[i]!r}")
            if position_looked and position_looked[-1] == i * 90:
                position_looked.pop()
            continue
        json_string = json.dumps(data, indent=4)
        if i==0:
            llm_answer1.append(json_string)
        else:
            llm_answer1.append(json_string)
    return llm_answer1,position_looked


def get_objects_boxes(llava_answer1,fig_name):
    global text_threshold
    box_info_list_sum = []
    json_objects = copy.deepcopy(llava_answer1)
    # Parse each JSON object and store in a dictionary
    angles_objects = {}
    for json_obj in json_objects:
        obj_dict = json.loads(json_obj)
        angle = obj_dict['Angle']
        objects = obj_dict['Objects']
        angles_objects[angle] = objects
    for i, (text_prompt_list, key_angle) in enumerate(zip(angles_objects.values(),angles_objects.keys())):
        image_path = f"navigation_images/{fig_name}+surroundings_angle_{key_angle}.png"
        text_prompt_list = list(set(text_prompt_list))
        text_prompt_list = [obj for obj in text_prompt_list if isinstance(obj, str) and obj.strip()]
        if len(text_prompt_list) == 0:
            box_info_list_sum.append([])
            continue
        text_prompt = text_prompt_list[0]
        result = ' . '.join([obj.lower() for obj in text_prompt_list]) + ' .'
        image_pil, image = load_image(image_path)
        image_pil.save(os.path.join(output_dir, f"raw_image_angle_{i}.jpg"))
        if use_ros2_detection:
            ros_client = get_detection_ros_client()
            image_np = np.array(image_pil, dtype=np.uint8)
            position_id = fig_name
            payload = ros_client.detect_objects(position_id, key_angle, result, image_np)
            box_info_list = convert_ros_detection_payload_to_box_info_list(payload)
            image_with_box = image_pil
        else:
            if token_spans is not None:
                text_threshold = None
                print("Using token_spans. Set the text_threshold to None.")
            boxes_filt, pred_phrases = get_grounding_output(
                model_dino, image, result, box_threshold, text_threshold, cpu_only=False, token_spans=eval(f"{token_spans}"),text_prompt=text_prompt
            )
            size = image_pil.size
            pred_dict = {
                "boxes": boxes_filt,
                "size": [size[1], size[0]],  # H,W
                "labels": pred_phrases,
            }

            image_with_box, _ , box_info_list = plot_boxes_to_image(image_pil, pred_dict)
        box_info_list_copy = copy.deepcopy(box_info_list)
        box_info_list_flag = np.zeros(len(box_info_list))
        box_info_list_real = []
        for j in range (0,len(box_info_list_copy)):
            if box_info_list_flag[j] ==1:
                continue
            if j == len(box_info_list_copy)-1:
                box_info_list_real.append(box_info_list_copy[j])
                break
            for k in range(j+1,len(box_info_list_copy)):
                if box_info_list_copy[j]['label'] == box_info_list_copy[k]['label']:
                    box_info_list_copy[j]['box'][0] = min(box_info_list_copy[j]['box'][0],box_info_list_copy[k]['box'][0])
                    box_info_list_copy[j]['box'][1] = min(box_info_list_copy[j]['box'][1],box_info_list_copy[k]['box'][1])
                    box_info_list_copy[j]['box'][2] = max(box_info_list_copy[j]['box'][2],box_info_list_copy[k]['box'][2])
                    box_info_list_copy[j]['box'][3] = max(box_info_list_copy[j]['box'][3],box_info_list_copy[k]['box'][3])
                    box_info_list_flag[k] = 1
            box_info_list_real.append(box_info_list_copy[j])

        # If Grounding DINO did not return a box for a requested label, do not
        # synthesize a full-image fallback box. Those oversized boxes dominate
        # the semantic/CLIP matching stage and bias candidates toward
        # window/wall/doorway artifacts.

        del_tmp = []
        for j in range(0,len(box_info_list_real)):
            text_exist_flag = 0
            for k in range(0,len(text_prompt_list)):
                if box_info_list_real[j]['label'].lower() == text_prompt_list[k].lower() :
                    text_exist_flag =1
                    break
            if text_exist_flag == 0:
                del_tmp.append(j)
        del_tmp.sort(reverse=True)
        for j in range(len(del_tmp)):
            del box_info_list_real[del_tmp[j]]
        box_info_list_sum.append(box_info_list_real)
        image_with_box.save(os.path.join(output_dir, f"pred_angle_{i}.jpg"))
    return box_info_list_sum


def get_objects(topomap,scene,position_looked,box_info_list_sum,semantic_observations,obj_dict):
    # define object
    ObjectInfo = namedtuple("ObjectInfo", ["label","angle", "obj_id", "category", "center", "sizes"])

    objects_info_filtered = []

    max_similar_objs_list = []

    current_position = np.array(topomap.now.position if topomap.now is not None else [0.0, 0.0, 0.0])
    empty_position = []
    for i,(angle_picture, box_info_list) in enumerate(zip(position_looked,box_info_list_sum)):
        topomap.now.similarity.append([0.0 for _ in range(len(final_goal_list))])
        empty_flag = 0
        semantic = semantic_observations[angle_picture//90]
        for box_info in box_info_list:
            label = box_info['label'].lower()
            x1, y1, x2, y2 = box_info['box']
            semantic_box = semantic[y1:y2, x1:x2]
            unique_labels = np.unique(semantic_box)
            filtered_objects = []
            for label_id in unique_labels:
                obj = scene.objects[label_id]
                object_info_filtered = ObjectInfo(
                    label=label,
                    angle=angle_picture,
                    obj_id=label_id,
                    category=obj.category.name().lower(),
                    center=obj.obb.center,
                    sizes=obj.obb.sizes
                )
                objects_info_filtered.append(object_info_filtered)
                filtered_objects.append(object_info_filtered)

            similarities = [(label,obj.angle, obj.obj_id, get_text_similarity(label, obj.category), obj.category, obj.center) for obj in filtered_objects]

            if similarities:
                max_similarity = max(similarities, key=lambda x: x[3])[3]
                max_similar_objs = [(label,angle, obj_id, simi, category, center) for label,angle, obj_id, simi, category, center in similarities if simi == max_similarity]
                if len(max_similar_objs) > 1:
                    closest_obj = min(max_similar_objs, key=lambda x: euclidean(current_position, x[5]))
                    max_similar_objs = [closest_obj]

                objects = []

                if len(topomap.used_id)!=0 and any(max_similar_objs[0][2] == item[0] for item in topomap.used_id):
                    item_to_remove = max_similar_objs[0][0]
                    json_origin = topomap.now.describe[i]
                    objects_angle = json.loads(json_origin)
                    objects_origin = objects_angle['Objects']
                    objects = [obj for obj in objects_origin if obj.lower() != item_to_remove.lower()]
                    obj_dict['Angle'] = angle_picture
                    obj_dict['Objects'] = objects
                    if len(objects) == 0:
                        empty_position.append(i)
                        empty_flag = 1
                        continue
                    topomap.now.describe[i] = json.dumps(obj_dict, indent=4)
                    continue
                else :
                    if not object_describe_multi_time:
                        topomap.used_id.append([max_similar_objs[0][2],max_similar_objs[0][3]])
                    if use_real_semetic:
                        json_origin = topomap.now.describe[i]
                        objects_angle = json.loads(json_origin)
                        objects_origin = objects_angle['Objects']
                        for k,obj in enumerate(objects_origin):
                            for j,similar_obj in enumerate(max_similar_objs):
                                if similar_obj[0].lower() == obj.lower():
                                    objects_angle['Objects'][k] = similar_obj[4]
                        max_similar_objs_list.append([(max_similar_objs[0][4],max_similar_objs[0][1],int(max_similar_objs[0][2]),max_similar_objs[0][3],max_similar_objs[0][4],max_similar_objs[0][5])])
                        topomap.now.describe[i] = json.dumps(objects_angle, indent=4)
                    else:
                        max_similar_objs_list.append(max_similar_objs)
                    print(max_similar_objs[0][4])
                    if use_pruning:
                        for k in range(0,len(final_goal_list)):
                            if get_text_similarity(final_goal_list[k], max_similar_objs[0][4]) + 0.1 * max(get_text_similarity(final_goal_list[k], 'door'), get_text_similarity(final_goal_list[k], 'door frame')) > topomap.now.similarity[i][k]:
                                topomap.now.similarity[i][k] = get_text_similarity(final_goal_list[k], max_similar_objs[0][4])
            else:
                max_similar_objs = []
            if empty_flag == 1:
                break
    return max_similar_objs_list,empty_position


def planning(place_describe,place_describe_cache,final_goal,trajectory,allowed_objects=None,allowed_objects_by_place=None):
    planning_start_time = time.perf_counter()
    effective_use_kv_cache = use_kv_cache and place_describe_cache is not None
    if effective_use_kv_cache and isinstance(place_describe_cache, tuple):
        if DynamicCache is None:
            print("[Debug] DynamicCache unavailable, falling back to non-KV planning")
            effective_use_kv_cache = False
        else:
            try:
                place_describe_cache = convert_legacy_kv_to_runtime_cache(place_describe_cache)
            except Exception as exc:
                print(f"[Debug] KV cache conversion failed, falling back to non-KV planning: {exc}")
                effective_use_kv_cache = False
    input_text = 'The above is a description of different places in different angles in the environment.'
    input_text+= f'Your can get to any place described in the json data. '
    input_text+= f'Your goal is to find the {final_goal}. Based on the above json data, please choose one specific object to travel to as your target. If your goal is already in the description, please choose it as the target.'
    if allowed_objects:
        allowed_objects_text = ', '.join(allowed_objects)
        input_text += f' You must choose an object only from this allowed list: {allowed_objects_text}.'
        if final_goal in allowed_objects:
            input_text += f' Since {final_goal} is in the allowed list, choose {final_goal}.'
        input_text += ' Do not output any object name that is not in the allowed list.'
    if allowed_objects_by_place:
        per_place_text = []
        for place_idx, labels in allowed_objects_by_place.items():
            if labels:
                per_place_text.append(f'Place {place_idx}: {", ".join(labels)}')
        if per_place_text:
            input_text += ' Valid choosable objects by place are: ' + ' ; '.join(per_place_text) + '.'
            input_text += ' The Objects field must contain exactly one object from the chosen place list.'
            input_text += ' The candidate lists are ordered from more semantically related to the goal to less related, so if you are uncertain, prefer earlier candidates.'
    if use_traj:
        input_text+= f'Here is the objects that you have traveled to before: {trajectory} Do not choose the objects that you have traveled to before as the target. '
    if pay_attention_to_door:
        if use_real_semetic:
            input_text+='Note that you can travel to door or door frame to other spaces if there are no clear evidence to choose the target. '
        else:
            input_text+='Note that you can travel to entrance or door frame to other spaces if there are no clear evidence to choose the target.'
    input_text+='''Return exactly one JSON object by referring to the following template.
            {"Place": x, "Angle": x, "Objects": ["xxxx"] }
            If your goal is already in the description, please choose it as the target. You should not output any explanation, markdown, prose, examples, or extra text before or after the JSON. Note that your should choose only one object in one angle of one place in the json data as the target.'''
    if not effective_use_kv_cache:
        prompt2 = build_chat_prompt(f"{place_describe}\n{input_text}")
        inputs2 = planner_tokenizer(prompt2, padding=True, return_tensors="pt").to(device0)
        with torch.no_grad():
            output2 = planner_model.generate(
                **inputs2,
                max_new_tokens=64,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=planner_tokenizer.pad_token_id,
            )
    else:
        # TODO: keep the existing KV-cache reuse path intact for the current
        # planner model family when use_kv_cache=True is enabled.
        conma_flag = 0
        prompt_pruning = build_chat_prompt(input_text)
        new_input_pruning = planner_tokenizer(prompt_pruning, padding=True, return_tensors="pt").to(device0)
        generated_tokens = []
        while True:
            with torch.no_grad():
                outputs = planner_model(input_ids=new_input_pruning['input_ids'],
                                # attention_mask=new_inputs['attention_mask'],
                                past_key_values=place_describe_cache,
                                use_cache=True)
            next_token = outputs.logits.argmax(dim=-1)[:, -1:]
            generated_tokens.append(int(next_token[0][0]))
            if int(next_token[0][0]) == 7:
                print('over')
                break
            if int(next_token[0][0]) == 97 and conma_flag ==0:
                conma_flag = 1

            place_describe_cache = outputs.past_key_values
            new_input_pruning = {'input_ids': next_token}
            if len(generated_tokens)>100:
                break

    if not effective_use_kv_cache:
        generated = output2[:, inputs2["input_ids"].shape[1]:]
        real_output2 = planner_tokenizer.decode(generated[0], skip_special_tokens=True)
        del output2
    else:
        real_output2 = planner_tokenizer.decode(generated_tokens, skip_special_tokens=True)
        if last_non_space_char(real_output2) == ']':
            real_output2 += '}'
        del generated_tokens
    torch.cuda.empty_cache()
    gc.collect()
    print(input_text)
    print(real_output2)

    llava_answer2 = real_output2.strip()
    planning_elapsed = time.perf_counter() - planning_start_time
    planning_mode = "kv" if effective_use_kv_cache else "no-kv"
    print(f"[Timing] planning mode={planning_mode} elapsed={planning_elapsed:.3f}s")

    return llava_answer2






def val_one_episode(topomap,sim,agent,start_point,start_rotation,final_goal_id,final_goal,distance):
    episode_start_time = time.perf_counter()
    final_goal = canonical_goal_name(final_goal)
    visible_ratio_threshold = float(os.environ.get("EFFICIENTNAV_VISIBLE_RATIO_THRESHOLD", "0.002"))
    min_visible_bbox_side_px = int(os.environ.get("EFFICIENTNAV_MIN_VISIBLE_BBOX_SIDE_PX", "24"))

    # ==========================================================================================================================================
    # INITIAL SIM
    # =================================================================================================================================================================

    agent_state = ThorAgentState(np.array(start_point, dtype=np.float32), float(start_rotation))
    agent.set_state(agent_state)

    # =================================================================================================================================================================
    # FIND SHORTEST PATH
    # ==========================================================================================================================================

    scene = sim.semantic_scene
    final_goal_label_ids = set()
    for semantic_idx, scene_object in enumerate(scene.objects):
        if semantic_idx == 0:
            continue
        if canonical_goal_name(scene_object.category.name()) == final_goal:
            final_goal_label_ids.add(semantic_idx)

    def get_object_position(object_id):
        obj = scene.objects[object_id]
        return obj.category.name(),obj.obb.center, obj.obb.sizes


    _,shortest_target_position, shortest_target_dims = get_object_position(final_goal_id)

    path = ThorShortestPath()
    path.requested_start = agent.state.position
    path.requested_end = shortest_target_position

    initial_pathfinder_start_time = time.perf_counter()
    found_path = sim.pathfinder.find_path(path)
    initial_pathfinder_elapsed = time.perf_counter() - initial_pathfinder_start_time
    print(
        f"[Timing] pathfinder goal-distance elapsed={initial_pathfinder_elapsed:.3f}s "
        f"found={found_path}"
    )
    path_points = path.points

    shortest_length = 0
    if found_path:
        for i, point in enumerate(path_points):
            if i==0 :
                continue
            else :
                shortest_length += math.sqrt((path_points[i][0]-path_points[i-1][0])**2+(path_points[i][2]-path_points[i-1][2])**2)
    real_distance = shortest_length
    print(f'real_distance:{real_distance}')


    # ==========================================================================================================================================
    # INITIAL PARAMETERS
    # ==========================================================================================================================================


    if not use_door_as_trajectory:
        trajectory = ' '
    else:
        trajectory = 'Door. Window.'

    sub_goal_history = []
    final_length = 0


    last_target_position = agent_state.position
    ## do not navigate to the same nodes
    last_key = []
    last_angle = []
    last_index = []
    target_tuple = None
    last_answer = ' '
    repeated_answer_count = 0
    episode_success = False
    visible_goal_target_position = None
    visible_goal_name = None
    visible_goal_id = None
    place_target_visit_counts = {}

    def get_current_described_labels_for_place(place_idx):
        node = topomap.find_node(topomap.root, f'Place {place_idx}')
        if node is None:
            return set()
        selected_indices = None
        if hasattr(topomap, "_get_selected_description_indices"):
            try:
                selected_indices = topomap._get_selected_description_indices(
                    node,
                    last_key,
                    last_index,
                    final_goal,
                )
            except Exception:
                selected_indices = None
        if selected_indices is None:
            describe_entries = node.describe
        else:
            describe_entries = [
                node.describe[i]
                for i in selected_indices
                if 0 <= i < len(node.describe)
            ]
        current_labels = set()
        for describe_json in describe_entries:
            try:
                describe_data = json.loads(describe_json)
            except Exception:
                continue
            for obj in describe_data.get("Objects", []):
                normalized = canonical_goal_name(str(obj).strip().lower())
                if normalized:
                    current_labels.add(normalized)
        return current_labels

    def order_labels_by_goal_relevance(labels):
        return sorted(
            labels,
            key=lambda label: (get_planner_label_priority(label, final_goal), label),
            reverse=True,
        )

    def get_transition_labels(labels):
        return [label for label in labels if canonical_goal_name(label) in transition_planner_labels]

    def collect_allowed_objects():
        allowed_by_place = collect_allowed_objects_by_place()
        scored_labels = {}
        for labels in allowed_by_place.values():
            for label in labels:
                score = get_planner_label_priority(label, final_goal)
                existing_score = scored_labels.get(label)
                if existing_score is None or score > existing_score:
                    scored_labels[label] = score
        return order_labels_by_goal_relevance(list(scored_labels.keys()))

    def collect_allowed_objects_by_place():
        allowed_by_place = {}
        for place_idx, place_candidates in enumerate(topomap.place_clip_id):
            current_labels = get_current_described_labels_for_place(place_idx)
            labels = set()
            for object_tuple in place_candidates:
                if len(object_tuple) == 0:
                    continue
                label = canonical_goal_name(str(object_tuple[0][0]).strip().lower())
                if label not in current_labels:
                    continue
                if not is_semantically_reasonable_planner_label(label):
                    continue
                labels.add(label)
            has_preferred = any(label not in low_value_planner_labels for label in labels)
            filtered_labels = [
                label for label in labels
                if (label not in low_value_planner_labels) or not has_preferred
            ]
            ordered_labels = order_labels_by_goal_relevance(filtered_labels)
            if place_target_visit_counts.get(place_idx, 0) > 0:
                ordered_labels = order_labels_by_goal_relevance(get_transition_labels(ordered_labels))
            allowed_by_place[place_idx] = ordered_labels
        return allowed_by_place

    def find_direct_goal_choice():
        for place_idx, place_candidates in enumerate(topomap.place_clip_id):
            current_labels = get_current_described_labels_for_place(place_idx)
            for object_tuple in place_candidates:
                if len(object_tuple) == 0:
                    continue
                candidate = object_tuple[0]
                candidate_label = canonical_goal_name(candidate[0])
                if candidate_label not in current_labels:
                    continue
                if candidate_label == final_goal:
                    return {
                        "Place": place_idx,
                        "Angle": int(candidate[1]),
                        "Objects": [final_goal],
                    }
        return None

    for epoch in range(0,30):
        length_this_epoch = 0.0
        sr = 0
        spl = 0.0
        target_index = final_goal_list.index(final_goal)
        place_describe_cache = None

        # ==========================================================================================================================================
        # GET OBSERVATION
        # ==========================================================================================================================================
        if topomap.current_inference > 0:
            nearest_length, nearest_position,nearest_node  = topomap.find_nearest_node(topomap.root,agent_state.position)
        else:
            nearest_length = 1000
        if group_node :
            skip_node = (nearest_length < hebing_threshould) and (topomap.current_inference > 0)
        else:
            skip_node = nearest_length < hebing_threshould+1 and topomap.current_inference > 0 and epoch ==0
        fig_name = f'big+{topomap.num_node}+{epoch}'
        if skip_node:
            topomap.now = nearest_node
        else:
            surroundings = []
            depth = []
            semantic_observations = []
            images = []
            images_per_row = 2
            fig, axes = plt.subplots(ceil(360 / 90 / images_per_row), images_per_row, figsize=(15, 15))

            for idx, angle in enumerate(range(0, 360, 90)):
                agent_state.rotation = float(angle)
                agent.set_state(agent_state)
                if observation_rotation_pause > 0:
                    print(
                        f"[Debug] observation rotation: angle={angle} pause={observation_rotation_pause:.2f}s"
                    )
                    time.sleep(observation_rotation_pause)
                sur = sim.get_sensor_observations()
                surroundings.append(sur)
                semantic_observations.append(sur["semantic_sensor"])
                color_image = sur["color_sensor"]
                depth.append(sur["depth_sensor"])
                image_path = f"navigation_images/{fig_name}+surroundings_angle_{angle}.png"
                imageio.imwrite(image_path, color_image)
                row, col = divmod(idx, images_per_row)

            image1 = I.open(f"navigation_images/{fig_name}+surroundings_angle_0.png").convert("RGB")
            image2 = I.open(f"navigation_images/{fig_name}+surroundings_angle_90.png").convert("RGB")
            image3 = I.open(f"navigation_images/{fig_name}+surroundings_angle_180.png").convert("RGB")
            image4 = I.open(f"navigation_images/{fig_name}+surroundings_angle_270.png").convert("RGB")


            image_size=672
            image1 = image1.resize((image_size,image_size), I.LANCZOS)
            image2 = image2.resize((image_size,image_size), I.LANCZOS)
            image3 = image3.resize((image_size,image_size), I.LANCZOS)
            image4 = image4.resize((image_size,image_size), I.LANCZOS)
            images = [image1,image2,image3,image4]

            ##put the images into LLava for first stage, and put the output to llava for second stage, xxx is the output
            # ==========================================================================================================================================
            # DESCRIBE IMAGE
            # ==========================================================================================================================================


            llava_answer1,position_looked = get_observation(images,depth)
            json_objects = copy.deepcopy(llava_answer1)
            obj_dict = {"Angle": 0, "Objects": []}

            llava_answer1 = []

            for json_obj in json_objects:
                obj_dict = json.loads(json_obj)
                obj_dict['Objects'] = list(set(obj_dict['Objects']))
                llava_answer1.append(json.dumps(obj_dict, indent=4))

            if use_kv_cache:
                empty_position = []
                for i in range(0, len(llava_answer1)):
                    objects_angle = json.loads(llava_answer1[i])
                    obj_dict = {'Place': topomap.num_node, **{key: value for key, value in objects_angle.items()}}
                    if len(obj_dict['Objects'])==0:
                        empty_position.append(i)
                    llava_answer1[i] = json.dumps(obj_dict, indent=4)

                empty_position.sort(reverse=True)
                for i in range(len(empty_position)):
                    del llava_answer1[empty_position[i]]
                    del llava_answer1[empty_position[i]]

            if topomap.current_inference==0:
                topomap.add_node(parent_key=None, key = 'Place 0', position = copy.deepcopy(agent_state.position), distance_to_parent = 0.0, picture = images, describe = llava_answer1,direction=None,waypoint=None)
                topomap.num_node += 1
                topomap.current_inference += 1
            else:
                # topomap.add_node(parent_key=None, key = f'Place {topomap.num_node}', position = target_position, distance_to_parent = 0.0, picture = [image1,image2,image3,image4], describe = llava_answer1,direction=None,waypoint=None)
                topomap.add_node(parent_key=None, key = f'Place {topomap.num_node}', position = copy.deepcopy(agent_state.position), distance_to_parent = 0.0, picture = images, describe = llava_answer1,direction=None,waypoint=None)
                topomap.num_node += 1
                topomap.current_inference += 1

            torch.cuda.empty_cache()
            gc.collect()

            # ==========================================================================================================================================
            # GET_OBJECTS_BOXES
            # ==========================================================================================================================================

            box_info_list_sum = get_objects_boxes(llava_answer1,fig_name)
            print(f"[Debug] box_info_list_sum: {box_info_list_sum}")

            # ==========================================================================================================================================
            # GET_OBJECTS
            # ==========================================================================================================================================
            max_similar_objs_list,empty_position = get_objects(
                topomap,
                scene,
                position_looked,
                box_info_list_sum,
                semantic_observations,
                copy.deepcopy(obj_dict),
            )
            print(f"[Debug] max_similar_objs_list: {max_similar_objs_list}")
            print(f"[Debug] empty_position: {empty_position}")

            topomap.place_clip_id.append(max_similar_objs_list)

            # ==========================================================================================================================================
            # LLM决定子目标
            # ==========================================================================================================================================
            llava_answer_concat = ' '

            for i in range(0, len(topomap.now.describe)):
                llava_answer_concat += topomap.now.describe[i]



        if use_pruning:
            similarity = topomap.get_similarity_threshould(topomap.root,last_key,last_index,target_index,final_goal)
            similarity.sort(reverse=True)
            if len(similarity) <= node_pruning_num:
                topomap.similarity_threshould[target_index] = similarity[-1]
            else:
                topomap.similarity_threshould[target_index] = similarity[node_pruning_num]
        if use_kv_cache and topomap.use_kv_cache and topomap.kv_cache_supported:
            topomap.used_groups = []
            place_describe,place_describe_cache= topomap.create_describe_and_cache(planner_model,topomap.root,last_key,last_index,target_index,final_goal)
        else:
            place_describe_cache = None
            place_describe= topomap.create_describe(topomap.root,last_key,last_index,target_index,final_goal)
        print(place_describe)

        allowed_objects = collect_allowed_objects()
        allowed_objects_by_place = collect_allowed_objects_by_place()
        print(f"[Debug] allowed planner objects: {allowed_objects}")
        print(f"[Debug] allowed planner objects by place: {allowed_objects_by_place}")
        direct_goal_choice = find_direct_goal_choice()
        if direct_goal_choice is not None:
            llava_answer2 = json.dumps(direct_goal_choice, ensure_ascii=False)
            print(f"[Debug] bypassing planner because goal is already observed: {llava_answer2}")
        else:
            llava_answer2 = planning(
                place_describe,
                place_describe_cache,
                final_goal,
                trajectory,
                allowed_objects,
                allowed_objects_by_place,
            )

        # ===================================================================================================================
        # GET SUB-GOAL
        # ===================================================================================================================

        planner_choice = parse_planner_response(llava_answer2, allowed_objects_by_place, final_goal)
        if planner_choice is None:
            print(f"[Debug] failed to recover planner response from raw text: {llava_answer2!r}")
            break
        json_str = json.dumps(planner_choice, ensure_ascii=False)
        print(json_str)
        if json_str == last_answer:
            repeated_answer_count += 1
        else:
            repeated_answer_count = 0
        last_answer = json_str
        if repeated_answer_count >= 3:
            print("Planner repeated the same target selection. Stopping this episode to avoid looping.")
            break
        try:
            data = planner_choice
            data_tmp = int(data["Place"])
        except:
            break

        target_place = data["Place"]
        angle_goal = data["Angle"]
        objects = data["Objects"]
        if not isinstance(objects, list):
            objects = [str(objects)]
        objects = [obj for obj in objects if isinstance(obj, str) and obj.strip()]
        if len(objects) == 0:
            break
        if len(objects) > 1:
            objects = [objects[0]]
        objects = [objects[0].strip().lower()]
        if int(target_place) >= len(topomap.place_clip_id):
            target_place = 0
        current_place_allowed = allowed_objects_by_place.get(int(target_place), [])
        if objects[0] not in current_place_allowed:
            remapped = False
            if final_goal in allowed_objects:
                for place_idx, labels in allowed_objects_by_place.items():
                    if final_goal in labels:
                        print(f"[Debug] planner object {objects[0]!r} not allowed for place {target_place}; remapping to goal {final_goal!r} in place {place_idx}")
                        target_place = place_idx
                        objects = [final_goal]
                        remapped = True
                        break
            if not remapped and current_place_allowed:
                print(f"[Debug] planner object {objects[0]!r} not allowed for place {target_place}; remapping to first allowed object {current_place_allowed[0]!r}")
                objects = [current_place_allowed[0]]
                remapped = True
            if not remapped:
                for place_idx, labels in allowed_objects_by_place.items():
                    if labels:
                        print(f"[Debug] planner place {target_place} has no allowed object match; remapping to place {place_idx} object {labels[0]!r}")
                        target_place = place_idx
                        objects = [labels[0]]
                        remapped = True
                        break
        print(f"[Debug] planner selected: place={target_place}, angle={angle_goal}, objects={objects}")

        last_angle = angle_goal
        target_node = topomap.find_node(topomap.root,f'Place {target_place}')
        print(f'last_key:{last_key}')
        objects_str = ', '.join(objects)
        result = f"An area of {objects_str}."
        sub_goal_history.append(objects_str)

        if (objects[0].lower() not in trajectory) and (objects[0] not in trajectory):
            trajectory += f'{objects[0]} in Place {target_place}.'

        # ==========================================================================================================================================
        # GET SUB-GOAL INFORMATION
        # ==========================================================================================================================================
        if int(target_place)>=len(topomap.place_clip_id):
            target_place = 0
        if len(topomap.place_clip_id) == 0:
            break
        place_id = topomap.place_clip_id[target_place]
        print(f"[Debug] place_clip_id[{target_place}] candidates: {place_id}")
        flag_tmp = 0
        target_tuple = None

        for object_tuple in place_id:
            if len(object_tuple)==0:
                continue
            if objects[0].lower() == object_tuple[0][0].lower() and angle_goal == object_tuple[0][1]:
                target_tuple = object_tuple[0]
                flag_tmp = 1
                break
            if objects[0].lower() == object_tuple[0][0].lower():
                target_tuple = object_tuple[0]
                flag_tmp = 1


        ## if the place id is wrong, check other places
        if flag_tmp ==0:
            print(f"[Debug] no direct match for {objects[0]!r} in place {target_place}; searching other places")
            for i,place_id in enumerate(topomap.place_clip_id):
                for object_tuple in place_id:
                    if len(object_tuple)==0:
                        continue
                    if objects[0].lower() == object_tuple[0][0].lower():
                        target_tuple = object_tuple[0]
                        flag_tmp = 1
                        target_place = i

        if flag_tmp ==0:
            print(f"[Debug] no exact object match for {objects[0]!r}; falling back to first available candidate")
            print(f"[Debug] all place_clip_id candidates: {topomap.place_clip_id}")
            for fallback_place, fallback_place_id in enumerate(topomap.place_clip_id):
                for object_tuple in fallback_place_id:
                    if len(object_tuple) == 0:
                        continue
                    target_tuple = object_tuple[0]
                    target_place = fallback_place
                    flag_tmp = 1
                    break
                if flag_tmp == 1:
                    break
        if flag_tmp == 0 or target_tuple is None:
            print(f"[Debug] failed to resolve target tuple for planner object {objects[0]!r}")
            break
        print(f"[Debug] resolved target tuple: {target_tuple} from place {target_place}")
        print(target_place)
        place_target_visit_counts[target_place] = place_target_visit_counts.get(target_place, 0) + 1

        agent_state = agent.get_state()
        topomap.now = topomap.find_node(topomap.root, f'Place {target_place}')

        if 'Place'+f' {target_place}' != topomap.now.key:
            path = ThorShortestPath()
            path.requested_start = agent.state.position
            path.requested_end = topomap.now.position

            pathfinder_start_time = time.perf_counter()
            found_path = sim.pathfinder.find_path(path)
            pathfinder_elapsed = time.perf_counter() - pathfinder_start_time
            print(
                f"[Timing] pathfinder to-place elapsed={pathfinder_elapsed:.3f}s "
                f"found={found_path}"
            )
            path_points = path.points

            if found_path:
                for i, point in enumerate(path_points):
                    if i==0 :
                        continue
                    else :
                        final_length += math.sqrt((path_points[i][0]-path_points[i-1][0])**2+(path_points[i][2]-path_points[i-1][2])**2)
                        length_this_epoch += math.sqrt((path_points[i][0]-path_points[i-1][0])**2+(path_points[i][2]-path_points[i-1][2])**2)

        observations = []
        semantic_observations = []
        # Preserve the recorded floor height from the simulator instead of
        # forcing all navigation targets onto y=0.0.
        agent_state.position = copy.deepcopy(topomap.now.position)
        agent_state.rotation = float(angle_goal)
        agent.set_state(agent_state)
        obs = sim.get_sensor_observations()
        observations.append(obs)
        semantic_observations.append(obs["semantic_sensor"])
        color_image = obs["color_sensor"]

        scene = sim.semantic_scene

        # ==========================================================================================================================================
        # FIND PATH
        # ==========================================================================================================================================

        last_angle = target_tuple[1]
        target_node = topomap.find_node(topomap.root,f'Place {target_place}')
        if delete_traj:
            for i in range(0,len(target_node.describe)):
                last_data = json.loads(target_node.describe[i])
                if last_angle == last_data["Angle"]:
                    last_key.append(f'Place {target_place}')
                    last_index.append(i)
                    target_node.state = 'recompute'
                    break

        print(f'final_goal:{final_goal},sub_goal:{target_tuple[0]},place:{target_place},trajectory:{trajectory}',)
        sub_target_id = target_tuple[2]

        def is_door(object_id):
            obj = scene.objects[object_id]
            return obj.category.name() == "door" or obj.category.name() == "door frame"

        if directly_find and epoch == 29:
            for i,place_id_tmp in enumerate(topomap.place_clip_id):
                for j,object_tmp in enumerate(place_id_tmp):
                    if object_tmp[0][0].lower() in final_goal.lower() or final_goal.lower() in object_tmp[0][0].lower() or (final_goal == 'sofa' and 'couch' in object_tmp[0][0].lower()) or (final_goal == 'tv' and 'television' in object_tmp[0][0].lower()):
                            sub_target_id = object_tmp[0][2]

        def detect_distance_ahead(agent_position, direction, step_size=0.25, max_distance=5.0):
            distance_traveled = 0.0
            current_position = np.array(agent_position)
            while distance_traveled < max_distance:
                next_position = current_position + direction * step_size
                if not sim.pathfinder.is_navigable(next_position):
                    break
                current_position = next_position
                distance_traveled += step_size
            return distance_traveled




        print(f'sub_target_id:{sub_target_id}')
        target_name,target_position, target_dims = get_object_position(sub_target_id)
        print(f'target:{target_tuple[4].lower()}')
        print(f'final_goal:{final_goal.lower()}')
        print(f'final_length:{final_length}')

        current_place = agent_state.position
        if target_position[0] == last_target_position[0] and target_position[2] == last_target_position[2]:
            if objects[0] in trajectory:
                continue
            else:
                trajectory += f'{objects[0]}. '
                continue
        last_target_position = copy.deepcopy(target_position)
        if target_position is not None:
            path = ThorShortestPath()
            path.requested_start = agent.state.position
            path.requested_end = target_position

            current_position = copy.deepcopy(agent.state.position)
            previous_position = copy.deepcopy(current_position)
            steps = 0
            total_distance_traveled = 0.0
            step_size = 0.25
            current_index = 0

            pathfinder_start_time = time.perf_counter()
            found_path = sim.pathfinder.find_path(path)
            pathfinder_elapsed = time.perf_counter() - pathfinder_start_time
            print(
                f"[Timing] pathfinder to-subgoal elapsed={pathfinder_elapsed:.3f}s "
                f"found={found_path}"
            )
            path_points = path.points
            if found_path:
                observations = []
                subgoal_visible_logged = False
                final_goal_visible_logged = False
                while current_index < len(path_points) - 1:
                    segment_start = current_position
                    segment_end = np.array(path_points[current_index + 1])  # 确保 segment_end 可写

                    direction = segment_end - segment_start
                    segment_distance = np.linalg.norm(direction)
                    if segment_distance <= step_size:
                        current_position = segment_end
                        current_index += 1
                    else:
                        direction /= segment_distance
                        current_position += direction * step_size

                    distance_to_target = np.linalg.norm(current_position - target_position)
                    if early_stop ==True:
                        stop_distance = 0.25
                    else:
                        stop_distance = 0.05
                    if distance_to_target <= stop_distance :
                        print("Agent is within 1m of the target. Stopping.")
                        break

                    agent_state = ThorAgentState(np.array(current_position, dtype=np.float32), float(agent_state.rotation))

                    if current_index < len(path_points) - 1:
                        next_point = np.array(path_points[current_index + 1])  # 确保 next_point 可写
                        direction_to_next = next_point - current_position
                        direction_to_next /= np.linalg.norm(direction_to_next)
                        agent_state.rotation = vector_to_yaw(direction_to_next)

                    agent.set_state(agent_state)

                    step_distance = np.linalg.norm(current_position - previous_position)
                    total_distance_traveled += step_distance
                    previous_position = current_position.copy()

                    observations = sim.get_sensor_observations()
                    semantic_frame = observations["semantic_sensor"]
                    if not subgoal_visible_logged:
                        subgoal_visible, subgoal_visible_ratio, subgoal_bbox_width, subgoal_bbox_height = is_object_clearly_visible(
                            semantic_frame,
                            sub_target_id,
                            visible_ratio_threshold,
                            min_visible_bbox_side_px,
                        )
                        if subgoal_visible:
                            print(
                                f"[Debug] sub-goal visible on screen: label={target_tuple[4].lower()} "
                                f"ratio={subgoal_visible_ratio:.6f} bbox={subgoal_bbox_width}x{subgoal_bbox_height}"
                            )
                            subgoal_visible_logged = True
                    if final_goal_label_ids and not final_goal_visible_logged:
                        best_goal_ratio = 0.0
                        best_goal_id = None
                        best_goal_bbox_width = 0
                        best_goal_bbox_height = 0
                        for goal_label_id in final_goal_label_ids:
                            goal_visible, goal_ratio, goal_bbox_width, goal_bbox_height = is_object_clearly_visible(
                                semantic_frame,
                                goal_label_id,
                                visible_ratio_threshold,
                                min_visible_bbox_side_px,
                            )
                            if not goal_visible:
                                continue
                            if goal_ratio > best_goal_ratio:
                                best_goal_ratio = goal_ratio
                                best_goal_id = goal_label_id
                                best_goal_bbox_width = goal_bbox_width
                                best_goal_bbox_height = goal_bbox_height
                        if best_goal_id is not None:
                            goal_name, _, _ = get_object_position(best_goal_id)
                            color_image = observations["color_sensor"]
                            detection_request_id = f"goal-visible-{topomap.num_node}-{epoch}-{steps + 1}"
                            rgb_goal_detection = detect_goal_in_current_view(
                                color_image,
                                canonical_goal_name(goal_name),
                                detection_request_id,
                                int(agent_state.rotation),
                            )
                            if rgb_goal_detection is not None:
                                print(
                                    f"[Debug] final goal visible on screen: label={canonical_goal_name(goal_name)} "
                                    f"ratio={best_goal_ratio:.6f} semantic_bbox={best_goal_bbox_width}x{best_goal_bbox_height} "
                                    f"rgb_bbox={rgb_goal_detection['width']}x{rgb_goal_detection['height']}"
                                )
                                final_goal_visible_logged = True
                                episode_success = True
                                visible_goal_id = best_goal_id
                                visible_goal_name, visible_goal_target_position, _ = get_object_position(best_goal_id)
                                bbox_debug_path = f"tmp/navigation_images/final_goal_visible_step_{steps + 1}.png"
                                if save_goal_bbox_debug(
                                    color_image,
                                    semantic_frame,
                                    best_goal_id,
                                    canonical_goal_name(goal_name),
                                    bbox_debug_path,
                                ):
                                    print(f"[Debug] saved final goal bbox: {bbox_debug_path}")
                                image_path = f"tmp/navigation_images/navigation_step_{steps + 1}.png"
                                imageio.imwrite(image_path, color_image)
                                steps += 1
                                break
                            print(
                                f"[Debug] semantic goal candidate rejected by RGB detector: "
                                f"label={canonical_goal_name(goal_name)} ratio={best_goal_ratio:.6f} "
                                f"semantic_bbox={best_goal_bbox_width}x{best_goal_bbox_height}"
                            )
                    color_image = observations["color_sensor"]
                    image_path = f"tmp/navigation_images/navigation_step_{steps + 1}.png"
                    imageio.imwrite(image_path, color_image)
                    steps += 1

                if episode_success:
                    print("[Debug] final goal detected; approaching its coordinates before stopping")
                    final_length += total_distance_traveled
                    length_this_epoch += total_distance_traveled
                    if visible_goal_target_position is not None:
                        approach_stop_distance = 1.5
                        approach_path = ThorShortestPath()
                        approach_path.requested_start = agent.state.position
                        approach_path.requested_end = visible_goal_target_position
                        approach_pathfinder_start_time = time.perf_counter()
                        approach_found_path = sim.pathfinder.find_path(approach_path)
                        approach_pathfinder_elapsed = time.perf_counter() - approach_pathfinder_start_time
                        print(
                            f"[Timing] pathfinder final-approach elapsed={approach_pathfinder_elapsed:.3f}s "
                            f"found={approach_found_path}"
                        )
                        if approach_found_path:
                            approach_points = approach_path.points
                            approach_current_position = copy.deepcopy(agent.state.position)
                            approach_previous_position = copy.deepcopy(approach_current_position)
                            approach_index = 0
                            approach_distance_traveled = 0.0
                            while approach_index < len(approach_points) - 1:
                                approach_segment_end = np.array(approach_points[approach_index + 1])
                                approach_direction = approach_segment_end - approach_current_position
                                approach_segment_distance = np.linalg.norm(approach_direction)
                                if approach_segment_distance <= step_size:
                                    approach_current_position = approach_segment_end
                                    approach_index += 1
                                else:
                                    approach_direction /= approach_segment_distance
                                    approach_current_position += approach_direction * step_size

                                distance_to_visible_goal = np.linalg.norm(approach_current_position - visible_goal_target_position)
                                if distance_to_visible_goal <= approach_stop_distance:
                                    print(f"[Debug] close enough to final goal coordinates: label={canonical_goal_name(visible_goal_name)} distance={distance_to_visible_goal:.3f}")
                                    break

                                agent_state = ThorAgentState(np.array(approach_current_position, dtype=np.float32), float(agent_state.rotation))
                                if approach_index < len(approach_points) - 1:
                                    approach_next_point = np.array(approach_points[approach_index + 1])
                                    approach_direction_to_next = approach_next_point - approach_current_position
                                    if np.linalg.norm(approach_direction_to_next) > 1e-6:
                                        approach_direction_to_next /= np.linalg.norm(approach_direction_to_next)
                                        agent_state.rotation = vector_to_yaw(approach_direction_to_next)
                                agent.set_state(agent_state)
                                approach_step_distance = np.linalg.norm(approach_current_position - approach_previous_position)
                                approach_distance_traveled += approach_step_distance
                                approach_previous_position = approach_current_position.copy()
                            final_length += approach_distance_traveled
                            length_this_epoch += approach_distance_traveled
                        else:
                            print("[Debug] no path found for final goal close approach")
                    sr = 1
                    spl = 1 if final_length == 0 else min(1, distance / max(final_length, 1e-6))
                    break

                if is_door(sub_target_id) and through_door:
                    print("Target object is a door. Detecting distance ahead...")
                    direction_to_target = yaw_to_vector(agent.state.rotation)
                    max_distance_ahead = detect_distance_ahead(agent.state.position, direction_to_target)
                    print(f"Maximum distance ahead in the current direction: {max_distance_ahead:.2f} meters")
                    move_steps = int(max_distance_ahead / 0.25)
                    if move_steps > 0:
                        for _ in range(move_steps+3):
                            agent.act("move_forward")
                            total_distance_traveled += 0.25
                final_length += total_distance_traveled
                length_this_epoch += total_distance_traveled
            else:
                print('No path found')
        if not(last_answer == llava_answer2 and use_pruning):
            last_answer = llava_answer2


        if episode_success:
            break

    episode_elapsed = time.perf_counter() - episode_start_time
    print(
        f"[Timing] episode elapsed={episode_elapsed:.3f}s sr={sr} "
        f"spl={spl:.4f} final_length={final_length:.4f}"
    )
    return sr,spl,real_distance,final_length




def val_auto():
    experiment_seed = int(os.environ.get("EFFICIENTNAV_EXPERIMENT_SEED", "7"))
    fixed_goal_instance_index_raw = os.environ.get("EFFICIENTNAV_FIXED_GOAL_INSTANCE_INDEX")
    fixed_start_index_raw = os.environ.get("EFFICIENTNAV_FIXED_START_INDEX")
    fixed_start_rotation_raw = os.environ.get("EFFICIENTNAV_FIXED_START_ROTATION")

    fixed_goal_instance_index = None if fixed_goal_instance_index_raw in (None, "") else int(fixed_goal_instance_index_raw)
    fixed_start_index = None if fixed_start_index_raw in (None, "") else int(fixed_start_index_raw)
    fixed_start_rotation = None if fixed_start_rotation_raw in (None, "") else float(fixed_start_rotation_raw)

    houses = load_procthor_houses(seed=experiment_seed, split=os.environ.get("EFFICIENTNAV_PROCTHOR_SPLIT", "train"))
    forced_house_index_raw = os.environ.get("EFFICIENTNAV_HOUSE_INDEX", "1")
    forced_house_index = None
    if forced_house_index_raw not in (None, ""):
        try:
            forced_house_index = int(forced_house_index_raw)
        except ValueError:
            print(f"[Debug] invalid EFFICIENTNAV_HOUSE_INDEX={forced_house_index_raw!r}; falling back to sequential houses")
            forced_house_index = None
    if forced_house_index is not None:
        if forced_house_index < 0 or forced_house_index >= len(houses):
            print(
                f"[Debug] EFFICIENTNAV_HOUSE_INDEX={forced_house_index} is out of range "
                f"(available: 0..{len(houses) - 1}); falling back to house 0"
            )
            forced_house_index = 0
        houses = [houses[forced_house_index]]
        print(f"[Debug] using forced house index={forced_house_index}")

    for i, house in enumerate(houses):
        SR = 0.0
        SPL = 0.0
        total_episode = 0
        total_length = 0.0
        total_length_sr = 0.0
        easy_SR = 0.0
        easy_SPL = 0.0
        easy_episode = 0
        easy_threshould = 6.0
        easy_length = 0.0
        medium_SR = 0.0
        medium_SPL = 0.0
        medium_episode = 0
        medium_length = 0.0
        hard_threshould = 9.0
        hard_SR = 0.0
        hard_SPL = 0.0
        hard_episode = 0
        hard_length = 0.0
        if i >= num_episode:
            break

        sim_settings = {
            "width": render_width,
            "height": render_height,
            "sensor_height": 1,
            "color_sensor": True,
            "depth_sensor": True,
            "semantic_sensor": True,
            "seed": experiment_seed,
            "enable_physics": False,
            "fov_horizontal": 90.0,
            "grid_size": 0.25,
            "house": house,
        }

        cfg = make_cfg(sim_settings)

        try:
            sim.close()
        except Exception:
            pass

        sim = ThorSim(cfg)

        random.seed(sim_settings["seed"])
        sim.seed(sim_settings["seed"])

        sim.initialize_agent(agent_id=0)
        agent = sim.agents[0]

        try:
            topo_map
            del topo_map
            gc.collect()
            print('scene change')
        except:
            pass

        selected_goal_name = choose_goal_name_for_house(sim.semantic_scene)
        if selected_goal_name is None:
            continue

        topomap = Navigation_map()
        topomap.planner_model = planner_model
        topomap.semantic_model = model_clip
        topomap.semantic_tokenizer = clip_tokenizer
        topomap.semantic_max_length = clip_model_max_length
        topomap.processor = planner_tokenizer
        topomap.use_kv_cache = use_kv_cache
        topomap.similarity_threshould = [0.0 for _ in range(len(final_goal_list))]
        topomap.similarity_times = [0 for _ in range(len(final_goal_list))]

        SR_eposion = []
        fail_eposion = []
        subgoal_found = []

        reachable_positions = sim._reachable_positions
        candidate_objects = [
            (idx, obj) for idx, obj in enumerate(sim.semantic_scene.objects)
            if idx != 0 and canonical_goal_name(obj.category.name()) == selected_goal_name
        ]
        if len(reachable_positions) == 0 or len(candidate_objects) == 0:
            continue

        for j in range(num_environment):
            if j >= num_environment:
                break
            if fixed_goal_instance_index is not None:
                if fixed_goal_instance_index < 0 or fixed_goal_instance_index >= len(candidate_objects):
                    print(
                        f"[Debug] EFFICIENTNAV_FIXED_GOAL_INSTANCE_INDEX={fixed_goal_instance_index} "
                        f"is out of range for {selected_goal_name} candidates (0..{len(candidate_objects)-1}); "
                        f"falling back to candidate 0"
                    )
                    selected_goal_instance_index = 0
                else:
                    selected_goal_instance_index = fixed_goal_instance_index
                final_goal_id, goal_object = candidate_objects[selected_goal_instance_index]
            else:
                final_goal_id, goal_object = random.choice(candidate_objects)
            final_goal = selected_goal_name
            if fixed_start_index is not None:
                if fixed_start_index < 0 or fixed_start_index >= len(reachable_positions):
                    print(
                        f"[Debug] EFFICIENTNAV_FIXED_START_INDEX={fixed_start_index} "
                        f"is out of range for reachable positions (0..{len(reachable_positions)-1}); "
                        f"falling back to reachable position 0"
                    )
                    selected_start_index = 0
                else:
                    selected_start_index = fixed_start_index
                start_point = copy.deepcopy(reachable_positions[selected_start_index])
                far_start_metadata = None
            else:
                selected_start_index, start_point, far_start_metadata = select_far_start_position(
                    reachable_positions,
                    goal_object.obb.center,
                )
                if start_point is None:
                    start_point = copy.deepcopy(random.choice(reachable_positions))
                    selected_start_index = None
                    far_start_metadata = None

            if fixed_start_rotation is not None:
                start_rotation = float(fixed_start_rotation)
            else:
                start_rotation = random.choice([0.0, 90.0, 180.0, 270.0])

            print(
                f"[Debug] experiment setup: seed={experiment_seed} "
                f"goal_instance_index={selected_goal_instance_index if fixed_goal_instance_index is not None else 'random'} "
                f"start_index={selected_start_index if selected_start_index is not None else 'random'} "
                f"start_rotation={start_rotation} "
                f"start_point=({start_point[0]:.3f}, {start_point[1]:.3f}, {start_point[2]:.3f})"
            )
            if far_start_metadata is not None:
                print(
                    f"[Debug] far-start selection: selected_distance={far_start_metadata['selected_distance']:.3f} "
                    f"min_required_distance={far_start_metadata['min_required_distance']:.3f} "
                    f"max_distance={far_start_metadata['max_distance']:.3f} "
                    f"candidate_count={far_start_metadata['candidate_count']} "
                    f"reachable_count={far_start_metadata['reachable_count']}"
                )
            path = ThorShortestPath()
            path.requested_start = start_point
            path.requested_end = goal_object.obb.center
            found_path = sim.pathfinder.find_path(path)
            if not found_path:
                continue
            geodesic_distance = 0.0
            for k in range(1, len(path.points)):
                geodesic_distance += math.sqrt(
                    (path.points[k][0] - path.points[k - 1][0]) ** 2
                    + (path.points[k][2] - path.points[k - 1][2]) ** 2
                )
            if geodesic_distance <= 0.0:
                continue
            euclidean_distance = math.sqrt(
                (start_point[0] - goal_object.obb.center[0]) ** 2
                + (start_point[2] - goal_object.obb.center[2]) ** 2
            )

            distance = geodesic_distance
            episode_wall_start_time = time.perf_counter()
            sr, spl, real_distance,final_length= val_one_episode(topomap,sim,agent,start_point,start_rotation,final_goal_id,final_goal,distance)
            episode_wall_elapsed = time.perf_counter() - episode_wall_start_time
            SR += sr
            SPL += spl
            total_episode +=1
            total_length += final_length
            print(
                f"[Timing] val_auto episode total elapsed={episode_wall_elapsed:.3f}s "
                f"goal={final_goal} sr={sr} spl={spl:.4f}"
            )
            if sr == 1:
                print(f"[Debug] episode success: final_goal={final_goal} sr={sr} spl={spl:.4f} final_length={final_length:.4f}")
                total_length_sr += final_length
                SR_eposion.append(j)
                subgoal_found.append(final_goal)
                print("[Debug] goal reached and visible. Holding current view. Press Ctrl-C to exit.")
                while True:
                    time.sleep(1.0)
            else:
                fail_eposion.append(final_goal)
            if distance < easy_threshould:
                easy_SR += sr
                easy_SPL += spl
                easy_episode +=1
                if sr == 1:
                    easy_length += final_length
            elif distance > hard_threshould:
                hard_SR += sr
                hard_SPL += spl
                hard_episode +=1
                if sr == 1:
                    hard_length += final_length
            else:
                medium_SR += sr
                medium_SPL += spl
                medium_episode += 1
                if sr == 1:
                    medium_length += final_length
            os.makedirs(f'output/{current_time}', exist_ok=True)
            file_name_result = f'output/{current_time}/results{i}_test.txt'
            with open(file_name_result, 'w') as file:
                file.write(f"SR: {SR}\n")
                file.write(f"SPL: {SPL}\n")
                file.write(f"Total Episodes: {total_episode}\n")
                file.write(f"Total Length: {total_length}\n")
                file.write(f"Easy SR: {easy_SR}\n")
                file.write(f"Easy SPL: {easy_SPL}\n")
                file.write(f"Easy Episodes: {easy_episode}\n")
                file.write(f"Easy Length: {easy_length}\n")
                file.write(f"Medium SR: {medium_SR}\n")
                file.write(f"Medium SPL: {medium_SPL}\n")
                file.write(f"Medium Episodes: {medium_episode}\n")
                file.write(f"Medium Length: {medium_length}\n")
                file.write(f"Hard SR: {hard_SR}\n")
                file.write(f"Hard SPL: {hard_SPL}\n")
                file.write(f"Hard Episodes: {hard_episode}\n")
                file.write(f"Hard Length: {hard_length}\n")
                file.write(f"SR eposion: {SR_eposion}\n")
                file.write(f"SR subgoal: {subgoal_found}\n")
                file.write(f"num_node: {topomap.num_node}\n")





val_auto()
