
import os
import gzip
import json
import re
import time
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, AutoProcessor
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
from sensor_msgs.msg import Image as ROSImage
from std_msgs.msg import String as ROSString

print(f"[EfficientNav] running file: {__file__}")
print(f"[EfficientNav] units module: {units.__file__}")

os.makedirs("navigation_images", exist_ok=True)
os.makedirs("tmp/navigation_images", exist_ok=True)

cuda_available = torch.cuda.is_available()
cuda_device_count = torch.cuda.device_count() if cuda_available else 0
primary_device = "cuda:0" if cuda_available and cuda_device_count > 0 else "cpu"
max_memory = {idx: "47GiB" for idx in range(cuda_device_count)} if cuda_available else None
planner_device_map = "auto" if cuda_device_count > 0 else None
planner_model_path = os.environ.get("EFFICIENTNAV_QWEN_PATH", "PATH/TO/Qwen3.5-0.8B")
planner_tokenizer = AutoTokenizer.from_pretrained(planner_model_path, trust_remote_code=True)
planner_processor = AutoProcessor.from_pretrained(planner_model_path, trust_remote_code=True)
if planner_tokenizer.pad_token is None:
    planner_tokenizer.pad_token = planner_tokenizer.eos_token
planner_model_kwargs = {
    "torch_dtype": torch.float16 if cuda_available else torch.float32,
    "low_cpu_mem_usage": True,
    "trust_remote_code": True,
}
if planner_device_map is not None:
    planner_model_kwargs["device_map"] = planner_device_map
    if max_memory:
        planner_model_kwargs["max_memory"] = max_memory
planner_model = AutoModelForImageTextToText.from_pretrained(planner_model_path, **planner_model_kwargs)
planner_supports_vision = hasattr(planner_processor, "image_processor")
use_ros2_detection = os.environ.get("EFFICIENTNAV_USE_ROS2_DETECTION", "0") == "1"
observation_rotation_pause = float(os.environ.get("EFFICIENTNAV_OBSERVATION_ROTATION_PAUSE", "0.25"))
ros2_detection_timeout_sec = float(os.environ.get("EFFICIENTNAV_ROS2_DETECTION_TIMEOUT", "30.0"))


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
local_model_path = os.environ.get("EFFICIENTNAV_CLIP_PATH", "PATH/TO/clip")
clip_tokenizer = CLIPTokenizer.from_pretrained(local_model_path)
model_clip = CLIPTextModel.from_pretrained(local_model_path).to(device0)

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
use_kv_cache = False
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


class DetectionROSClient(Node):
    def __init__(self):
        super().__init__("efficientnav_detection_client")
        self.prompt_pub = self.create_publisher(ROSString, "/detection/prompt", 10)
        self.image_pub = self.create_publisher(ROSImage, "/camera/image_raw", 10)
        self.result_sub = self.create_subscription(ROSString, "/detection/json", self._result_callback, 10)
        self._results = {}

    def _result_callback(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        position_id = str(payload.get("position_id", ""))
        angle = int(payload.get("angle", -1))
        if not position_id or angle < 0:
            return
        self._results[(position_id, angle)] = payload

    def publish_detection_request(self, position_id, angle, prompt, image_np):
        payload = {
            "prompt": prompt,
            "position_id": position_id,
            "angle": int(angle),
        }
        prompt_msg = ROSString()
        prompt_msg.data = json.dumps(payload)
        self.prompt_pub.publish(prompt_msg)

        image_msg = ROSImage()
        image_msg.header.frame_id = "camera_link"
        image_msg.height = int(image_np.shape[0])
        image_msg.width = int(image_np.shape[1])
        image_msg.encoding = "rgb8"
        image_msg.is_bigendian = False
        image_msg.step = int(image_np.shape[1] * 3)
        image_msg.data = image_np.tobytes()
        self.image_pub.publish(image_msg)
        print(
            f"[Debug] published detection request: position_id={position_id} angle={int(angle)} "
            f"prompt={prompt!r}"
        )

    def wait_for_detection_result(self, position_id, angle, timeout_sec=None):
        if timeout_sec is None:
            timeout_sec = ros2_detection_timeout_sec
        deadline = time.time() + timeout_sec
        key = (str(position_id), int(angle))
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if key in self._results:
                print(
                    f"[Debug] received detection result: position_id={position_id} angle={int(angle)}"
                )
                return self._results.pop(key)
        raise TimeoutError(f"Timed out waiting for detection result for {position_id=} {angle=}")


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


def get_observation(images,depth):
    if not planner_supports_vision:
        qwen_answer = []
        position_looked = []
        for i in range(0,4):
            if depth[i].mean() <= depth_threshould:
                print(f"[Debug] get_observation skip angle={i * 90} reason=depth")
                continue
            position_looked.append(i * 90)
            qwen_answer.append(json.dumps({"Angle": i * 90, "Objects": ["door frame"]}, indent=4))
        return qwen_answer, position_looked

    observation_instruction = '''You need to make a purposeful observation of the image from the current perspective.
Then describe the main larger solid objects in the image in a short statement and follow the following format:
{ "Angle": 0, "Objects": ["Object name", "Object name"] }
Here are some things you should be aware of:
1. Entrances or doorways to other spaces in the room count as objects, which you need to describe. But do not describe doors.
2. Objects that are too small need no description.
3. You should describe the same object only once. You can describe 4 objects in the image at most.
4. Only output description follow the format, other content is not output.
5. Do not describe objects in the mirror.'''

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
        start = json_data[i].find('{')
        end = json_data[i].rfind('}') + 1
        json_str = json_data[i][start:end]

        try:
            data = json.loads(json_str)
        except Exception:
            print(f"[Debug] get_observation skip angle={i * 90} reason=json_parse raw={json_data[i]!r}")
            if position_looked and position_looked[-1] == i * 90:
                position_looked.pop()
            continue
        data["Angle"] = i * 90
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
            ros_client.publish_detection_request(position_id, key_angle, result, image_np)
            payload = ros_client.wait_for_detection_result(position_id, key_angle)
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
    def get_text_embedding(text):
        inputs = clip_tokenizer(text, return_tensors='pt').to(device0)
        with torch.no_grad():
            text_embedding = model_clip(**inputs).last_hidden_state
            text_embedding = text_embedding.mean(dim=1)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        return text_embedding.cpu().numpy()

    def get_similarity(text1, text2):
        vec1 = get_text_embedding(text1)
        vec2 = get_text_embedding(text2)
        similarity = 1 - cosine(vec1[0], vec2[0])
        return similarity

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

            similarities = [(label,obj.angle, obj.obj_id, get_similarity(label, obj.category), obj.category, obj.center) for obj in filtered_objects]

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
                            if get_similarity(final_goal_list[k], max_similar_objs[0][4]) + 0.1* max(get_similarity(final_goal_list[k], 'door'),get_similarity(final_goal_list[k], 'door frame'))> topomap.now.similarity[i][k]:
                                topomap.now.similarity[i][k] = get_similarity(final_goal_list[k], max_similar_objs[0][4])
            else:
                max_similar_objs = []
            if empty_flag == 1:
                break
    return max_similar_objs_list,empty_position


def planning(place_describe,place_describe_cache,final_goal,trajectory,allowed_objects=None,allowed_objects_by_place=None):
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
    if use_traj:
        input_text+= f'Here is the objects that you have traveled to before: {trajectory} Do not choose the objects that you have traveled to before as the target. '
    if pay_attention_to_door:
        if use_real_semetic:
            input_text+='Note that you can travel to door or door frame to other spaces if there are no clear evidence to choose the target. '
        else:
            input_text+='Note that you can travel to entrance or door frame to other spaces if there are no clear evidence to choose the target.'
    input_text+='''Return json data by referring to the following template.
            {"Place": x, "Angle": x, "Objects": ["xxxx"] }
            If your goal is already in the description, please choose it as the target. You should not output any information other than this json data. Note that your should choose only one object in one angle of one place in the json data as the target.'''
    if not use_kv_cache:
        prompt2 = build_chat_prompt(f"{place_describe}\n{input_text}")
        inputs2 = planner_tokenizer(prompt2, padding=True, return_tensors="pt").to(device0)
        with torch.no_grad():
            output2 = planner_model.generate(**inputs2, max_new_tokens=200, pad_token_id=planner_tokenizer.pad_token_id)
    else:
        # TODO: keep the existing KV-cache reuse path intact, but adapt this branch
        # for Qwen-compatible cache handling separately when use_kv_cache=True is needed.
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

    if not use_kv_cache:
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

    return llava_answer2






def val_one_episode(topomap,sim,agent,start_point,start_rotation,final_goal_id,final_goal,distance):
    final_goal = canonical_goal_name(final_goal)
    visible_ratio_threshold = 1e-4

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

    found_path = sim.pathfinder.find_path(path)
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

    def collect_allowed_objects():
        allowed = []
        seen = set()
        for place_candidates in topomap.place_clip_id:
            for object_tuple in place_candidates:
                if len(object_tuple) == 0:
                    continue
                label = str(object_tuple[0][0]).strip().lower()
                if label in trusted_planner_labels and label not in seen:
                    allowed.append(label)
                    seen.add(label)
        return allowed

    def collect_allowed_objects_by_place():
        allowed_by_place = {}
        for place_idx, place_candidates in enumerate(topomap.place_clip_id):
            labels = []
            seen = set()
            for object_tuple in place_candidates:
                if len(object_tuple) == 0:
                    continue
                label = str(object_tuple[0][0]).strip().lower()
                if label in trusted_planner_labels and label not in seen:
                    labels.append(label)
                    seen.add(label)
            allowed_by_place[place_idx] = labels
        return allowed_by_place

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
            max_similar_objs_list,empty_position = get_objects(topomap,scene,position_looked,box_info_list_sum,semantic_observations,obj_dict)
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
            similarity = topomap.get_similarity_threshould(topomap.root,last_key,last_index,target_index)
            similarity.sort(reverse=True)
            if len(similarity) <= node_pruning_num:
                topomap.similarity_threshould[target_index] = similarity[-1]
            else:
                topomap.similarity_threshould[target_index] = similarity[node_pruning_num]
        if use_kv_cache:
            topomap.used_groups = []
            place_describe,place_describe_cache= topomap.create_describe_and_cache(planner_model,topomap.root,last_key,last_index,target_index)
        else:
            place_describe= topomap.create_describe(topomap.root,last_key,last_index,target_index)
        print(place_describe)

        allowed_objects = collect_allowed_objects()
        allowed_objects_by_place = collect_allowed_objects_by_place()
        print(f"[Debug] allowed planner objects: {allowed_objects}")
        print(f"[Debug] allowed planner objects by place: {allowed_objects_by_place}")
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

        json_data = llava_answer2
        start = json_data.find('{')
        end = json_data.rfind('}') + 1
        json_str = json_data[start:end]
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
            data = json.loads(json_str)
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

        agent_state = agent.get_state()
        topomap.now = topomap.find_node(topomap.root, f'Place {target_place}')

        if 'Place'+f' {target_place}' != topomap.now.key:
            path = ThorShortestPath()
            path.requested_start = agent.state.position
            path.requested_end = topomap.now.position

            found_path = sim.pathfinder.find_path(path)
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

            found_path = sim.pathfinder.find_path(path)
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
                        subgoal_visible_ratio = float(np.mean(semantic_frame == sub_target_id))
                        if subgoal_visible_ratio > visible_ratio_threshold:
                            print(f"[Debug] sub-goal visible on screen: label={target_tuple[4].lower()} ratio={subgoal_visible_ratio:.6f}")
                            subgoal_visible_logged = True
                    if final_goal_label_ids and not final_goal_visible_logged:
                        best_goal_ratio = 0.0
                        best_goal_id = None
                        for goal_label_id in final_goal_label_ids:
                            goal_ratio = float(np.mean(semantic_frame == goal_label_id))
                            if goal_ratio > best_goal_ratio:
                                best_goal_ratio = goal_ratio
                                best_goal_id = goal_label_id
                        if best_goal_id is not None and best_goal_ratio > visible_ratio_threshold:
                            goal_name, _, _ = get_object_position(best_goal_id)
                            print(f"[Debug] final goal visible on screen: label={canonical_goal_name(goal_name)} ratio={best_goal_ratio:.6f}")
                            final_goal_visible_logged = True
                            episode_success = True
                            visible_goal_id = best_goal_id
                            visible_goal_name, visible_goal_target_position, _ = get_object_position(best_goal_id)
                            color_image = observations["color_sensor"]
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
                        approach_found_path = sim.pathfinder.find_path(approach_path)
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

    return sr,spl,real_distance,final_length




def val_auto():
    houses = load_procthor_houses(seed=7, split=os.environ.get("EFFICIENTNAV_PROCTHOR_SPLIT", "train"))

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
            "width": 1024,
            "height": 1024,
            "sensor_height": 1,
            "color_sensor": True,
            "depth_sensor": True,
            "semantic_sensor": True,
            "seed": 7,
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

        topomap = Navigation_map()
        topomap.planner_model = planner_model
        topomap.semantic_model = model_clip
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
            if idx != 0 and obj.category.name() in final_goal_list
        ]
        if len(reachable_positions) == 0 or len(candidate_objects) == 0:
            continue

        for j in range(num_environment):
            if j >= num_environment:
                break
            final_goal_id, goal_object = random.choice(candidate_objects)
            final_goal = canonical_goal_name(goal_object.category.name())
            start_point = copy.deepcopy(random.choice(reachable_positions))
            start_rotation = random.choice([0.0, 90.0, 180.0, 270.0])
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
            sr, spl, real_distance,final_length= val_one_episode(topomap,sim,agent,start_point,start_rotation,final_goal_id,final_goal,distance)
            SR += sr
            SPL += spl
            total_episode +=1
            total_length += final_length
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
