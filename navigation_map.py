import json
import re
import torch
import numpy as np
from typing import List, Tuple, Optional
import time
import math
import gc
import copy
from transformers.cache_utils import Cache, DynamicCache

width_weight = 0.001
gpu_node_num = 20
use_pruning = True
layer_threshold = 5
low_information_memory_labels = {"wall", "floor", "ceiling"}
frontier_memory_labels = {"doorway", "door frame", "door", "window"}
structural_phrase_tokens = {
    "a", "an", "the", "of", "with", "corner", "wall", "floor", "ceiling",
    "tiled", "pattern"
}

class TreeNode:
    def __init__(self, key: str, position: List[float], direction: float, waypoint = None, distance_to_parent: float = 0.0, parent: Optional['TreeNode'] = None, picture = None, describe = None):
        self.key = key
        self.position = position
        self.direction = direction
        self.waypoint = waypoint
        self.distance_to_parent = distance_to_parent
        self.picture = picture
        self.parent = parent
        self.children = []
        self.describe = describe
        self.describe_kv = []
        self.similarity = []
        self.current_inference = 0
        self.state = 'explorable'
        self.group = None
        self.describe_kv_signature = None

    def add_child(self, child_node: 'TreeNode'):
        child_node.parent = self
        self.children.append(child_node)

    def to_dict(self):
        return {
            'key': self.key,
            'position': self.position,
            'direction': self.direction,
            'waypoint': self.waypoint,
            'distance_to_parent': self.distance_to_parent,
            'children': [child.to_dict() for child in self.children]
        }

    @staticmethod
    def from_dict(data: dict, parent: Optional['TreeNode'] = None) -> 'TreeNode':
        node = TreeNode(
            data['key'],
            tuple(data['position']),
            data['direction'],
            data['waypoint'],
            data['distance_to_parent'],
            parent
        )
        node.children = [TreeNode.from_dict(child, node) for child in data['children']]
        return node
    
    def find_connections(self):
        connection_map = ' '
        if not self.children:
            return connection_map
        else:
            for i in range(len(self.children)):
                connection_map += f'{self.key} is connected to {self.children[i].key}. '
            for i in range(len(self.children)):
                connection_map += self.children[i].find_connections()
            return connection_map

class Navigation_map:
    def __init__(self, root: Optional[TreeNode] = None):
        self.planner_model = None
        self.semantic_model = None
        self.semantic_tokenizer = None
        self.semantic_max_length = None
        self.processor = None
        self.use_kv_cache = False
        self.root = root
        self.now = root
        self.current_inference = 0
        self.num_node = 0
        self.similarity_threshould = None
        self.similarity_times = None
        self.store_in_cpu = []
        self.store_in_gpu = []
        self.store_in_gpu_score = []
        self.used_id = []
        self.device_map = None
        self.used_groups = []
        self.place_clip_id = []
        self.kv_cache_supported = True

    def disable_kv_cache(self, reason: str):
        self.use_kv_cache = False
        self.kv_cache_supported = False
        print(f"[Debug] disabling KV cache fallback: {reason}")

    def _normalize_similarity_text(self, text):
        normalized = str(text or "").strip().lower()
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized

    def _canonical_similarity_label(self, label):
        normalized = self._normalize_similarity_text(label)
        alias_map = {
            "television": "tv",
            "tvstand": "tv",
            "couch": "sofa",
            "armchair": "chair",
            "diningchair": "chair",
            "houseplant": "plant",
        }
        return alias_map.get(normalized, normalized)

    def _compact_similarity_label(self, label):
        normalized = self._normalize_similarity_text(label)
        if not normalized:
            return ""

        keyword_groups = [
            ({"toilet"}, "toilet"),
            ({"sink"}, "sink"),
            ({"doorway", "door", "doors", "doorframe", "frame"}, "doorway"),
            ({"window", "windows"}, "window"),
            ({"mirror"}, "mirror"),
            ({"television", "tv", "tvstand"}, "tv"),
            ({"sofa", "couch"}, "sofa"),
            ({"chair", "chairs", "armchair", "diningchair"}, "chair"),
            ({"table", "diningtable", "desk"}, "table"),
            ({"plant", "houseplant", "vase"}, "plant"),
            ({"statue", "sculpture"}, "statue"),
            ({"lamp"}, "lamp"),
            ({"bed"}, "bed"),
            ({"wall", "walls"}, "wall"),
            ({"floor", "floors"}, "floor"),
            ({"ceiling", "ceilings"}, "ceiling"),
        ]
        tokens = [token for token in re.findall(r"[a-z0-9]+", normalized) if token]
        token_set = set(tokens)
        for keywords, canonical in keyword_groups:
            if token_set & keywords:
                return canonical

        filtered_tokens = [
            token for token in tokens
            if token not in {
                "there", "is", "are", "the", "a", "an", "of", "in", "on", "at",
                "to", "from", "with", "and", "behind", "side", "image", "room",
                "outside", "inside", "left", "right", "made", "up", "that",
                "this", "these", "those", "reflecting", "closed", "open", "silver",
                "light", "blue", "grey", "gray", "rectangular", "tiles", "tile",
            }
        ]
        if not filtered_tokens:
            return ""
        return " ".join(filtered_tokens[:3])

    def _description_signature_text(self, raw_description):
        labels = self._extract_description_labels(raw_description)
        if not labels:
            return ""
        compact_labels = []
        seen = set()
        for label in labels:
            compact_label = self._compact_similarity_label(label)
            if compact_label and compact_label not in seen:
                compact_labels.append(compact_label)
                seen.add(compact_label)
        return " ".join(compact_labels)

    def _node_description_text(self, node, final_goal_label=None, last_key=None, last_index=None):
        if node is None or not node.describe:
            return ""
        if last_key is not None and last_index is not None and final_goal_label is not None:
            selected_indices = self._get_selected_description_indices(node, last_key, last_index, final_goal_label)
        else:
            selected_indices = list(range(len(node.describe)))
        parts = [
            self._description_signature_text(node.describe[i])
            for i in selected_indices
            if 0 <= i < len(node.describe)
        ]
        return self._normalize_similarity_text(" ".join(parts))

    def _encode_similarity_text(self, text):
        normalized_text = self._normalize_similarity_text(text)
        if not normalized_text:
            return None
        if self.semantic_model is None or self.semantic_tokenizer is None:
            return None
        model_device = next(self.semantic_model.parameters()).device
        max_length = self.semantic_max_length
        if max_length is None:
            tokenizer_max_length = getattr(self.semantic_tokenizer, "model_max_length", None)
            if isinstance(tokenizer_max_length, int) and 0 < tokenizer_max_length < 100000:
                max_length = tokenizer_max_length
        tokenizer_kwargs = {"return_tensors": "pt", "truncation": True}
        if max_length is not None:
            tokenizer_kwargs["max_length"] = max_length
        inputs = self.semantic_tokenizer(normalized_text, **tokenizer_kwargs).to(model_device)
        with torch.no_grad():
            text_embedding = self.semantic_model(**inputs).last_hidden_state
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).to(text_embedding.dtype)
                text_embedding = (text_embedding * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            else:
                text_embedding = text_embedding.mean(dim=1)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        return text_embedding[0].detach().cpu().numpy()

    def _similarity_score(self, text_a, text_b):
        embedding_a = self._encode_similarity_text(text_a)
        embedding_b = self._encode_similarity_text(text_b)
        if embedding_a is not None and embedding_b is not None:
            similarity = float(np.dot(embedding_a, embedding_b))
            if not math.isnan(similarity):
                return similarity

        tokens_a = set(re.findall(r"[a-z0-9]+", self._normalize_similarity_text(text_a)))
        tokens_b = set(re.findall(r"[a-z0-9]+", self._normalize_similarity_text(text_b)))
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

    def _collect_group_texts(self, node, current_node):
        group_texts = {}
        if node is None:
            return group_texts
        if node.key != current_node.key and node.group is not None:
            node_text = self._node_description_text(node)
            if node_text:
                group_texts.setdefault(node.group, []).append(node_text)
        for child in node.children:
            child_texts = self._collect_group_texts(child, current_node)
            for group_id, texts in child_texts.items():
                group_texts.setdefault(group_id, []).extend(texts)
        return group_texts

    def _get_group_prefix_text(self, group, node, current_node):
        prefix_parts = []
        if node.group == group and node.key != current_node.key:
            node_text = self._node_description_text(node)
            if node_text:
                prefix_parts.append(node_text)
        for child in node.children:
            child_prefix = self._get_group_prefix_text(group, child, current_node)
            if child_prefix:
                prefix_parts.append(child_prefix)
        return " ".join(part for part in prefix_parts if part)

    def _build_cache_from_text(self, describe_text, past_key_values=None):
        normalized_text = self._normalize_similarity_text(describe_text)
        if not normalized_text:
            return None
        conversation_kv = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": normalized_text},
                ],
            }
        ]
        prompt_kv = self.processor.apply_chat_template(
            conversation_kv,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs_kv = self.processor(prompt_kv, padding=True, return_tensors="pt").to("cuda:0")
        runtime_cache = self._legacy_to_dynamic_cache(past_key_values)
        with torch.no_grad():
            output_kv = self.planner_model(
                input_ids=inputs_kv["input_ids"],
                use_cache=True,
                past_key_values=runtime_cache,
            )
        return self._extract_legacy_kv_cache(
            output_kv.past_key_values,
            inputs_kv["input_ids"].shape[1],
        )

    def _extract_legacy_kv_cache(self, past_key_values, seq_len: int):
        legacy_cache = []
        for layer in getattr(past_key_values, "layers", []):
            if hasattr(layer, "keys") and hasattr(layer, "values"):
                key_tensor = layer.keys
                value_tensor = layer.values
            else:
                raise TypeError(
                    f"unsupported cache layer type for EfficientNav KV reuse: {type(layer).__name__}"
                )
            legacy_cache.append(
                (
                    key_tensor[:, :, -seq_len:, :],
                    value_tensor[:, :, -seq_len:, :],
                )
            )
        if not legacy_cache:
            raise TypeError(
                f"unsupported past_key_values type for EfficientNav KV reuse: {type(past_key_values).__name__}"
            )
        return tuple(legacy_cache)

    def _legacy_to_dynamic_cache(self, legacy_cache):
        if legacy_cache is None or isinstance(legacy_cache, Cache):
            return legacy_cache
        if not isinstance(legacy_cache, tuple):
            raise TypeError(f"unsupported legacy cache type: {type(legacy_cache).__name__}")
        cache = DynamicCache()
        for layer_idx, layer_cache in enumerate(legacy_cache):
            if not isinstance(layer_cache, tuple) or len(layer_cache) != 2:
                raise TypeError(f"invalid layer cache format at layer {layer_idx}")
            key_states, value_states = layer_cache
            cache.update(key_states, value_states, layer_idx)
        return cache

    def add_node(self, parent_key: str, key: str, position: List[float], direction: float, waypoint, distance_to_parent: float,picture,describe):
        if not self.root:
            self.root = TreeNode(key, position, direction, waypoint, distance_to_parent,None,picture,describe)
            self.now = self.root
            self.now.group = 0
        else:
            parent_node = self.now
            # parent_node = self.find_node(self.root, parent_key)
            if parent_node:
                child = TreeNode(key, position, direction, waypoint, distance_to_parent, parent_node,picture,describe)
                parent_node.add_child(child)
            else:
                raise ValueError("Parent key not found in the tree")
            self.now = child
            if self.use_kv_cache and self.kv_cache_supported and self.planner_model is not None and self.processor is not None:
                try:
                    self.get_node_group(self.planner_model,self.now)
                except Exception as exc:
                    self.disable_kv_cache(f"group assignment failed: {exc}")
        if self.use_kv_cache and self.kv_cache_supported and self.planner_model is not None and self.processor is not None:
            try:
                self.compute_kv(self.now, list(range(len(self.now.describe))))
            except Exception as exc:
                self.disable_kv_cache(f"compute_kv failed: {exc}")

    def compute_kv(self,node,num_node):
        describe = ' '
        group_kv = None
        if self.num_node > 1 and node.group is not None:
            group_prefix_text = self._get_group_prefix_text(node.group, self.root, node)
            if group_prefix_text:
                group_kv = self._build_cache_from_text(group_prefix_text)
        for i in num_node:
            describe += node.describe[i]
        node.describe_kv = self._build_cache_from_text(describe, past_key_values=group_kv)
        node.describe_kv_signature = tuple(num_node)
        if self.device_map == None:
            self.device_map = [[tensor[0].device,tensor[1].device] for tensor in node.describe_kv]
        if node.key not in self.store_in_gpu:
            node.describe_kv = tuple((tensor[0].to('cpu'),tensor[1].to('cpu')) for tensor in node.describe_kv)
        # import pdb;pdb.set_trace()
    
    def load_kv_to_gpu(self,node):
        node.describe_kv = tuple((tensor[0].to(self.device_map[i][0]),tensor[1].to(self.device_map[i][1])) for i,tensor in enumerate(node.describe_kv))

    def find_node(self, node: TreeNode, key: str) -> Optional[TreeNode]:
        if node.key == key:
            return node
        for child in node.children:
            found_node = self.find_node(child, key)
            if found_node:
                return found_node
        return None
    
    def find_nearest_node(self, node: TreeNode, position) -> Optional[TreeNode]:
        nearest_length = 1000
        nearest_position = None
        nearest_node = None
        for child in node.children:
            length,nearest_position_child,child_node = self.find_nearest_node(child, position)
            if length < nearest_length:
                nearest_length = length
                nearest_position = nearest_position_child
                nearest_node = child_node
        length = math.sqrt((position[0]-node.position[0])**2+(position[1]-node.position[1])**2+(position[2]-node.position[2])**2)
        if length < nearest_length:
            nearest_length = length
            nearest_position = node.position
            nearest_node = node
        return nearest_length,nearest_position,nearest_node

    def get_path(self, start_key: str, end_key: str) -> List[Tuple[str, Tuple[float, float]]]:
        start_node = self.find_node(self.root, start_key)
        end_node = self.find_node(self.root, end_key)
        if not start_node or not end_node:
            raise ValueError("One or both keys not found in the tree")
        
        # Get path from start_node to root
        path_to_root_from_start = []
        node = start_node
        while node:
            path_to_root_from_start.append(node)
            node = node.parent
        path_to_root_from_start.reverse()
        
        # Get path from end_node to root
        path_to_root_from_end = []
        node = end_node
        while node:
            path_to_root_from_end.append(node)
            node = node.parent

        # Find common ancestor
        i = 0
        while (i < len(path_to_root_from_start) and i < len(path_to_root_from_end) and
               path_to_root_from_start[i] == path_to_root_from_end[i]):
            i += 1

        common_ancestor_index = i - 1

        # Path from start to common ancestor
        path = path_to_root_from_start[:common_ancestor_index + 1]

        # Path from common ancestor to end (in reverse)
        path.extend(reversed(path_to_root_from_end[common_ancestor_index + 1:]))

        return [(node.key, node.position) for node in path]

    def save_tree(self, filename: str):
        if not self.root:
            raise ValueError("Tree is empty")
        with open(filename, 'w') as file:
            json.dump(self.root.to_dict(), file)

    def load_tree(self, filename: str):
        with open(filename, 'r') as file:
            data = json.load(file)
            self.root = TreeNode.from_dict(data)

    def print_tree(self, node: Optional[TreeNode] = None, level: int = 0):
        if node is None:
            node = self.root
        print('  ' * level + f"({node.key}) Position: {node.position}, Direction: {node.direction}, Distance to parent: {node.distance_to_parent}")
        for child in node.children:
            self.print_tree(child, level + 1)

    def _get_consumed_description_indices(self, node, last_key, last_index):
        consumed_indices = set()
        if node.key in last_key:
            for j in range(len(last_key)):
                if node.key == last_key[j]:
                    consumed_indices.add(last_index[j])
        return consumed_indices

    def _extract_description_labels(self, raw_description):
        try:
            description_data = json.loads(raw_description)
        except Exception:
            return []

        labels = []
        for obj in description_data.get("Objects", []):
            label = str(obj).strip().lower()
            if label:
                labels.append(label)
        return labels

    def _is_low_information_label(self, label, final_goal_label):
        normalized_label = str(label).strip().lower()
        normalized_goal = str(final_goal_label).strip().lower()
        if normalized_label == normalized_goal:
            return False
        if normalized_label in frontier_memory_labels:
            return False
        if normalized_label in low_information_memory_labels:
            return True
        tokens = [token for token in re.split(r"[^a-z0-9]+", normalized_label) if token]
        if any(token in {"door", "doorway", "window"} for token in tokens):
            return False
        if any(token in low_information_memory_labels for token in tokens):
            non_structural_tokens = [token for token in tokens if token not in structural_phrase_tokens]
            return len(non_structural_tokens) == 0
        return False

    def _is_productive_description(self, raw_description, final_goal_label):
        labels = self._extract_description_labels(raw_description)
        if not labels:
            return True
        for label in labels:
            if not self._is_low_information_label(label, final_goal_label):
                return True
        return False

    def _get_selected_description_indices(self, node, last_key, last_index, final_goal_label):
        consumed_indices = self._get_consumed_description_indices(node, last_key, last_index)
        active_indices = [i for i in range(len(node.describe)) if i not in consumed_indices]
        productive_indices = [
            i for i in active_indices
            if self._is_productive_description(node.describe[i], final_goal_label)
        ]
        if productive_indices:
            return productive_indices
        return active_indices


    def create_describe(self,node,last_key,last_index,target_index,final_goal_label):
        if use_pruning:
            selected = any(np.array([row[target_index]-width_weight*math.sqrt((node.position[0]-self.now.position[0])**2+(node.position[2]-self.now.position[2])**2) for row in node.similarity]) > self.similarity_threshould[target_index])  or any([any(np.array([row[target_index]-width_weight*math.sqrt((children.position[0]-self.now.position[0])**2+(children.position[2]-self.now.position[2])**2) for row in children.similarity]) > self.similarity_threshould[target_index] )for children in node.children])
            if node.parent != None:
                selected = selected or any(np.array([row[target_index]-width_weight*math.sqrt((node.parent.position[0]-self.now.position[0])**2+(node.parent.position[2]-self.now.position[2])**2) for row in node.parent.similarity]) > self.similarity_threshould[target_index])
        else:
            selected = True
        describe = ' '
        if selected:
            describe = f'{node.key} : '
            selected_indices = self._get_selected_description_indices(node, last_key, last_index, final_goal_label)
            if selected_indices:
                for i in selected_indices:
                    describe += node.describe[i]
            else:
                describe = ' '
        for child in node.children:
            describe += self.create_describe(child,last_key,last_index,target_index,final_goal_label)
            
        return describe
    
    def get_similarity_threshould(self,node,last_key,last_index,target_index,final_goal_label):
        similarity_child=[]
        similarities = [0.0]
        selected_indices = self._get_selected_description_indices(node, last_key, last_index, final_goal_label)
        for i in selected_indices:
            similarities.append(node.similarity[i][target_index]-width_weight*math.sqrt((node.position[0]-self.now.position[0])**2+(node.position[2]-self.now.position[2])**2))
        
        for child in node.children:
            similarity_child = self.get_similarity_threshould(child,last_key,last_index,target_index,final_goal_label)
        for similar in similarity_child:
            similarities.append(similar)
        return similarities

    def find_token_length(self,node,tokenizer):
        length = []
        prompt = ' '
        for i in range(len(node.describe)):
            prompt += node.describe[i]
        prompt_token = tokenizer(prompt)
        length.append(len(prompt_token['input_ids']))
        for i in range(len(node.children)):
            length_child = self.find_token_length(node.children[i],tokenizer)
            for i in range(len(length_child)):
                length.append(length_child[0])
        return length
    

    def get_node_group(self,model,node):
        threshold = 0.30
        group_texts = self._collect_group_texts(self.root, node)
        if not group_texts:
            node.group = 0
            return
        describe_current = self._node_description_text(node)
        best_group = None
        best_score = -1.0
        for group_id, text_parts in group_texts.items():
            group_text = self._normalize_similarity_text(" ".join(text_parts))
            score = self._similarity_score(describe_current, group_text)
            if score > best_score:
                best_score = score
                best_group = group_id
        if best_group is not None and best_score >= threshold:
            node.group = best_group
        else:
            node.group = (max(group_texts.keys()) + 1) if group_texts else 0

    
    def _create_describe_for_cache(self,node,last_key,last_index,target_index,final_goal_label):
        if use_pruning:
            selected = any(np.array([row[target_index]-width_weight*math.sqrt((node.position[0]-self.now.position[0])**2+(node.position[2]-self.now.position[2])**2) for row in node.similarity]) > self.similarity_threshould[target_index]) or node.group in self.used_groups
            selected = selected or any([any(np.array([row[target_index]-width_weight*math.sqrt((children.position[0]-self.now.position[0])**2+(children.position[2]-self.now.position[2])**2) for row in children.similarity]) > self.similarity_threshould[target_index] )for children in node.children])
            if node.parent != None:
                selected = selected or any(np.array([row[target_index]-width_weight*math.sqrt((node.parent.position[0]-self.now.position[0])**2+(node.parent.position[2]-self.now.position[2])**2) for row in node.parent.similarity]) > self.similarity_threshould[target_index])
        else:
            selected = True

        describe = ' '
        if selected:
            selected_indices = self._get_selected_description_indices(node, last_key, last_index, final_goal_label)
            if selected_indices:
                self.used_groups.append(node.group)
                for i in selected_indices:
                    describe += node.describe[i]
        for child in node.children:
            describe += self._create_describe_for_cache(child,last_key,last_index,target_index,final_goal_label)
        return describe

    def create_describe_and_cache(self,model,node,last_key,last_index,target_index,final_goal_label):
        describe = self._create_describe_for_cache(node,last_key,last_index,target_index,final_goal_label)
        describe_kv = self._build_cache_from_text(describe) if describe.strip() else None
        return describe,describe_kv
