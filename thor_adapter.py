import math
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import prior
from ai2thor.controller import Controller


def load_procthor_houses(seed=7, split="train", shuffle=True):
    dataset = prior.load_dataset("procthor-10k")
    houses = []
    if split is not None:
        try:
            houses = list(dataset[split])
        except (KeyError, TypeError, IndexError):
            houses = []

    if not houses:
        try:
            for value in dataset.values():
                houses.extend(list(value))
        except AttributeError:
            for split_name in ("train", "val", "test"):
                try:
                    houses.extend(list(dataset[split_name]))
                except (KeyError, TypeError, IndexError):
                    continue
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(houses)
    return houses


def _position_to_array(position_dict):
    return np.array(
        [position_dict["x"], position_dict.get("y", 0.0), position_dict["z"]],
        dtype=np.float32,
    )


def yaw_to_vector(yaw_degrees):
    radians = math.radians(yaw_degrees)
    return np.array([math.sin(radians), 0.0, math.cos(radians)], dtype=np.float32)


def vector_to_yaw(direction):
    # TODO: confirm the exact AI2-THOR yaw convention for every house type.
    return float((math.degrees(math.atan2(direction[0], direction[2])) + 360.0) % 360.0)


def canonical_goal_name(name):
    normalized = (name or "unknown").lower()
    alias_map = {
        "television": "tv",
        "tvstand": "tv",
        "chair": "chair",
        "armchair": "chair",
        "diningchair": "chair",
        "sofa": "sofa",
        "couch": "sofa",
        "bed": "bed",
        "toilet": "toilet",
        "houseplant": "plant",
        "plant": "plant",
    }
    return alias_map.get(normalized, normalized)


@dataclass
class ThorAgentState:
    position: np.ndarray
    rotation: float = 0.0


class ThorCategory:
    def __init__(self, name):
        self._name = canonical_goal_name(name or "unknown")

    def name(self):
        return self._name


class ThorOBB:
    def __init__(self, center, sizes):
        self.center = np.array(center, dtype=np.float32)
        self.sizes = np.array(sizes, dtype=np.float32)


class ThorObject:
    def __init__(self, label, metadata):
        self.label = label
        self.object_id = metadata.get("objectId", f"object_{label}")
        self.category = ThorCategory(metadata.get("objectType", "unknown"))
        center = metadata.get("axisAlignedBoundingBox", {}).get("center", metadata.get("position", {}))
        size = metadata.get("axisAlignedBoundingBox", {}).get("size", {"x": 0.25, "y": 0.25, "z": 0.25})
        self.obb = ThorOBB(
            [
                center.get("x", 0.0),
                center.get("y", 0.0),
                center.get("z", 0.0),
            ],
            [
                size.get("x", 0.25),
                size.get("y", 0.25),
                size.get("z", 0.25),
            ],
        )


class ThorSemanticScene:
    def __init__(self, objects):
        self.objects = objects


class ThorShortestPath:
    def __init__(self):
        self.requested_start = None
        self.requested_end = None
        self.points = []


class ThorPathfinder:
    def __init__(self, sim):
        self.sim = sim

    def find_path(self, path):
        return self.sim._find_path(path)

    def is_navigable(self, position):
        return self.sim._is_navigable(position)


class ThorAgent:
    def __init__(self, sim):
        self.sim = sim
        self.state = ThorAgentState(np.zeros(3, dtype=np.float32), 0.0)

    def get_state(self):
        return ThorAgentState(self.state.position.copy(), float(self.state.rotation))

    def set_state(self, state):
        self.state = ThorAgentState(np.array(state.position, dtype=np.float32), float(state.rotation))
        event = self.sim.controller.step(
            action="Teleport",
            position={
                "x": float(self.state.position[0]),
                "y": float(self.state.position[1]),
                "z": float(self.state.position[2]),
            },
            rotation={"x": 0.0, "y": float(self.state.rotation), "z": 0.0},
            horizon=0.0,
            standing=True,
        )
        self.sim.last_event = event
        metadata = event.metadata.get("agent", {})
        if "position" in metadata and "rotation" in metadata:
            self.state.position = _position_to_array(metadata["position"])
            self.state.rotation = float(metadata["rotation"]["y"])
        return event

    def act(self, action_name):
        action_map = {
            "move_forward": "MoveAhead",
            "turn_left": "RotateLeft",
            "turn_right": "RotateRight",
        }
        event = self.sim.controller.step(action=action_map.get(action_name, action_name))
        self.sim.last_event = event
        metadata = event.metadata.get("agent", {})
        if "position" in metadata and "rotation" in metadata:
            self.state.position = _position_to_array(metadata["position"])
            self.state.rotation = float(metadata["rotation"]["y"])
        if action_name == "move_forward":
            metadata = event.metadata.get("agent", {})
            self.state.position = _position_to_array(metadata["position"])
            self.state.rotation = float(metadata["rotation"]["y"])
        return event


class ThorSim:
    def __init__(self, settings):
        self.settings = settings
        self.house = settings["house"]
        self.controller = Controller(
            scene=self.house,
            width=settings["width"],
            height=settings["height"],
            fieldOfView=settings["fov_horizontal"],
            gridSize=settings.get("grid_size", 0.25),
            renderDepthImage=settings.get("depth_sensor", True),
            renderInstanceSegmentation=settings.get("semantic_sensor", True),
            agentMode="default",
            snapToGrid=False,
            visibilityDistance=5.0,
        )
        self.last_event = self.controller.last_event
        self.agents = [ThorAgent(self)]
        self.pathfinder = ThorPathfinder(self)
        self.semantic_scene = None
        self._reachable_positions = []
        self._reachable_lookup = {}
        self._refresh_scene_cache()

    def close(self):
        self.controller.stop()

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def initialize_agent(self, agent_id=0):
        return self.agents[agent_id]

    def get_sensor_observations(self):
        event = self.last_event or self.controller.last_event
        return {
            "color_sensor": event.frame,
            "depth_sensor": event.depth_frame if getattr(event, "depth_frame", None) is not None else np.zeros(event.frame.shape[:2], dtype=np.float32),
            "semantic_sensor": self._build_semantic_frame(event),
        }

    def reset(self, house):
        self.house = house
        self.last_event = self.controller.reset(scene=house)
        self._refresh_scene_cache()

    def _refresh_scene_cache(self):
        metadata_objects = self.last_event.metadata.get("objects", [])
        objects = [ThorObject(0, {"objectId": "__background__", "objectType": "background"})]
        self.object_id_to_label = {"__background__": 0}
        for index, metadata in enumerate(metadata_objects, start=1):
            objects.append(ThorObject(index, metadata))
            self.object_id_to_label[metadata["objectId"]] = index
        self.semantic_scene = ThorSemanticScene(objects)
        event = self.controller.step(action="GetReachablePositions")
        self.last_event = event
        self._reachable_positions = [
            _position_to_array(pos) for pos in (event.metadata.get("actionReturn") or [])
        ]
        self._reachable_lookup = {
            self._grid_key(pos): pos for pos in self._reachable_positions
        }

    def _build_semantic_frame(self, event):
        # TODO: Unlike Habitat's dense semantic sensor, AI2-THOR only exposes
        # visible instance segmentation. Pixels without a visible instance stay
        # as background label 0, so downstream bbox+unique matching may be
        # noisier around thin objects and occlusion boundaries.
        semantic = np.zeros(event.frame.shape[:2], dtype=np.int32)

        instance_segmentation_frame = getattr(event, "instance_segmentation_frame", None)
        color_to_object_id = getattr(event, "color_to_object_id", None)
        if instance_segmentation_frame is not None and color_to_object_id:
            color_frame = np.asarray(instance_segmentation_frame)
            if color_frame.ndim == 3 and color_frame.shape[:2] == semantic.shape:
                encoded = (
                    color_frame[..., 0].astype(np.int32) << 16
                    | color_frame[..., 1].astype(np.int32) << 8
                    | color_frame[..., 2].astype(np.int32)
                )
                for color, object_id in color_to_object_id.items():
                    if isinstance(color, str):
                        continue
                    try:
                        color_key = tuple(color)
                    except TypeError:
                        continue
                    if len(color_key) != 3:
                        continue
                    encoded_color = (
                        int(color_key[0]) << 16
                        | int(color_key[1]) << 8
                        | int(color_key[2])
                    )
                    label = self.object_id_to_label.get(object_id, 0)
                    semantic[encoded == encoded_color] = label
                return semantic

        instance_masks = getattr(event, "instance_masks", None) or {}
        # Paint larger masks first so smaller visible instances can overwrite
        # them at boundaries, which is closer to the expected visible-instance
        # labeling used by bbox crop + np.unique().
        for object_id, mask in sorted(
            instance_masks.items(),
            key=lambda item: int(np.count_nonzero(item[1])),
            reverse=True,
        ):
            label = self.object_id_to_label.get(object_id, 0)
            semantic[np.asarray(mask, dtype=bool)] = label
        return semantic

    def _grid_key(self, position):
        return (
            round(float(position[0]), 2),
            round(float(position[1]), 2),
            round(float(position[2]), 2),
        )

    def _nearest_reachable(self, position):
        if not self._reachable_positions:
            return np.array(position, dtype=np.float32)
        target = np.array(position, dtype=np.float32)
        return min(
            self._reachable_positions,
            key=lambda pos: np.linalg.norm(pos[[0, 2]] - target[[0, 2]]) + 2.0 * abs(float(pos[1] - target[1])),
        )

    def _is_navigable(self, position):
        target = np.array(position, dtype=np.float32)
        return any(
            np.linalg.norm(pos[[0, 2]] - target[[0, 2]]) < 0.26 and abs(float(pos[1] - target[1])) < 0.6
            for pos in self._reachable_positions
        )

    def _neighbor_indices(self, index, grid_size):
        current = self._reachable_positions[index]
        neighbors = []
        horizontal_limit = grid_size * 1.6
        vertical_limit = 0.6
        for neighbor_index, candidate in enumerate(self._reachable_positions):
            if neighbor_index == index:
                continue
            horizontal_distance = np.linalg.norm(candidate[[0, 2]] - current[[0, 2]])
            vertical_distance = abs(float(candidate[1] - current[1]))
            if horizontal_distance <= horizontal_limit and vertical_distance <= vertical_limit:
                neighbors.append(neighbor_index)
        return neighbors

    def _find_path(self, path):
        grid_size = float(self.settings.get("grid_size", 0.25))
        start = self._nearest_reachable(path.requested_start)
        end = self._nearest_reachable(path.requested_end)
        start_key = self._grid_key(start)
        end_key = self._grid_key(end)
        start_index = None
        end_index = None
        for idx, position in enumerate(self._reachable_positions):
            key = self._grid_key(position)
            if key == start_key and start_index is None:
                start_index = idx
            if key == end_key and end_index is None:
                end_index = idx
        if start_index is None or end_index is None:
            path.points = [np.array(start, dtype=np.float32), np.array(end, dtype=np.float32)]
            return False
        queue = deque([start_index])
        parents = {start_index: None}
        while queue:
            current = queue.popleft()
            if current == end_index:
                break
            for neighbor in self._neighbor_indices(current, grid_size):
                if neighbor not in parents:
                    parents[neighbor] = current
                    queue.append(neighbor)
        if end_index not in parents:
            path.points = [np.array(start, dtype=np.float32), np.array(end, dtype=np.float32)]
            return False
        key_path = []
        cursor = end_index
        while cursor is not None:
            key_path.append(cursor)
            cursor = parents[cursor]
        key_path.reverse()
        path.points = [np.array(self._reachable_positions[index], dtype=np.float32) for index in key_path]
        return True
