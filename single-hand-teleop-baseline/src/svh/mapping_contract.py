from __future__ import annotations

"""H2O 标签、训练 checkpoint 与实时 SVH 9 通道映射的语义指纹。"""

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from svh.svh_layout import SVH_9CH_NAMES


MAPPING_CONTRACT_VERSION = "svh9-label-v2-open-release"
H2O_LABEL_GESTURE_CONTEXT_POLICY = "stateless_raw_gesture_as_stable_proxy"
RUNTIME_GESTURE_CONTEXT_POLICY = "consecutive_gesture_stabilizer"

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 前五个模块全部参与指纹；H2O adapter 只选取会改变 pose-only 标签的常量与函数，
# 避免单纯调整数据枚举、manifest 排版或诊断文本就让现役 checkpoint 失效。
_MAPPING_ALGORITHM_AST_TARGETS: Dict[str, tuple[str, ...] | None] = {
    "src/features/geometry_utils.py": None,
    "src/features/hand_features.py": None,
    "src/gesture/rule_based_gesture.py": (
        "_mean",
        "infer_gesture_raw",
        "GestureStabilizer",
    ),
    "src/control/control_representation.py": None,
    "src/svh/svh_adapter.py": None,
    "experiments/intent_prediction/intent_prediction/h2o_adapter.py": (
        "H2O_POSE_VALUE_COUNT",
        "H2O_RIGHT_VALID_INDEX",
        "H2O_RIGHT_POINTS_START",
        "HAND_LANDMARK_COUNT",
        "SVH_CHANNEL_COUNT",
        "read_h2o_right_hand_pose",
        "project_h2o_points_normalized",
        "canonicalize_h2o_camera_xy",
        "h2o_frame_to_svh9",
    ),
}

_RUNTIME_CONTEXT_DEFAULTS: Dict[str, Any] = {
    "stable_gesture_min_consecutive": 2,
    "stable_unknown_consecutive": 1,
}

# 该值是版本的一部分，不能在公式变化后原地更新；公式或上下文变化必须先升级
# MAPPING_CONTRACT_VERSION，再重标、重训和重做 retention gate。
FROZEN_MAPPING_IMPLEMENTATION_SHA256_BY_VERSION: Dict[str, str] = {
    MAPPING_CONTRACT_VERSION: "9373dc0857df09609a7a06179afa7e933f918d276384211646cf152c07d690ce",
}

# 只纳入会改变 9 通道 normalized target 的参数；输出路径、UDP 端口、日志和
# 推理开关不属于标签语义，修改它们不应让模型失效。
_MAPPING_DEFAULTS: Dict[str, Any] = {
    "pinch_distance_norm_threshold": 0.45,
    "pinch_open_ratio_min": 0.75,
    "pinch_support_curl_max": 0.65,
    "open_ratio_threshold": 0.85,
    "open_mean_curl_max": 0.45,
    "fist_ratio_threshold": 0.85,
    "fist_mean_curl_min": 0.45,
    "fist_compact_ratio_threshold": 0.65,
    "control_grasp_open_ref": 0.02,
    "control_grasp_closed_ref": 0.55,
    "control_pinch_open_ref": 0.45,
    "control_pinch_closed_ref": 0.08,
    "control_hand_open_ratio_open_ref": 0.95,
    "control_hand_open_ratio_closed_ref": 0.25,
    "control_pinch_index_open_ref": 0.05,
    "control_pinch_index_closed_ref": 0.35,
    "control_open_release_enabled": False,
    "control_open_release_start_ratio": 0.85,
    "control_open_release_full_ratio": 0.95,
    "svh_enable_gesture_fallback": False,
    "svh_preview_layout": "svh_9ch",
    "svh_preview_channel_count": 9,
    "svh_position_open_value": 0.0,
    "svh_position_closed_value": 1.0,
    "svh_thumb_grasp_scale": 0.85,
    "svh_thumb_opposition_scale": 0.75,
    "svh_pinch_support_scale": 0.20,
    "svh_open_spread_scale": 0.25,
    "svh_grasp_spread_scale": 0.05,
    "svh_pinch_spread_scale": 0.10,
}


def mapping_contract_payload(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """返回可序列化的有效映射语义，不包含运行期无关配置。"""

    return {
        "version": MAPPING_CONTRACT_VERSION,
        "single_right_hand": True,
        "channel_order": list(SVH_9CH_NAMES),
        "parameters": {
            key: cfg.get(key, default)
            for key, default in sorted(_MAPPING_DEFAULTS.items())
        },
    }


def mapping_contract_sha256(cfg: Dict[str, Any]) -> str:
    encoded = json.dumps(
        mapping_contract_payload(cfg),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class _DocstringStripper(ast.NodeTransformer):
    """只去掉文档字符串；注释本来就不会进入 AST。"""

    @staticmethod
    def _without_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            return body[1:]
        return body

    def visit_Module(self, node: ast.Module) -> ast.AST:
        self.generic_visit(node)
        node.body = self._without_docstring(node.body)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        node.body = self._without_docstring(node.body)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        self.generic_visit(node)
        node.body = self._without_docstring(node.body)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        self.generic_visit(node)
        node.body = self._without_docstring(node.body)
        return node


def _assigned_names(node: ast.stmt) -> set[str]:
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        return set()
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return {target.id for target in targets if isinstance(target, ast.Name)}


def _canonical_algorithm_ast(path: Path, selected_names: tuple[str, ...] | None) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    if selected_names is not None:
        selected = set(selected_names)
        body: list[ast.stmt] = []
        found: set[str] = set()
        for node in tree.body:
            node_name = getattr(node, "name", None)
            assignment_names = _assigned_names(node)
            if node_name in selected or assignment_names & selected:
                body.append(node)
                if node_name in selected:
                    found.add(str(node_name))
                found.update(assignment_names & selected)
        missing = sorted(selected - found)
        if missing:
            raise RuntimeError(f"映射算法指纹找不到符号 {missing}：{path}")
        tree = ast.Module(body=body, type_ignores=[])
    normalized = _DocstringStripper().visit(tree)
    ast.fix_missing_locations(normalized)
    return ast.dump(normalized, annotate_fields=True, include_attributes=False)


def mapping_algorithm_sha256(*, project_root: Path | None = None) -> str:
    """计算标签/运行时映射公式的 AST 指纹，忽略注释、空白和文档字符串。"""

    root = (project_root or _PROJECT_ROOT).resolve()
    digest = hashlib.sha256()
    for relative_path, selected_names in _MAPPING_ALGORITHM_AST_TARGETS.items():
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"映射算法指纹缺少源码：{path}")
        canonical = _canonical_algorithm_ast(path, selected_names)
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(canonical.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def mapping_implementation_payload(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """补充旧 mapping SHA 未覆盖的公式和手势上下文语义。"""

    return {
        "version": MAPPING_CONTRACT_VERSION,
        "algorithm_ast_sha256": mapping_algorithm_sha256(),
        "h2o_label_gesture_context_policy": H2O_LABEL_GESTURE_CONTEXT_POLICY,
        "runtime_gesture_context_policy": RUNTIME_GESTURE_CONTEXT_POLICY,
        "runtime_gesture_parameters": {
            key: cfg.get(key, default)
            for key, default in sorted(_RUNTIME_CONTEXT_DEFAULTS.items())
        },
    }


def mapping_implementation_sha256(cfg: Dict[str, Any]) -> str:
    encoded = json.dumps(
        mapping_implementation_payload(cfg),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def expected_mapping_implementation_sha256(
    version: str = MAPPING_CONTRACT_VERSION,
) -> str:
    try:
        return FROZEN_MAPPING_IMPLEMENTATION_SHA256_BY_VERSION[version]
    except KeyError as exc:
        raise ValueError(f"未知 mapping implementation 版本：{version}") from exc


def assert_mapping_implementation_compatible(
    cfg: Dict[str, Any],
    artifact_contract: Dict[str, Any] | None = None,
) -> str:
    """拒绝同一版本下的算法/去抖参数漂移，并兼容缺少新字段的既有 v2 工件。"""

    frozen = expected_mapping_implementation_sha256()
    artifact_expected = (artifact_contract or {}).get("mapping_implementation_sha256")
    if artifact_expected is not None and artifact_expected != frozen:
        raise ValueError(
            "工件 mapping implementation SHA-256 不属于当前冻结版本："
            f"artifact={artifact_expected}, frozen={frozen}"
        )
    current = mapping_implementation_sha256(cfg)
    if current != frozen:
        raise ValueError(
            "mapping implementation SHA-256 不匹配；标签公式、实时映射或手势上下文已漂移，"
            "必须升级 mapping contract 版本并重新生成标签/训练/评估："
            f"frozen={frozen}, current={current}"
        )
    return current
