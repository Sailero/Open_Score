"""Paths and process ownership for the current HAD experiment."""
from __future__ import annotations
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
import uuid

PROFILE = "main"

EXPECTED_ENV = {"had_core_version": "had-workbench-2.1.0",
                "physics_protocol": "rebuild-calibrated-v3-r7-target-initialization"}
SKY_EXPECTED_ENV = {"had_core_version": "skyhad-4.3.0", "physics_protocol": "skyhad-4.3"}
EXPERIMENT_ARMS = ("A", "B", "C")


def checked_experiment_arm(experiment_arm):
    if experiment_arm is not None and experiment_arm not in EXPERIMENT_ARMS:
        raise ValueError("experiment_arm must be A, B, C or None")
    return experiment_arm


def stable_environment_identity(value):
    """Compare scientific identity independently of the machine's install path."""
    if isinstance(value, dict):
        return {key: stable_environment_identity(item) for key, item in value.items() if key != "had_module"}
    if isinstance(value, (list, tuple)):
        return [stable_environment_identity(item) for item in value]
    return value


def scientific_config(config):
    """Freeze learning/task settings while allowing runtime placement to change."""
    runtime = {"output", "device", "use_cuda", "resume", "run", "report_interval",
               "progress_interval", "resume_interval", "progress", "live_curves",
               "launch_command", "launch_argv", "launch_platform", "working_directory", "python_executable", "concurrency", "skip_final_eval"}
    return json.loads(json.dumps(stable_environment_identity({key: value for key, value in config.items()
        if key not in runtime and not key.startswith("_")}), ensure_ascii=False, allow_nan=False))


def is_sky_environment(env_args=None):
    return (env_args or {}).get("env_agent_type") == "UAV_fixedwing"


def check_environment(env_args=None):
    """Require the physics selected explicitly by the retained task configuration."""
    expected = SKY_EXPECTED_ENV if is_sky_environment(env_args) else EXPECTED_ENV
    try:
        import had_env
        if is_sky_environment(env_args):
            from had_env.config import CORE_VERSION, PHYSICS_PROTOCOL
        else:
            from had_env.core.version import CORE_VERSION, PHYSICS_PROTOCOL
    except ImportError as error:
        raise RuntimeError(f"Install the selected HAD environment {expected}; help/status/plot remain available") from error
    actual = dict(had_core_version=CORE_VERSION, physics_protocol=PHYSICS_PROTOCOL)
    if actual != expected:
        raise ValueError(f"Environment mismatch: expected {expected}, found {actual}; no sampling started")
    return dict(actual, library_version=getattr(had_env, "__version__", CORE_VERSION),
                had_module=str(Path(had_env.__file__).resolve()))


def register_environment(output, config, env_info=None, parameter_snapshot=None):
    """Retain a fixed-wing experiment's real physics and reject mixed protocols."""
    root = Path(output)
    root = root.parent if root.name == "had" else root
    env_args = dict(config.get("env_args") or {})
    if not is_sky_environment(env_args):
        return read_json(root / "experiment.json", {})
    if root.name == PROFILE:
        raise ValueError("Fixed-wing SkyHAD requires its independent output; existing main data are preserved")
    identity = check_environment(env_args)
    env_info = dict(env_info or {})
    parameters = parameter_snapshot if parameter_snapshot is not None else config.get("environment_parameters")
    features = dict((parameters or {}).get("feature_spec") or {})
    for key in (
        "entity_shape", "n_actions", "feature_names", "spatial_dim", "had_position_indices",
        "had_velocity_indices", "had_position_scale", "had_velocity_scale", "rigid_features",
        "had_quaternion_indices", "had_angular_velocity_indices", "had_angular_velocity_scale",
        "actor_entity_shape", "observer_entity_shape"):
        value = env_info.get(key, config.get(key))
        if value is not None:
            features[key] = value
    metadata = dict(schema="had-current-experiment-v1", profile=root.name,
                    main_method=config["method"], main_method_display="LEAF" if config["method"] == "regir_nomem" else config["method"],
                    seeds=[int(config["seed"])], budget_steps=int(config["t_max"]),
                    official_checkpoint="final", env_identity=identity, env_args=env_args,
                    feature_spec=features, environment_parameters=parameters,
                    effective_config=(parameters or {}).get("effective_config"),
                    train_N=[4, 6, 8, 10], train_K=[1, 2, 3],
                    validation_configs=[[4, 4, 2], [6, 6, 2], [8, 8, 2], [10, 10, 3]],
                    validation_points=50, validation_episodes_per_config=25,
                    validation_scenes=[9500, 9599], final_episodes_per_config=300,
                    final_scenes=[9000, 9299])
    metadata = json.loads(json.dumps(metadata, ensure_ascii=False, allow_nan=False))
    experiment_arm = checked_experiment_arm(config.get("experiment_arm"))
    if experiment_arm is not None:
        if config.get("preset") != "sky_friendly_v1":
            raise ValueError("Experiment arms require preset sky_friendly_v1")
        common_args = {key: value for key, value in metadata["env_args"].items() if key != "birth_curriculum"}
        common_parameters = {key: value for key, value in (metadata["environment_parameters"] or {}).items()
                             if key != "birth_curriculum"}
        common = {key: metadata[key] for key in ("env_identity", "feature_spec", "train_N", "train_K",
            "validation_configs", "validation_points", "validation_episodes_per_config", "validation_scenes",
            "final_episodes_per_config", "final_scenes", "official_checkpoint")}
        common.update(env_args=common_args, environment_parameters=common_parameters)
        arm = dict(experiment_arm=experiment_arm, config=scientific_config(config),
                   env_args=metadata["env_args"], environment_parameters=metadata["environment_parameters"],
                   model_directory=str(run_directory(root, config["method"], config["seed"],
                       experiment_arm=experiment_arm).relative_to(root)).replace("\\", "/"))
        manifest = dict(schema="had-current-experiment-v2", profile=root.name,
            preset=config["preset"], main_method=config["method"], main_method_display=metadata["main_method_display"],
            seeds=metadata["seeds"], budget_steps=metadata["budget_steps"],
            env_identity=identity, feature_spec=features, common=common, arms={experiment_arm: arm})
        manifest = json.loads(json.dumps(manifest, ensure_ascii=False, allow_nan=False))
        path = root / "experiment.json"
        deadline = time.monotonic() + 15
        while True:
            with process_lock(root / ".experiment.metadata.lock") as owned:
                if owned:
                    retained = read_json(path)
                    if path.exists() and retained is None:
                        raise ValueError("Existing experiment metadata are unreadable; existing data are preserved")
                    if retained is not None:
                        for key in ("schema", "profile", "preset", "main_method", "seeds", "budget_steps", "common"):
                            if stable_environment_identity(retained.get(key)) != stable_environment_identity(manifest[key]):
                                raise ValueError(f"Existing experiment {key} differs; existing data are preserved")
                        previous_arm = retained.get("arms", {}).get(experiment_arm)
                        if previous_arm is not None and stable_environment_identity(previous_arm) != stable_environment_identity(arm):
                            raise ValueError(f"Existing experiment arm {experiment_arm} configuration differs; existing data are preserved")
                        if previous_arm is not None:
                            return retained
                        retained["arms"][experiment_arm] = arm
                        manifest = retained
                    elif any((environment_directory(root) / f"{stream}.csv").exists()
                             for stream in ("episodes", "learning", "progress", "trajectories")):
                        raise ValueError("Existing records lack armed experiment identity; existing data are preserved")
                    atomic_json(path, manifest)
                    return manifest
            if time.monotonic() >= deadline:
                raise RuntimeError("Experiment metadata are busy; no identity changes were written")
            time.sleep(.05)
    path = root / "experiment.json"
    retained = read_json(path)
    if path.exists() and retained is None:
        raise ValueError("Existing experiment metadata are unreadable; existing data are preserved")
    if retained is not None:
        for key in ("profile", "main_method", "seeds", "budget_steps", "env_identity", "env_args", "feature_spec", "environment_parameters"):
            if stable_environment_identity(retained.get(key)) != stable_environment_identity(metadata[key]):
                raise ValueError(f"Existing experiment {key} differs; existing data are preserved")
        return retained
    directory = environment_directory(root)
    if any((directory / f"{stream}.csv").exists() for stream in ("episodes", "learning", "progress", "trajectories")):
        raise ValueError("Existing records lack fixed-wing experiment identity; existing data are preserved")
    atomic_json(path, metadata)
    return metadata


def is_profile(output):
    root = Path(output)
    return root.name == "had" or (root / "had").is_dir()

def environment_directory(output, env="had"):
    if env != "had":
        raise ValueError("The active experiment supports HAD only")
    root = Path(output)
    return root if root.name == "had" else root / "had"

def ensure_layout(output):
    root = Path(output)
    directory = environment_directory(root)
    if root != directory and not directory.exists() and any((root / f"{stream}.csv").is_file() for stream in ("episodes", "learning", "progress", "trajectories")):
        raise RuntimeError("Legacy flat CSV output must be migrated explicitly; existing records are preserved")
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def run_directory(output, method, seed, env="had", run="train", *, experiment_arm=None):
    directory = environment_directory(output, env) / str(method)
    if checked_experiment_arm(experiment_arm) is not None:
        directory = directory / "arms" / experiment_arm
    return directory / "models" / f"seed_{int(seed)}"

def new_artifact_id():
    return uuid.uuid4().hex

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.pending")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default

@contextmanager
def process_lock(path):
    """One scheduler/evaluator owns this file until its process exits."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    locked = False
    try:
        if os.name == "nt":
            import msvcrt
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                locked = True
            except OSError:
                pass
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except OSError:
                pass
        yield locked
    finally:
        if locked:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()

def logical_devices(value):
    devices = tuple(int(v.strip()) for v in str(value).split(",") if v.strip())
    if not devices or len(set(devices)) != len(devices) or min(devices) < 0:
        raise ValueError("Devices must be distinct non-negative logical CUDA indices")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and max(devices) >= len(visible.split(",")):
        raise ValueError("Device index is outside inherited CUDA_VISIBLE_DEVICES")
    return devices

def physical_device(logical):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    return visible.split(",")[int(logical)].strip() if visible else str(logical)
