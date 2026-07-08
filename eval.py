from __future__ import annotations
import argparse
import json
import os
import torch
from transformers import AutoConfig
from deepspec.eval.dspark import Gemma4DSparkEvaluator, Qwen3DSparkEvaluator
from deepspec.eval.eagle3 import Gemma4Eagle3Evaluator, Qwen3Eagle3Evaluator
from deepspec.utils import CustomJSONEncoder, device_count

EVALUATORS = {
    "Qwen3DSparkModel": Qwen3DSparkEvaluator,
    "Gemma4DSparkModel": Gemma4DSparkEvaluator,
    "Qwen3Eagle3Model": Qwen3Eagle3Evaluator,
    "Gemma4Eagle3Model": Gemma4Eagle3Evaluator,
    "Eagle3DraftModel": Qwen3Eagle3Evaluator,
}

TASKS = [
    ("gsm8k", 500),
    ("math500", 500),
    ("aime25",30),
    ("humaneval", 164),
    ("mbpp", 256),
    ("livecodebench", 500),
    ("mt-bench", 80),
    ("alpaca", 500),
    ("arena-hard-v2", 500),
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_name_or_path", type=str, required=True)
    parser.add_argument("--draft_name_or_path",type=str,required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help=("Confidence-head early-stop threshold. Confidence calibration metrics are collected only when this is 0.0."),
    )
    parser.add_argument("--tensorboard-dir", type=str, default=None)
    parser.add_argument("--step", type=int, default=None,help=("step for tensorboard logging"),)
    parser.add_argument("--seed", type=int, default=980406)
    parser.add_argument(
        "--nproc",
        type=int,
        default=None,
        help=(
            "Number of local processes to spawn. Defaults to the number of visible "
            "accelerator devices. Useful on Ascend NPUs when torch.npu.device_count() "
            "reports all physical devices but only a subset is available to the job."
        ),
    )
    args = parser.parse_args()
    args.tasks = list(TASKS)
    return args


def main(local_rank: int, args):
    if local_rank == 0:
        print(json.dumps(args, indent=4, cls=CustomJSONEncoder), flush=True)
    draft_config = AutoConfig.from_pretrained(args.draft_name_or_path)
    evaluator_cls = EVALUATORS[draft_config.architectures[0]]
    evaluator = evaluator_cls(local_rank, args)
    evaluator.evaluate()
    evaluator.clean_up()

def _resolve_nproc(args_nproc: int | None) -> int:
    """Resolve the number of local eval processes.

    Priority:
      1. --nproc CLI argument.
      2. DEEPSPEC_LOCAL_WORLD_SIZE environment variable.
      3. Count of ASCEND_RT_VISIBLE_DEVICES (NPU only).
      4. device_count().
    """
    if args_nproc is not None:
        return args_nproc
    env_nproc = os.environ.get("DEEPSPEC_LOCAL_WORLD_SIZE")
    if env_nproc is not None:
        return int(env_nproc)
    visible_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if visible_devices is not None:
        return len(visible_devices.split(","))
    return device_count()


if __name__ == "__main__":
    args = parse_args()
    nprocs = _resolve_nproc(args.nproc)
    assert nprocs > 0, f"nproc must be positive, got {nprocs}"
    # init_dist() derives the local world size from device_count() by default.
    # When we restrict evaluation to a subset of devices, tell init_dist the
    # actual number of spawned processes so HCCL sees a consistent world size.
    os.environ["DEEPSPEC_LOCAL_WORLD_SIZE"] = str(nprocs)
    print(f"[eval] spawning {nprocs} local process(es)", flush=True)
    torch.multiprocessing.spawn(
        main,
        args=(args,),
        nprocs=nprocs,
    )
