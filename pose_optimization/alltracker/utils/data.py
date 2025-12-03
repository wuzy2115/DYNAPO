import torch
import dataclasses
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, Optional, Dict


@dataclass(eq=False)
class VideoData:
    """
    Dataclass for storing video tracks data.
    """

    video: torch.Tensor  # B,S,C,H,W
    trajs: torch.Tensor  # B,S,N,2
    visibs: torch.Tensor  # B,S,N
    valids: Optional[torch.Tensor] = None  # B,S,N
    seq_name: Optional[str] = None
    dname: Optional[str] = None
    aug_video: Optional[torch.Tensor] = None


def collate_fn(batch):
    """
    Collate function for video tracks data.
    """
    video = torch.stack([b.video for b in batch], dim=0)
    trajs = torch.stack([b.trajs for b in batch], dim=0)
    visibs = torch.stack([b.visibs for b in batch], dim=0)
    seq_name = [b.seq_name for b in batch]
    dname = [b.dname for b in batch]

    return VideoData(
        video=video,
        trajs=trajs,
        visibs=visibs,
        seq_name=seq_name,
        dname=dname,
    )


def collate_fn_train(batch):
    """
    Collate function for video tracks data during training.
    """
    gotit = [gotit for _, gotit in batch]
    video = torch.stack([b.video for b, _ in batch], dim=0)
    trajs = torch.stack([b.trajs for b, _ in batch], dim=0)
    visibs = torch.stack([b.visibs for b, _ in batch], dim=0)
    valids = torch.stack([b.valids for b, _ in batch], dim=0)
    seq_name = [b.seq_name for b, _ in batch]
    dname = [b.dname for b, _ in batch]

    return (
        VideoData(
            video=video,
            trajs=trajs,
            visibs=visibs,
            valids=valids,
            seq_name=seq_name,
            dname=dname,
        ),
        gotit,
    )


def try_to_cuda(t: Any) -> Any:
    """
    Try to move the input variable `t` to a cuda device.

    Args:
        t: Input.

    Returns:
        t_cuda: `t` moved to a cuda device, if supported.
    """
    try:
        t = t.float().cuda()
    except AttributeError:
        pass
    return t


def dataclass_to_cuda_(obj):
    """
    Move all contents of a dataclass to cuda inplace if supported.

    Args:
        batch: Input dataclass.

    Returns:
        batch_cuda: `batch` moved to a cuda device, if supported.
    """
    for f in dataclasses.fields(obj):
        setattr(obj, f.name, try_to_cuda(getattr(obj, f.name)))
    return obj
