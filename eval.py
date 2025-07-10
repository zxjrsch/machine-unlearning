import glob
import os
from pathlib import Path

import torch
from torchinfo import summary

from model import MaskingGCN


class Eval:
    def __init__(self, gcn_checkpoint: Path) -> None:
        self.gcn = self.load_checkpoints(gcn_checkpoint)

    def load_checkpoints(self, path: Path) -> MaskingGCN:
        gcn = MaskingGCN()
        gcn.load_state_dict(torch.load(path))
        return gcn
    
    def inspect_model(self) -> None:
        summary(self.gcn)
    
if __name__ == '__main__':
    checkpoint_path = Path(sorted(glob.glob(os.path.join('checkpoints/gcn', '*.pt')))[0])
    eval = Eval(gcn_checkpoint=checkpoint_path)
    eval.inspect_model()