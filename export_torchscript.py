"""
Export TorchScript
python export_torchscript.py \
    --variant dpt_hybrid \
    --precision float16 \
    --output 
"""

import argparse

import cv2
import torch
from torchvision.transforms import Compose

from dpt.models import DPTDepthModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet

torch.backends.cudnn.benchmark = True

class ExportTorchScript:
    def __init__(self):
        self.parse_args()
        self.init_model()
        
    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--variant', type=str, required=True, choices=['dpt_hybrid'])
        parser.add_argument('--model_weights', type=str, default=None) 
        parser.add_argument('--precision', type=str, default='float16')
        parser.add_argument('--output', type=str, required=True)
        self.args = parser.parse_args()
        
    def init_model(self):
        self.device = 'cuda'
        self.precision = {'float32': torch.float32, 'float16': torch.float16}[self.args.precision]

        default_models = {
            "dpt_hybrid": "weights/dpt_hybrid-midas-d889a10e.pt",
        }

        if self.args.model_weights is None:
            self.args.model_weights = default_models[self.args.variant]
            model_path = self.args.model_weights

        if self.args.variant == "dpt_hybrid": #DPT-Hybrid
            self.model = DPTDepthModel(
                path=model_path,
                backbone="vitb_rn50_384",
                non_negative=True,
                enable_attention_hooks=False,
            )
            self.net_w, self.net_h = 384, 384
            resize_mode="minimal"
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        else:
            print(f"model_type '{self.args.variant}' not implemented, use: --model_type large")
            assert False

        self.transform = Compose(
            [
                Resize(
                    self.net_w,
                    self.net_h,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method=resize_mode,
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                normalization,
                PrepareForNet(),
            ]
        )

        self.model = self.model.to(device=self.device, dtype=self.precision).eval()
        self.model = torch.jit.script(self.model)

        if self.args.precision == 'float16':
            output_filename = "MiDaSv3_" + self.args.variant + "_fp16_" + self.args.output
        else:
            output_filename = "MiDaSv3_" + self.args.variant + "_fp32_" + self.args.output

        print(output_filename)
        self.model.save(output_filename)

        print("####### [Start:] Test Print Model Code #######")
        test_model = torch.jit.load(output_filename)
        print(test_model.code)
        print("####### [End:] Test Print Model Code #######")

if __name__ == '__main__':
    ExportTorchScript()
