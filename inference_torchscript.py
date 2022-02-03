"""
Export TorchScript
python inference_torchscript.py \
    --variant dpt_hybrid \
    --precision float16 \
    --output 
"""

import argparse
import glob
import os

import cv2
import torch
from torchvision.transforms import Compose

import util.io
from dpt.transforms import NormalizeImage, PrepareForNet, Resize


def run(input_path, output_path, checkpoint):
    # Run MonoDepthNN to compute depth maps.
    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # set Tensor params
    net_w = net_h = 384
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    # load torchscript
    model = torch.jit.load(checkpoint).to(device)
    model = torch.jit.freeze(model)
    model = model.eval()

    # get input
    img_names = glob.glob(os.path.join(input_path, "*"))
    num_images = len(img_names)

    # create output folder
    os.makedirs(output_path, exist_ok=True)

    print("start processing")
    for ind, img_name in enumerate(img_names):
        if os.path.isdir(img_name):
            continue

        print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))
        # input

        img = util.io.read_image(img_name)

        img_input = transform({"image": img})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
            sample = sample.half()

            prediction = model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        # import matplotlib.pyplot as plt
        # plt.imshow(prediction)
        # plt.show()

        filename = os.path.join(
            output_path, os.path.splitext(os.path.basename(img_name))[0]
        )
        print(filename)
        util.io.write_depth(
            filename, prediction, bits=2, absolute_depth=False
        )

    print("finished")

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    run(args.input_path, args.output_path, args.checkpoint)

