import json
import torch
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import ToTensor


scene_idx = 0

with open('./output/CLEVR_scenes.json') as f:
    scenes = json.load(f)['scenes']

scene = scenes[scene_idx]
image = Image.open(Path('./output/images', scene['image_filename'])).convert('RGB')
ww, hh = image.size

bboxes = [(int(obj['x']*ww), int(obj['y']*hh), int(obj['x']*ww)+int(obj['width']*ww), int(obj['y']*hh)+int(obj['height']*hh)) for obj in scene['objects']]

bboxes = torch.tensor(bboxes)
image = ToTensor()(image) * 255

image = draw_bounding_boxes(image.type(torch.uint8), bboxes)

plt.imshow(image.permute(1,2,0))
plt.show()
