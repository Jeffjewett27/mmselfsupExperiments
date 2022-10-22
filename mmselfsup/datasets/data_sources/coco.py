import os
from PIL import Image
import glob

from ..builder import DATASOURCES

@DATASOURCES.register_module
class COCO(object):

    def __init__(self, root, list_file=None):
        self.fns = glob.glob(root + '/*g')

    def get_length(self):
        return len(self.fns)

    def get_sample(self, idx):
        img = Image.open(self.fns[idx])
        img = img.convert('RGB')
        return img
