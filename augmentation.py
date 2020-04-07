import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import sys
from PIL import Image, ImageEnhance, ImageOps
import numbers
import random

class SubPolicy(object):
    def __init__(
            self,
            p1,
            operation1,
            magnitude_idx1,
            p2,
            operation2,
            magnitude_idx2,
            fillcolor=(255, 255, 255),
#             fillcolor=(128, 128, 128),
    ):
        ranges = {
            "shearx": np.linspace(0, 0.3, 10),
            "sheary": np.linspace(0, 0.3, 10),
            "translatex": np.linspace(0, 0.3, 10),
            "translatey": np.linspace(0, 0.3, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10,
#             "cutout": np.round(np.linspace(0, 20, 10), 0).astype(np.int),
            "cutout": np.round(np.linspace(100, 300, 10), 0).astype(np.int),
        }
        

        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(
                rot, Image.new("RGBA", rot.size, (255,) * 4), rot
            ).convert(img.mode)

        func = {
            "shearx": lambda img, magnitude, random_dir: img.transform(
                img.size,
                Image.AFFINE,
                (1, magnitude * random_dir, 0, 0, 1, 0),
                Image.NEAREST,
                fillcolor=fillcolor,
            ),
            "sheary": lambda img, magnitude, random_dir: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, magnitude * random_dir, 1, 0),
                Image.NEAREST,
                fillcolor=fillcolor,
            ),
            "translatex": lambda img, magnitude, random_dir: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, magnitude * img.size[0] * random_dir, 0, 1, 0),
                fillcolor=fillcolor,
            ),
            "translatey": lambda img, magnitude, random_dir: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, 0, 1, magnitude * img.size[1] * random_dir),
                fillcolor=fillcolor,
            ),
            "rotate": lambda img, magnitude, random_dir: rotate_with_fill(img, magnitude * random_dir),
#             "rotate": lambda img, magnitude, random_dir: img.rotate(magnitude * random_dir),
            "color": lambda img, magnitude, random_dir: ImageEnhance.Color(img).enhance(
                1 + magnitude * random_dir
            ),
            "posterize": lambda img, magnitude, random_dir: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude, random_dir: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude, random_dir: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random_dir
            ),
            "sharpness": lambda img, magnitude, random_dir: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random_dir
            ),
            "brightness": lambda img, magnitude, random_dir: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random_dir
            ),
            "autocontrast": lambda img, magnitude, random_dir: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude, random_dir: ImageOps.equalize(img),
            "invert": lambda img, magnitude, random_dir: ImageOps.invert(img),
            "cutout": lambda img, magnitude, random_dir: Cutout(magnitude)(img),
        }
    
        self.p1 = float(p1)
        self._operation1_name = operation1
        self.operation1 = func[operation1.lower()]
        self.magnitude1 = ranges[operation1.lower()][int(magnitude_idx1)]
        self.p2 = float(p2)
        self._operation2_name = operation2
        self.operation2 = func[operation2.lower()]
        self.magnitude2 = ranges[operation2.lower()][int(magnitude_idx2)]

    def __call__(self, img, rand_p1=None, rand_p2=None, random_dir=None):
#         if random.random() < self.p1:
#             img = self.operation1(img, self.magnitude1)
#         if random.random() < self.p2:
#             img = self.operation2(img, self.magnitude2)
        if rand_p1 == None:
            rand_p1 = np.random.random()
        if rand_p2 == None:
            rand_p2 = np.random.random()
        if len(random_dir) != 2:
            random_dir = np.random.choice([-1, 1], size=2)
            
        if rand_p1 < self.p1:
            img = self.operation1(img, self.magnitude1, random_dir[0])
        if rand_p2 < self.p2:
            img = self.operation2(img, self.magnitude2, random_dir[1])
        return img

    def __repr__(self):
        return f"{self._operation1_name} with p:{self.p1} and magnitude:{self.magnitude1} \t" \
            f"{self._operation2_name} with p:{self.p2} and magnitude:{self.magnitude2} \n"


class RandAugment:
    """
    # randaugment is adaptived from UDA tensorflow implementation:
    # https://github.com/jizongFox/uda
    """

    @classmethod
    def get_trans_list(cls):
        trans_list = [
            'ShearX', 'TranslateX', 'TranslateY', 'ShearY', 'Rotate',
            ]
        trans_list = [
            'Invert', 'Sharpness', 'AutoContrast', 'Posterize',
            'ShearX', 'TranslateX', 'TranslateY', 'ShearY', 'Rotate',
            'Equalize', 'Contrast', 'Color', 'Solarize', 'Brightness']
#         trans_list = [
#             'Invert', 'Cutout', 'Sharpness', 'AutoContrast', 'Posterize',
#             'ShearX', 'TranslateX', 'TranslateY', 'ShearY', 'Rotate',
#             'Equalize', 'Contrast', 'Color', 'Solarize', 'Brightness']
        return trans_list

    @classmethod
    def get_rand_policies(cls):
        op_list = []
        for trans in cls.get_trans_list():
            for magnitude in range(1, 10):
                op_list += [(0.5, trans, magnitude)]
        policies = []
        for op_1 in op_list:
            for op_2 in op_list:
                policies += [[op_1, op_2]]
#         return policies
        return np.array(policies)

    def __init__(self) -> None:
        super().__init__()
        self._policies = self.get_rand_policies()

    def __call__(self, img):
        randomly_chosen_policy = self._policies[random.randint(0, len(self._policies) - 1)]
        policy = SubPolicy(*randomly_chosen_policy[0], *randomly_chosen_policy[1])
        print(policy)
        return policy(img)

    def __repr__(self):
        return "Random Augment Policy"
    
    def get_instance_policy(self):
        randomly_chosen_policy = self._policies[np.random.randint(0, len(self._policies) - 1)]
        policy = SubPolicy(*randomly_chosen_policy[0], *randomly_chosen_policy[1])
        return policy
    
    def get_batch_policy(self, batch_size):
        policy_batch = []
        randomly_chosen_policy_batch = self._policies[np.random.randint(0, len(self._policies) - 1, size=batch_size)]
        policy_batch = [SubPolicy(*policy[0], *policy[1]) for policy in randomly_chosen_policy_batch]
#         policy = SubPolicy(*randomly_chosen_policy[0], *randomly_chosen_policy[1])
        return policy_batch

class Cutout:

    def __init__(self, size=16) -> None:
        self.size = size

    def _create_cutout_mask(self, img_height, img_width, num_channels, size):
        """Creates a zero mask used for cutout of shape `img_height` x `img_width`.
        Args:
          img_height: Height of image cutout mask will be applied to.
          img_width: Width of image cutout mask will be applied to.
          num_channels: Number of channels in the image.
          size: Size of the zeros mask.
        Returns:
          A mask of shape `img_height` x `img_width` with all ones except for a
          square of zeros of shape `size` x `size`. This mask is meant to be
          elementwise multiplied with the original image. Additionally returns
          the `upper_coord` and `lower_coord` which specify where the cutout mask
          will be applied.
        """
        # assert img_height == img_width

        # Sample center where cutout mask will be applied
        height_loc = np.random.randint(low=0, high=img_height)
        width_loc = np.random.randint(low=0, high=img_width)

        size = int(size)
        # Determine upper right and lower left corners of patch
        upper_coord = (max(0, height_loc - size // 2), max(0, width_loc - size // 2))
        lower_coord = (
            min(img_height, height_loc + size // 2),
            min(img_width, width_loc + size // 2),
        )
        mask_height = lower_coord[0] - upper_coord[0]
        mask_width = lower_coord[1] - upper_coord[1]
        assert mask_height > 0
        assert mask_width > 0

        mask = np.ones((img_height, img_width, num_channels))
        zeros = np.zeros((mask_height, mask_width, num_channels))
        mask[upper_coord[0]: lower_coord[0], upper_coord[1]: lower_coord[1], :] = zeros
        return mask, upper_coord, lower_coord

    def __call__(self, pil_img):
        pil_img = pil_img.copy()
        img_height, img_width, num_channels = (*pil_img.size, 3)
        _, upper_coord, lower_coord = self._create_cutout_mask(
            img_height, img_width, num_channels, self.size
        )
        pixels = pil_img.load()  # create the pixel map
        for i in range(upper_coord[0], lower_coord[0]):  # for every col:
            for j in range(upper_coord[1], lower_coord[1]):  # For every row
                pixels[i, j] = (125, 122, 113, 0)  # set the colour accordingly
        return pil_img

class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, data):
        img, tar = data
        h, w = None, None
        th, tw = self.size
        
        if img.size != tar.size:
            print(img.size, tar.size)
            raise ValueError('Images must be same size')
        else:
            w, h = img.size
            
        if w == tw and h == th:
            return img, tar
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        
        crop_img = img.crop((x1, y1, x1 + tw, y1 + th)) 
        crop_tar = tar.crop((x1, y1, x1 + tw, y1 + th)) 
        
#         crop_img = img[..., y1:y1 + th, x1:x1 + tw]
#         crop_tar = tar[..., y1:y1 + th, x1:x1 + tw]
#             output.append(tensor[..., y1:y1 + th, x1:x1 + tw].contiguous())
        
        return crop_img, crop_tar

