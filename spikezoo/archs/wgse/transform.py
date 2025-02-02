import torch
import torch.nn.functional as F
from math import sin, cos, pi
import numbers
import random


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, y):
        for t in self.transforms:
            x, y = t(x, y)
        return x, y

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomCrop(object):
    """Crop the tensor at a random location.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(x, output_size):
        w, h = x.shape[2], x.shape[1]
        th, tw = output_size
        assert(th <= h)
        assert(tw <= w)
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        return i, j, th, tw

    def __call__(self, x, y):
        """
            x: [C x H x W] Tensor to be rotated.
        Returns:
            Tensor: Cropped tensor.
        """
        i, j, h, w = self.get_params(x, self.size)

        return x[:, i:i + h, j:j + w], y[:, i:i + h, j:j + w]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomRotationFlip(object):
    """Rotate the image by angle.
    """

    def __init__(self, degrees, p_hflip=0.5, p_vflip=0.5):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.p_hflip = p_hflip
        self.p_vflip = p_vflip

    @staticmethod
    def get_params(degrees, p_hflip, p_vflip):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])
        angle_rad = angle * pi / 180.0

        M_original_transformed = torch.FloatTensor([[cos(angle_rad), -sin(angle_rad), 0],
                                                    [sin(angle_rad), cos(angle_rad), 0],
                                                    [0, 0, 1]])

        if random.random() < p_hflip:
            M_original_transformed[:, 0] *= -1

        if random.random() < p_vflip:
            M_original_transformed[:, 1] *= -1

        M_transformed_original = torch.inverse(M_original_transformed)

        M_original_transformed = M_original_transformed[:2, :].unsqueeze(dim=0)  # 3 x 3 -> N x 2 x 3
        M_transformed_original = M_transformed_original[:2, :].unsqueeze(dim=0)

        return M_original_transformed, M_transformed_original

    def __call__(self, x, y):
        """
            x: [C x H x W] Tensor to be rotated.
        Returns:
            Tensor: Rotated tensor.
        """
        assert(len(x.shape) == 3)

        M_original_transformed, M_transformed_original = self.get_params(self.degrees, self.p_hflip, self.p_vflip)
        affine_gridx = F.affine_grid(M_original_transformed, x.unsqueeze(dim=0).shape, align_corners=False)
        transformedx = F.grid_sample(x.unsqueeze(dim=0), affine_gridx, align_corners=False)

        affine_gridy = F.affine_grid(M_original_transformed, y.unsqueeze(dim=0).shape, align_corners=False)
        transformedy = F.grid_sample(y.unsqueeze(dim=0), affine_gridy, align_corners=False)

        return transformedx.squeeze(dim=0), transformedy.squeeze(dim=0)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', p_flip={:.2f}'.format(self.p_hflip)
        format_string += ', p_vlip={:.2f}'.format(self.p_vflip)
        format_string += ')'
        return format_string