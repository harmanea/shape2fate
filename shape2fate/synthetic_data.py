import numpy as np
from scipy import ndimage
from torch.utils.data import IterableDataset

from .sim import illumination_pattern
from .otf import ModelOTF
from .reconstruction import cpu_reconstruct
from .parameters import AcquisitionParameters, ReconstructionParameters


class SyntheticDataset(IterableDataset):
    def __init__(self, contrast_fg_range: tuple[float,float] = (0.0, 1.0), contrast_bg_range: tuple[float,float] = (0.0, 1.0)):
        self.patch_size = 128
        
        self.contrast_fg_range = contrast_fg_range
        self.contrast_bg_range = contrast_bg_range        

        self.frequency = 0.17
        self.amplitude = 1.0

        self.ap = AcquisitionParameters(image_size=self.patch_size // 2, na=1.49, pixel_size=0.07, wavelength=512)
        self.rp = ReconstructionParameters(wiener_parameter=0.1)

        self.otf = ModelOTF(self.ap)
        self.otf_mult = self.otf.draw(self.patch_size)

        self.perlin = PerlinNoise(self.patch_size, 1)

    def _simulate_sim(self, image):
        angle0 = np.random.uniform(0, np.pi * 2)
        phase_offsets = np.random.uniform(0, np.pi * 2, 3)

        shifts = [(self.frequency * self.patch_size * np.sin(angle0 + i * np.pi / 3),
                   self.frequency * self.patch_size * np.cos(angle0 + i * np.pi / 3))
                  for i in range(3)]

        illumination = np.stack([illumination_pattern(angle0 + i // 3 * np.pi / 3,
                                                      self.frequency,
                                                      phase_offsets[i // 3] + (i % 3) * np.pi * 2 / 3,
                                                      self.amplitude,
                                                      self.patch_size)
                                 for i in range(9)])

        fg_c = np.random.uniform(*self.contrast_fg_range)
        bg_c = np.random.uniform(*self.contrast_bg_range)
        foreground = 250 + fg_c * 500
        background = 50 + bg_c * 50

        high_res_image = (image * foreground + background) * self.perlin()

        ix = np.fft.fft2(illumination * high_res_image)
        hix = self.otf_mult * ix
        dhix = hix.reshape(9, 2, self.patch_size // 2, 2, self.patch_size // 2).sum((1, 3)) / 4
        low_res_images = np.random.poisson(np.fft.ifft2(dhix).real).astype(np.float64)

        noisy_shifts = [np.random.triangular((y - 0.25, x - 0.25), (y, x), (y + 0.25, x + 0.25)) for y, x in shifts]
        noisy_phase_offsets = np.random.normal(phase_offsets, np.pi / 6)
        noisy_amplitudes = np.random.normal(self.amplitude, 0.1, 3)

        reconstruction = cpu_reconstruct(low_res_images, self.otf, noisy_shifts, noisy_phase_offsets, noisy_amplitudes, self.ap, self.rp)
        return (reconstruction - np.mean(reconstruction)) / np.std(reconstruction), illumination, low_res_images 


class SyntheticCCPDataset(SyntheticDataset):
    def __init__(self,min_n: int = 5, max_n: int = 15, radius: float = 2.5,
        contrast_fg_range: tuple[float,float] = (0.0, 1.0),
        contrast_bg_range: tuple[float,float] = (0.0, 1.0)):
        super().__init__(
            contrast_fg_range=contrast_fg_range,
            contrast_bg_range=contrast_bg_range
        )

        self.patch_size = 128

        # Possible positions
        yy, xx = np.mgrid[15:self.patch_size - 1:16, 15:self.patch_size - 1:16]
        self.yy = yy.flatten()
        self.xx = xx.flatten()

        if not (0 <= min_n < max_n <= 49):
            raise ValueError("Require 0 ? min_n < max_n ? 49")
        self.min_n, self.max_n = min_n, max_n
            
        self.max_offset = 8
        assert self.max_offset < 16

        # Beta params
        self.beta_a = 2
        self.beta_b = 1

        # CCP shape params
        self.radius = radius
        self.thickness = 1.0

        # Patch positions
        self.yyy, self.xxx = np.mgrid[:self.patch_size, :self.patch_size]

    def __iter__(self):
        while True:
            yield self.data_sample()

    def data_sample(self):
        # Generate positions and classes
        n = np.random.randint(self.min_n, self.max_n)
        indices = np.random.choice(len(self.yy), size=n, replace=False)
        offsets = np.random.uniform(-self.max_offset, self.max_offset, (n, 2))
        positions = np.column_stack([self.yy[indices], self.xx[indices]]) + offsets
        classes = np.random.beta(self.beta_a, self.beta_b, n) * 0.9 + 0.1

        # Generate a simulated HR image and output target
        target_distance = classes * self.radius
        distance = np.hypot(self.yyy[..., None] - positions[:, 0], self.xxx[..., None] - positions[:, 1])
        abs_distance = np.abs(distance - target_distance)
        parts = np.where(abs_distance > self.thickness, 0,
                         np.log(np.interp(abs_distance / self.thickness, [0, 1], [np.e, 1])))
        full_image = np.sum(parts, -1)

        distances = np.maximum(classes - distance / ((1 - classes) * 2 + self.radius + self.thickness * 2), 0)
        y = np.minimum(np.sum(distances, -1), 1)

        # Generate simulated SIM image
        x,_,_ = super()._simulate_sim(full_image)

        return x, y


class SyntheticAdipocytesCCPDataset(SyntheticCCPDataset):
    def __init__(self):
        super().__init__()

        self.radius = 3.5
        self.max_offset = 7
        self.beta_a = 1.5

    def data_sample(self):
        # Generate positions and classes
        n = np.random.randint(self.min_n, self.max_n)
        indices = np.random.choice(len(self.yy), size=n, replace=False)
        offsets = np.random.uniform(-self.max_offset, self.max_offset, (n, 2))
        positions = np.column_stack([self.yy[indices], self.xx[indices]]) + offsets
        classes = np.random.beta(self.beta_a, self.beta_b, n) * 0.9 + 0.1

        # Generate offset for each low-resolution frame
        speeds = np.random.exponential(1.5 - classes, n)
        angles = np.random.uniform(0, np.pi * 2, n)
        motion_vectors = np.column_stack([np.sin(angles), np.cos(angles)]) * speeds[:, None]
        trajectories = positions + np.linspace(-motion_vectors, motion_vectors, 9)

        # Generate a simulated HR image and output target
        target_distance = classes * self.radius
        distance = np.hypot(self.yyy[..., None, None] - trajectories[..., 0],
                            self.xxx[..., None, None] - trajectories[..., 1])
        abs_distance = np.abs(distance - target_distance)
        parts = np.where(abs_distance > self.thickness, 0,
                         np.log(np.interp(abs_distance / self.thickness, [0, 1], [np.e, 1])))
        full_images = np.sum(parts, -1)

        distances = np.maximum(classes - distance[:, :, 4] / ((1 - classes) * 2 + self.radius + self.thickness * 2), 0)
        y = np.minimum(np.sum(distances, -1), 1)

        # Generate simulated SIM image
        x,_,_ = super()._simulate_sim(full_images.transpose((2, 0, 1)))

        return x, y


class SyntheticExocytosisDataset(SyntheticDataset):
    def __init__(self):
        super().__init__()

        self.patch_size = 128

        # Possible positions
        yy, xx = np.mgrid[15:self.patch_size - 1:16, 15:self.patch_size - 1:16]
        self.yy = yy.flatten()
        self.xx = xx.flatten()

        self.min_vesicles_n, self.max_vesicles_n = 2, 8
        self.max_offset = 5
        assert self.max_offset < 16

        # Patch positions
        self.yyy, self.xxx = np.mgrid[:self.patch_size, :self.patch_size]

        self.primary_perlin = PerlinNoise(self.patch_size, 1)
        self.secondary_perlin = PerlinNoise(self.patch_size, 4)
        self.ternary_perlin = PerlinNoise(self.patch_size, 8)
        self.perlin = lambda: self.primary_perlin() + 0.75 * self.secondary_perlin() + 0.5 * self.ternary_perlin()

    def __iter__(self):
        while True:
            yield self.data_sample()

    def data_sample(self):
        # Generate positions and shapes
        n = np.random.randint(self.min_vesicles_n, self.max_vesicles_n) + 1
        indices = np.random.choice(len(self.yy), size=n, replace=False)
        offsets = np.random.uniform(-self.max_offset, self.max_offset, (n, 2))
        positions = np.column_stack([self.yy[indices], self.xx[indices]]) + offsets

        explosion_position = positions[0]
        vesicle_positions = positions[1:]

        angles = np.random.uniform(0, np.pi * 2, n - 1)
        lengths = np.random.uniform(1, 12, (n - 1, 1))
        midpoint_offsets = np.random.uniform(-5, 5, (n - 1, 1))

        starts = vesicle_positions + np.column_stack([np.sin(angles), np.cos(angles)]) * lengths
        midpoints = vesicle_positions + np.column_stack(
            [np.sin(angles + np.pi / 2), np.cos(angles + np.pi / 2)]) * midpoint_offsets
        ends = vesicle_positions - np.column_stack([np.sin(angles), np.cos(angles)]) * lengths

        speeds = np.random.rayleigh(0.123, n - 1)
        point_shifts = np.random.normal(0, speeds[:, None, None, None], (n - 1, 9, 3, 2))

        # Generate a simulated HR image and output target
        images = np.zeros((n - 1, 9, self.patch_size, self.patch_size))

        for i, a, b, c, s in zip(range(n - 1), starts, midpoints, ends, point_shifts):
            intensity = np.random.uniform(0.75, 1.25)
            t = np.arange(0, 1, 1 / np.hypot(*(a - c)))
            for j, (a_shift, b_shift, c_shift) in enumerate(s):
                P0 = (a + a_shift)[:, None] * t + (1 - t) * (b + b_shift)[:, None]
                P1 = (b + b_shift)[:, None] * t + (1 - t) * (c + c_shift)[:, None]

                draw_points = P0 * t + (1 - t) * P1

                self._draw_points(images[i, j], draw_points)
                images[i, j] *= intensity / np.max(images[i, j])

        dilated_binary_images = np.zeros((2, n - 1, self.patch_size, self.patch_size))
        for i in range(n - 1):
            dilated_binary_images[0, i] = ndimage.binary_dilation(images[i, 0] > 0)
            dilated_binary_images[1, i] = ndimage.binary_dilation(dilated_binary_images[0, i])

        overlaps = np.sum(dilated_binary_images[1], 0) > 1
        binary_image = np.where(overlaps, 0, np.sum(dilated_binary_images[0], 0))

        images = np.sum(images, 0)

        edt = ndimage.distance_transform_edt(binary_image)
        y0 = np.where(binary_image, 0.1 + edt / 3.0, 0)

        draw_filopodium = np.random.choice([True, False], p=[0.2, 0.8])
        if draw_filopodium:
            thickness = np.random.choice([1, 2, 3, 4, 5])
            intensity = np.random.uniform(0.05, 0.25)

            position = np.random.choice(4)
            match position:
                case 0:  # top
                    start_position = np.array((0, np.random.random() * self.patch_size))
                case 1:  # bottom
                    start_position = np.array((self.patch_size - 1, np.random.random() * self.patch_size))
                case 2:  # left
                    start_position = np.array((np.random.random() * self.patch_size, 0))
                case 3:  # right
                    start_position = np.array((np.random.random() * self.patch_size, self.patch_size - 1))

            end_position = np.random.random(2) * self.patch_size

            t = np.arange(0, 1, 1 / np.hypot(*(start_position - end_position)))
            draw_points = start_position[:, None] * t + (1 - t) * end_position[:, None]

            temp = np.zeros((self.patch_size, self.patch_size))
            self._draw_points(temp, draw_points)
            temp = ndimage.grey_dilation(temp, (thickness, thickness))
            images += intensity * temp / np.max(temp)

        draw_explosion = np.random.choice([True, False], p=[0.3, 0.7])
        if draw_explosion:
            time_offset = np.random.choice([-3, -2, -1, 0, 1, 2, 3], p=[0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1])
            intensity = np.random.uniform(0.75, 1.25)

            radii = []
            for i in range(9):
                if 0 <= i + time_offset < 9:
                    explosion_offsets = np.random.normal(0, (i + time_offset) * 0.75 + 1, (2, 200))
                    draw_points = explosion_position[:, None] + explosion_offsets

                    temp = np.zeros((self.patch_size, self.patch_size))
                    self._draw_points(temp, draw_points)
                    images[i] += intensity * temp / np.max(temp)

                    radii.append(np.hypot(*explosion_offsets))

            d = np.percentile(radii, 90)
            distance = np.hypot(self.yyy - explosion_position[0], self.xxx - explosion_position[1])
            y1 = np.where(distance > d, 0, (1 - distance / d) * 0.9 + 0.1)

        else:
            y1 = np.zeros((self.patch_size, self.patch_size))

        x,_,_ = self._simulate_sim(images)
        y = np.stack([y0, y1])

        return x, y

    def _draw_points(self, img, pos):
        y, x = np.clip(pos, 0, self.patch_size - 1)

        x1 = np.floor(x).astype(int)
        x2 = np.ceil(x).astype(int)

        y1 = np.floor(y).astype(int)
        y2 = np.ceil(y).astype(int)

        w11 = (x2 - x) * (y2 - y)
        w12 = (x2 - x) * (y - y1)
        w21 = (x - x1) * (y2 - y)
        w22 = (x - x1) * (y - y1)

        w11 = np.where((w11 == 0) & (w12 == 0) & (w21 == 0) & (w22 == 0), 1, w11)

        img[y1, x1] += np.sqrt(w11)
        img[y1, x2] += np.sqrt(w12)
        img[y2, x1] += np.sqrt(w21)
        img[y2, x2] += np.sqrt(w22)

        xx = np.round(x).astype(int)
        yy = np.round(y).astype(int)

        img[yy, xx] += 0.5


class PerlinNoise:
    """Adapted from: https://github.com/pvigier/perlin-numpy"""

    _SQRT_2_INV = 2 ** -0.5

    def __init__(self, size: int, resolution: int):
        """
        :param size: length of both dimensions of the generated noise image
        :param resolution: number of noise periods to generate along each axis
        """

        meshgrid = np.mgrid[0:resolution:resolution / size, 0:resolution:resolution / size]
        self.grid = np.stack(meshgrid) % 1

        self.t = self._fade(self.grid)
        self.d = size // resolution
        self.sample_size = (resolution + 1, resolution + 1)

    def __call__(self) -> np.ndarray:
        """
        Returned values are always in the [0, 1] range with ~0.5 mean
        """
        angles = 2 * np.pi * np.random.random_sample(self.sample_size)
        gradients = np.dstack((np.cos(angles), np.sin(angles))).repeat(self.d, 0).repeat(self.d, 1)

        n00 = (np.dstack((self.grid[0], self.grid[1])) * gradients[:-self.d, :-self.d]).sum(2)
        n10 = (np.dstack((self.grid[0] - 1, self.grid[1])) * gradients[self.d:, :-self.d]).sum(2)
        n01 = (np.dstack((self.grid[0], self.grid[1] - 1)) * gradients[:-self.d, self.d:]).sum(2)
        n11 = (np.dstack((self.grid[0] - 1, self.grid[1] - 1)) * gradients[self.d:, self.d:]).sum(2)

        n0 = n00 * (1 - self.t[0]) + n10 * self.t[0]
        n1 = n01 * (1 - self.t[0]) + n11 * self.t[0]

        return 0.5 + self._SQRT_2_INV * (n0 * (1 - self.t[1]) + n1 * self.t[1])

    @staticmethod
    def _fade(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3
