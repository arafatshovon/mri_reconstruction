from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as etree

from fastmri import fft2c, ifft2c

from src.utils.fft import apply_mask, to_tensor
from src.utils.mask import MaskFunc


def et_query(
    root: etree.Element,
    qlist: Sequence[str],
    namespace: str = "http://www.ismrm.org/ISMRMRD",
):
    s = "."
    prefix = "ismrmrd_namespace"
    ns = {prefix: namespace}
    for el in qlist:
        s = s + f"//{prefix}:{el}"
    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")
    return str(value.text)


class FastMRIRawDataSample(NamedTuple):
    fname: str
    slice_ind: int
    metadata: Dict[str, Any]


def Raw_sample_filter(raw_sample):
    return True


class VarNetSample(NamedTuple):
    masked_kspace: torch.Tensor
    mask: torch.Tensor
    target: torch.Tensor
    fname: str
    slice_num: int
    max_value: float
    crop_size: Tuple[int, int]


class DataTransform:
    """
    Data Transformer for training HUMUS-Net model.
    """

    def __init__(
        self,
        uniform_train_resolution: Union[List[int], Tuple[int]],
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
    ):
        self.mask_func = mask_func
        self.use_seed = use_seed
        self.uniform_train_resolution = uniform_train_resolution

    def _crop_if_needed(self, image):
        w_from = h_from = 0

        if self.uniform_train_resolution[0] < image.shape[-3]:
            w_from = (image.shape[-3] - self.uniform_train_resolution[0]) // 2
            w_to = w_from + self.uniform_train_resolution[0]
        else:
            w_to = image.shape[-3]

        if self.uniform_train_resolution[1] < image.shape[-2]:
            h_from = (image.shape[-2] - self.uniform_train_resolution[1]) // 2
            h_to = h_from + self.uniform_train_resolution[1]
        else:
            h_to = image.shape[-2]

        return image[..., w_from:w_to, h_from:h_to, :]

    def _pad_if_needed(self, image):
        pad_w = self.uniform_train_resolution[0] - image.shape[-3]
        pad_h = self.uniform_train_resolution[1] - image.shape[-2]

        if pad_w > 0:
            pad_w_left = pad_w // 2
            pad_w_right = pad_w - pad_w_left
        else:
            pad_w_left = pad_w_right = 0

        if pad_h > 0:
            pad_h_left = pad_h // 2
            pad_h_right = pad_h - pad_h_left
        else:
            pad_h_left = pad_h_right = 0

        return (
            torch.nn.functional.pad(
                image.permute(0, 3, 1, 2),
                (pad_h_left, pad_h_right, pad_w_left, pad_w_right),
                "reflect",
            )
            .permute(0, 2, 3, 1)
        )

    def _to_uniform_size(self, kspace):
        image = ifft2c(kspace)
        image = self._crop_if_needed(image)
        image = self._pad_if_needed(image)
        kspace = fft2c(image)
        return kspace

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, int, float, torch.Tensor]:
        is_testing = target is None

        kspace = kspace.astype(np.complex64)

        if len(kspace.shape) == 2:
            kspace = np.expand_dims(kspace, axis=0)
        elif len(kspace.shape) == 4:
            h, w = kspace.shape[-2:]
            kspace = np.reshape(kspace, (-1, h, w))
        assert len(kspace.shape) == 3

        if not is_testing:
            target = target.astype(np.float32)
            target = to_tensor(target)
            max_value = attrs["max"].astype(np.float32)
        else:
            target = torch.tensor(0)
            max_value = 0.0

        kspace = to_tensor(kspace)

        if not is_testing:
            kspace = self._to_uniform_size(kspace)
        else:
            if self.uniform_train_resolution[0] < kspace.shape[-3]:
                image = ifft2c(kspace)
                h_from = (image.shape[-3] - self.uniform_train_resolution[0]) // 2
                h_to = h_from + self.uniform_train_resolution[0]
                image = image[..., h_from:h_to, :, :]
                kspace = fft2c(image)

        seed = None if not self.use_seed else tuple(map(ord, fname))
        acq_start = attrs["padding_left"]
        acq_end = attrs["padding_right"]

        if not is_testing:
            crop_size = torch.tensor([target.shape[0], target.shape[1]])
        else:
            crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        if self.mask_func:
            masked_kspace, mask, _ = apply_mask(
                kspace, self.mask_func, seed=seed, padding=(acq_start, acq_end)
            )
        else:
            masked_kspace = kspace
            shape = np.array(kspace.shape)
            num_cols = shape[-2]
            shape[:-3] = 1
            mask_shape = [1] * len(shape)
            mask_shape[-2] = num_cols
            mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
            mask = mask.reshape(*mask_shape)
            mask[:, :, :acq_start] = 0
            mask[:, :, acq_end:] = 0

        return (
            masked_kspace,
            mask.byte(),
            target,
            fname,
            slice_num,
            max_value,
            crop_size,
        )


class Mri_Data(Dataset):
    def __init__(self, files, transform, challenge):
        self.file = files
        self.transform = transform
        self.challenge = challenge
        self.start = 0
        self.end = None

    def _retrive_metadata(self, hf):
        et_root = etree.fromstring(hf["ismrmrd_header"][()])

        enc = ["encoding", "encodedSpace", "matrixSize"]
        enc_size = (
            int(et_query(et_root, enc + ["x"])),
            int(et_query(et_root, enc + ["y"])),
            int(et_query(et_root, enc + ["z"])),
        )
        rec = ["encoding", "reconSpace", "matrixSize"]
        recon_size = (
            int(et_query(et_root, rec + ["x"])),
            int(et_query(et_root, rec + ["y"])),
            int(et_query(et_root, rec + ["z"])),
        )

        lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
        enc_limits_center = int(et_query(et_root, lims + ["center"]))
        enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

        padding_left = enc_size[1] // 2 - enc_limits_center
        padding_right = padding_left + enc_limits_max

        metadata = {
            "padding_left": padding_left,
            "padding_right": padding_right,
            "encoding_size": enc_size,
            "recon_size": recon_size,
        }
        return metadata

    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        masked_kspace, mask, target, slice_num, max_value, crop_size = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        with h5py.File(self.file[index], "r") as hf:
            kspace = hf["kspace"]
            if self.challenge == "multicoil":
                rec_esc = hf["reconstruction_rss"]
            else:
                rec_esc = hf["reconstruction_rss"]

            attr = dict(hf.attrs)
            attr.update(self._retrive_metadata(hf))
            f_name = self.file[index].split("/")[-1]

            for idx, element in enumerate(kspace):
                kspace_t, mask_t, target_t, fname, slc_num, val, crop = self.transform(
                    element, None, rec_esc[idx], attr, f_name, idx
                )
                masked_kspace.append(kspace_t)
                mask.append(mask_t)
                target.append(target_t)
                slice_num.append(slc_num)
                max_value.append(val)
                crop_size.append(crop)

        return (
            torch.stack(masked_kspace[self.start : self.end]),
            torch.stack(mask[self.start : self.end]),
            torch.stack(target[self.start : self.end]),
            torch.tensor(slice_num[self.start : self.end]),
            torch.tensor(max_value[self.start : self.end]),
            torch.stack(crop_size[self.start : self.end]),
        )
