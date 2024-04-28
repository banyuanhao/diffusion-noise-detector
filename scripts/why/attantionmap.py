import sys
sys.path.append('/home/banyh2000/odfn')
from scripts.models.diffuserpipeline import StableDiffusionPipeline
import torch
import random
import numpy as np
from typing import TypeVar
T = TypeVar('T')
from PIL import Image
from pathlib import Path
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
from scripts.utils.utils_odfn import variance_index_sorted, seeds_plus, seeds_plus_dict
from typing import List, Generic, TypeVar, Callable, Union, Any
import functools
import itertools
from diffusers.models.attention_processor import Attention
import torch.nn as nn

trace: Type[DiffusionHeatMapHooker] = DiffusionHeatMapHooker
__all__ = ['ObjectHooker', 'ModuleLocator', 'UNetCrossAttentionLocator']


ModuleType = TypeVar('ModuleType')


from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import Attention

def auto_device(obj: T = torch.device('cpu')) -> T:
    if isinstance(obj, torch.device):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        return obj.to('cuda')

def set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    gen = torch.Generator(device=auto_device())
    gen.manual_seed(seed)

    return gen

def replace(latent_source, latent_target, bounding_box_latent_source, bounding_box_latent_target):
    """_summary_

    Args:
        latent_source (_type_): _description_
        latent_target (_type_): _description_
        bounding_box_latent_source (_type_): _description_
        bounding_box_latent_target (_type_): _description_

    Returns:
        _type_: _description_
    """
    x_s, y_s, width_s, height_s = bounding_box_latent_source
    x_t, y_t, width_t, height_t = bounding_box_latent_target
    latent_target[:, :, y_t:y_t+height_t, x_t:x_t+width_t] = latent_source[:, :, y_s:y_s+height_s, x_s:x_s+width_s]
    return latent_target

class ObjectHooker(Generic[ModuleType]):
    def __init__(self, module: ModuleType):
        self.module: ModuleType = module
        self.hooked = False
        self.old_state = dict()

    def __enter__(self):
        self.hook()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unhook()

    def hook(self):
        if self.hooked:
            raise RuntimeError('Already hooked module')

        self.old_state = dict()
        self.hooked = True
        self._hook_impl()

        return self

    def unhook(self):
        if not self.hooked:
            raise RuntimeError('Module is not hooked')

        for k, v in self.old_state.items():
            if k.startswith('old_fn_'):
                setattr(self.module, k[7:], v)

        self.hooked = False
        self._unhook_impl()

        return self

    def monkey_patch(self, fn_name, fn, strict: bool = True):
        try:
            self.old_state[f'old_fn_{fn_name}'] = getattr(self.module, fn_name)
            setattr(self.module, fn_name, functools.partial(fn, self.module))
        except AttributeError:
            if strict:
                raise

    def monkey_super(self, fn_name, *args, **kwargs):
        return self.old_state[f'old_fn_{fn_name}'](*args, **kwargs)

    def _hook_impl(self):
        raise NotImplementedError

    def _unhook_impl(self):
        pass



class ModuleLocator(Generic[ModuleType]):
    def locate(self, model: nn.Module) -> List[ModuleType]:
        raise NotImplementedError
    
class UNetCrossAttentionLocator(ModuleLocator[Attention]):
    def __init__(self, restrict: bool = None, locate_middle_block: bool = False):
        self.restrict = restrict
        self.layer_names = []
        self.locate_middle_block = locate_middle_block

    def locate(self, model: UNet2DConditionModel) -> List[Attention]:
        """
        Locate all cross-attention modules in a UNet2DConditionModel.

        Args:
            model (`UNet2DConditionModel`): The model to locate the cross-attention modules in.

        Returns:
            `List[Attention]`: The list of cross-attention modules.
        """
        self.layer_names.clear()
        blocks_list = []
        up_names = ['up'] * len(model.up_blocks)
        down_names = ['down'] * len(model.down_blocks)

        for unet_block, name in itertools.chain(
                zip(model.up_blocks, up_names),
                zip(model.down_blocks, down_names),
                zip([model.mid_block], ['mid']) if self.locate_middle_block else [],
        ):
            if 'CrossAttn' in unet_block.__class__.__name__:
                blocks = []

                for spatial_transformer in unet_block.attentions:
                    for transformer_block in spatial_transformer.transformer_blocks:
                        blocks.append(transformer_block.attn2)

                blocks = [b for idx, b in enumerate(blocks) if self.restrict is None or idx in self.restrict]
                names = [f'{name}-attn-{i}' for i in range(len(blocks)) if self.restrict is None or i in self.restrict]
                blocks_list.extend(blocks)
                self.layer_names.extend(names)

        return blocks_list

class ObjectHooker(Generic[ModuleType]):
    def __init__(self, module: ModuleType):
        self.module: ModuleType = module
        self.hooked = False
        self.old_state = dict()

    def __enter__(self):
        self.hook()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unhook()

    def hook(self):
        if self.hooked:
            raise RuntimeError('Already hooked module')

        self.old_state = dict()
        self.hooked = True
        self._hook_impl()

        return self

    def unhook(self):
        if not self.hooked:
            raise RuntimeError('Module is not hooked')

        for k, v in self.old_state.items():
            if k.startswith('old_fn_'):
                setattr(self.module, k[7:], v)

        self.hooked = False
        self._unhook_impl()

        return self

    def monkey_patch(self, fn_name, fn, strict: bool = True):
        try:
            self.old_state[f'old_fn_{fn_name}'] = getattr(self.module, fn_name)
            setattr(self.module, fn_name, functools.partial(fn, self.module))
        except AttributeError:
            if strict:
                raise

    def monkey_super(self, fn_name, *args, **kwargs):
        return self.old_state[f'old_fn_{fn_name}'](*args, **kwargs)

    def _hook_impl(self):
        raise NotImplementedError

    def _unhook_impl(self):
        pass

class DiffusionHeatMapHooker(AggregateHooker):
    def __init__(
            self,
            pipeline: Union[StableDiffusionPipeline, StableDiffusionXLPipeline],
            low_memory: bool = False,
            load_heads: bool = False,
            save_heads: bool = False,
            data_dir: str = None
    ):
        self.all_heat_maps = RawHeatMapCollection()
        h = (pipeline.unet.config.sample_size * pipeline.vae_scale_factor)
        self.latent_hw = 4096 if h == 512 or h == 1024 else 9216  # 64x64 or 96x96 depending on if it's 2.0-v or 2.0
        locate_middle = load_heads or save_heads
        self.locator = UNetCrossAttentionLocator(restrict={0} if low_memory else None, locate_middle_block=locate_middle)
        self.last_prompt: str = ''
        self.last_image: Image = None
        self.time_idx = 0
        self._gen_idx = 0

        modules = [
            UNetCrossAttentionHooker(
                x,
                self,
                layer_idx=idx,
                latent_hw=self.latent_hw,
                load_heads=load_heads,
                save_heads=save_heads,
                data_dir=data_dir
            ) for idx, x in enumerate(self.locator.locate(pipeline.unet))
        ]

        modules.append(PipelineHooker(pipeline, self))

        if type(pipeline) == StableDiffusionXLPipeline:
            modules.append(ImageProcessorHooker(pipeline.image_processor, self))

        super().__init__(modules)
        self.pipe = pipeline

    def time_callback(self, *args, **kwargs):
        self.time_idx += 1

    @property
    def layer_names(self):
        return self.locator.layer_names

    def to_experiment(self, path, seed=None, id='.', subtype='.', **compute_kwargs):
        # type: (Union[Path, str], int, str, str, Dict[str, Any]) -> GenerationExperiment
        """Exports the last generation call to a serializable generation experiment."""

        return GenerationExperiment(
            self.last_image,
            self.compute_global_heat_map(**compute_kwargs).heat_maps,
            self.last_prompt,
            seed=seed,
            id=id,
            subtype=subtype,
            path=path,
            tokenizer=self.pipe.tokenizer,
        )

    def compute_global_heat_map(self, prompt=None, factors=None, head_idx=None, layer_idx=None, normalize=False):
        # type: (str, List[float], int, int, bool) -> GlobalHeatMap
        """
        Compute the global heat map for the given prompt, aggregating across time (inference steps) and space (different
        spatial transformer block heat maps).

        Args:
            prompt: The prompt to compute the heat map for. If none, uses the last prompt that was used for generation.
            factors: Restrict the application to heat maps with spatial factors in this set. If `None`, use all sizes.
            head_idx: Restrict the application to heat maps with this head index. If `None`, use all heads.
            layer_idx: Restrict the application to heat maps with this layer index. If `None`, use all layers.

        Returns:
            A heat map object for computing word-level heat maps.
        """
        heat_maps = self.all_heat_maps

        if prompt is None:
            prompt = self.last_prompt

        if factors is None:
            factors = {0, 1, 2, 4, 8, 16, 32, 64}
        else:
            factors = set(factors)

        all_merges = []
        x = int(np.sqrt(self.latent_hw))

        with auto_autocast(dtype=torch.float32):
            for (factor, layer, head), heat_map in heat_maps:
                if factor in factors and (head_idx is None or head_idx == head) and (layer_idx is None or layer_idx == layer):
                    heat_map = heat_map.unsqueeze(1)
                    # The clamping fixes undershoot.
                    all_merges.append(F.interpolate(heat_map, size=(x, x), mode='bicubic').clamp_(min=0))

            try:
                maps = torch.stack(all_merges, dim=0)
            except RuntimeError:
                if head_idx is not None or layer_idx is not None:
                    raise RuntimeError('No heat maps found for the given parameters.')
                else:
                    raise RuntimeError('No heat maps found. Did you forget to call `with trace(...)` during generation?')

            maps = maps.mean(0)[:, 0]
            maps = maps[:len(self.pipe.tokenizer.tokenize(prompt)) + 2]  # 1 for SOS and 1 for padding

            if normalize:
                maps = maps / (maps[1:-1].sum(0, keepdim=True) + 1e-6)  # drop out [SOS] and [PAD] for proper probabilities

        return GlobalHeatMap(self.pipe.tokenizer, prompt, maps)


class UNetCrossAttentionHooker(ObjectHooker[Attention]):
    def __init__(
            self,
            module: Attention,
            parent_trace: 'trace',
            context_size: int = 77,
            layer_idx: int = 0,
            latent_hw: int = 9216,
            load_heads: bool = False,
            save_heads: bool = False,
            data_dir: Union[str, Path] = None,
    ):
        super().__init__(module)
        self.heat_maps = parent_trace.all_heat_maps
        self.context_size = context_size
        self.layer_idx = layer_idx
        self.latent_hw = latent_hw

        self.load_heads = load_heads
        self.save_heads = save_heads
        self.trace = parent_trace

        if data_dir is not None:
            data_dir = Path(data_dir)
        else:
            data_dir = cache_dir() / 'heads'

        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def _unravel_attn(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        # x shape: (heads, height * width, tokens)
        """
        Unravels the attention, returning it as a collection of heat maps.

        Args:
            x (`torch.Tensor`): cross attention slice/map between the words and the tokens.
            value (`torch.Tensor`): the value tensor.

        Returns:
            `List[Tuple[int, torch.Tensor]]`: the list of heat maps across heads.
        """
        h = w = int(math.sqrt(x.size(1)))
        maps = []
        x = x.permute(2, 0, 1)

        with auto_autocast(dtype=torch.float32):
            for map_ in x:
                map_ = map_.view(map_.size(0), h, w)
                # For Instruct Pix2Pix, divide the map into three parts: text condition, image condition and unconditional,
                # and only keep the text condition part, which is first of the three parts(as per diffusers implementation).
                if map_.size(0) == 24:
                    map_ = map_[:((map_.size(0) // 3)+1)]  # Filter out unconditional and image condition
                else:
                    map_ = map_[map_.size(0) // 2:] #  # Filter out unconditional
                maps.append(map_)

        maps = torch.stack(maps, 0)  # shape: (tokens, heads, height, width)
        return maps.permute(1, 0, 2, 3).contiguous()  # shape: (heads, tokens, height, width)

    def _save_attn(self, attn_slice: torch.Tensor):
        torch.save(attn_slice, self.data_dir / f'{self.trace._gen_idx}.pt')

    def _load_attn(self) -> torch.Tensor:
        return torch.load(self.data_dir / f'{self.trace._gen_idx}.pt')

    def __call__(
            self,
            attn: Attention,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
    ):
        """Capture attentions and aggregate them."""
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross is not None:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # DAAM save heads
        if self.save_heads:
            self._save_attn(attention_probs)
        elif self.load_heads:
            attention_probs = self._load_attn()

        # compute shape factor
        factor = int(math.sqrt(self.latent_hw // attention_probs.shape[1]))
        self.trace._gen_idx += 1

        # skip if too large
        if attention_probs.shape[-1] == self.context_size and factor != 8:
            # shape: (batch_size, 64 // factor, 64 // factor, 77)
            maps = self._unravel_attn(attention_probs)

            for head_idx, heatmap in enumerate(maps):
                self.heat_maps.update(factor, self.layer_idx, head_idx, heatmap)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

    def _hook_impl(self):
        self.original_processor = self.module.processor
        self.module.set_processor(self)

    def _unhook_impl(self):
        self.module.set_processor(self.original_processor)

    @property
    def num_heat_maps(self):
        return len(next(iter(self.heat_maps.values())))
    
i = 9999
seed_source = seeds_plus[variance_index_sorted[i]]
seed_target = seeds_plus[variance_index_sorted[i]]

prompt_source = "A woman carries a stylish handbag on her left shoulder, with the handbag on the left"
prompt_target = "A woman carries a stylish handbag on her shoulder."


model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True).to(device)

locator = UNetCrossAttentionLocator(restrict=None, locate_middle_block=False)
cross_attention_layers = locator.locate(pipe.unet)
# print(cross_attention_layers)

count = 0
def record_attention(module, input, output):
    print("record_attention")
    
## register hooks in the cross-attention layers
for layer in cross_attention_layers:
    layer.register_forward_hook(record_attention)
    
## run the model
out = out_source, latents_source_final = pipe(prompt=prompt_source, generator=set_seed(seed_source), output_type = "latent and pil")
print(count)