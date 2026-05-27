from __future__ import annotations

import copy
import torch
import torch.nn as nn
from transformers import PreTrainedModel

from adaptive_pruning.utils.prunable_llm import PrunableLLM


class TorchRouter(nn.Module):
    """
    Translates an SKLearn LogisticRegression model to an explicit FP32 PyTorch 
    forward-prop module for unified execution. 
    """
    def __init__(self, clf, dtype=torch.float32, device="cpu"):
        super().__init__()
        weights = torch.from_numpy(clf.coef_)
        bias = torch.from_numpy(clf.intercept_)
        
        num_classes, hidden_dim = weights.shape
        self.linear = nn.Linear(hidden_dim, num_classes)
        self.linear.weight.data = weights.to(dtype=dtype, device=device)
        self.linear.bias.data = bias.to(dtype=dtype, device=device)
        
        self.register_buffer("classes_", torch.from_numpy(clf.classes_).to(device=device))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.linear(hidden_states)
        argmax_idx = logits.argmax(dim=-1)
        return self.classes_[argmax_idx]


class RouterState:
    """Holds routing decision for the current pass to be shared across layer wrappers."""
    def __init__(self, router: nn.Module):
        self.router = router
        self.active_expert = None
        self._cached_expert_id: int | None = None

    def evaluate(self, hidden_states: torch.Tensor, past_key_value) -> bool:
        # Check if this is a prefill phase (new sequence) to trigger routing.
        is_new_seq = past_key_value is None
        if not is_new_seq:
            if hasattr(past_key_value, 'get_seq_length'):
                is_new_seq = past_key_value.get_seq_length(0) == 0
            elif isinstance(past_key_value, tuple) and len(past_key_value) > 0 and len(past_key_value[0]) > 0:
                is_new_seq = past_key_value[0][0].shape[-2] == 0

        # Route if it's a completely new sequence OR if we lack an active expert.
        should_route = is_new_seq or self.active_expert is None
        if should_route:
            router_input = hidden_states[:, -1, :].to(torch.float32)
            self.active_expert = self.router(router_input)

            # The single sync that we cannot avoid: we need the Python int to
            # index the experts list. Do it ONCE here, not per-layer per-token.
            unique_experts = torch.unique(self.active_expert)
            if unique_experts.numel() > 1:
                raise NotImplementedError(
                    "Branched routing of multiple experts during a single batched operation "
                    "is not supported with HF DynamicCache. Please reduce batch size to 1 "
                    "or use homogeneous batches."
                )
            self._cached_expert_id = int(unique_experts[0].item())

        return should_route

    @property
    def expert_id(self) -> int:
        if self._cached_expert_id is None:
            raise RuntimeError("Router wasn't evaluated!")
        return self._cached_expert_id

    def reset(self) -> None:
        self.active_expert = None
        self._cached_expert_id = None


class RoutedLayerWrapper(nn.Module):
    """
    Wraps a single layer at a specific index. Selects the underlying expert module 
    dynamically based on the global RouterState.
    """
    def __init__(self, layer_idx: int, split_layer: int, expert_branches: nn.ModuleList, router_state: RouterState):
        super().__init__()
        self.layer_idx = layer_idx
        self.split_layer = split_layer
        self.expert_branches = expert_branches
        self.router_state = router_state
        self.branched_layer_idx = layer_idx - split_layer
        
        # Native Python list for blisteringly fast O(1) pointer lookups during decode!
        # nn.ModuleList.__getitem__ creates heavy string dict lookups under the hood.
        self._fast_branches = [branch[self.branched_layer_idx] for branch in expert_branches]

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        # Only evaluate router on the first branched layer!
        if self.layer_idx == self.split_layer:
            past_key_value = kwargs.get('past_key_value', None)
            self.router_state.evaluate(hidden_states, past_key_value)
            
        expert_id = self.router_state.expert_id
        
        # Avoid PyTorch nn.ModuleList overhead per-token
        layer = self._fast_branches[expert_id]
        
        return layer(hidden_states, *args, **kwargs)


class LayerRangePruner:
    """Applies and restores pruning for a specific layer range."""
    HEAD_DIM = 128

    def __init__(self, layers: list[nn.Module], start_layer: int, end_layer: int, config, device=None):
        self.layers = layers
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.device = torch.device(device) if device else next(layers[0].parameters()).device
        self._orig_num_heads = config.num_attention_heads
        self._orig_hidden = config.hidden_size
        self._orig_intermediate = config.intermediate_size
        self._deltas: dict | None = None
        self._pruned_unstr: bool = False
        self._active_expert: dict | None = None

    def apply(self, expert: dict, *, unstr: bool = False) -> None:
        if self._active_expert is not None:
            self.restore()

        self._pruned_unstr = unstr
        self._deltas = {}

        for idx in range(self.start_layer, self.end_layer):
            ld = expert["layers"][idx]
            layer = self.layers[idx]
            dev = layer.self_attn.q_proj.weight.device

            attn_mask = ld["attn_mask"].to(dev)
            mlp_mask = ld["mlp_mask"].to(dev)
            attn_mask_exp = attn_mask.repeat_interleave(self.HEAD_DIM)
            retained_attn = torch.where(attn_mask_exp)[0]
            removed_attn = torch.where(~attn_mask_exp)[0]
            retained_mlp = torch.where(mlp_mask)[0]
            removed_mlp = torch.where(~mlp_mask)[0]

            delta: dict = {
                "o_proj.bias": layer.self_attn.o_proj.bias.data.clone(),
                "down_proj.bias": layer.mlp.down_proj.bias.data.clone(),
                "q_removed": layer.self_attn.q_proj.weight.data[removed_attn],
                "k_removed": layer.self_attn.k_proj.weight.data[removed_attn],
                "v_removed": layer.self_attn.v_proj.weight.data[removed_attn],
                "up_removed": layer.mlp.up_proj.weight.data[removed_mlp],
                "gate_removed": layer.mlp.gate_proj.weight.data[removed_mlp],
            }
            if not unstr:
                delta["o_removed_cols"] = (
                    layer.self_attn.o_proj.weight.data[:, removed_attn]
                )
                delta["down_removed_cols"] = (
                    layer.mlp.down_proj.weight.data[:, removed_mlp]
                )
            self._deltas[idx] = delta

            if unstr:
                mask_attn = attn_mask_exp.unsqueeze(-1)
                layer.self_attn.q_proj.weight.data *= mask_attn
                layer.self_attn.k_proj.weight.data *= mask_attn
                layer.self_attn.v_proj.weight.data *= mask_attn
                mask_mlp = mlp_mask.unsqueeze(-1)
                layer.mlp.up_proj.weight.data *= mask_mlp
                layer.mlp.gate_proj.weight.data *= mask_mlp
            else:
                layer.self_attn.q_proj.weight.data = (
                    layer.self_attn.q_proj.weight.data[retained_attn])
                layer.self_attn.k_proj.weight.data = (
                    layer.self_attn.k_proj.weight.data[retained_attn])
                layer.self_attn.v_proj.weight.data = (
                    layer.self_attn.v_proj.weight.data[retained_attn])
                layer.self_attn.o_proj.weight.data = (
                    layer.self_attn.o_proj.weight.data[:, retained_attn])
                retain_heads = int(attn_mask.sum().item())
                n_attn = int(attn_mask_exp.sum().item())
                layer.self_attn.num_heads = retain_heads
                layer.self_attn.hidden_size = retain_heads * self.HEAD_DIM
                layer.self_attn.q_proj.out_features = n_attn
                layer.self_attn.k_proj.out_features = n_attn
                layer.self_attn.v_proj.out_features = n_attn
                layer.self_attn.o_proj.in_features = n_attn

                layer.mlp.up_proj.weight.data = (
                    layer.mlp.up_proj.weight.data[retained_mlp])
                layer.mlp.gate_proj.weight.data = (
                    layer.mlp.gate_proj.weight.data[retained_mlp])
                layer.mlp.down_proj.weight.data = (
                    layer.mlp.down_proj.weight.data[:, retained_mlp])
                n_mlp = int(mlp_mask.sum().item())
                layer.mlp.intermediate_size = n_mlp
                layer.mlp.up_proj.out_features = n_mlp
                layer.mlp.gate_proj.out_features = n_mlp
                layer.mlp.down_proj.in_features = n_mlp

            layer.self_attn.o_proj.bias.data = ld["attn_bias"].to(dev)
            layer.mlp.down_proj.bias.data = ld["mlp_bias"].to(dev)

        self._active_expert = expert

    def restore(self) -> None:
        if self._active_expert is None:
            return

        expert = self._active_expert
        unstr = self._pruned_unstr
        H = self._orig_hidden
        I = self._orig_intermediate

        for idx in range(self.start_layer, self.end_layer):
            delta = self._deltas[idx]
            ld = expert["layers"][idx]
            layer = self.layers[idx]
            dev = layer.self_attn.q_proj.weight.device

            attn_mask_exp = ld["attn_mask"].repeat_interleave(self.HEAD_DIM)
            retained_attn = torch.where(attn_mask_exp)[0].to(dev)
            removed_attn = torch.where(~attn_mask_exp)[0].to(dev)
            retained_mlp = torch.where(ld["mlp_mask"])[0].to(dev)
            removed_mlp = torch.where(~ld["mlp_mask"])[0].to(dev)

            if unstr:
                layer.self_attn.q_proj.weight.data[removed_attn] = delta["q_removed"]
                layer.self_attn.k_proj.weight.data[removed_attn] = delta["k_removed"]
                layer.self_attn.v_proj.weight.data[removed_attn] = delta["v_removed"]
                layer.mlp.up_proj.weight.data[removed_mlp] = delta["up_removed"]
                layer.mlp.gate_proj.weight.data[removed_mlp] = delta["gate_removed"]
            else:
                dtype = layer.self_attn.q_proj.weight.dtype
                in_qkv = layer.self_attn.q_proj.in_features
                in_mlp = layer.mlp.up_proj.in_features

                def _rebuild_rows(current, removed, out_dim, in_dim, ret, rem):
                    full = torch.empty(out_dim, in_dim, dtype=dtype, device=dev)
                    full[ret] = current
                    full[rem] = removed
                    return full

                def _rebuild_cols(current, removed, out_dim, orig_in, ret, rem):
                    full = torch.empty(out_dim, orig_in, dtype=dtype, device=dev)
                    full[:, ret] = current
                    full[:, rem] = removed
                    return full

                layer.self_attn.q_proj.weight.data = _rebuild_rows(
                    layer.self_attn.q_proj.weight.data, delta["q_removed"],
                    H, in_qkv, retained_attn, removed_attn)
                layer.self_attn.k_proj.weight.data = _rebuild_rows(
                    layer.self_attn.k_proj.weight.data, delta["k_removed"],
                    H, in_qkv, retained_attn, removed_attn)
                layer.self_attn.v_proj.weight.data = _rebuild_rows(
                    layer.self_attn.v_proj.weight.data, delta["v_removed"],
                    H, in_qkv, retained_attn, removed_attn)
                layer.self_attn.o_proj.weight.data = _rebuild_cols(
                    layer.self_attn.o_proj.weight.data, delta["o_removed_cols"],
                    H, H, retained_attn, removed_attn)
                layer.mlp.up_proj.weight.data = _rebuild_rows(
                    layer.mlp.up_proj.weight.data, delta["up_removed"],
                    I, in_mlp, retained_mlp, removed_mlp)
                layer.mlp.gate_proj.weight.data = _rebuild_rows(
                    layer.mlp.gate_proj.weight.data, delta["gate_removed"],
                    I, in_mlp, retained_mlp, removed_mlp)
                layer.mlp.down_proj.weight.data = _rebuild_cols(
                    layer.mlp.down_proj.weight.data, delta["down_removed_cols"],
                    H, I, retained_mlp, removed_mlp)

                layer.self_attn.num_heads = self._orig_num_heads
                layer.self_attn.hidden_size = H
                layer.self_attn.q_proj.out_features = H
                layer.self_attn.k_proj.out_features = H
                layer.self_attn.v_proj.out_features = H
                layer.self_attn.o_proj.in_features = H
                layer.mlp.intermediate_size = I
                layer.mlp.up_proj.out_features = I
                layer.mlp.gate_proj.out_features = I
                layer.mlp.down_proj.in_features = I

            layer.self_attn.o_proj.bias.data = delta["o_proj.bias"]
            layer.mlp.down_proj.bias.data = delta["down_proj.bias"]

        self._active_expert = None
        self._deltas = None

    def commit(self) -> None:
        self._active_expert = None
        self._deltas = None

    @property
    def active_expert(self) -> dict | None:
        return self._active_expert


class DynamicRoutedLayerWrapper(nn.Module):
    """
    Wraps a single layer at a specific index and dynamically applies masks
    to the downstream layers when a new expert is routed.
    """
    def __init__(
        self,
        layer_idx: int,
        split_layer: int,
        base_layer: nn.Module,
        router_state: RouterState,
        dynamic_pruner: LayerRangePruner,
        experts: list[dict],
        dynamic_unstr: bool,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.split_layer = split_layer
        self.base_layer = base_layer
        self.router_state = router_state
        self.dynamic_pruner = dynamic_pruner
        self.experts = experts
        self.dynamic_unstr = dynamic_unstr

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        if self.layer_idx == self.split_layer:
            past_key_value = kwargs.get('past_key_value', None)
            should_route = self.router_state.evaluate(hidden_states, past_key_value)
            if should_route:
                expert_id = self.router_state.expert_id
                self.dynamic_pruner.apply(self.experts[expert_id], unstr=self.dynamic_unstr)

        return self.base_layer(hidden_states, *args, **kwargs)


class RoutedLLM(nn.Module):
    """
    Variant 1: Static pruning.

    Configured, natively running HF model (LlamaForCausalLM), which 
    utilizes a routing to experts mechanism after a specified layer.
    """
    def __init__(
        self, 
        model: PreTrainedModel, 
        split_layer: int, 
        base_expert_path: str,
        expert_paths: list[str], 
        router_model_path: str,
        device: str | torch.device = "cuda:0"
    ):
        super().__init__()
        self.model = model
        self.device = torch.device(device)
        self.split_layer = split_layer + 1 # +1 because standard indexing means 0..split_layer-1 are base layers, we want 0..split_layer to be base layers
        
        print(f"[RoutedLLM] Loading Sklearn router...")
        clf = torch.load(router_model_path, map_location=self.device, weights_only=False)
        router = TorchRouter(clf, device=self.device, dtype=torch.float32)

        print("[RoutedLLM] Starting to allocate dedicated weights for experts.")
        temp_pruner = PrunableLLM(self.model, device=self.device)
        
        expert_branches = nn.ModuleList()
        # Important: The order of `expert_paths` MUST correspond to the numerical 
        # indexing in the Router classifier model. (e.g., 0: boolq, 1: hella..)
        for idx, expert_path in enumerate(expert_paths):
            print(f"[RoutedLLM] Precompiling Expert {idx}: {expert_path}")
            
            expert_data = torch.load(expert_path, map_location=self.device, weights_only=False)
            temp_pruner.prune(expert_data, unstr=False)
            
            # Deepcopy creates a hard copy of weights exclusively for layers >= split_layer,
            # disconnected from the original object. This prevents deltas conflicts.
            branch = nn.ModuleList([
                copy.deepcopy(self.model.model.layers[i]) for i in range(split_layer, len(self.model.model.layers))
            ])
            expert_branches.append(branch)
            temp_pruner.unprune()
            
        print(f"[RoutedLLM] Applying generalized Base model ({base_expert_path}) to initial layers [0..{split_layer-1}].")
        base_expert_data = torch.load(base_expert_path, map_location=self.device, weights_only=False)
        temp_pruner.prune(base_expert_data, unstr=False)
        
        # Module Construction (Attach it directly to the HF pipeline as individual layer wrappers)
        shared_layers = list(self.model.model.layers[:split_layer])
        
        router_state = RouterState(router)
        routed_layers = []
        for i in range(split_layer, len(self.model.model.layers)):
            routed_layers.append(RoutedLayerWrapper(
                layer_idx=i, 
                split_layer=split_layer, 
                expert_branches=expert_branches, 
                router_state=router_state
            ))
            
        # Overwrite native layers with our list. LlamaModel loop will run consecutively,
        # but layers after split_layer will dynamically choose the expert weights!
        self.model.model.layers = nn.ModuleList(shared_layers + routed_layers)
        print("[RoutedLLM] Integrated. Model ready with native HF Generate support.")
        
    def forward(self, *args, **kwargs):
        """Pushes operations to the main LlamaForCausalLM"""
        return self.model(*args, **kwargs)
        
    def generate(self, *args, **kwargs):
        """Pushes operations to the main LlamaForCausalLM"""
        return self.model.generate(*args, **kwargs)


class DynamicRoutedLLM(nn.Module):
    """
    Variant 2: Dynamic pruning.

    Layers [0..split_layer-1] are permanently pruned using the general expert.
    Layers [split_layer..L-1] stay dense, and expert masks are applied dynamically
    at the start of each new sequence (prefill) based on router output.
    """
    def __init__(
        self,
        model: PreTrainedModel,
        split_layer: int,
        base_expert_path: str,
        expert_paths: list[str],
        router_model_path: str,
        device: str | torch.device = "cuda:0",
        dynamic_unstr: bool = True,
    ):
        super().__init__()
        self.model = model
        self.device = torch.device(device)
        self.split_layer = split_layer + 1 # +1 because standard indexing means 0..split_layer-1 are base layers, we want 0..split_layer to be base layers
        self.dynamic_unstr = dynamic_unstr

        print("[DynamicRoutedLLM] Loading Sklearn router...")
        clf = torch.load(router_model_path, map_location=self.device, weights_only=False)
        router = TorchRouter(clf, device=self.device, dtype=torch.float32)

        # Capture original layers before wrapping.
        full_layers = list(self.model.model.layers)

        print("[DynamicRoutedLLM] Applying base expert to shared layers.")
        base_expert = torch.load(base_expert_path, map_location=self.device, weights_only=False)
        base_pruner = LayerRangePruner(
            layers=full_layers,
            start_layer=0,
            end_layer=split_layer,
            config=self.model.config,
            device=self.device,
        )
        base_pruner.apply(base_expert, unstr=False)
        base_pruner.commit()

        print("[DynamicRoutedLLM] Loading dynamic experts (masks + bias deltas).")
        experts = [torch.load(p, map_location=self.device, weights_only=False) for p in expert_paths]

        dynamic_pruner = LayerRangePruner(
            layers=full_layers,
            start_layer=split_layer,
            end_layer=len(full_layers),
            config=self.model.config,
            device=self.device,
        )

        shared_layers = full_layers[:split_layer]
        router_state = RouterState(router)
        routed_layers = []
        for i in range(split_layer, len(full_layers)):
            routed_layers.append(DynamicRoutedLayerWrapper(
                layer_idx=i,
                split_layer=split_layer,
                base_layer=full_layers[i],
                router_state=router_state,
                dynamic_pruner=dynamic_pruner,
                experts=experts,
                dynamic_unstr=dynamic_unstr,
            ))

        self.model.model.layers = nn.ModuleList(shared_layers + routed_layers)
        self._dynamic_pruner = dynamic_pruner
        print("[DynamicRoutedLLM] Integrated. Model ready with dynamic pruning.")

    def unprune_dynamic(self) -> None:
        """Restore dense weights for the dynamic (split_layer..L-1) range."""
        self._dynamic_pruner.restore()

    def forward(self, *args, **kwargs):
        """Pushes operations to the main LlamaForCausalLM"""
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """Pushes operations to the main LlamaForCausalLM"""
        auto_unprune = kwargs.pop("auto_unprune", True)
        try:
            return self.model.generate(*args, **kwargs)
        finally:
            if auto_unprune:
                self._dynamic_pruner.restore()
