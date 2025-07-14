# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional, Tuple

import torch
from flashinfer.comm.trtllm_alltoall import MnnvlMoe as MnnvlMoe
from flashinfer.comm.trtllm_alltoall import MoEAlltoallInfo as MoEAlltoallInfo

# from vllm.distributed.device_communicators.vllm_mnnvl_compat import (
#             vLLMToMPIShim)

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import (get_dp_group, get_tp_group)
from vllm.distributed.device_communicators.all2all import (
    FlashInferAllToAllManager)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input)

_alltoall_manager = None
# print('xx'*100)
# print(get_tp_group().cpu_group)
# _flashinfer_mnnvlmoe = FlashInferAllToAllManager(get_tp_group().cpu_group)

# from flashinfer.comm.mnnvl import MpiComm
# print(f"MpiComm was to: {MpiComm}")
# MpiComm = vLLMToMPIShim
# print(f"MpiComm Repaced to: {MpiComm}")

def get_local_sizes(local_tokens):
    cu_sizes = get_forward_context().dp_metadata.cu_tokens_across_dp_cpu
    sizes = [cu_sizes[0].item()]
    for i in range(1, len(cu_sizes)):
        sizes.append((cu_sizes[i] - cu_sizes[i - 1]).item())
    max_num_tokens = envs.VLLM_MOE_DP_CHUNK_SIZE
    sizes_chunked = [max_num_tokens] * len(sizes)
    if local_tokens < max_num_tokens:
        # When the number of local tokens is less than max_num_tokens, all other
        # ranks will also have fewer than max_num_tokens. The remaining tokens
        # are accounted for as residual.
        sizes_chunked = [x % max_num_tokens for x in sizes]

    return sizes_chunked


class FlashInferCutlassMoEPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):

    def __init__(
        self,
        quant_dtype: Optional[torch.dtype] = None,
        per_channel_quant: bool = False,
        block_shape: Optional[list[int]] = None,
        ep_rank: int = 0,
        ep_size: int = 1,
    ):
        super().__init__()
        self.per_channel_quant = per_channel_quant
        self.block_shape = block_shape
        self.quant_dtype = quant_dtype
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        # self.alltoall_info = None

    @property
    def alltoall_info(self) -> MoEAlltoallInfo:
        return self.alltoall_info

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> Optional[int]:
        return None

    def topk_indices_dtype(self) -> Optional[torch.dtype]:
        return None

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],  # Not used
        a2_scale: Optional[torch.Tensor],  # Not used
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        a1_gscale: torch.Tensor,
        use_dp: Optional[bool] = True,
        local_tokens: int = -1,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor], Optional[torch.Tensor]]:


        print('xx'*100)
        print(get_tp_group().cpu_group)
        print(get_tp_group().rank_in_group)
        print(get_tp_group().world_size)
        _flashinfer_mnnvlmoe = FlashInferAllToAllManager(get_tp_group().cpu_group)
        _flashinfer_mnnvlmoe.initialize(
            world_size=get_tp_group().world_size,
            rank=get_tp_group().rank_in_group,
        )
        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            # TODO: this only works for topK=1, will need to update for topK>1
            assert topk == 1, \
                "apply_router_weight_on_input is only implemented for topk=1"
            a1.mul_(topk_weights.to(a1.dtype))
        if not use_dp:
            a1q, a1q_scale = moe_kernel_quantize_input(
                a1,
                a1_gscale,
                quant_config.quant_dtype,
                self.per_channel_quant,
                self.block_shape,
                is_fp4_scalar_swizzled=True,
            )
        else:
            #TODO(shuw): make env var
            enable_flashinfer_fp4_allgather = True
            enable_flashinfer_alltoall = False

            if enable_flashinfer_alltoall:
                global_num_tokens_cpu = get_forward_context(
                ).dp_metadata.cu_tokens_across_dp_cpu[-1]
                top_k = topk_ids.size(1)
                x, topk_ids, topk_weights, alltoall_info = flashinfer_alltoall_dispatch(
                    # TODO(shuw): need to consider chunking for global_num_tokens_cpu
                    global_num_tokens_cpu,
                    a1,
                    topk_ids,
                    topk_weights,
                    top_k,
                    num_experts,
                    self.ep_rank,
                    self.ep_size,
                )
                self.alltoall_info = alltoall_info

            a1q, a1q_scale = moe_kernel_quantize_input(
                a1,
                a1_gscale,
                quant_config.quant_dtype,
                self.per_channel_quant,
                self.block_shape,
                is_fp4_scalar_swizzled=False  # delay swizzle to after comm
            )

            if enable_flashinfer_fp4_allgather:
                topk_weights, topk_ids, a1q, a1q_scale = \
                    get_dp_group().all_gatherv([topk_weights, topk_ids, a1q, a1q_scale],
                                            dim=0,
                                            sizes=get_local_sizes(local_tokens))

            if enable_flashinfer_alltoall:
                print("all2allcalling"*100)
                a1q = MnnvlMoe.mnnvl_moe_alltoallv(a1q, self.alltoall_info,
                                                   self.alltoall_workspace,
                                                   self.ep_rank, self.ep_size)
                a1q_scale = MnnvlMoe.mnnvl_moe_alltoallv(
                    a1q_scale, alltoall_info, self.alltoall_workspace,
                    self.ep_rank, self.ep_size)

            from flashinfer import fp4_swizzle_blockscale
            a1_m, a1_n = a1q.shape
            a1q_scale = fp4_swizzle_blockscale(a1q_scale, a1_m, a1_n * 2)

        return a1q, a1q_scale, None, topk_ids, topk_weights

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
        use_dp: bool = False,
        local_tokens: int = -1,
    ) -> None:
        if use_dp:
            # TODO(shuw): env var later
            enable_flashinfer_fp4_allgather = True
            enable_flashinfer_alltoall = False
            if enable_flashinfer_fp4_allgather:
                fused_expert_output = get_dp_group().reduce_scatterv(
                    fused_expert_output,
                    dim=0,
                    sizes=get_local_sizes(local_tokens),
                )

            if enable_flashinfer_alltoall:
                top_k = topk_ids.size(1)
                token_count = fused_expert_output.shape[0]
                _ = flashinfer_alltoall_combine(
                    fused_expert_output,
                    # TODO(shuw): need to consider chunking for global_num_tokens_cpu
                    self.alltoall_info,
                    ep_rank=self.ep_rank,
                    ep_size=self.ep_size,
                    top_k=top_k,
                    token_count=token_count,
                )
        output.copy_(fused_expert_output)


def flashinfer_alltoall_dispatch(
    global_num_tokens_cpu: list[int],
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    top_k: int,
    num_experts: int,
    ep_rank: int,
    ep_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, MoEAlltoallInfo]:
    # TODO(shuw): add later
    # assert (
    #     ensure_alltoall_workspace_initialized()
    # ), "FlashInfer AllToAll workspace not available"

    # gather router info
    # Assume same number of tokens across all devices if global_num_tokens_cpu is None
    max_num_token = max(global_num_tokens_cpu
                        ) if global_num_tokens_cpu is not None else x.shape[0]
    topk_ids = torch.nn.functional.pad(
        topk_ids, (0, 0, 0, max_num_token - topk_ids.shape[0]), "constant",
        num_experts)
    topk_weights = torch.nn.functional.pad(
        topk_weights, (0, 0, 0, max_num_token - topk_weights.shape[0]))
    gathered_topk_ids, gathered_topk_weights = (get_dp_group().all_gatherv(
        [topk_ids, topk_weights]))
    gathered_topk_ids = torch.flatten(gathered_topk_ids.contiguous(),
                                      start_dim=0,
                                      end_dim=-2)
    gathered_topk_weights = torch.flatten(gathered_topk_weights.contiguous(),
                                          start_dim=0,
                                          end_dim=-2)
    gathered_target_rank_ids = _flashinfer_mnnvlmoe.compute_target_rank_id(
        gathered_topk_ids, num_experts, ep_size)
    alltoall_info, topk_ids, topk_weights = (
        _flashinfer_mnnvlmoe.mnnvl_moe_alltoallv_prepare(
            gathered_target_rank_ids,
            None,
            gathered_topk_ids,
            gathered_topk_weights,
            max_num_token,
            num_experts,
            top_k,
            ep_rank,
            ep_size,
        ))

    x = _flashinfer_mnnvlmoe.mnnvl_moe_alltoallv(
        x, alltoall_info, _alltoall_manager.workspace_tensor, ep_rank, ep_size)

    return x, topk_ids, topk_weights, alltoall_info


def flashinfer_alltoall_combine(
    output: torch.Tensor,
    alltoall_info: MoEAlltoallInfo,
    top_k: int,
    ep_rank: int,
    ep_size: int,
    token_count: int,
):
    # TODO(shuw): add later
    # assert (
    #     ensure_alltoall_workspace_initialized()
    # ), "FlashInfer AllToAll workspace not available"
    return _flashinfer_mnnvlmoe.mnnvl_moe_alltoallv_combine(
        output,
        alltoall_info,
        _alltoall_manager.workspace_tensor,
        ep_rank=ep_rank,
        ep_size=ep_size,
        top_k=top_k,
        token_count=token_count,
    )
