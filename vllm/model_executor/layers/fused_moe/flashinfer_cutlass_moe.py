# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional, Dict

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_prepare_finalize import (
    FlashInferCutlassMoEPrepareAndFinalize)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

from vllm.utils import round_up

logger = init_logger(__name__)

from typing import TYPE_CHECKING

try:
    from flashinfer import fp4_quantize as fp4_quantize
    from flashinfer.fused_moe import cutlass_fused_moe as cutlass_fused_moe
except ImportError:
    if not TYPE_CHECKING:
        cutlass_fused_moe = None

has_flashinfer_cutlass_fused_moe = cutlass_fused_moe is not None


def _valid_flashinfer_fused_moe(hidden_states: torch.Tensor, w1: torch.Tensor,
                                w2: torch.Tensor) -> bool:
    """
    Check if the given problem size is supported by the DeepGemm grouped
    gemm kernel.  All of M, N, K and the quantization block_shape must be
    aligned by `dg.get_m_alignment_for_contiguous_layout()`.
    """
    if not has_flashinfer_cutlass_fused_moe:
        logger.debug(
            "FlashInferExperts disabled: flashinfer_cutlass_fused_moe not available."
        )
        return False
    # Data type checks
    if (w1.dtype != torch.uint8 or w2.dtype != torch.uint8
            or hidden_states.dtype
            not in [torch.float32, torch.float16, torch.bfloat16]):
        logger.debug(
            f"FlashInferExperts disabled: w1/w2 must be torch.uint8 (got w1={w1.dtype}, w2={w2.dtype}), "
            f"hidden_states must be float32, float16, or bfloat16 (got {hidden_states.dtype})."
        )
        return False
    return True

#TODO(shuw): refactor out
class FlashInferExperts(mk.FusedMoEPermuteExpertsUnpermute):

    def __init__(self,
        use_nvfp4_w4a4: bool = False,
        use_fp8_w8a8: bool = False,
        use_dp: bool=False,
        ep_rank: int=0,
        ep_size: int=1,
        tp_rank: int=0,
        tp_size: int=1,
    ):
        super().__init__(
            FusedMoEQuantConfig(
                quant_dtype=torch.uint8,
                per_act_token_quant=False,
                block_shape=None,
            ))
        # super().__init__()
        import pdb
        # pdb.set_trace()
        self.use_nvfp4_w4a4 = use_nvfp4_w4a4
        self.use_fp8_w8a8 = use_fp8_w8a8
        self.ep_rank=ep_rank
        self.ep_size=ep_size
        self.tp_rank=tp_rank
        self.tp_size=tp_size
        self.use_dp=use_dp

    @property
    def activation_formats(
        self
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        return (mk.FusedMoEActivationFormat.Standard,
                mk.FusedMoEActivationFormat.Standard)

    def supports_expert_map(self) -> bool:
        return False
        
    def supports_chunking(self) -> bool:
        #TODO(shuw): support chunking later
        return False

    def workspace_shapes(
        self, a: torch.Tensor, aq: torch.Tensor, M: int, N: int, K: int,
        topk: int, global_num_experts: int, local_num_experts: int
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], torch.dtype]:
        # We use global_num_experts due to how moe_align_block_size handles
        # expert_maps.
        """
        Compute the shapes for the temporary and final outputs of the two gemms
        and activation in the fused expert function.  Since the gemms are
        independent, the workspace for the first gemm can be shared with the
        workspace for the last gemm.

        Returns a tuple of:
        - workspace13 shape tuple: must be large enough to hold the
          result of either expert gemm.
        - workspace2 shape tuple: must be large enough to hold the
          result of the activation function.
        - output shape tuple: must be exact size of the final gemm output.
        - Workspace type: The dtype to use for the workspace tensors.
        - Note: in order for activation chunking to work, the first dimension
          of each tuple must be the number of tokens.
        """        
        # num_experts = global_num_experts
        # block_m = self.block_shape[0]
        # M_sum = (M * topk) + num_experts * (block_m - 1)
        # M_sum = round_up(M_sum, block_m)
        # workspace1 = ()
        workspace2 = ()
        output_shape = a.shape
        workspace_dtype = a.dtype
        workspace1 = output_shape

        return (workspace1, workspace2, output_shape, workspace_dtype)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: Optional[torch.Tensor],
        w1_scale: Optional[torch.Tensor],
        w2_scale: Optional[torch.Tensor],
        w1_zp: Optional[torch.Tensor],
        w2_zp: Optional[torch.Tensor],
        a1q_scale: Optional[torch.Tensor], # ALLERT here, a1q_scale is a1_scale
        a2_scale: Optional[torch.Tensor],
        workspace13:Optional[torch.Tensor],
        workspace2:Optional[torch.Tensor],
        expert_num_tokens: Optional[torch.Tensor],
        # extra_expert_args: Optional[dict]=None,
        topk_weights: torch.Tensor,
        g1_alphas: torch.Tensor,
        g2_alphas: torch.Tensor,
        # input_sf: torch.Tensor,
        a1_scale: torch.Tensor,
        out_dtype: torch.dtype,
    ):
        # Flashinfer CUTLASS kernel takes scalar global scales,
        # min because inv_scale.
        if self.use_nvfp4_w4a4:
            # assert 'g1_alphas' in extra_expert_args
            # assert 'g2_alphas' in extra_expert_args
            # assert 'out_dtype' in extra_expert_args

            quant_scales = [
                torch.min(a1_scale),
                w1_scale.view(torch.int32),
                g1_alphas,
                torch.min(a2_scale),
                w2_scale.view(torch.int32),
                g2_alphas,
            ]
            # TODO(shuw): later make output into flashfiner api
            import pdb
            # print(f"self.ep_size:{self.ep_size}")
            # print(f"self.ep_rank:{self.ep_rank}")

            # print(f"self.tp_size:{self.tp_size}")
            # print(f"self.tp_rank:{self.tp_rank}")
            # print(f"hidden_states: dtype={hidden_states.dtype}, shape={hidden_states.shape}")
            # print(f"topk_ids: dtype={topk_ids.dtype}, shape={topk_ids.shape}")
            # print(f"topk_weights: dtype={topk_weights.dtype}, shape={topk_weights.shape}")
            # print(f"w1: dtype={w1.dtype}, shape={w1.shape}")
            # print(f"w2: dtype={w2.dtype}, shape={w2.shape}")
            # print(f"out_dtype: {out_dtype}")
            # if isinstance(quant_scales, (list, tuple)):
            #     for i, qs in enumerate(quant_scales):
            #         if hasattr(qs, 'dtype') and hasattr(qs, 'shape'):
            #             print(f"quant_scales[{i}]: dtype={qs.dtype}, shape={qs.shape}")
            #         else:
            #             print(f"quant_scales[{i}]: {qs}")
            # else:
            #     print(f"quant_scales: {quant_scales}")
            # if hasattr(a1q_scale, 'dtype') and hasattr(a1q_scale, 'shape'):
            #     print(f"a1q_scale: dtype={a1q_scale.dtype}, shape={a1q_scale.shape}")
            # else:
            #     print(f"a1q_scale: {a1q_scale}")
            # print(f"ep_size: {self.ep_size}, ep_rank: {self.ep_rank}, tp_size: {self.tp_size}, tp_rank: {self.tp_rank}")
            output = cutlass_fused_moe(
                hidden_states,
                topk_ids.to(torch.int),
                topk_weights,
                # FlashInfer API requires weight to be long for nvfp4
                w1.view(torch.long),
                w2.view(torch.long),
                output_dtype=out_dtype,
                quant_scales=quant_scales,
                input_sf=a1q_scale,
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
                tp_size=self.tp_size,
                tp_rank=self.tp_rank,
            )[0]
            # print("eee"*100)
            # return output
        else:
            raise ValueError("Only nvfp4 quantization is currently supported.")



# class FlashInferCutlassKernels(mk.FusedMoEModularKernel):

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         w1: torch.Tensor,
#         w2: torch.Tensor,
#         topk_ids: torch.Tensor,
#         topk_weights: torch.Tensor,
#         inplace: bool = False,
#         activation: str = "silu",
#         global_num_experts: int = -1,
#         expert_map: Optional[torch.Tensor] = None,
#         w1_scale: Optional[torch.Tensor] = None,
#         w2_scale: Optional[torch.Tensor] = None,
#         w1_zp: Optional[torch.Tensor] = None,
#         w2_zp: Optional[torch.Tensor] = None,
#         a1_scale: Optional[torch.Tensor] = None,
#         a2_scale: Optional[torch.Tensor] = None,
#         expert_num_tokens: Optional[torch.Tensor] = None,
#         g1_alphas: torch.Tensor = None,
#         g2_alphas: torch.Tensor = None,
#         input_sf: torch.Tensor = None,
#         out_dtype: torch.dtype = None,
#         ep_rank: Optional[int] = 0,
#         ep_size: Optional[int] = 1,
#         tp_rank: Optional[int] = 0,
#         tp_size: Optional[int] = 1,
#         use_dp: bool = False,
#         apply_router_weight_on_input: bool = False,
#     ) -> torch.Tensor:
#         a1 = hidden_states
#         output = a1 if inplace else torch.zeros_like(a1)

#         # flashinfer kernel don't need partition topk_ids and top_weights
#         # just to quantization in prepare

#         (a1q, a1q_scale, expert_num_tokens, _expert_topk_ids,
#          _expert_topk_weights) = self.prepare_finalize.prepare(
#              a1, a1_scale, a2_scale, topk_weights, topk_ids,
#              global_num_experts, expert_map, apply_router_weight_on_input,
#              use_dp)

#         # TODO(shuw): no chunk atm
#         fused_out = self.fused_experts.apply(
#             a1q,
#             w1,
#             w2,
#             _expert_topk_ids,
#             _expert_topk_weights,
#             activation,
#             global_num_experts,
#             w1_scale,
#             w2_scale,
#             w1_zp,
#             w2_zp,
#             a1_scale,
#             a2_scale,
#             expert_num_tokens,
#             g1_alphas,
#             g2_alphas,
#             a1q_scale,
#             out_dtype,
#             ep_rank,
#             ep_size,
#             tp_rank,
#             tp_size,
#         )
#         self.prepare_finalize.finalize(output, fused_out, topk_weights,
#                                        topk_ids, apply_router_weight_on_input,
#                                        use_dp)
#         return output


# def flashinfer_cutlass_fused_moe_nvfp4(
#     hidden_states: torch.Tensor,
#     topk_weights: torch.Tensor,
#     topk_ids: torch.Tensor,
#     w1: torch.Tensor,
#     w2: torch.Tensor,
#     w1_scale: torch.Tensor,
#     w2_scale: torch.Tensor,
#     a1_scale: torch.Tensor,
#     a2_scale: torch.Tensor,
#     g1_alphas: torch.Tensor,
#     g2_alphas: torch.Tensor,
#     inplace: bool = False,
#     activation: str = "silu",
#     global_num_experts: int = -1,
#     # TODO: put in class init
#     ep_rank: Optional[int] = 0,
#     ep_size: Optional[int] = 1,
#     tp_rank: Optional[int] = 0,
#     tp_size: Optional[int] = 1,
#     use_dp: bool = False,
#     apply_router_weight_on_input=False,
# ) -> torch.Tensor:
#     import pdb
#     pdb.set_trace()
#     fn = mk.FusedMoEModularKernel(
#         FlashInferCutlassMoEPrepareAndFinalizeNoEP(
#             quant_dtype=torch.uint8,  #meaning 2x e2m1 packed in one
#         ),
#         FlashInferExperts(),
#     )
#     # quant_scales computed in the prepare
#     extra_expert_args = {
#         'g1_alphas' : g1_alphas,
#         'g2_alphas' : g2_alphas,
#         'ep_rank': ep_rank,
#         'ep_size': ep_size,
#         'tp_rank': tp_rank,
#         'tp_size': ep_size,
#         'out_dtype': hidden_states.dtype,
#     }
#     extra_prepare_args = {
#         'use_dp': use_dp
#     }
#     extra_finalize_args = {
#         'use_dp': use_dp
#     }    
#     return fn(
#         hidden_states,
#         w1,
#         w2,
#         topk_ids,
#         topk_weights,
#         inplace,
#         activation,
#         global_num_experts,
#         w1_scale=w1_scale,
#         w2_scale=w2_scale,
#         a1_scale=a1_scale,
#         a2_scale=a2_scale,
#         apply_router_weight_on_input=apply_router_weight_on_input,
#         extra_expert_args=extra_expert_args,
#         extra_prepare_args=extra_prepare_args,
#         extra_finalize_args=extra_finalize_args,
#         # g1_alphas=g1_alphas,
#         # g2_alphas=g2_alphas,
#         # ep_rank=ep_rank,
#         # ep_size=ep_size,
#         # tp_rank=tp_rank,
#         # tp_size=tp_size,
#         # # use_dp=use_dp,
#         # out_dtype=hidden_states.dtype,
#     )
