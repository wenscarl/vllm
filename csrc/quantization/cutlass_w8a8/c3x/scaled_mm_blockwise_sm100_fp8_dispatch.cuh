#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass_extensions/gemm/dispatch_policy.hpp"
#include "cutlass_extensions/gemm/collective/collective_builder.hpp"

#include "cutlass_gemm_caller.cuh"

namespace vllm {

using namespace cute;

template <typename OutType, typename MmaTileShape, typename PerSmTileShape,
          typename EpilogueTileShape, typename ScalesPerTile,
          int TileSizeM_ = 128, class ClusterShape = Shape<_1, _1, _1>>
struct cutlass_3x_gemm_fp8_blockwise {
  using TileSizeM = Int<TileSizeM_>;

  using ElementAB = cutlass::float_e4m3_t;

  using ElementA = ElementAB;
  using LayoutA = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  using ElementB = ElementAB;
  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using ElementC = void;
  using ElementD = OutType;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  //   using StrideC = StrideD;
  using LayoutC = LayoutD;
  static constexpr int AlignmentC = AlignmentD;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementBlockScale = float;

  // MMA and Cluster Tile Shapes
  // Shape of the tile computed by tcgen05 MMA, could be across 2 SMs if Cluster
  // Shape %2 == 0 using MmaTileShape_MNK = Shape<_128,_128,_128>;
  static constexpr int ScaleMsPerTile = size<0>(ScalesPerTile{});
  static constexpr int ScaleGranularityM =
      size<0>(MmaTileShape{}) / ScaleMsPerTile;
  static constexpr int ScaleGranularityN =
      size<1>(MmaTileShape{}) / size<1>(ScalesPerTile{});
  static constexpr int ScaleGranularityK =
      size<2>(MmaTileShape{}) / size<2>(ScalesPerTile{});

  // Shape of the threadblocks in a cluster
  using ClusterShape_MNK = Shape<_1, _1, _1>;

  using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<
      ScaleGranularityM, ScaleGranularityN, ScaleGranularityK,
      cute::UMMA::Major::MN, cute::UMMA::Major::K>;

  using LayoutSFA =
      decltype(ScaleConfig::deduce_layoutSFA());  // Layout type for SFA matrix
                                                  // operand
  using LayoutSFB =
      decltype(ScaleConfig::deduce_layoutSFB());  // Layout type for SFB matrix
                                                  // operand
  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  using AtomThrShape = Shape<_1, _1, _1>;

  // clang-format off
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      PerSmTileShape,
      ClusterShape,
      EpilogueTileShape,
      ElementAccumulator,
      ElementCompute,
      ElementC,
      LayoutC,
      AlignmentC,
      ElementD,
      LayoutD,
      AlignmentD,
      cutlass::epilogue::TmaWarpSpecialized1Sm
  >::CollectiveOp;
  
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      cute::tuple<LayoutA, LayoutSFA>,
      AlignmentA,
      ElementB,
      cute::tuple<LayoutB, LayoutSFB>,
      AlignmentB,
      ElementAccumulator,
      MmaTileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
      >,
      cutlass::gemm::KernelTmaWarpSpecializedBlockwise1SmSm100
  >::CollectiveOp;
  // clang-format on

  using KernelType = enable_sm100_or_later<cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>>;

  struct GemmKernel : public KernelType {};
};

template <typename Gemm>
void cutlass_gemm_caller_blockwise(torch::Tensor& out, torch::Tensor const& a,
                                   torch::Tensor const& b,
                                   torch::Tensor const& a_scales,
                                   torch::Tensor const& b_scales) {
  using GemmKernel = typename Gemm::GemmKernel;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using LayoutSFA = typename Gemm::LayoutSFA;
  using LayoutSFB = typename Gemm::LayoutSFB;
  using ScaleConfig = typename Gemm::ScaleConfig;

  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;

  int32_t m = a.size(0), n = b.size(1), k = a.size(1);
  auto prob_shape = cute::make_shape(m, n, k, 1);

  StrideA a_stride;
  StrideB b_stride;
  StrideC c_stride;
  a_stride =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  b_stride =
      cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  c_stride =
      cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));

  LayoutSFA layout_SFA =
      ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
  LayoutSFB layout_SFB =
      ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));

  auto a_ptr = static_cast<ElementAB*>(a.data_ptr());
  auto b_ptr = static_cast<ElementAB*>(b.data_ptr());
  auto a_scales_ptr = static_cast<float*>(a_scales.data_ptr());
  auto b_scales_ptr = static_cast<float*>(b_scales.data_ptr());

  typename GemmKernel::MainloopArguments mainloop_args{
      a_ptr,        a_stride,   b_ptr,        b_stride,
      a_scales_ptr, layout_SFA, b_scales_ptr, layout_SFB};

  auto c_ptr = static_cast<ElementD*>(out.data_ptr());
  typename GemmKernel::EpilogueArguments epilogue_args{
      {}, c_ptr, c_stride, c_ptr, c_stride};
  epilogue_args.thread.alpha = 1.0f;
  c3x::cutlass_gemm_caller<GemmKernel>(a.device(), prob_shape, mainloop_args,
                                       epilogue_args);
}

template <typename OutType>
void cutlass_gemm_blockwise_sm100_fp8_dispatch(torch::Tensor& out,
                                               torch::Tensor const& a,
                                               torch::Tensor const& b,
                                               torch::Tensor const& a_scales,
                                               torch::Tensor const& b_scales) {
  auto m = a.size(0);
  auto k = a.size(1);
  auto n = b.size(1);

  // Define tile shapes based on the value of m
  if (m <= 128) {
    cutlass_gemm_caller_blockwise<cutlass_3x_gemm_fp8_blockwise<
        OutType, Shape<_64, _128, _128>, Shape<_64, _128, _128>,
        Shape<_64, _64>, Shape<_64, _1, _1>>>(out, a, b, a_scales, b_scales);
  } else {
    cutlass_gemm_caller_blockwise<cutlass_3x_gemm_fp8_blockwise<
        OutType, Shape<_128, _128, _128>, Shape<_128, _128, _128>,
        Shape<_128, _64>, Shape<_128, _1, _1>>>(out, a, b, a_scales, b_scales);
  }
}

}  // namespace vllm
