#pragma once

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace onnxruntime {
namespace contrib {

namespace transformers {

// Base class of gready search implementation that is common for both GPT-2 and Bart/T5.
template <typename T>
class GreedySearchBase {
 public:
  GreedySearchBase(OpKernelContextInternal& context,
                   const SessionState& decoder_session_state,
                   concurrency::ThreadPool* thread_pool,
                   void* cuda_stream,
                   IConsoleDumper* cuda_dumper,
                   BeamSearchParameters& params,
                   const BeamSearchDeviceHelper::TopkFunc& topk_func,
                   const BeamSearchDeviceHelper::ProcessLogitsFunc<T>& process_logits_func,
                   const BeamSearchDeviceHelper::DeviceCopyFunc<float>& device_copy_func)
      : context_(context),
        decoder_session_state_(decoder_session_state),
        thread_pool_(thread_pool),
        implicit_inputs_(context_.GetImplicitInputs()),
        cuda_stream_(cuda_stream),
        cuda_dumper_(cuda_dumper),
        parameters_(&params),
        cpu_allocator_(nullptr),
        temp_space_allocator_(nullptr),
        topk_func_(topk_func),
        process_logits_func_(process_logits_func),
        device_copy_func_(device_copy_func) {
    parameters_->ParseFromInputs(&context);

    cpu_allocator_ = decoder_session_state.GetExecutionProviders()
                         .Get(onnxruntime::kCpuExecutionProvider)
                         ->GetAllocator(0, OrtMemTypeDefault);
  }

  // Initialize by validating all the inputs, and allocating the output tensors.
  Status Initialize();

  // Validate inputs.
  Status CheckInputs(const OpKernelContextInternal& context);

 protected:
  // Process logits and append next tokens to sequences.
  Status GenerateNextToken(const OrtValue& logits,
                           gsl::span<int32_t>& beam_next_tokens,
                           gsl::span<int32_t>& beam_indices,
                           BeamSearchState<T>& beam_state,
                           BeamSearchCpuState& cpu_state,
                           int counter);

  // Calculate scores from logits, then apply filtering and select next token for each beam.
  Status ProcessLogits(const OrtValue& logits,  // logits output of subgraph
                       BeamSearchState<T>& beam_state,
                       BeamSearchCpuState& cpu_state,
                       AllocatorPtr& allocator,
                       int counter);

  bool IsCuda() const { return cuda_stream_ != nullptr; }

  const IConsoleDumper* GetConsoleDumper() const { return IsCuda() ? cuda_dumper_ : &(cpu_dumper_); }

  OpKernelContextInternal& context_;

  const SessionState& decoder_session_state_;

  concurrency::ThreadPool* thread_pool_;

  const std::vector<const OrtValue*>& implicit_inputs_;

  void* cuda_stream_;

  IConsoleDumper* cuda_dumper_;
  CpuTensorConsoleDumper cpu_dumper_;

  BeamSearchParameters* parameters_;

  LogitsProcessorList logits_processors_;

  //std::unique_ptr<BeamSearchScorer> beam_scorer_;

  AllocatorPtr cpu_allocator_;
  AllocatorPtr temp_space_allocator_;

  // Device specific functions
  BeamSearchDeviceHelper::TopkFunc topk_func_;
  BeamSearchDeviceHelper::ProcessLogitsFunc<T> process_logits_func_;
  BeamSearchDeviceHelper::DeviceCopyFunc<float> device_copy_func_;
};

template <typename T>
Status GreedySearchBase<T>::CheckInputs(const OpKernelContextInternal& context) {
  // Input shapes:
  //   input_ids  : (batch_size, sequence_length)

  const Tensor* input_ids = context.Input<Tensor>(0);
  const auto& dims = input_ids->Shape().GetDims();
  if (dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'input_ids' is expected to have 2 dimensions, got ",
                           dims.size());
  }

  return Status::OK();
}

template <typename T>
Status GreedySearchBase<T>::Initialize() {
  ORT_RETURN_IF_ERROR(context_.GetTempSpaceAllocator(&temp_space_allocator_));

#define CHECK_SCALAR_INPUT(name, index, required)                                                                 \
  auto* name##_tensor = context_.Input<Tensor>(index);                                                            \
  if (name##_tensor) {                                                                                            \
    if (!name##_tensor->Shape().IsScalar()) {                                                                     \
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "'BeamSearch' input " #name " should be a scalar. Got shape of ", \
                             name##_tensor->Shape());                                                             \
    }                                                                                                             \
  } else if (required) {                                                                                          \
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "'BeamSearch' input " #name " is required");                        \
  }

  CHECK_SCALAR_INPUT(max_length, 1, true);

  CHECK_SCALAR_INPUT(min_length, 2, false);

  ORT_RETURN_IF_ERROR(CheckInputs(context_));

  // This flag will be updated later when the scores output exists.
  parameters_->output_scores = false;

  // Greedy search
  parameters_->no_repeat_ngram_size = 0;

  if (!IsCuda()) {
    // Logits processor is used in CPU only. In CUDA, cuda kernels are used instead.
    // Initialize processsors after CheckInputs so that parameters_->vocab_mask is ready.
    logits_processors_.Init(*parameters_);
  }

  return Status::OK();
}

template <typename T>
Status GreedySearchBase<T>::ProcessLogits(
    const OrtValue& logits,
    BeamSearchState<T>& beam_state,
    BeamSearchCpuState& cpu_state,
    AllocatorPtr& allocator,
    int counter) {
  return process_logits_func_(logits, &beam_state, &cpu_state, &(cpu_state.sequences), allocator,
                              thread_pool_, &logits_processors_, beam_scorer_.get(),
                              parameters_, counter, cuda_stream_, GetConsoleDumper());
}

template <typename T>
Status GreedySearchBase<T>::GenerateNextToken(
    const OrtValue& logits,
    gsl::span<int32_t>& beam_next_tokens,
    gsl::span<int32_t>& beam_indices,
    BeamSearchState<T>& beam_state,
    BeamSearchCpuState& cpu_state,
    int counter) {
  // Process logits to get next token scores
  ORT_RETURN_IF_ERROR(ProcessLogits(logits, beam_state, cpu_state, temp_space_allocator_, counter));

  gsl::span<float>& beam_scores = beam_scorer_->GetNextScores();
  // It is optional to clone beam_scores. Change it to use same buffer also works for CPU:
  //    beam_state.beam_scores = beam_scores
  // Here we make a copy to reduce the coupling with little cost (the buffer size is small).
  ORT_RETURN_IF_ERROR(device_copy_func_(beam_state.beam_scores, beam_scores, cuda_stream_, DeviceCopyDirection::hostToDevice));

  beam_next_tokens = beam_scorer_->GetNextTokens();
  beam_indices = beam_scorer_->GetNextIndices();

#ifdef DEBUG_BEAM_SEARCH
  cpu_dumper_.Print("beam_scores after scorer", beam_scores.data(), parameters_->batch_size, parameters_->num_beams);
  cpu_dumper_.Print("beam_next_tokens after scorer", beam_next_tokens.data(), parameters_->batch_size, parameters_->num_beams);
  cpu_dumper_.Print("beam_indices after scorer", beam_indices.data(), parameters_->batch_size, parameters_->num_beams);
#endif

  cpu_state.sequences.AppendNextTokenToSequences(beam_indices, beam_next_tokens);

#ifdef DEBUG_BEAM_SEARCH
  cpu_state.sequences.PrintSequences(&cpu_dumper_);
#endif
  return Status::OK();
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
