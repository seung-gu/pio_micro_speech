/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "feature_provider.h"

#include "audio_provider.h"
#include "micro_features_micro_features_generator.h"
#include "micro_features_micro_model_settings.h"

// initializing feature_size_, feature_data_, is_first_run in base class 
//  -> int feature_size_, int8_t* feature_data_, bool is_first_run;
// int feature_size_ = feature_size = kFeatureElementCount, int8_t* feature_data = feature_buffer
// constexpr int kFeatureElementCount = (kFeatureSliceSize(40) * kFeatureSliceCount(49));
// int8_t* feature_data = feature_buffer[kFeatureElementCount]; => feature_buffer[40*49=1960]
//   
// connecting int8_t feature_buffer[kFeatureElementCount] in micro_speech 
//  to int8_t* feature_data_ in feature_provider.h
FeatureProvider::FeatureProvider(int feature_size, int8_t* feature_data)  
    : feature_size_(feature_size),
      feature_data_(feature_data),
      is_first_run_(true) {
  // Initialize the feature data to default values.
  for (int n = 0; n < feature_size_; ++n) {
    feature_data_[n] = 0;
  }
}

FeatureProvider::~FeatureProvider() {}

// in micro_speech : 
// feature_provider->PopulateFeatureData(
//                    error_reporter, previous_time, current_time, &how_many_new_slices);
TfLiteStatus FeatureProvider::PopulateFeatureData(
    tflite::ErrorReporter* error_reporter, int32_t last_time_in_ms,
    int32_t time_in_ms, int* how_many_new_slices) {
  if (feature_size_ != kFeatureElementCount) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Requested feature_data_ size %d doesn't match %d",
                         feature_size_, kFeatureElementCount);
    return kTfLiteError;
  }

  // Quantize the time into steps as long as each window stride, so we can
  // figure out which audio data we need to fetch.
  // kFeatureSliceStrideMs = 20,
  // ex) last_time_in_ms : 500ms, time_in_ms : 540ms,
  //     last_step = 500 / 20 = 25, current_step = 540 / 20 = 27
  //     sclices_needed = 27 - 25 = 2
  const int last_step = (last_time_in_ms / kFeatureSliceStrideMs);  
  const int current_step = (time_in_ms / kFeatureSliceStrideMs);

  int slices_needed = current_step - last_step;
  // If this is the first call, make sure we don't use any cached information.
  if (is_first_run_) {
    TfLiteStatus init_status = InitializeMicroFeatures(error_reporter);
    if (init_status != kTfLiteOk) {
      return init_status;
    }
    is_first_run_ = false;
    slices_needed = kFeatureSliceCount;
  }
  // kFeatureSliceCount = 49
  if (slices_needed > kFeatureSliceCount) {
    slices_needed = kFeatureSliceCount;
  }
  *how_many_new_slices = slices_needed;

  // ex) slices_to_keep = kFeatureSliceCount - slices_needed(2) = 49 - 2 = 47
  //     slices_to_drop = 49 - 47 = 2
  const int slices_to_keep = kFeatureSliceCount - slices_needed;
  const int slices_to_drop = kFeatureSliceCount - slices_to_keep;
  // If we can avoid recalculating some slices, just move the existing data
  // up in the spectrogram, to perform something like this:
  // last time = 80ms          current time = 120ms
  // +-----------+             +-----------+
  // | data@20ms |         --> | data@60ms |
  // +-----------+       --    +-----------+
  // | data@40ms |     --  --> | data@80ms |
  // +-----------+   --  --    +-----------+
  // | data@60ms | --  --      |  <empty>  |
  // +-----------+   --        +-----------+
  // | data@80ms | --          |  <empty>  |
  // +-----------+             +-----------+

  // shift 20ms x slices
  // ex) slices_to_keep : 47, loop 0~46
  //     dest_slice_data* = feature_data_ + (dest_slice * kFeatureSliceSize
  //     feature_data_[1960], kFeatureSliceSize = 40
  //     dest_slice_data : 0, 40, 80, 120 ...
  //     src_slice = dest_slice + slices_to_drop = 0 + 2, 1 + 2, ...
  //     src_slice_data : 80, 120, 160, 200 ... 
  if (slices_to_keep > 0) {
    for (int dest_slice = 0; dest_slice < slices_to_keep; ++dest_slice) {
      // int8_t* feature_data_ 
      int8_t* dest_slice_data =
          feature_data_ + (dest_slice * kFeatureSliceSize);
      const int src_slice = dest_slice + slices_to_drop;
      const int8_t* src_slice_data =
          feature_data_ + (src_slice * kFeatureSliceSize);
      for (int i = 0; i < kFeatureSliceSize; ++i) {
        dest_slice_data[i] = src_slice_data[i];
      }
    }
  }
  // Any slices that need to be filled in with feature data have their
  // appropriate audio data pulled, and features calculated for that slice.
  // ex) slices_to_keep : 47
  //     kFeatureSliceCount = 49
  //     new_slice : 47, 48
  //     current_step : 27
  //     new_step = (27 - 49 + 1) + 47(,48) = 26(,27)
  if (slices_needed > 0) {
    for (int new_slice = slices_to_keep; new_slice < kFeatureSliceCount;
         ++new_slice) {
      const int new_step = (current_step - kFeatureSliceCount + 1) + new_slice;
      const int32_t slice_start_ms = (new_step * kFeatureSliceStrideMs);
      int16_t* audio_samples = nullptr;
      int audio_samples_size = 0;
      // TODO(petewarden): Fix bug that leads to non-zero slice_start_ms
      GetAudioSamples(error_reporter, (slice_start_ms > 0 ? slice_start_ms : 0),
                      kFeatureSliceDurationMs, &audio_samples_size,
                      &audio_samples);
      if (audio_samples_size < kMaxAudioSampleSize) {
        TF_LITE_REPORT_ERROR(error_reporter,
                             "Audio data size %d too small, want %d",
                             audio_samples_size, kMaxAudioSampleSize);
        return kTfLiteError;
      }
      int8_t* new_slice_data = feature_data_ + (new_slice * kFeatureSliceSize);
      size_t num_samples_read;
      TfLiteStatus generate_status = GenerateMicroFeatures(
          error_reporter, audio_samples, audio_samples_size, kFeatureSliceSize,
          new_slice_data, &num_samples_read);
      if (generate_status != kTfLiteOk) {
        return generate_status;
      }
    }
  }
  return kTfLiteOk;
}
