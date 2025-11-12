# Kelly25 Voice Model Validation Report

**Generated:** 2025-10-08T15:01:02.809699

## Model Information

- **Model Path:** kelly25_model_output\best_model.pth
- **Device:** cpu
- **Configuration:** {'vocab_size': 256, 'hidden_dim': 128, 'learning_rate': 0.0001, 'batch_size': 8, 'epochs': 50, 'checkpoint_interval': 5, 'sample_rate': 22050}

## Validation Results

### Basic Generation

**Status:** ✅ PASS


### Multiple Samples

**Status:** ✅ PASS

**Samples:** 5
- Duration: 5.00s, Range: [-0.097, 0.093]
- Duration: 5.00s, Range: [-0.099, 0.095]
- Duration: 5.00s, Range: [-0.097, 0.093]
- Duration: 5.00s, Range: [-0.103, 0.099]
- Duration: 5.00s, Range: [-0.101, 0.097]

### Audio Quality

**Status:** ✅ PASS

**Metrics:**
- length: 110250
- duration_seconds: 5.0
- min_value: -0.10217487812042236
- max_value: 0.09798970073461533
- mean_value: -0.0005635680863633752
- std_value: 0.02619929052889347
- rms: 0.02620534971356392
- dynamic_range: 0.2001645863056183
- dominant_frequency: 245.0
- frequency_analysis: SUCCESS
- has_signal: True
- is_normalized: True
- has_clipping: False

### Model Performance

**Status:** ✅ PASS

**Metrics:**
- total_parameters: 457340203
- trainable_parameters: 457340203
- model_size_mb: 1744.6144218444824
- hidden_dimension: 128
- vocab_size: 256
- generation_time_seconds: 0.023616480827331542
- generations_per_second: 42.34331132192617

## Summary

**Tests Passed:** 4/4
**Success Rate:** 100.0%

🎉 **All tests passed!** The Kelly25 voice model is ready for use.
