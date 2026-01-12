# Milestone 3 – Model Evaluation & Tuning

## Objective
Improve baseline CNN performance through systematic model tuning and evaluation.

## Baseline Model (v1)
- Validation Accuracy: ~78%
- Architecture: 2 Conv blocks

## Experiment 1 – Batch Normalization (v2)
- Result: Performance degraded
- Conclusion: BatchNorm not suitable for this dataset and architecture

## Final Model – Deeper CNN (v3)
- Added one additional Conv block
- Increased Dense layer capacity
- Validation Accuracy: ~92–93%

## Outcome
- Improved class separation
- Reduced confusion between similar instruments
- Model v3 selected for deployment

## Notes
- Trained model saved as `.keras` locally (not uploaded)
