# Folder Cleanup Plan

## Current Status
Directory has ~100+ files including scripts, models, data, and outputs from experiments.

---

## Files to **KEEP** (Essential)

### Core System Files
```
✅ blend_knob_v2_with_swing.py       # FINAL working blend knob system
✅ meter_conditional_lstm.py          # Model C architecture
✅ api.py                             # API interface
✅ classify_terminal.py               # Useful classification utility
✅ ConvNet_SNFPR_model.pth            # CNN classifier model
```

### Essential Documentation
```
✅ BLEND_KNOB_TWO_LEVEL_SWING.md     # Complete pipeline documentation
✅ BLEND_KNOB_TEST_RESULTS.md        # Test results
✅ MODEL_C_DOCUMENTATION.md           # Model C research
✅ FINAL_MODEL.md                     # Production model info
✅ PROJECT_DOCUMENTATION.md           # Overall project docs
✅ README.md                          # Main readme
```

### Essential Directories
```
✅ models/                            # Trained models
✅ drums/                             # Drum samples
✅ tabla/                             # Tabla samples
✅ generated_blend_v2/                # Latest blend outputs
✅ input/                             # Test inputs
```

---

## Files to **ARCHIVE** (Move to `_archive/` folder)

### Old Versions & Experiments
```
📦 blend_knob_generator.py           # OLD VERSION (before two-level swing)
📦 classify_and_generate_modelc.py   # Intermediate version
📦 jugalbandi_standalone.py          # Old standalone version
```

### Batch Processing Scripts (Multiple versions)
```
📦 batch_classify.py
📦 batch_classify_long_files.py
📦 batch_classify_standalone.py
📦 classify_no_flask.py
```

### Export Scripts (Multiple experimental versions)
```
📦 export_60_40_split.py
📦 export_drum_mapping.py
📦 export_model_audio.py
📦 export_model_audio_improved.py
📦 export_proper_drum_mapping.py
📦 export_tabla_and_drums.py
📦 export_with_context.py
📦 export_with_swing.py
📦 export_two_level_swing.py         # Keep for reference but archive
```

### Analysis & Comparison Scripts
```
📦 compare_all_models.py
📦 compare_audio_outputs.py
📦 compare_models.py
```

### Diagnostic Scripts
```
📦 diagnose_drum_timing.py
📦 diagnose_durations.py
📦 diagnose_swing_issue.py
```

### Test Scripts
```
📦 test_*.py (all test scripts)
```

### Training Scripts (Archive but keep for reference)
```
📦 train_*.py (all training scripts)
📦 lstm_data_prep.py
📦 lstm_model.py
📦 prepare_training_data.py
📦 lstm_training_data.pkl
```

### Utility Scripts
```
📦 convert_aif_to_wav.py
📦 fix_drum_samples.py
📦 taal_utils.py
📦 tabla_rule_correction.py
```

### Old Documentation
```
📦 BLEND_KNOB_PIPELINE.md            # Superseded by TWO_LEVEL_SWING version
📦 BAR_AWARE_PIPELINE.md             # Research doc (archive)
📦 EMBEDDING_CONFUSION_RESEARCH.md   # Research doc (archive)
📦 METER_DETECTION_ANALYSIS.md       # Research doc (archive)
📦 PROTOTYPE_FEASIBILITY.md          # Research doc (archive)
📦 SWING_DRUM_MAPPING_SUMMARY.md     # Research doc (archive)
📦 TIMING_VERIFICATION_SUMMARY.md    # Research doc (archive)
```

### Data Files (Review & Archive)
```
📦 classification_review_samples.csv
📦 classification_review_samples_2.csv
📦 corrected_labels.csv
📦 post_processed_results.csv
📦 training_data_classified.json
📦 training_data_corrected*.json
📦 teental_dugun_classification.json
```

### Text Logs
```
📦 training_log*.txt (all training logs)
📦 complete_model_comparison.txt
📦 model_comparison_results.txt
```

---

## Directories to **ARCHIVE**

```
📦 generated_audio/                  # Old generation outputs
📦 generated_audio_*/ (all variants)
📦 generated_drums/
📦 generated_proper_drums/
📦 generated_tabla_and_drums/
📦 generated_two_level_swing/
📦 bar_slices/
📦 downbeat_slices/
📦 segmented_bars/
📦 training_data/
📦 classified_long_files/
📦 drums_fixed/
```

---

## Files to **DELETE** (Redundant/Temporary)

### Environment Files (Keep one, delete duplicates)
```
❌ environment_minimal.yml            # Keep environment.yml only
❌ environment_production.yml
```

### System Files
```
❌ .DS_Store
❌ __pycache__/
```

### Redundant Config
```
❌ .env.example (if not needed)
❌ .gitignore (if already in repo)
```

---

## Proposed Folder Structure After Cleanup

```
tabla-to-drumset_SG/
├── README.md
├── environment.yml
│
├── 📁 core/                          # Core system files
│   ├── blend_knob_v2_with_swing.py
│   ├── meter_conditional_lstm.py
│   ├── api.py
│   └── classify_terminal.py
│
├── 📁 models/                        # Trained models
│   ├── best_bar_aware_lstm.pth
│   └── ConvNet_SNFPR_model.pth
│
├── 📁 samples/                       # Audio samples
│   ├── drums/
│   └── tabla/
│
├── 📁 docs/                          # Essential documentation
│   ├── BLEND_KNOB_TWO_LEVEL_SWING.md
│   ├── BLEND_KNOB_TEST_RESULTS.md
│   ├── MODEL_C_DOCUMENTATION.md
│   ├── FINAL_MODEL.md
│   └── PROJECT_DOCUMENTATION.md
│
├── 📁 outputs/                       # Generated outputs
│   └── generated_blend_v2/
│
├── 📁 input/                         # Test inputs
│
└── 📁 _archive/                      # Archived experiments
    ├── scripts/
    ├── old_docs/
    ├── old_outputs/
    └── training_artifacts/
```

---

## Cleanup Steps

### Step 1: Create Archive Structure
```bash
mkdir -p _archive/{scripts,old_docs,old_outputs,training_artifacts,data}
```

### Step 2: Move Scripts
```bash
# Old versions
mv blend_knob_generator.py _archive/scripts/
mv classify_and_generate_modelc.py _archive/scripts/

# Batch processing
mv batch_classify*.py _archive/scripts/
mv classify_no_flask.py _archive/scripts/

# Export variants
mv export_*.py _archive/scripts/

# Analysis
mv compare*.py _archive/scripts/
mv diagnose*.py _archive/scripts/

# Tests
mv test_*.py _archive/scripts/

# Training
mv train_*.py _archive/training_artifacts/
mv lstm_*.py _archive/scripts/
mv prepare_training_data.py _archive/training_artifacts/

# Utilities
mv convert_aif_to_wav.py _archive/scripts/
mv fix_drum_samples.py _archive/scripts/
mv taal_utils.py _archive/scripts/
mv tabla_rule_correction.py _archive/scripts/
mv jugalbandi_standalone.py _archive/scripts/
```

### Step 3: Move Old Documentation
```bash
mv BAR_AWARE_PIPELINE.md _archive/old_docs/
mv BLEND_KNOB_PIPELINE.md _archive/old_docs/
mv EMBEDDING_CONFUSION_RESEARCH.md _archive/old_docs/
mv METER_DETECTION_ANALYSIS.md _archive/old_docs/
mv PROTOTYPE_FEASIBILITY.md _archive/old_docs/
mv SWING_DRUM_MAPPING_SUMMARY.md _archive/old_docs/
mv TIMING_VERIFICATION_SUMMARY.md _archive/old_docs/
```

### Step 4: Move Data Files
```bash
mv *.csv _archive/data/
mv *.json _archive/data/
mv *.pkl _archive/training_artifacts/
mv *.txt _archive/training_artifacts/
```

### Step 5: Move Old Output Directories
```bash
mv generated_audio* _archive/old_outputs/
mv generated_drums _archive/old_outputs/
mv generated_proper_drums _archive/old_outputs/
mv generated_tabla_and_drums _archive/old_outputs/
mv generated_two_level_swing _archive/old_outputs/
mv bar_slices _archive/old_outputs/
mv downbeat_slices _archive/old_outputs/
mv segmented_bars _archive/old_outputs/
mv classified_long_files _archive/old_outputs/
mv drums_fixed _archive/old_outputs/
mv training_data _archive/training_artifacts/
```

### Step 6: Delete Redundant Files
```bash
rm -rf __pycache__/
rm .DS_Store
rm environment_minimal.yml
rm environment_production.yml
```

### Step 7: Reorganize Core Files
```bash
mkdir -p core samples/drums samples/tabla docs outputs

# Move core scripts
mv blend_knob_v2_with_swing.py core/
mv meter_conditional_lstm.py core/
mv api.py core/
mv classify_terminal.py core/

# Move samples
mv drums/* samples/drums/
mv tabla/* samples/tabla/
rmdir drums tabla

# Move docs
mv BLEND_KNOB_TWO_LEVEL_SWING.md docs/
mv BLEND_KNOB_TEST_RESULTS.md docs/
mv MODEL_C_DOCUMENTATION.md docs/
mv FINAL_MODEL.md docs/
mv PROJECT_DOCUMENTATION.md docs/

# Move outputs
mv generated_blend_v2 outputs/
```

---

## Estimated Space Savings

### Current Usage
```
Total: ~2-3 GB (with all generated audio)
```

### After Cleanup
```
Active files: ~500 MB
Archive: ~2 GB (compressed)
Deleted: ~200 MB
```

### Result
Clean working directory with essential files easily accessible!

---

## Safety Notes

1. **DO NOT DELETE** anything from `models/` directory
2. **DO NOT DELETE** `drums/` or `tabla/` sample folders
3. **KEEP** at least one copy of training logs for reference
4. **BACKUP** before running cleanup if unsure
5. **Archive, don't delete** research documentation

---

## Execute Cleanup?

Once approved, I can execute this cleanup plan step by step with confirmation at each stage.