# Testing Checklist for New Features

Use this checklist to verify all new features are working correctly.

## Prerequisites
- [ ] Python 3.11+ installed
- [ ] Virtual environment activated
- [ ] Dependencies installed: `pip install -e .`
- [ ] `.env` file configured with at least one provider's API keys
- [ ] OpenAI API key configured

## 1. Sample Data Loader

### Via Quick Start Script
- [ ] Run `python quick_start.py`
- [ ] Verify sample PDF is generated in `data/pdfs/`
- [ ] Verify PDF is automatically indexed
- [ ] Verify test query runs successfully
- [ ] Check trace URLs are displayed (if providers configured)

### Via UI
- [ ] Run `streamlit run app.py`
- [ ] Navigate to "Quick Start" page
- [ ] Click "Load Sample Data" button
- [ ] Verify success message appears
- [ ] Click "View Sample Queries" button
- [ ] Verify 5+ sample queries are displayed

## 2. Performance Benchmarking

### Setup
- [ ] Navigate to "🚀 Benchmarking" page
- [ ] Verify active providers are listed
- [ ] Select 2-3 providers to benchmark

### Run Benchmark
- [ ] Set iterations to 5
- [ ] Enter test query: "What is observability?"
- [ ] Click "Run Benchmark"
- [ ] Verify baseline latency is measured
- [ ] Verify each provider is benchmarked
- [ ] Verify progress bar updates

### Results
- [ ] Verify "Average Latency by Provider" chart displays
- [ ] Verify "SDK Overhead by Provider" chart displays
- [ ] Verify detailed metrics table shows:
  - [ ] Average latency
  - [ ] Median latency
  - [ ] P95 latency
  - [ ] P99 latency
  - [ ] Overhead percentage
  - [ ] Error rate
- [ ] Click "Export Results"
- [ ] Verify `results/benchmark_results.json` is created

## 3. Automatic Capability Detection

### Setup
- [ ] Navigate to "🔍 Auto-Detection" page
- [ ] Verify active providers are listed
- [ ] Select 2-3 providers to test

### Run Detection
- [ ] Enable "Auto-fill comparison matrix"
- [ ] Set confidence threshold to 0.7
- [ ] Click "Detect Capabilities"
- [ ] Verify progress bar updates
- [ ] Verify "Detection complete!" message

### Results
- [ ] Verify detection results are displayed for each provider
- [ ] Check each result shows:
  - [ ] ✅/❌ status icon
  - [ ] Criterion name
  - [ ] Confidence percentage (colored)
  - [ ] Evidence text
- [ ] Verify "Updated X matrix entries" message
- [ ] Click "Save Matrix"
- [ ] Verify `results/comparison_matrix.json` is created/updated
- [ ] Click "Export Detection Results"
- [ ] Verify `results/capability_detection.json` is created

## 4. Enhanced Comparison Matrix

### Navigation
- [ ] Navigate to "Comparison Matrix" page
- [ ] Verify platform legend is displayed
- [ ] Verify category tabs are present

### Auto-filled Data
- [ ] Check if any criteria are auto-filled (from capability detection)
- [ ] Verify auto-filled entries have notes like "Auto-detected (confidence: X%)"
- [ ] Verify you can still manually edit auto-filled values

### New Categories
- [ ] Verify "Debugging RCA" tab exists
- [ ] Verify "Security Compliance" tab exists
- [ ] Verify "Optimization Dev" tab exists
- [ ] Verify "Agentic Logic" tab exists

### New Criteria
Check these new criteria are present:
- [ ] Trace Diff / Side-by-Side
- [ ] AI Failure Summarization
- [ ] Automated Root Cause Analysis
- [ ] Issue Clustering
- [ ] Time-Series Performance Compare
- [ ] Outlier Detection
- [ ] PII/Compliance Shield
- [ ] Prompt Injection Guard
- [ ] Security Analyzers
- [ ] Auto-Prompt Fix (AutoTune)
- [ ] Hot Reload (Active Debug)
- [ ] Local-first / Offline mode
- [ ] Session Replays
- [ ] State/Memory View
- [ ] Thread Management
- [ ] Browser Agent Visualization

### Export
- [ ] Click "Save Matrix"
- [ ] Verify success message
- [ ] Click "Export Markdown"
- [ ] Verify `results/comparison_matrix.md` is created
- [ ] Click "Show Rankings"
- [ ] Verify radar chart displays
- [ ] Verify detailed rankings table displays

## 5. Enhanced Quick Start Page

### Visual Elements
- [ ] Verify centered title "🚀 PDF Knowledge Explorer"
- [ ] Verify description text is displayed
- [ ] Verify 4 capability cards are shown:
  - [ ] 📚 RAG Engine
  - [ ] 🕵️ Observability
  - [ ] 🧪 Scenarios
  - [ ] 📊 Analytics

### Workflow
- [ ] Verify 4 workflow steps are displayed
- [ ] Verify architecture diagram (Graphviz) renders

### Sample Data Section
- [ ] If sample data exists: verify "✓ Sample data loaded" message
- [ ] If no sample data: verify "Load Sample Data" button
- [ ] Verify "Next Steps" info box is displayed

## 6. Integration Tests

### Full Workflow
- [ ] Load sample data
- [ ] Index sample PDF
- [ ] Run a query in "Query Interface"
- [ ] Verify trace URLs are displayed
- [ ] Run benchmarks on 2+ providers
- [ ] Run capability detection on same providers
- [ ] Check comparison matrix is auto-filled
- [ ] Add manual ratings to matrix
- [ ] Export all results (benchmark, detection, matrix)

### Error Handling
- [ ] Try running benchmark with no providers selected (should be disabled)
- [ ] Try running detection with no providers selected (should be disabled)
- [ ] Try loading sample data twice (should skip or confirm)

## 7. Documentation

### Files Created
- [ ] Verify `docs/IMPROVEMENTS.md` exists and is comprehensive
- [ ] Verify `docs/SUMMARY.md` exists with visual diagrams
- [ ] Verify `docs/SUMMARY_RU.md` exists (Russian version)
- [ ] Verify `README.md` has new features section

### Code Documentation
- [ ] Check `src/evaluations/benchmark.py` has docstrings
- [ ] Check `src/evaluations/capability_detector.py` has docstrings
- [ ] Check `src/utils/sample_data.py` has docstrings

## 8. Dependencies

### Installation
- [ ] Run `pip install -e .`
- [ ] Verify `reportlab` is installed
- [ ] Verify no dependency conflicts

### Import Tests
```python
# Run in Python REPL
from src.evaluations.benchmark import PerformanceBenchmarker
from src.evaluations.capability_detector import CapabilityDetector
from src.utils.sample_data import initialize_sample_data
# All should import without errors
```

## Issues Found

Document any issues here:

1. 
2. 
3. 

## Overall Assessment

- [ ] All features work as expected
- [ ] Documentation is clear and helpful
- [ ] UI is intuitive and responsive
- [ ] Exports work correctly
- [ ] No critical bugs found

## Sign-off

- Tested by: _______________
- Date: _______________
- Version: _______________
- Status: ☐ Pass  ☐ Fail  ☐ Needs Review
