# Screenshots for the submission PDF

Two folders, two sections of the report.

## `lab/`

The BITS Virtual Lab evidence. Run `bash scripts/lab_run.sh` on the lab VM and
capture the terminal at the end. The frame needs to show:

* the `RESULTS - trained on <hostname> [EC2 i-...]` banner
* the six models with all six metrics
* the `ALL CHECKS PASSED` line

The EC2 instance id is the part that makes this evidence rather than decoration,
since it cannot be produced from a laptop.

## `app/`

The deployed Streamlit app. One capture per tab is plenty:

1. Sidebar with the upload control and the model dropdown open
2. Metrics tab
3. Confusion matrix and classification report tab
4. Compare all models tab
5. Predictions tab

## Both folders

Images are embedded in filename order, so prefix them `01-`, `02-` to control
the sequence. Anything ending `.png`, `.jpg`, `.jpeg`, `.gif` or `.webp` is
picked up. Then run:

```bash
python scripts/make_pdf.py
```
