# R0 best-rationale runtime overlay

This ignored staging directory is created by
`scripts/prepare_r0_best_rationale_overlay.py`.  It overlays the immutable R0
ensemble image with the selected `mal-direct-lora-epoch2` rationale adapter,
the exact hash-bound training/evaluation prompt, and a new runtime manifest.

Do not commit staged weights, prompts copied for the image, or completion
records from this directory.
