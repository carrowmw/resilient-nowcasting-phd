Created three ablation configs. Here's the experimental design:

Config	What it tests	Changed vs. current (train.yaml)


train_ablation_old_arch.yaml
Is it the model capacity?	h=32, L=2, e=16, bs=32 (dark-brook-37 arch) + new inductive pipeline


train_ablation_small_batch.yaml
Is it the batch size?	bs=32 only; keeps h=64, L=3, e=32


train_ablation_high_dropout.yaml
Can regularization rescue it?	dropout=0.3 (3×); keeps everything else large


How to interpret results:

If Ablation C (old arch) recovers spatial quality → capacity is the main culprit
If Ablation D (small batch) recovers → batch size is driving sharper minima
If Ablation E (high dropout) recovers → the large arch can work but needs regularization
If C recovers but D/E don't → the old arch is simply the right size for 81 nodes
After training, run python scripts/compare_runs.py --all to compare all runs including the new ablations.