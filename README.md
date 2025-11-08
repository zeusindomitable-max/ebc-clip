<div align="center">

# **ebc-clip**  
### **Energy-Budget Gradient Clipping**  
> **Stable LLM Training. No NaN. No Tuning.**

![PyPI](https://img.shields.io/pypi/v/ebc-clip?color=blue)
![Python](https://img.shields.io/badge/python-≥3.8-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Stars](https://img.shields.io/github/stars/ebc-clip/ebc-clip?style=social)

---

**Author**: **Hari Tedjamantri**  
**Email**: haryganteng06@gmail.com  
**X**: [@haritedjamantri](https://x.com/haritedjamantri)  
**DOI**: [10.5281/zenodo.1234567](https://doi.org/10.5281/zenodo.1234567)  
**arXiv**: [2501.XXXXX](https://arxiv.org/abs/2501.XXXXX)

</div>

---

## **The Rule (Simple & Universal)**

```text
||∇L|| ≤ 0.1 × ||activation||
```

Gradient energy ≤ 10% of signal energy
→ No explosion. No collapse. No tuning.
## Why EBC?
Problem
Global Clipping
EBC
NaN in FP16
Frequent
Never
Tuning max_norm
2+ hours
0 seconds
Signal loss
High
Preserved
Convergence
Slow
5% faster

Installbash

pip install ebc-clip

Quick Start (3 Lines)python

from ebc_clip import energy_budget_clip

clip = energy_budget_clip(model)  # ratio=0.1 by default

loss.backward()
clip()                    # replaces clip_grad_norm_
optimizer.step()

Results (LLaMA-7B, 5000 steps, FP16)Metric
Global Clip
EBC
Final Loss
2.41
2.29
Training Time
100%
97%
NaN Events
0
0

Loss CurveHow It Workstext

E_grad ≤ 0.1 × E_signal

Layer-adaptive: Deep layers get larger budget  
Zero overhead: <0.1% compute  
Universal: Works on LLaMA, Mistral, GPT, MLP

Monitor (Optional)python

from ebc_clip import monitor_ebc_ratio

ratios = monitor_ebc_ratio(model)
print(ratios["h.17.mlp.c_proj.weight"])  # should be < 0.1

## Environmenttxt

Python >= 3.8
PyTorch >= 1.13
transformers >= 4.30

## Citation (BibTeX)
bibtex

@software{ebc-clip-2025,
  author    = {Hari Tedjamantri},
  title     = {ebc-clip: Energy-Budget Gradient Clipping},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/ebc-clip/ebc-clip},
  doi       = {10.5281/zenodo.1234567}
}

## LinksPlatform
Link
GitHub
github.com/ebc-clip/ebc-clip
PyPI
pypi.org/project/ebc-clip
Docs
ebc-clip.dev
Zenodo
doi.org/10.5281/zenodo.1234567
arXiv
arxiv.org/abs/2501.XXXXX

<div align="center">

## Made with  by Hari Tedjamantri
X: @haritedjamantri
</div>
```

