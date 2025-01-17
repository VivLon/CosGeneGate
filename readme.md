# CosGeneGate Selects Multi-functional and Credible Biomarkers for Single-cell Analysis

This is the official repo for our marker gene selection project.

# Installation

To prepare the model, please use:

```
pip install -r requirements.txt
```

To install stg, please use our forked mode from [this link](https://github.com/runopti/stg):

```
git clone https://github.com/HelloWorldLTY/stg.git
cd stg/python
pip install -e ./
```

To install the R environment for the deconvolution analysis, please use:

```
install.packages("CosGeneGateDeconv")
```

or

```
install.packages('devtools')
devtools::install_github('VivLon/CosGeneGate/tree/main/CosGeneGateDeconv')
library(CosGeneGateDeconv)
```

# Tutorials

Please use the file **Tutorial_marker_selection.ipynb** as a tutorial for running our algorithm. The demo dataset can be found in [this folder](https://drive.google.com/drive/folders/1kEK6MPGejnpXMIthULP66ytE4teLgKx9?usp=drive_link).

# Benchmark

Please refer the original repos for the implementation of our benchmarked methods: [COSG](https://github.com/genecell/COSG), [NS-Forest](https://github.com/JCVenterInstitute/NSForest), [scGeneFit](https://github.com/solevillar/scGeneFit-python), [STG](https://github.com/runopti/stg), and [scMAGS](https://github.com/doganlab/scmags).

# Acknowledgements

We refer the codes from [STG](https://github.com/runopti/stg), [CSCORE](https://github.com/ChangSuBiostats/CS-CORE_python), and [COSG](https://github.com/genecell/COSG).

Thanks for their great work!

# Citation

```
@article{liu2025cosgenegate,
  title={CosGeneGate selects multi-functional and credible biomarkers for single-cell analysis},
  author={Liu, Tianyu and Long, Wenxin and Cao, Zhiyuan and Wang, Yuge and He, Chuan Hua and Zhang, Le and Strittmatter, Stephen M and Zhao, Hongyu},
  journal={Briefings in Bioinformatics},
  volume={26},
  number={1},
  pages={bbae626},
  year={2025},
  publisher={Oxford University Press}
}
```
