# ADvisor
✨ Official implementation of ADvisor, proposed in the ACL 2026 Industry Track paper:

Pre-Deployment Advertisement Ranking under Data Scarcity via Context-Aware Criteria Generation with VLMs

[![ACL Paper](https://img.shields.io/badge/ACL%20Paper-2026%20Industry%20Track-blue.svg)](https://aclanthology.org/2026.acl-industry.28/)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)


## 🚀 Quick Start

### 🔗 Using Similar Brand Information
```bash
python main.py --use_cross_brand_for_features
```

### 🎯 Using Only Target Brand Information
```bash
python main.py
```

## Data Format
The expected directory structure is as follows:

data/
├── train_<brand_id>.csv

├── test_<brand_id>.csv

├── caption_dict.json
├── caption_embeddings.pkl
├── brand_embeddings.pkl
└── brand_descriptions.json
  
Each train_<brand_id>.csv and test_<brand_id>.csv file should contain advertisement-level information and target metric columns used for ranking.

Example:

train_brandA.csv
test_brandA.csv
train_brandB.csv
test_brandB.csv
...

Optional files such as captions, caption embeddings, brand embeddings, and brand descriptions can be used to provide additional context for VLM-based criteria generation and scoring.
