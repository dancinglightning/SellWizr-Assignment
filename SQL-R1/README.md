<div align="center">
<h1>SQL-R1: Training Natural Language to SQL Reasoning Model By Reinforcement Learning</h1>
</div>

<div align="center">
<p>
    <a href="https://github.com/MPX0222">Peixian Ma</a><sup>1,2</sup>&nbsp;&nbsp;
    <a href="https://idea-finai.github.io/SQL-R1/">Xialie Zhuang</a><sup>1,3</sup>&nbsp;&nbsp;
    <a href="https://idea-finai.github.io/SQL-R1/">Chengjin Xu</a><sup>1,4</sup>&nbsp;&nbsp;
    <a href="https://idea-finai.github.io/SQL-R1/">Xuhui Jiang</a><sup>1,4</sup>&nbsp;&nbsp;
    <a href="https://idea-finai.github.io/SQL-R1/">Ran Chen</a><sup>1</sup>&nbsp;&nbsp;
    <a href="https://idea-finai.github.io/SQL-R1/">Jian Guo</a><sup>1</sup>
</p>

<p>
    <sup>1</sup>IDEA Research, International Digital Economy Academy
    <sup>2</sup>The Hong Kong University of Science and Technology (Guangzhou)
    <sup>3</sup>University of Chinese Academy of Science
    <sup>4</sup>DataArc Tech Ltd.
</p>
</div>

<div align="center">
<p>
<i class="fa-solid fa-envelope"></i> Contact: <a href="mailto:pma929@connect.hkust-gz.edu.cn">pma929@connect.hkust-gz.edu.cn</a>, <a href="mailto:xuchengjin@idea.edu.cn">xuchengjin@idea.edu.cn</a>
</p>
</div>


<div align="center" style="display: flex; gap: 5px; justify-content: center;">
<a href="https://dataarctech.github.io/SQL-R1/"><img src="https://img.shields.io/badge/🏠_Homepage-blue?style=for-the-badge"/></a>
<a href="https://arxiv.org/abs/2504.08600"><img src="https://img.shields.io/badge/arXiv-red?style=for-the-badge&logo=arxiv"/></a>
<a href="https://github.com/DataArcTech/SQL-R1"><img src="https://img.shields.io/badge/GitHub-black?style=for-the-badge&logo=github"/></a>
<a href="https://huggingface.co/MPX0222forHF/SQL-R1-7B"><img src="https://img.shields.io/badge/HuggingFace-FF9D00?style=for-the-badge&logo=huggingface"/></a>
<a href="https://www.modelscope.cn/models/MPX0222/SQL-R1-7B"><img src="https://img.shields.io/badge/🤖_ModelScope-946CE6?style=for-the-badge"/></a>
<a href="https://github.com/DataArcTech/SQL-R1/stargazers"><img src="https://img.shields.io/github/stars/DataArcTech/SQL-R1?style=for-the-badge&color=white"/></a>
</div>
<br>

<div align="center" style="display: flex; gap: 5px; justify-content: center;">
  <p>
  <b>🔥  Our work is accepted by NeurIPS 2025. Welcome to star and cite our work! ✨</b> 
  </p>
</div>

---

## 📋 SellWizr Assignment: RL Integration for Text-to-SQL

This repository extends SQL-R1 with an optimized setup for running on Kaggle with limited GPU resources (16GB T4).

### Chosen Paper & Rationale

**Paper**: [SQL-R1: Training Natural Language to SQL Reasoning Model By Reinforcement Learning](https://arxiv.org/abs/2504.08600)

**Why SQL-R1?**
1. **State-of-the-art results**: 88.6% on Spider, 67.1% on BIRD benchmarks
2. **GRPO Algorithm**: Uses Group Relative Policy Optimization (no critic model), saving ~50% GPU memory vs PPO
3. **3B Model Available**: Qwen2.5-Coder-3B fits within 16GB memory budget
4. **Complete RL Pipeline**: Full implementation of reward computation and policy updates

### RL Integration Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    GRPO Training Loop                            │
├─────────────────────────────────────────────────────────────────┤
│  1. Data Loader → Load prompts with ground truth SQL            │
│  2. Actor Rollout → vLLM generates N responses per prompt       │
│  3. Reward Manager → Compute scores via synsql.py               │
│  4. GRPO Advantage → Normalize rewards per-group (no critic!)   │
│  5. Policy Update → PPO-style clipped objective + KL penalty    │
│  6. Loop → Repeat for configured epochs                         │
└─────────────────────────────────────────────────────────────────┘
```

### Reward Design

| Component | Correct | Incorrect | Description |
|-----------|:-------:|:---------:|-------------|
| **Format** | +1 | -1 | Valid `<think>...</think><answer>...</answer>` |
| **Execution** | +2 | -2 | SQL executes without errors |
| **Result Match** | +3 | -3 | Query results match gold SQL |
| **Length Bonus** | 0-1.5 | 0 | Encourages concise reasoning |

**Total Score Range**: -6 to +7.5

### Training on Kaggle

```bash
# Option 1: Use the Jupyter notebook
# Upload notebooks/SQL_R1_Kaggle_Training.ipynb to Kaggle

# Option 2: Use the training script
sh sh/train_kaggle.sh
```

**Memory Optimizations for 16GB GPU:**
- Batch size: 2
- Gradient checkpointing: Enabled
- CPU offloading: Parameters, gradients, optimizer states
- Sequence length: 1024 prompt + 512 response
- vLLM memory: 30% GPU allocation

### Key Files

| File | Purpose |
|------|---------|
| `notebooks/SQL_R1_Kaggle_Training.ipynb` | Complete Kaggle notebook |
| `sh/train_kaggle.sh` | Optimized training script |
| `verl/utils/reward_score/synsql.py` | Reward computation |
| `verl/trainer/ppo/core_algos.py` | GRPO advantage estimation |
| `verl/trainer/main_ppo.py` | Main training entry point |

---

## 📖 Overview

Natural Language to SQL (NL2SQL) enables intuitive interactions with databases by transforming natural language queries into structured SQL statements.  Despite recent advancements in enhancing human-computer interaction within database applications, significant challenges persist, particularly regarding the inference performance in complex scenarios involving multi-table joins and nested queries. Current methodologies primarily utilize supervised fine-tuning (SFT) to train the NL2SQL model, which may limit adaptability and interpretability in new environments (e.g., finance and healthcare). In order to enhance the reasoning performance of the NL2SQL model in the above complex situations, we introduce SQL-R1, a novel NL2SQL reasoning model trained by the reinforcement learning (RL) algorithms. We design a specialized RL-based reward function tailored for NL2SQL tasks and discussed the impact of cold start on the effectiveness of intensive training. In addition, we achieve competitive accuracy using only a tiny amount of synthetic NL2SQL data for augmented training and further explore data engineering for RL. In existing experiments, SQL-R1 achieves execution accuracy of 88.6\% and 67.1\% on the benchmark Spider and BIRD, respectively.

<div align="center">
<img src="images/overview.png" alt="SQL-R1 Overview" width="800"/>
<p align="center">
Figure 1: Demonstration of our work.
</p>

</div>


## 📚 Citations

```bibtex
@article{ma2025sql,
  title={SQL-R1: Training Natural Language to SQL Reasoning Model By Reinforcement Learning},
  author={Ma, Peixian and Zhuang, Xialie and Xu, Chengjin and Jiang, Xuhui and Chen, Ran and Guo, Jian},
  journal={arXiv preprint arXiv:2504.08600},
  year={2025}
}
```

## 📰 News

- **[2025.09.18]** 🎉 SQL-R1 is accept by NeurIPS 2025! We will soon update the full version of the paper and poster. Welcome to star and cite our work!
- **[2025.05.27]** 🎉 We have released the full version of SQL-R1.
- **[2025.05.21]** 🎉 We have released our model weights on HuggingFace! Check out the [Model Weights](#-model-weights) section below.
- **[2025.04.11]** 📑 Our paper is now available on [arXiv](https://arxiv.org/abs/2504.08600).


## 🚀 Coming Soon Checklist

- [x] 📝 Update the camera-ready version of the paper, homepage and poster. coming sooooon!
- [x] 📊 Release model weights on HuggingFace and ModelScope
- [x] 🔧 Open source training code and RL dataset
- [x] 📝 Detailed documentation
- [x] 🛠️ Environment setup guide


## 🤖 Model Weights

We are excited to release our SQL-R1 model weights! You can find them on HuggingFace:

| Model  | Size | HuggingFace Link | ModelScope Link |
|-------------|-------------|------|------|
| SQL-R1 (3B) | 3B | [🤗 Download](https://huggingface.co/MPX0222forHF/SQL-R1-3B) | - |
| SQL-R1 (7B) | 7B | [🤗 Download](https://huggingface.co/MPX0222forHF/SQL-R1-7B) | [🤖 Download](https://www.modelscope.cn/models/MPX0222/SQL-R1-7B) |
| SQL-R1 (14B) | 14B | [🤗 Download](https://huggingface.co/MPX0222forHF/SQL-R1-14B) | - |


## 📑 Documentation Structure

This repository is organized as follows:

```
SQL-R1/
├── 📁 data/                             # Datasets and Databases
│   ├── 📁 Spider/      
│   └── 📁 BIRD/        
├── 📁 models/                           # Foundation models or checkpoints
│   ├── 📁 Qwen2.5-Coder-3B-Instruct/   
│   └── 📁 Qwen2.5-Coder-7B-Instruct/   
├── 📁 db_info/                          # Database information files (Just for inference)
├── 📁 example_data/                     # Example data (Training)
├── 📁 sh/                               # Scripts for data processing, training, inference and evaluation
│   ├── 📄 train.sh
│   ├── 📄 inference.sh
│   ├── 📄 eval_spider.sh
│   └── 📄 eval_bird.sh
├── 📁 src/                              # Source code
│   ├── 📁 data_preprocess/
│   ├── 📁 evaluations/
│   ├── 📁 utils/
│   ├── 📄 inference.py
│   └── 📄 evaluation_*.py
├── 📁 verl/                             # Verl reinforcement learning framework
├── 📄 requirements.txt
└── 📄 README.md
```


## 🛠️ Environment Setup

> [!NOTE]
> Before getting started, make sure your computing environment supports the following settings:
> - Environment: Python 3.9+
> - CUDA Version: 12.0+ (for verl and vllm integration)
> - GPU Prerequisites: 8 x 80GB+ GPU (for training) / 2 x 40GB GPU (for inference)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/MPX0222/SQL-R1.git
cd SQL-R1
```

2. Create and activate a virtual environment (recommended):
```bash
conda create -n sqlr1 python=3.9
```

3. Install dependencies:
```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.6.3 ray
pip install flash-attn --no-build-isolation
pip install -e .  # For verl integration
pip install wandb IPython matplotlib sqlparse func_timeout nltk ijson
```

4. Download the model weights from HuggingFace and put them in the `models/` directory. 

5. For training, copy the training dataset in the `example_data` directory to the `data` directory. For inference, copy the database information in the `db_info` directory (including files forSpider-dev, Spider-test and BIRD-dev) to the related dataset (`data/Spider`, `data/BIRD`) directory.

## 🚀 Quick Start

1. Run training:
```bash
sh sh/train.sh
```

2. Run inference:
```bash
sh sh/inference.sh
```

3. Run evaluation:
```bash
# evaluate spider
sh sh/eval_spider.sh
# evaluate bird
sh sh/eval_bird.sh
```

## 🌟 Applications

SQL-R1 can be effectively utilized in the following key areas:

1. **Foundation Model for Workflow Systems**
   - Serves as a base model for complex database operations
   - Enables seamless integration with existing workflow automation systems
   - Supports customization and fine-tuning for specific business processes
   - Provides robust API endpoints for system integration

2. **Enhanced NL2SQL Interpretability**
   - Generates detailed explanations for SQL query construction
   - Provides step-by-step reasoning for query transformation
   - Helps users understand the relationship between natural language and SQL syntax
   - Offers visualization of query execution plans

3. **Self-Evolving NL2SQL Agent Checkpoint**
   - Acts as a foundation checkpoint for continuous learning
   - Enables iterative improvement through feedback loops
   - Supports transfer learning for domain-specific adaptations
   - Facilitates model versioning and performance tracking

This may be added in the future work. If you have any ideas, please feel free to contact us. 

## Thanks for

We thank [OmniSQL](https://github.com/RUCKBReasoning/OmniSQL) and follow their evaluation code and database information retrieval code. We have adapted and modified their evaluation scripts for our project.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=DataArcTech/SQL-R1&type=Date)](https://www.star-history.com/#DataArcTech/SQL-R1&Date)

