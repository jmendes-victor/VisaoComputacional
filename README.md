## 📋 Descrição

Este projeto implementa e compara diferentes estratégias de aprendizado contínuo:

- **FT (Fine-Tuning)**: Treinamento sequencial sem mitigação de esquecimento
- **ER (Experience Replay)**: Replay de amostras antigas durante o treinamento
- **OURS (AKD)**: Método proposto com Attention Knowledge Distillation
- **JT (Joint Training)**: Upper bound - treinamento com todos os dados

### Métricas Avaliadas

- **AACC**: Average Accuracy (média das acurácias finais)
- **BWT**: Backward Transfer (mede o esquecimento catastrófico)
- **IM**: Intransigence Measure (diferença para o upper bound)

## 🚀 Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/jmendes-victor/VisaoComputacional
cd VisaoComputacional
```

### 2. Crie um ambiente virtual (recomendado)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

## 📁 Estrutura do Projeto

```
.
├── main.py                 # Script principal
├── requirements.txt        # Dependências
├── README.md              # Este arquivo
├── src/
│   ├── __init__.py
│   ├── config.py          # Configurações e hiperparâmetros
│   ├── utils.py           # Funções utilitárias
│   ├── dataset.py         # Dataset e DataLoader
│   ├── model.py           # Arquitetura CLAM-SB
│   ├── memory.py          # Buffer de replay
│   ├── losses.py          # Funções de perda (KD, AKD)
│   ├── trainer.py         # Engine de treinamento
│   └── joint_trainer.py   # Joint Training
└── local_data/
    ├── csv/
    │   └── sce_E3/        # Arquivos CSV das tarefas
    └── embeddings/        # Embeddings pré-computados (.npy)
```

## 📊 Dados

Os dados consistem em embeddings pré-computados de WSIs. A estrutura esperada:

- **CSV**: Arquivos com colunas `Patient` (ID) e `GT` (ground truth/classe)
- **Embeddings**: Arquivos `.npy` com nome `{Patient}.npy`

### Cenário E3 (3 tarefas)

| Tarefa | Classes | Arquivos |
|--------|---------|----------|
| 1 | 2, 4 | 2_4_train.csv, 2_4_test.csv |
| 2 | 0, 5 | 0_5_train.csv, 0_5_test.csv |
| 3 | 1, 3 | 3_1_train.csv, 3_1_test.csv |

## ▶️ Execução

Execute o experimento completo:

```bash
python main.py
```

O script irá:
1. Carregar os datasets das 3 tarefas
2. Treinar cada método sequencialmente
3. Avaliar e exibir relatório comparativo

### Saída Esperada

```
===================================================================================================================
                               RELATÓRIO DE REPRODUÇÃO - COMPARAÇÃO COM O ARTIGO                                   
===================================================================================================================
MÉTODO          | MEU AACC   | PAPER      | MEU BWT    | PAPER      | MEU IM     | PAPER     
-------------------------------------------------------------------------------------------------------------------
Fine-Tuning     | 0.XXXX     | 0.3167     | -0.XXXX    | -0.8701    | -0.XXXX    | -0.1942   
ER (Replay)     | 0.XXXX     | 0.5388     | -0.XXXX    | -0.3869    | -0.XXXX    | -0.0942   
Ours (AKD)      | 0.XXXX     | 0.5926     | -0.XXXX    | -0.4056    | -0.XXXX    | -0.1604   
-------------------------------------------------------------------------------------------------------------------
Joint Training  | 0.XXXX     | 0.7311     | N/A        | N/A        | N/A        | N/A       
===================================================================================================================
```

## ⚙️ Configurações

Os hiperparâmetros podem ser ajustados em `src/config.py`:

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `SEED` | 42 | Seed para reprodutibilidade |
| `NUM_CLASSES` | 6 | Número total de classes |
| `DIM_FEATURES` | 512 | Dimensão dos embeddings |
| `BUFFER_SIZE` | 42 | Tamanho do buffer de replay |
| `AKD_LAMBDA` | 1.0 | Peso da perda AKD |
| `KD_LAMBDA` | 1.0 | Peso da perda KD |

## 🔬 Detalhes Técnicos

### Modelo CLAM-SB

O modelo usa atenção com gate para agregar instâncias:

1. **Feature Extraction**: Linear + ReLU + Dropout
2. **Gated Attention**: Mecanismo de atenção para ponderar instâncias
3. **Classification**: Camada linear para predição

### Knowledge Distillation

- **KD Loss**: KL-divergence entre logits do estudante e professor
- **AKD Loss**: KL-divergence entre distribuições de atenção

## 📝 Licença

Este projeto é para fins educacionais.

## 📚 Referências

- CLAM: Data Efficient and Weakly Supervised Computational Pathology
- Continual Learning for Medical Image Analysis
