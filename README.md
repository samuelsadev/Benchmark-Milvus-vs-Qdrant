# Milvus vs Qdrant: Hybrid Vector Search Benchmark

Benchmark comparativo de bancos de dados vetoriais para busca híbrida (dense + sparse + RRF + ColBERT reranking).

**Engines:** Milvus v2.6.16 vs Qdrant v1.12.0  
**Embeddings:** BGE-M3 (1024d dense + sparse)  
**Reranking:** ColBERT v2 com MaxSim em GPU  

📄 [Artigo no Medium](https://medium.com/@samuelsa.dev/milvus-vs-qdrant-comparei-na-pr%C3%A1tica-dois-bancos-vetoriais-para-busca-h%C3%ADbrida-em-produ%C3%A7%C3%A3o-1edd0691f848?sharedUserId=samuelsa.dev)

---

## Arquitetura do Pipeline

```
Query → [Busca Densa HNSW] ─┐
                             ├→ [RRF Fusion] → Top-100 → [ColBERT GPU] → Top-10
Query → [Busca Esparsa]  ───┘
```

1. **Busca densa** — BGE-M3, 1024d, HNSW, COSINE
2. **Busca esparsa** — BGE-M3 sparse, Inner Product
3. **Fusão RRF** — nativa de cada engine (k=60)
4. **Reranking ColBERT** — MaxSim late interaction em GPU
5. **Filtro regional** — aplicado previamente em todas as buscas

---

## Estrutura do Repositório

```
benchmark_milvus_vs_qdrant/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/                  # Dados de entrada (não versionado)
├── results/               # Resultados das execuções (não versionado)
├── src/                   # Código-fonte do benchmark
│   ├── config.py                    # Configurações centralizadas
│   ├── benchmark_base.py            # Classes base e métricas
│   ├── benchmark_milvus.py          # Benchmark Milvus
│   ├── benchmark_qdrant.py          # Benchmark Qdrant
│   ├── benchmark_v31_dual_dense.py  # Dual dense source (sumário vs tópico)
│   ├── benchmark_v31_observability.py # Observabilidade por pod
│   ├── regional_filter.py           # Filtro prévio por região
│   ├── colbert_rerank.py            # ColBERT MaxSim reranking
│   ├── concurrency_sweep.py         # Varredura de concorrência
│   ├── ef_search_sweep.py           # Varredura de ef_search
│   ├── simulation_matrix.py         # Matriz de simulações (produto cruzado)
│   ├── generate_ground_truth.py     # Geração de ground truth exato
│   ├── gt_relevance.py              # GT-Relevância por categoria
│   ├── prepare_data.py              # Preparação de dados
│   ├── download_embeddings.py       # Download de embeddings do S3
│   └── run_benchmark.py             # Runner principal
└── infra/                 # Manifests Kubernetes
    ├── qdrant-full.yaml
    ├── milvus-full.yaml
    └── services.yaml
```

---

## Pré-requisitos

- Python 3.10+
- GPU NVIDIA com CUDA (para ColBERT reranking)
- Kubernetes cluster (EKS, GKE ou local) com Qdrant e Milvus deployados
- ~100 GB de armazenamento para índices

---

## Instalação

```bash
git clone https://github.com/seu-usuario/benchmark_milvus_vs_qdrant.git
cd benchmark_milvus_vs_qdrant
pip install -r requirements.txt
```

---

## Configuração

Defina os endpoints dos bancos vetoriais via variáveis de ambiente:

```bash
export QDRANT_HOST=<endpoint_qdrant>
export MILVUS_HOST=<endpoint_milvus>
```

Para deploy local, use os manifests em `infra/`:

```bash
kubectl apply -f infra/qdrant-full.yaml
kubectl apply -f infra/milvus-full.yaml
kubectl apply -f infra/services.yaml
```

---

## Preparação dos Dados

O benchmark espera os seguintes arquivos no diretório `data/`:

| Arquivo | Descrição |
|---------|-----------|
| `documents.json` | Lista de documentos com campos `id`, `doc_id`, `region`, `category`, `doc_type`, `text` |
| `dense_embeddings.npy` | Array NumPy (N, 1024) com embeddings densos BGE-M3 |
| `dense_embeddings_sumario.npy` | Embeddings densos do sumário (para dual dense) |
| `dense_embeddings_topico.npy` | Embeddings densos do tópico (para dual dense) |
| `sparse_embeddings.json` | Lista de dicts `{indices: [...], values: [...]}` |
| `queries.json` | Lista de queries com campos `id`, `doc_id`, `region`, `category`, `text` |
| `query_dense.npy` | Array NumPy (Q, 1024) com embeddings das queries |
| `query_sparse.json` | Embeddings esparsos das queries |
| `ground_truth.json` | Lista de listas com IDs dos top-100 exatos por query |

Para gerar dados a partir dos seus embeddings:

```bash
cd src/
python prepare_data.py
python generate_ground_truth.py
```

---

## Execução

### Benchmark Completo

```bash
cd src/
python run_benchmark.py
```

### Execução por Fase

```bash
# Fase 1 — Matriz de simulações (2 engines × 2 dense × 3 RRF = 12 configs)
python run_benchmark.py --phase matrix

# Fase 1 (V3.1) — Dual dense corrigido
python benchmark_v31_dual_dense.py

# Fase 3 — Varredura ef_search (impacto no Recall vs latência)
python run_benchmark.py --phase ef_search

# Fase 5 — Varredura de concorrência (1 a 64 clientes)
python run_benchmark.py --phase concurrency

# Observabilidade por pod (com métricas CloudWatch)
python benchmark_v31_observability.py

# ColBERT reranking (latência em GPU)
python run_benchmark.py --phase colbert

# GT-Relevância (Precision@10 por categoria)
python run_benchmark.py --phase gt_relevance
```

---

## Métricas Avaliadas

| Métrica | Tipo | Descrição |
|---------|------|-----------|
| **Recall@100** | Primária | Fidelidade ao GT (alimenta o reranker) |
| Recall@10 | Secundária | Fidelidade ao GT no top-10 |
| Precision@10 | Secundária | Relevância de negócio (mesma categoria) |
| NDCG@10 | Secundária | Qualidade de ordenação |
| MRR@10 | Secundária | Posição do primeiro documento relevante |
| QPS | Performance | Queries por segundo |
| Latência P50/P95/P99 | Performance | Percentis de tempo de resposta |
| RAM / Disco | Footprint | Consumo de recursos |

---

## Rigor Estatístico

- 5 execuções por configuração (1ª descartada como warm-up)
- IC 95% via bootstrap (1.000 amostras sobre queries)
- Percentis P50, P95, P99
- Sementes fixas e versionadas (seed=42)
- Ground truth calculado por região via busca exata (brute-force)

---

## Limitações Conhecidas

1. **RRF ponderado** — Nenhuma engine suporta pesos variáveis via API nativa. Ambas usam RRF com pesos iguais (k=60).
2. **REST vs gRPC** — Qdrant testado via REST (porta gRPC não exposta no setup); Milvus via gRPC nativo.
3. **Assimetria de CPU** — Qdrant single-pod (4 cores, 24 GB); Milvus multi-pod (query node com 4 cores, 8 GB).
4. **WeightedRanker ≠ RRF** — O `WeightedRanker` do Milvus faz fusão linear por score, não é RRF.


