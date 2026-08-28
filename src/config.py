"""
Configurações centralizadas para o Benchmark V3 - Hybrid Search Benchmark
=========================================================================

MÉTRICA PRIMÁRIA: Recall@100

Princípio: Como o ColBERT reranqueia os top-100, a métrica primária
de infraestrutura é o Recall@100 do banco vetorial, é o conjunto
de candidatos que alimenta o reranker. O Recall@10 do banco,
isoladamente, não decide, pois o top-10 é reconstruído pelo reranking.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path


# =============================================================================
# MÉTRICAS PRIMÁRIAS (V3)
# =============================================================================
# Recall@100 é a métrica PRIMÁRIA de decisão de infraestrutura
# Recall@10 NÃO decide (o top-10 é reconstruído pelo reranking ColBERT)

PRIMARY_METRIC = "recall_at_100"

SECONDARY_METRICS = [
    "recall_at_10",
    "precision_at_10",
    "ndcg_at_10",
    "mrr_at_10",
    "latency_p95_ms",
    "qps",
]

# =============================================================================
# ESTATÍSTICA (V3 - IC 95%)
# =============================================================================
BOOTSTRAP_SAMPLES = 1000  # Número de amostras bootstrap para IC 95%
CONFIDENCE_LEVEL = 0.95
N_RUNS_MIN = 5  # Mínimo de runs para latência/QPS
WARMUP_RUNS = 1  # Runs de warmup a descartar

# =============================================================================
# GROUND TRUTH HÍBRIDO (V3 - GT-Fidelidade)
# =============================================================================
# Para cada query, roda-se a fórmula real do sistema (denso + esparso + RRF)
# sem nenhuma aproximação, força bruta, comparando contra os documentos
# da região da query, um a um.

GT_FIDELITY_K = 100  # Top-100 para GT-Fidelidade
GT_RELEVANCE_K = 10  # Top-10 para GT-Relevância
GT_EXCLUDE_SELF = True  # Excluir próprio documento da query dos resultados

# Pesos RRF para varredura
RRF_WEIGHTS = [
    (0.6, 0.4),  # Denso-dominante
    (0.5, 0.5),  # Balanceado
    (0.4, 0.6),  # Esparso-dominante
]

# Constante k do RRF
RRF_K = 60


@dataclass
class QdrantConfig:
    """Configuração do Qdrant"""
    host: str = "localhost"  # Sobrescrever via QDRANT_HOST
    port: int = 6333
    grpc_port: int = 6334
    timeout: int = 300
    collection_name: str = "benchmark_v3"


@dataclass
class MilvusConfig:
    """Configuração do Milvus"""
    host: str = "localhost"  # Sobrescrever via MILVUS_HOST
    port: int = 19530
    timeout: int = 300
    collection_name: str = "benchmark_v3"


@dataclass
class EmbeddingConfig:
    """Configuração dos embeddings BGE-M3"""
    model_name: str = "BAAI/bge-m3"
    dense_dim: int = 1024  # BGE-M3 dense dimension
    sparse_enabled: bool = True
    max_length: int = 512
    batch_size: int = 32
    device: str = "cuda" if __import__('torch').cuda.is_available() else "cpu"


@dataclass
class HNSWConfig:
    """Configuração do índice HNSW (idêntico para ambos engines)"""
    m: int = 16
    ef_construction: int = 128
    ef_search_values: List[int] = field(default_factory=lambda: [64, 128, 256])


@dataclass
class HybridSearchConfig:
    """Configuração da busca híbrida"""
    # Pesos RRF para varredura (V3 - matriz de simulações)
    rrf_weights: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.6, 0.4),  # Denso-dominante
        (0.5, 0.5),  # Balanceado
        (0.4, 0.6),  # Esparso-dominante
    ])
    rrf_k: int = 60
    fusion_method: str = "rrf"
    top_k_retrieval: int = 100  # Candidatos para reranking (GT-Fidelidade)
    top_k_final: int = 10  # Top-10 final após reranking


@dataclass
class ColBERTConfig:
    """Configuração do reranking ColBERT"""
    model_name: str = "colbert-ir/colbertv2.0"
    dim: int = 128
    top_k_rerank: int = 100  # Recebe do primeiro estágio
    top_k_final: int = 10
    enabled: bool = True
    device: str = "auto"  # "auto", "cuda" ou "cpu"


@dataclass
class DataConfig:
    """Configuração dos dados"""
    # S3 bucket com embeddings
    s3_bucket: str = "your-benchmark-bucket"
    s3_region: str = "us-east-1"

    # Paths locais dos embeddings
    embeddings_dir: str = str(Path(__file__).parent.parent / "data")
    embeddings_dir_alt: str = str(Path(__file__).parent.parent / "data")

    # Arquivos de embeddings (BGE-M3 v1.0)
    denso_sumario_file: str = "dense_embeddings_sumario.npy"
    denso_topico_file: str = "dense_embeddings_topico.npy"
    esparso_file: str = "sparse_embeddings.json"
    colbert_dir: str = "colbert_sumario"

    # S3 URLs
    s3_base_url: str = "https://your-bucket.s3.your-region.amazonaws.com"

    # Metadados
    metadata_fields: List[str] = field(default_factory=lambda: [
        "doc_id",
        "region",
        "category_l1",
        "category",
        "category_l3",
        "doc_type",
        "data_cadastro_final",
        "data_conclusao_final",
    ])

    # Campo para filtro prévio obrigatório (V3 - sempre aplicado)
    filter_field: str = "region"
    regional_filter_enabled: bool = True  # Filtro regional é OBRIGATÓRIO

    # Campo para GT-Relevância
    relevance_field: str = "category"

    # Regiões conhecidas
    fiscal_regions: List[str] = field(default_factory=lambda: [
        f"region_{i:02d}" for i in range(1, 13)  # 12 regiões
    ])


@dataclass
class BenchmarkConfig:
    """Configuração principal do benchmark V3"""
    # Dados
    data_dir: str = str(Path(__file__).parent.parent / "data")
    results_dir: str = str(Path(__file__).parent.parent / "results")

    # Dados
    num_documents: int = 86_551  # Total de documentos
    num_queries: int = 1000  # V3: 1000 queries held-out

    # Queries held-out (V3)
    queries_seed: int = 42  # Semente fixa para reprodutibilidade
    queries_stratify_by: str = "category"  # Estratificação
    queries_exclude_self: bool = True  # Excluir próprio documento (self-match)

    # Execução
    num_runs: int = 5
    warmup_queries: int = 10
    cooldown_seconds: int = 5

    # Ground truth
    use_exact_search: bool = True
    use_hybrid_gt: bool = True  # V3: GT por fusão exata (denso + esparso + RRF)

    # Métricas - PRIMÁRIA: Recall@100
    primary_metric: str = "recall_at_100"

    metrics: List[str] = field(default_factory=lambda: [
        # PRIMÁRIO
        "recall_at_100",
        "recall_at_100_ci_lower",
        "recall_at_100_ci_upper",
        # SECUNDÁRIOS
        "recall_at_10",
        "precision_at_10",
        "ndcg_at_10",
        "mrr_at_10",
        # LATÊNCIA
        "latency_mean_ms",
        "latency_std_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        # THROUGHPUT
        "qps",
        # FOOTPRINT
        "ram_gb",
        "disk_gb",
    ])


@dataclass
class Config:
    """Configuração completa do benchmark V3"""
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    hnsw: HNSWConfig = field(default_factory=HNSWConfig)
    hybrid: HybridSearchConfig = field(default_factory=HybridSearchConfig)
    colbert: ColBERTConfig = field(default_factory=ColBERTConfig)
    data: DataConfig = field(default_factory=DataConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    @classmethod
    def from_env(cls) -> "Config":
        """Carrega configuração de variáveis de ambiente"""
        config = cls()

        # Override com variáveis de ambiente se existirem
        if os.getenv("QDRANT_HOST"):
            config.qdrant.host = os.getenv("QDRANT_HOST")
        if os.getenv("MILVUS_HOST"):
            config.milvus.host = os.getenv("MILVUS_HOST")
        if os.getenv("NUM_DOCUMENTS"):
            config.benchmark.num_documents = int(os.getenv("NUM_DOCUMENTS"))
        if os.getenv("NUM_RUNS"):
            config.benchmark.num_runs = int(os.getenv("NUM_RUNS"))
        if os.getenv("NUM_QUERIES"):
            config.benchmark.num_queries = int(os.getenv("NUM_QUERIES"))

        return config


# Instância padrão
DEFAULT_CONFIG = Config()


def get_config() -> Config:
    """Retorna configuração (de env ou padrão)"""
    return Config.from_env()
