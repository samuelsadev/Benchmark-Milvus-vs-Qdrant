#!/usr/bin/env python3
"""
Benchmark do Milvus para busca híbrida (dense + sparse).

Milvus v2.4+ suporta:
- Vetores esparsos nativos
- Busca híbrida com RRF/Weighted reranking
- BM25 nativo (v2.5+)

Versão atual no EKS: v2.6.16 (suporta todas as features)

IMPORTANTE (V3): Filtro Regional Obrigatório
============================================
- O filtro por região é sempre aplicado previamente
- A busca exata é feita dentro do subconjunto da região da query
- Todas as buscas DEVEM ser filtradas pela região da query
"""
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

# Suprimir warnings de deprecação do PyMilvus ORM
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pymilvus')

from pymilvus import (
    connections,
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    MilvusException,
)

from config import Config, get_config
from benchmark_base import BaseBenchmark, BenchmarkResult
from regional_filter import RegionalFilterManager, create_milvus_filter


class MilvusBenchmark(BaseBenchmark):
    """Benchmark para Milvus com busca híbrida

    IMPORTANTE (V3): Filtro regional é OBRIGATÓRIO em todas as buscas.
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.collection: Optional[Collection] = None
        self.collection_name = config.milvus.collection_name
        self.version = None
        self.supports_sparse = False
        self.regional_filter: Optional[RegionalFilterManager] = None
        self._current_ef_search = config.hnsw.ef_search_values[0]

    def connect(self):
        """Conecta ao Milvus"""
        print(f"🔗 Conectando ao Milvus: {self.config.milvus.host}:{self.config.milvus.port}")

        connections.connect(
            alias="default",
            host=self.config.milvus.host,
            port=self.config.milvus.port,
            timeout=self.config.milvus.timeout,
        )

        # Verificar versão
        self.version = utility.get_server_version()
        print(f"✅ Conectado! Versão: {self.version}")

        # Verificar suporte a sparse (v2.4+)
        version_str = self.version.lstrip('v')
        parts = version_str.split('.')
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0

        self.supports_sparse = (major > 2) or (major == 2 and minor >= 4)

        if self.supports_sparse:
            print(f"✅ Milvus {self.version} suporta busca híbrida (dense + sparse + RRF)")
        else:
            print(f"⚠️ AVISO: Milvus {self.version} NÃO suporta vetores esparsos nativos!")
            print(f"   Requer v2.4+ para busca híbrida completa.")
            print(f"   Este benchmark usará apenas busca densa.")

        # Verificar se filtro regional está habilitado
        if self.config.data.regional_filter_enabled:
            print("✅ Filtro regional HABILITADO - todas as buscas serão filtradas por região")

    def create_collection(self):
        """Cria collection com ou sem suporte a sparse"""
        print(f"📦 Criando collection: {self.collection_name}")

        # Deletar se existir
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            print(f"   Collection anterior deletada")

        # Schema - V3: Inclui region para filtro prévio obrigatório
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="dense", dtype=DataType.FLOAT_VECTOR, dim=self.config.embedding.dense_dim),
            FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=100),
            # V3: Campo para filtro regional obrigatório
            FieldSchema(name="region", dtype=DataType.VARCHAR, max_length=50),
            # V3: Campo para GT-Relevância
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=200),
        ]

        # Adicionar campo sparse se suportado
        if self.supports_sparse:
            fields.append(
                FieldSchema(name="sparse", dtype=DataType.SPARSE_FLOAT_VECTOR)
            )
            print(f"   Adicionando campo sparse (Milvus {self.version})")

        schema = CollectionSchema(fields=fields, description="Hybrid Search Benchmark V3")

        self.collection = Collection(
            name=self.collection_name,
            schema=schema,
        )

        print(f"✅ Collection criada (sparse={'✓' if self.supports_sparse else '✗'})")

    def set_regional_filter(self, regional_filter: RegionalFilterManager):
        """
        Define o gerenciador de filtro regional.

        O filtro regional é OBRIGATÓRIO em V3 - todas as buscas
        devem ser filtradas pela região da query.
        """
        self.regional_filter = regional_filter
        print(f"✅ Filtro regional configurado: {len(regional_filter.documents):,} documentos mapeados")

    def ingest_data(
        self,
        documents: List[Dict],
        embeddings: np.ndarray,
        sparse_embeddings: List[Dict]
    ) -> float:
        """
        Ingere dados no Milvus.
        Retorna tempo de ingestão em segundos.

        IMPORTANTE (V3): Armazena region para filtro prévio.
        """
        print(f"📥 Ingerindo {len(documents):,} documentos no Milvus...")

        batch_size = 1000
        start_time = time.time()

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_sparse = sparse_embeddings[i:i + batch_size] if self.supports_sparse else None

            # Preparar dados
            ids = [doc['id'] for doc in batch_docs]
            dense_vectors = batch_embeddings.tolist()
            doc_types = [doc.get('doc_type', '') for doc in batch_docs]
            # V3: Campo para filtro regional obrigatório
            regions = [doc.get('region', 'UNKNOWN') for doc in batch_docs]
            # V3: Campo para GT-Relevância
            categories = [doc.get('category', 'UNKNOWN') for doc in batch_docs]

            data = [ids, dense_vectors, doc_types, regions, categories]

            # Adicionar sparse se suportado
            if self.supports_sparse and batch_sparse:
                sparse_vectors = []
                for sp in batch_sparse:
                    # Milvus espera dict {index: value}
                    sparse_dict = {int(idx): float(val) for idx, val in zip(sp['indices'], sp['values'])}
                    sparse_vectors.append(sparse_dict)
                data.append(sparse_vectors)

            self.collection.insert(data)

            if (i + batch_size) % 10000 == 0:
                print(f"   {i + batch_size:,} documentos inseridos...")

        # Flush para persistir
        self.collection.flush()

        ingest_time = time.time() - start_time
        print(f"✅ Ingestão concluída em {ingest_time:.2f}s")
        return ingest_time

    def build_index(self) -> float:
        """
        Constrói índice HNSW.
        Retorna tempo de build em segundos.
        """
        print("🔨 Construindo índice HNSW...")

        start_time = time.time()

        # Índice para vetor denso
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {
                "M": self.config.hnsw.m,
                "efConstruction": self.config.hnsw.ef_construction,
            }
        }

        self.collection.create_index(
            field_name="dense",
            index_params=index_params,
        )

        # Índice para sparse se suportado
        if self.supports_sparse:
            sparse_index_params = {
                "metric_type": "IP",  # Inner Product para sparse
                "index_type": "SPARSE_INVERTED_INDEX",
                "params": {"drop_ratio_build": 0.2},
            }
            self.collection.create_index(
                field_name="sparse",
                index_params=sparse_index_params,
            )

        # Carregar collection
        self.collection.load()

        build_time = time.time() - start_time
        print(f"✅ Índice construído em {build_time:.2f}s")
        return build_time

    def search(
        self,
        query_embedding: np.ndarray,
        query_sparse: Optional[Dict],
        top_k: int,
        ef_search: Optional[int] = None,
        query_region: Optional[str] = None
    ) -> Tuple[List[int], float]:
        """
        Busca apenas com vetor denso.

        IMPORTANTE (V3): Se query_region fornecida, aplica filtro regional.
        """
        start_time = time.time()

        # Usar ef_search fornecido ou padrão
        ef = ef_search if ef_search is not None else self._current_ef_search

        # HNSW requer ef >= limit
        effective_ef = max(ef, top_k)

        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": effective_ef},
        }

        # V3: Construir filtro regional OBRIGATÓRIO
        expr = None
        if self.config.data.regional_filter_enabled and query_region:
            expr = create_milvus_filter(query_region)

        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="dense",
            param=search_params,
            limit=top_k,
            expr=expr,  # V3: Filtro regional obrigatório
        )

        latency = (time.time() - start_time) * 1000  # ms
        ids = [hit.id for hit in results[0]]

        return ids, latency

    def hybrid_search(
        self,
        dense_embedding: np.ndarray,
        sparse_embedding: Optional[Dict],
        top_k: int,
        fusion_method: str = "rrf",
        ef_search: Optional[int] = None,
        query_region: Optional[str] = None
    ) -> Tuple[List[int], float]:
        """
        Executa busca híbrida (dense + sparse) com fusão RRF.

        Se Milvus < 2.4, usa apenas busca densa.
        Se Milvus >= 2.4, usa hybrid_search com RRF.

        IMPORTANTE (V3): Filtro regional é OBRIGATÓRIO.
        Todas as buscas devem ser filtradas pela região da query.
        """
        # Usar ef_search fornecido ou padrão
        ef = ef_search if ef_search is not None else self._current_ef_search

        if not self.supports_sparse or sparse_embedding is None:
            return self.search(dense_embedding, None, top_k, ef_search=ef, query_region=query_region)

        start_time = time.time()

        # Busca híbrida com RRF (Milvus 2.4+)
        from pymilvus import AnnSearchRequest, RRFRanker

        # HNSW requer ef >= limit
        prefetch_limit = top_k * 2
        effective_ef = max(ef, prefetch_limit)

        # V3: Filtro regional obrigatório
        expr = None
        if self.config.data.regional_filter_enabled and query_region:
            expr = create_milvus_filter(query_region)

        # Request para vetor denso com ef_search configurável
        dense_search = AnnSearchRequest(
            data=[dense_embedding.tolist()],
            anns_field="dense",
            param={
                "metric_type": "COSINE",
                "params": {"ef": effective_ef},
            },
            limit=prefetch_limit,
            expr=expr,  # V3: Filtro regional obrigatório
        )

        # Request para vetor esparso
        sparse_dict = {int(idx): float(val) for idx, val in zip(
            sparse_embedding['indices'], sparse_embedding['values']
        )}

        sparse_search = AnnSearchRequest(
            data=[sparse_dict],
            anns_field="sparse",
            param={
                "metric_type": "IP",
                "params": {},
            },
            limit=prefetch_limit,
            expr=expr,  # V3: Filtro regional obrigatório
        )

        # Fusão RRF
        ranker = RRFRanker(k=self.config.hybrid.rrf_k)

        results = self.collection.hybrid_search(
            reqs=[dense_search, sparse_search],
            rerank=ranker,
            limit=top_k,
        )

        latency = (time.time() - start_time) * 1000  # ms
        ids = [hit.id for hit in results[0]]

        return ids, latency

    def get_storage_path(self) -> str:
        """Retorna caminho de armazenamento do índice (para medir footprint)."""
        return f"/var/lib/milvus/data/{self.collection_name}"

    def set_ef_search(self, ef_search: int):
        """Define o ef_search atual para os métodos de busca"""
        self._current_ef_search = ef_search

    def cleanup(self):
        """Remove collection de teste"""
        try:
            utility.drop_collection(self.collection_name)
            print(f"🗑️ Collection {self.collection_name} removida")
        except:
            pass

        connections.disconnect("default")


def run_milvus_benchmark(
    config: Config = None,
    ef_search: int = None,
    regional_filter: 'RegionalFilterManager' = None
) -> BenchmarkResult:
    """
    Executa benchmark completo do Milvus.

    V3: Filtro regional é OBRIGATÓRIO.
    """
    if config is None:
        config = get_config()

    benchmark = MilvusBenchmark(config)

    # Carregar dados
    import json
    from pathlib import Path

    data_dir = Path(config.benchmark.data_dir)

    with open(data_dir / "documents.json") as f:
        documents = json.load(f)

    dense_embeddings = np.load(data_dir / "dense_embeddings.npy")

    with open(data_dir / "sparse_embeddings.json") as f:
        sparse_embeddings = json.load(f)

    with open(data_dir / "queries.json") as f:
        queries = json.load(f)

    query_dense = np.load(data_dir / "query_dense.npy")

    with open(data_dir / "query_sparse.json") as f:
        query_sparse = json.load(f)

    with open(data_dir / "ground_truth.json") as f:
        ground_truth = json.load(f)

    # V3: Inicializar filtro regional se não fornecido
    if regional_filter is None and config.data.regional_filter_enabled:
        print("\n🗺️ Inicializando filtro regional...")
        regional_filter = RegionalFilterManager(
            {str(doc.get('doc_id', doc.get('id', ''))): doc for doc in documents},
            config
        )

    # Executar benchmark
    benchmark.connect()

    # V3: Configurar filtro regional no benchmark
    if regional_filter:
        benchmark.set_regional_filter(regional_filter)

    benchmark.create_collection()

    ingest_time = benchmark.ingest_data(documents, dense_embeddings, sparse_embeddings)
    build_time = benchmark.build_index()

    # Definir cenário baseado no suporte
    scenario = "hybrid_search_rrf" if benchmark.supports_sparse else "dense_only_hnsw"

    result = benchmark.run_benchmark(
        queries=queries,
        query_embeddings=query_dense,
        query_sparse=query_sparse if benchmark.supports_sparse else [None] * len(queries),
        ground_truth=ground_truth,
        scenario=scenario,
        num_runs=config.benchmark.num_runs,
        warmup_queries=config.benchmark.warmup_queries,
        ef_search=ef_search,
        regional_filter=regional_filter,  # V3: Filtro regional
    )

    # Adicionar tempos e metadata
    result.ingest_time_seconds = ingest_time
    result.build_index_time_seconds = build_time
    result.num_documents = len(documents)
    result.dense_dim = dense_embeddings.shape[1]
    result.config['milvus_version'] = benchmark.version
    result.config['supports_sparse'] = benchmark.supports_sparse

    print(result.summary())

    # Salvar resultado
    results_dir = Path(config.benchmark.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    ef_display = ef_search if ef_search is not None else config.hnsw.ef_search_values[0]
    result.to_json(results_dir / f"milvus_hybrid_search_ef{ef_display}.json")

    return result


if __name__ == "__main__":
    run_milvus_benchmark()
