#!/usr/bin/env python3
"""
Benchmark do Qdrant para busca híbrida (dense + sparse).

O Qdrant v1.12.0 suporta nativamente:
- Vetores densos com HNSW
- Vetores esparsos
- Busca híbrida com fusão (prefetch + RRF)

IMPORTANTE (V3): Filtro Regional Obrigatório
============================================
- O filtro por região é sempre aplicado previamente
- A busca exata é feita dentro do subconjunto da região da query
- Todas as buscas DEVEM ser filtradas pela região da query
"""
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from config import Config, get_config
from benchmark_base import BaseBenchmark, BenchmarkResult
from regional_filter import RegionalFilterManager, create_qdrant_filter


class QdrantBenchmark(BaseBenchmark):
    """Benchmark para Qdrant com busca híbrida

    IMPORTANTE (V3): Filtro regional é OBRIGATÓRIO em todas as buscas.
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.client: Optional[QdrantClient] = None
        self.collection_name = config.qdrant.collection_name
        self.regional_filter: Optional[RegionalFilterManager] = None
        self._current_ef_search = config.hnsw.ef_search_values[0]

    def connect(self):
        """Conecta ao Qdrant via HTTP/gRPC"""
        print(f"🔗 Conectando ao Qdrant: {self.config.qdrant.host}:{self.config.qdrant.port}")

        self.client = QdrantClient(
            host=self.config.qdrant.host,
            port=self.config.qdrant.port,
            timeout=self.config.qdrant.timeout,
            prefer_grpc=False,  # Usar REST (porta gRPC 6334 não está exposta)
        )

        # Verificar conexão
        collections = self.client.get_collections()
        print(f"✅ Conectado! Collections existentes: {len(collections.collections)}")

        # Verificar se filtro regional está habilitado
        if self.config.data.regional_filter_enabled:
            print("✅ Filtro regional HABILITADO - todas as buscas serão filtradas por região")

    def create_collection(self):
        """Cria collection com suporte a vetores densos e esparsos"""
        print(f"📦 Criando collection: {self.collection_name}")

        # Deletar se existir
        try:
            self.client.delete_collection(self.collection_name)
            print(f"   Collection anterior deletada")
        except:
            pass

        # Criar collection com vetores densos e esparsos
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=self.config.embedding.dense_dim,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=False,
                    )
                )
            },
            hnsw_config=models.HnswConfigDiff(
                m=self.config.hnsw.m,
                ef_construct=self.config.hnsw.ef_construction,
            ),
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=0,  # Indexar imediatamente
            ),
        )

        print(f"✅ Collection criada com vetores densos ({self.config.embedding.dense_dim}d) e esparsos")

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
        Ingere dados no Qdrant.
        Retorna tempo de ingestão em segundos.

        IMPORTANTE (V3): Armazena region no payload para filtro prévio.
        """
        print(f"📥 Ingerindo {len(documents):,} documentos no Qdrant...")

        batch_size = 1000
        start_time = time.time()

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_sparse = sparse_embeddings[i:i + batch_size]

            points = []
            for j, (doc, dense, sparse) in enumerate(zip(batch_docs, batch_embeddings, batch_sparse)):
                point = models.PointStruct(
                    id=doc['id'],
                    vector={
                        "dense": dense.tolist(),
                        "sparse": models.SparseVector(
                            indices=sparse['indices'],
                            values=sparse['values'],
                        )
                    },
                    payload={
                        'text': doc.get('text', '')[:500],  # Limitar payload
                        'doc_type': doc.get('doc_type', ''),
                        'doc_id': doc.get('doc_id', ''),
                        # V3: Campo para filtro regional obrigatório
                        'region': doc.get('region', 'UNKNOWN'),
                        # V3: Campo para GT-Relevância
                        'category': doc.get('category', 'UNKNOWN'),
                    }
                )
                points.append(point)

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

            if (i + batch_size) % 10000 == 0:
                print(f"   {i + batch_size:,} documentos inseridos...")

        ingest_time = time.time() - start_time
        print(f"✅ Ingestão concluída em {ingest_time:.2f}s")
        return ingest_time

    def build_index(self) -> float:
        """
        Aguarda construção do índice HNSW.
        Retorna tempo de build em segundos.
        """
        print("🔨 Aguardando construção do índice...")

        start_time = time.time()

        # Qdrant constrói índice automaticamente, mas podemos forçar otimização
        self.client.update_collection(
            collection_name=self.collection_name,
            optimizer_config=models.OptimizersConfigDiff(
                indexing_threshold=20000,  # Valor padrão
            ),
        )

        # Aguardar até o índice estar pronto
        while True:
            info = self.client.get_collection(self.collection_name)
            if info.status == models.CollectionStatus.GREEN:
                break
            time.sleep(1)

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

        # Usar ef_search fornecido ou padrão do config
        ef = ef_search if ef_search is not None else self._current_ef_search

        # Construir filtro regional se habilitado
        query_filter = None
        if self.config.data.regional_filter_enabled and query_region:
            query_filter = create_qdrant_filter(query_region)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=("dense", query_embedding.tolist()),
            limit=top_k,
            search_params=models.SearchParams(
                hnsw_ef=ef,
            ),
            query_filter=query_filter,  # V3: Filtro regional obrigatório
        )

        latency = (time.time() - start_time) * 1000  # ms
        ids = [point.id for point in results]

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

        O Qdrant usa prefetch para busca multi-estágio:
        1. Prefetch: busca inicial em cada vetor
        2. Fusion: combina resultados via RRF

        IMPORTANTE (V3): Filtro regional é OBRIGATÓRIO.
        Todas as buscas devem ser filtradas pela região da query.
        """
        start_time = time.time()

        # Usar ef_search fornecido ou padrão
        ef = ef_search if ef_search is not None else self._current_ef_search

        if sparse_embedding is None or not sparse_embedding.get('indices'):
            # Fallback para busca densa apenas
            return self.search(dense_embedding, None, top_k, ef_search=ef, query_region=query_region)

        # V3: Construir filtro regional OBRIGATÓRIO
        query_filter = None
        if self.config.data.regional_filter_enabled and query_region:
            query_filter = create_qdrant_filter(query_region)

        # Busca híbrida com prefetch e ef_search configurável
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                # Prefetch denso com ef_search
                models.Prefetch(
                    query=dense_embedding.tolist(),
                    using="dense",
                    limit=top_k * 2,  # Buscar mais para fusão
                    params=models.SearchParams(hnsw_ef=ef),
                ),
                # Prefetch esparso
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_embedding['indices'],
                        values=sparse_embedding['values'],
                    ),
                    using="sparse",
                    limit=top_k * 2,
                ),
            ],
            query=models.FusionQuery(
                fusion=models.Fusion.RRF,  # Reciprocal Rank Fusion
            ),
            limit=top_k,
            query_filter=query_filter,  # V3: Filtro regional OBRIGATÓRIO
        )

        latency = (time.time() - start_time) * 1000  # ms
        ids = [point.id for point in results.points]

        return ids, latency

    def get_storage_path(self) -> str:
        """Retorna caminho de armazenamento do índice (para medir footprint)."""
        return f"/var/lib/qdrant/collections/{self.collection_name}"

    def set_ef_search(self, ef_search: int):
        """Define o ef_search atual para os métodos de busca"""
        self._current_ef_search = ef_search

    def cleanup(self):
        """Remove collection de teste"""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"🗑️ Collection {self.collection_name} removida")
        except:
            pass


def run_qdrant_benchmark(
    config: Config = None,
    ef_search: int = None,
    regional_filter: 'RegionalFilterManager' = None
) -> BenchmarkResult:
    """
    Executa benchmark completo do Qdrant.

    V3: Filtro regional é OBRIGATÓRIO.
    """
    if config is None:
        config = get_config()

    benchmark = QdrantBenchmark(config)

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

    result = benchmark.run_benchmark(
        queries=queries,
        query_embeddings=query_dense,
        query_sparse=query_sparse,
        ground_truth=ground_truth,
        scenario="hybrid_search_rrf",
        num_runs=config.benchmark.num_runs,
        warmup_queries=config.benchmark.warmup_queries,
        ef_search=ef_search,
        regional_filter=regional_filter,  # V3: Filtro regional
    )

    # Adicionar tempos
    result.ingest_time_seconds = ingest_time
    result.build_index_time_seconds = build_time
    result.num_documents = len(documents)
    result.dense_dim = dense_embeddings.shape[1]

    print(result.summary())

    # Salvar resultado
    results_dir = Path(config.benchmark.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    ef_display = ef_search if ef_search is not None else config.hnsw.ef_search_values[0]
    result.to_json(results_dir / f"qdrant_hybrid_search_ef{ef_display}.json")

    return result


if __name__ == "__main__":
    run_qdrant_benchmark()
