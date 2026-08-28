#!/usr/bin/env python3
"""
Filtro Prévio por Região - Benchmark V3
========================================

- O filtro por região é sempre aplicado previamente
- A busca exata é feita dentro do subconjunto da região da query
- Metadados incluem region

O filtro prévio é OBRIGATÓRIO em todas as buscas.
"""
import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

from config import Config, get_config


# Regiões conhecidas
REGIONS = [f"region_{i:02d}" for i in range(1, 13)]  # 12 regiões


@dataclass
class RegionalPartition:
    """Informação sobre uma partição regional."""
    region_id: str
    document_count: int = 0
    doc_ids: List[str] = field(default_factory=list)
    indices: np.ndarray = None  # Índices dos documentos na partição
    dense_embeddings: np.ndarray = None
    sparse_embeddings: List[Dict] = None

    def __post_init__(self):
        if self.indices is not None:
            self.indices = np.array(self.indices, dtype=np.int32)


@dataclass
class RegionalFilterConfig:
    """Configuração do filtro regional."""
    enabled: bool = True
    filter_field: str = "region"
    regions: List[str] = field(default_factory=lambda: REGIONS)

    # Estatísticas
    documents_by_region: Dict[str, int] = field(default_factory=dict)
    queries_by_region: Dict[str, int] = field(default_factory=dict)


class RegionalFilterManager:
    """
    Gerenciador de filtro por região.

    Responsável por:
    - Particionar documentos por região
    - Aplicar filtro prévio nas buscas
    - Calcular ground truth por região
    - Configurar índices para busca filtrada
    """

    def __init__(
        self,
        documents: Dict[str, Dict],
        config: Config = None
    ):
        """
        Inicializa o gerenciador de filtro regional.

        Args:
            documents: Dicionário doc_id -> documento com metadados
            config: Configuração do benchmark
        """
        self.config = config or get_config()
        self.documents = documents
        self.filter_config = RegionalFilterConfig()

        # Mapeamentos
        self.doc_to_region: Dict[str, str] = {}  # doc_id -> região
        self.region_to_docs: Dict[str, List[str]] = defaultdict(list)  # região -> [doc_ids]
        self.doc_to_index: Dict[str, int] = {}  # doc_id -> índice global
        self.region_to_indices: Dict[str, np.ndarray] = {}  # região -> índices

        # Construir mapeamentos
        self._build_mappings()

    def _build_mappings(self):
        """Constrói mapeamentos de documento -> região."""
        print(f"\n   Construindo mapeamentos regionais para {len(self.documents)} documentos...")

        for idx, (doc_nr, doc) in enumerate(self.documents.items()):
            region = doc.get(self.filter_config.filter_field, 'UNKNOWN')

            self.doc_to_region[doc_nr] = region
            self.region_to_docs[region].append(doc_nr)
            self.doc_to_index[doc_nr] = idx

        # Converter para arrays
        for region, docs in self.region_to_docs.items():
            indices = [self.doc_to_index[doc] for doc in docs]
            self.region_to_indices[region] = np.array(indices, dtype=np.int32)

        # Estatísticas
        self.filter_config.documents_by_region = {
            region: len(docs)
            for region, docs in self.region_to_docs.items()
        }

        print(f"   Documentos por região:")
        for region in sorted(self.filter_config.documents_by_region.keys()):
            count = self.filter_config.documents_by_region[region]
            print(f"      {region}: {count:,}")

    def get_region(self, doc_id: str) -> str:
        """Retorna a região de um documento."""
        return self.doc_to_region.get(doc_id, 'UNKNOWN')

    def get_documents_in_region(self, region: str) -> List[str]:
        """Retorna todos os documentos de uma região."""
        return self.region_to_docs.get(region, [])

    def get_indices_in_region(self, region: str) -> np.ndarray:
        """Retorna índices dos documentos de uma região."""
        return self.region_to_indices.get(region, np.array([], dtype=np.int32))

    def get_document_count(self, region: str) -> int:
        """Retorna o número de documentos em uma região."""
        return len(self.region_to_docs.get(region, []))

    def build_filter_for_query(
        self,
        query_doc_id: str
    ) -> Dict:
        """
        Constrói filtro para uma query baseado em sua região.

        O filtro é OBRIGATÓRIO - toda busca deve ser filtrada pela
        região da query.
        """
        region = self.get_region(query_doc_id)

        return {
            'field': self.filter_config.filter_field,
            'value': region,
            'match': 'must'  # Filtro obrigatório
        }

    def filter_embeddings_by_region(
        self,
        embeddings: np.ndarray,
        region: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filtra embeddings para uma região específica.

        Returns:
            (filtered_embeddings, indices) - embeddings filtrados e índices originais
        """
        indices = self.get_indices_in_region(region)

        if len(indices) == 0:
            return np.array([]), np.array([])

        return embeddings[indices], indices

    def filter_sparse_by_region(
        self,
        sparse_embeddings: List[Dict],
        region: str
    ) -> Tuple[List[Dict], np.ndarray]:
        """
        Filtra embeddings esparsos para uma região específica.

        Returns:
            (filtered_sparse, indices) - embeddings esparsos filtrados e índices
        """
        indices = self.get_indices_in_region(region)

        if len(indices) == 0:
            return [], np.array([])

        return [sparse_embeddings[i] for i in indices], indices

    def calculate_regional_ground_truth(
        self,
        query_doc_id: str,
        dense_embeddings: np.ndarray,
        sparse_embeddings: List[Dict],
        query_dense: np.ndarray,
        query_sparse: Dict,
        rrf_weight: Tuple[float, float] = (0.5, 0.5),
        rrf_k: int = 60,
        top_k: int = 100,
        exclude_self: bool = True
    ) -> List[str]:
        """
        Calcula ground truth (busca exata) para uma query DENTRO de sua região.

        Para cada query, roda-se a fórmula real do sistema (denso + esparso + RRF)
        sem nenhuma aproximação, força bruta, comparando contra os documentos
        da região da query, um a um.
        """
        # Obter região da query
        region = self.get_region(query_doc_id)

        # Filtrar embeddings para a região
        regional_dense, regional_indices = self.filter_embeddings_by_region(
            dense_embeddings, region
        )
        regional_sparse, _ = self.filter_sparse_by_region(
            sparse_embeddings, region
        )

        if len(regional_indices) == 0:
            return []

        # Calcular similaridades
        # Denso: cosseno
        query_dense_norm = query_dense / (np.linalg.norm(query_dense) + 1e-10)
        regional_dense_norm = regional_dense / (np.linalg.norm(regional_dense, axis=1, keepdims=True) + 1e-10)
        dense_scores = np.dot(regional_dense_norm, query_dense_norm)

        # Esparso: produto interno
        sparse_scores = np.zeros(len(regional_indices))
        if query_sparse and regional_sparse:
            query_indices = set(query_sparse.get('indices', []))
            query_values = {i: v for i, v in zip(query_sparse.get('indices', []),
                                                  query_sparse.get('values', []))}

            for idx, sp in enumerate(regional_sparse):
                score = 0.0
                for i, v in zip(sp.get('indices', []), sp.get('values', [])):
                    if i in query_indices:
                        score += query_values.get(i, 0) * v
                sparse_scores[idx] = score

        # Calcular ranks
        dense_ranks = np.argsort(-dense_scores) + 1  # 1-indexed
        sparse_ranks = np.argsort(-sparse_scores) + 1

        # Inverter para obter rank de cada documento
        dense_rank_inv = np.zeros(len(dense_ranks), dtype=np.int32)
        sparse_rank_inv = np.zeros(len(sparse_ranks), dtype=np.int32)

        for rank, doc_idx in enumerate(dense_ranks, 1):
            dense_rank_inv[doc_idx - 1] = rank
        for rank, doc_idx in enumerate(sparse_ranks, 1):
            sparse_rank_inv[doc_idx - 1] = rank

        # RRF ponderado
        w_denso, w_esparso = rrf_weight
        rrf_scores = (
            w_denso / (rrf_k + dense_rank_inv) +
            w_esparso / (rrf_k + sparse_rank_inv)
        )

        # Top-k
        top_k_indices = np.argsort(-rrf_scores)[:top_k]

        # Converter para doc_id
        doc_ids = [
            list(self.documents.keys())[regional_indices[i]]
            for i in top_k_indices
        ]

        # Excluir self se necessário
        if exclude_self and query_doc_id in doc_ids:
            doc_ids.remove(query_doc_id)
            # Adicionar próximo
            if len(top_k_indices) < len(rrf_scores):
                next_idx = top_k_indices[-1] + 1
                if next_idx < len(regional_indices):
                    doc_ids.append(
                        list(self.documents.keys())[regional_indices[next_idx]]
                    )

        return doc_ids

    def get_regional_stats(self) -> Dict:
        """Retorna estatísticas regionais."""
        return {
            'total_documents': len(self.documents),
            'documents_by_region': self.filter_config.documents_by_region,
            'num_regions': len(self.filter_config.documents_by_region),
            'filter_field': self.filter_config.filter_field
        }

    def validate_regional_coverage(
        self,
        queries: List[Dict]
    ) -> Dict:
        """Valida cobertura regional das queries."""
        queries_by_region = defaultdict(int)

        for query in queries:
            query_nr = query.get('doc_id', '')
            region = self.get_region(query_nr)
            queries_by_region[region] += 1

        self.filter_config.queries_by_region = dict(queries_by_region)

        return {
            'total_queries': len(queries),
            'queries_by_region': dict(queries_by_region),
            'regions_with_queries': len(queries_by_region),
            'regions_without_queries': len(self.region_to_docs) - len(queries_by_region)
        }


def create_qdrant_filter(region: str) -> Dict:
    """
    Cria filtro para Qdrant no formato correto.
    """
    return {
        'must': [
            {
                'key': 'region',
                'match': {'value': region}
            }
        ]
    }


def create_milvus_filter(region: str) -> str:
    """
    Cria filtro para Milvus no formato correto.
    """
    return f'region == "{region}"'


def load_documents_with_regions(data_path: str) -> Dict[str, Dict]:
    """
    Carrega documentos com campo region.
    """
    import pandas as pd

    data_path = Path(data_path)

    if data_path.suffix == '.parquet':
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    documents = {}

    for _, row in df.iterrows():
        doc_nr = str(row.get('doc_id', ''))
        if doc_nr:
            documents[doc_nr] = {
                'doc_id': doc_nr,
                'region': row.get('region', 'UNKNOWN'),
                'category': row.get('category', 'UNKNOWN'),
            }

    return documents


def test_regional_filter():
    """Testa o filtro regional."""
    print("\n" + "=" * 70)
    print("TESTE: Filtro Prévio por Região")
    print("=" * 70)

    # Criar documentos de teste
    test_docs = {}
    for i in range(100):
        region = f"region_{(i % 10) + 1:02d}"
        test_docs[f"doc_{i}"] = {
            'doc_id': f"doc_{i}",
            'region': region,
            'category': f"tipo_{i % 5}"
        }

    # Criar gerenciador
    manager = RegionalFilterManager(test_docs)

    # Testar mapeamentos
    print("\nEstatísticas regionais:")
    stats = manager.get_regional_stats()
    print(f"   Total de documentos: {stats['total_documents']}")
    print(f"   Regiões: {stats['num_regions']}")

    # Testar filtro
    test_query = "doc_5"
    filter_config = manager.build_filter_for_query(test_query)
    print(f"\nFiltro para {test_query}:")
    print(f"   Região: {filter_config['value']}")

    # Testar índices regionais
    region = filter_config['value']
    indices = manager.get_indices_in_region(region)
    print(f"   Documentos na região: {len(indices)}")

    print("\n✅ Teste passou!")

    return manager


if __name__ == "__main__":
    test_regional_filter()
