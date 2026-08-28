#!/usr/bin/env python3
"""
GT-Relevância por category - Benchmark V3
=====================================================

Conforme benchmark specification Seção 5.1.2:
"GT-Relevância (binária, para Precision@10 e NDCG@10)"

Pergunta que responde: o top-10 final (após o ColBERT) está preenchido 
com manifestações do mesmo assunto que a query, na taxonomia do sistema 
(category)?

Relevância binária: um item do top-10 é relevante (1) se tem o mesmo 
category da query; senão, 0. Sem graduação por hierarquia.

Métricas:
• Precision@10 = (nº de itens com o mesmo nível_2) / 10
• NDCG@10 = usa os mesmos rótulos binários, mas credita colocar os 
  relevantes nas primeiras posições. Mede a qualidade de ordenação 
  do ColBERT.
"""
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

from config import Config, get_config


@dataclass
class RelevanceResult:
    """Resultado de GT-Relevância para uma query."""
    query_doc_id: str
    query_category: str
    top_10_doc_ids: List[str]
    top_10_tipos_servico: List[str]
    
    # Relevância binária por item
    relevance_labels: List[int]  # 1 se mesmo category, 0 caso contrário
    
    # Métricas
    precision_at_10: float
    ndcg_at_10: float
    
    # Contagem
    relevant_count: int  # Número de itens relevantes no top-10
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class GTRelevanceReport:
    """Relatório completo de GT-Relevância."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    num_queries: int = 0
    total_relevant_items: int = 0
    
    # Métricas agregadas
    mean_precision_at_10: float = 0.0
    mean_ndcg_at_10: float = 0.0
    
    # IC 95%
    precision_ci_lower: float = 0.0
    precision_ci_upper: float = 0.0
    ndcg_ci_lower: float = 0.0
    ndcg_ci_upper: float = 0.0
    
    # Por category
    metrics_by_tipo: Dict[str, Dict] = field(default_factory=dict)
    
    # Resultados individuais
    results: List[RelevanceResult] = field(default_factory=list)
    
    # Distribuição de tipos
    tipo_distribution: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "num_queries": self.num_queries,
            "total_relevant_items": self.total_relevant_items,
            "mean_precision_at_10": self.mean_precision_at_10,
            "mean_ndcg_at_10": self.mean_ndcg_at_10,
            "precision_ci": [self.precision_ci_lower, self.precision_ci_upper],
            "ndcg_ci": [self.ndcg_ci_lower, self.ndcg_ci_upper],
            "metrics_by_tipo": self.metrics_by_tipo,
            "results": [r.to_dict() for r in self.results],
            "tipo_distribution": self.tipo_distribution
        }
    
    def to_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def summary(self) -> str:
        """Retorna resumo formatado."""
        lines = [
            "",
            "=" * 80,
            "GT-RELEVÂNCIA POR category",
            "=" * 80,
            f"Timestamp: {self.timestamp}",
            f"Queries avaliadas: {self.num_queries}",
            f"Total de itens relevantes: {self.total_relevant_items}",
            "",
            "MÉTRICAS AGREGADAS:",
            "-" * 80,
            f"  Precision@10: {self.mean_precision_at_10:.4f} IC [{self.precision_ci_lower:.4f}, {self.precision_ci_upper:.4f}]",
            f"  NDCG@10:      {self.mean_ndcg_at_10:.4f} IC [{self.ndcg_ci_lower:.4f}, {self.ndcg_ci_upper:.4f}]",
            "",
        ]
        
        if self.metrics_by_tipo:
            lines.append("MÉTRICAS POR category:")
            lines.append("-" * 80)
            lines.append(f"  {'Tipo':<40} {'Count':>8} {'P@10':>10} {'NDCG@10':>10}")
            lines.append("  " + "-" * 70)
            
            for tipo, metrics in sorted(self.metrics_by_tipo.items(), key=lambda x: -x[1]['count']):
                lines.append(f"  {tipo:<40} {metrics['count']:>8} {metrics['precision_at_10']:>10.4f} {metrics['ndcg_at_10']:>10.4f}")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)


class GTRelevanceEvaluator:
    """
    Avaliador de GT-Relevância.
    
    Avalia se o top-10 final (pós-reranking) está preenchido com 
    manifestações do mesmo category da query.
    """
    
    def __init__(self, documents: Dict[str, Dict], config: Config = None):
        """
        Inicializa o avaliador.
        
        Args:
            documents: Dicionário doc_id -> documento com metadados
            config: Configuração do benchmark
        """
        self.config = config or get_config()
        self.documents = documents
        
        # Mapear doc_id -> category
        self.tipo_servico_map: Dict[str, str] = {}
        
        for doc_nr, doc in documents.items():
            tipo = doc.get('category', 'UNKNOWN')
            self.tipo_servico_map[doc_nr] = tipo
        
        print(f"   GT-Relevância: {len(self.tipo_servico_map)} documentos indexados")
    
    def get_tipo_servico(self, doc_id: str) -> str:
        """Retorna o category de uma manifestação."""
        return self.tipo_servico_map.get(doc_id, 'UNKNOWN')
    
    def is_relevant(self, query_tipo: str, doc_tipo: str) -> bool:
        """
        Determina se um documento é relevante para uma query.
        
        Relevância binária: documento é relevante se tem o mesmo 
        category da query.
        """
        return query_tipo == doc_tipo and query_tipo != 'UNKNOWN'
    
    def calculate_precision_at_10(self, relevance_labels: List[int]) -> float:
        """
        Calcula Precision@10.
        
        Precision@10 = (nº de itens relevantes) / 10
        """
        return sum(relevance_labels) / 10.0
    
    def calculate_ndcg_at_10(self, relevance_labels: List[int]) -> float:
        """
        Calcula NDCG@10.
        
        NDCG = DCG / IDCG
        DCG = Σ (rel_i / log2(i+2))
        IDCG = DCG ideal (todos os relevantes nas primeiras posições)
        """
        def dcg(scores: List[int]) -> float:
            return sum(score / np.log2(i + 2) for i, score in enumerate(scores))
        
        # DCG do ranking atual
        dcg_value = dcg(relevance_labels)
        
        # IDCG: melhor caso (todos os relevantes primeiro)
        num_relevant = sum(relevance_labels)
        ideal_scores = [1] * min(num_relevant, 10) + [0] * max(0, 10 - num_relevant)
        idcg_value = dcg(ideal_scores)
        
        if idcg_value == 0:
            return 0.0
        
        return dcg_value / idcg_value
    
    def evaluate_query(
        self,
        query_doc_id: str,
        top_10_doc_ids: List[str]
    ) -> RelevanceResult:
        """
        Avalia uma única query.
        
        Args:
            query_doc_id: ID da query
            top_10_doc_ids: Top-10 documentos retornados (pós-reranking)
            
        Returns:
            RelevanceResult com métricas
        """
        # Tipo da query
        query_tipo = self.get_tipo_servico(query_doc_id)
        
        # Tipos dos documentos retornados
        top_10_tipos = [self.get_tipo_servico(doc_nr) for doc_nr in top_10_doc_ids]
        
        # Relevância binária
        relevance_labels = [
            1 if self.is_relevant(query_tipo, doc_tipo) else 0
            for doc_tipo in top_10_tipos
        ]
        
        # Métricas
        precision = self.calculate_precision_at_10(relevance_labels)
        ndcg = self.calculate_ndcg_at_10(relevance_labels)
        
        return RelevanceResult(
            query_doc_id=query_doc_id,
            query_category=query_tipo,
            top_10_doc_ids=top_10_doc_ids,
            top_10_tipos_servico=top_10_tipos,
            relevance_labels=relevance_labels,
            precision_at_10=precision,
            ndcg_at_10=ndcg,
            relevant_count=sum(relevance_labels)
        )
    
    def evaluate_all(
        self,
        queries: List[Dict],
        search_results: Dict[str, List[str]],
        bootstrap_samples: int = 1000,
        seed: int = 42
    ) -> GTRelevanceReport:
        """
        Avalia todas as queries.
        
        Args:
            queries: Lista de queries (cada uma com doc_id)
            search_results: Dicionário query_doc_id -> top_10_doc_ids
            bootstrap_samples: Número de amostras para IC 95%
            seed: Semente para reprodutibilidade
            
        Returns:
            GTRelevanceReport completo
        """
        print(f"\n   Avaliando GT-Relevância para {len(queries)} queries...")
        
        report = GTRelevanceReport(num_queries=len(queries))
        
        # Métricas por query
        precisions = []
        ndcgs = []
        
        # Métricas por tipo
        tipo_metrics = defaultdict(lambda: {'precisions': [], 'ndcgs': []})
        tipo_counts = defaultdict(int)
        
        for query in queries:
            query_nr = str(query.get('doc_id', query.get('id', '')))
            
            if query_nr not in search_results:
                continue
            
            top_10 = search_results[query_nr]
            
            if len(top_10) < 10:
                # Preencher com vazios se necessário
                top_10 = top_10 + [''] * (10 - len(top_10))
            
            result = self.evaluate_query(query_nr, top_10[:10])
            
            report.results.append(result)
            precisions.append(result.precision_at_10)
            ndcgs.append(result.ndcg_at_10)
            
            # Agregar por tipo
            query_tipo = result.query_category
            tipo_metrics[query_tipo]['precisions'].append(result.precision_at_10)
            tipo_metrics[query_tipo]['ndcgs'].append(result.ndcg_at_10)
            tipo_counts[query_tipo] += 1
        
        # Calcular médias
        if precisions:
            report.mean_precision_at_10 = np.mean(precisions)
            report.mean_ndcg_at_10 = np.mean(ndcgs)
            report.total_relevant_items = sum(r.relevant_count for r in report.results)
            
            # IC 95% via bootstrap
            np.random.seed(seed)
            
            precision_bootstrap = []
            ndcg_bootstrap = []
            
            for _ in range(bootstrap_samples):
                idx = np.random.choice(len(precisions), size=len(precisions), replace=True)
                precision_bootstrap.append(np.mean(np.array(precisions)[idx]))
                ndcg_bootstrap.append(np.mean(np.array(ndcgs)[idx]))
            
            report.precision_ci_lower = np.percentile(precision_bootstrap, 2.5)
            report.precision_ci_upper = np.percentile(precision_bootstrap, 97.5)
            report.ndcg_ci_lower = np.percentile(ndcg_bootstrap, 2.5)
            report.ndcg_ci_upper = np.percentile(ndcg_bootstrap, 97.5)
            
            # Métricas por tipo
            for tipo, metrics in tipo_metrics.items():
                report.metrics_by_tipo[tipo] = {
                    'count': len(metrics['precisions']),
                    'precision_at_10': np.mean(metrics['precisions']),
                    'ndcg_at_10': np.mean(metrics['ndcgs'])
                }
            
            report.tipo_distribution = dict(tipo_counts)
        
        print(f"   Precision@10 médio: {report.mean_precision_at_10:.4f}")
        print(f"   NDCG@10 médio: {report.mean_ndcg_at_10:.4f}")
        
        return report


def generate_conference_spreadsheet(
    queries: List[Dict],
    search_results: Dict[str, List[str]],
    documents: Dict[str, Dict],
    output_path: str,
    sample_size: int = 100,
    seed: int = 42
) -> str:
    """
    Gera planilha .xlsx para conferência humana.
    
    Conforme benchmark specification Fase 3:
    "Planilha de conferência humana: para cada simulação, gera-se um .xlsx 
    com uma linha por query contendo o doc_id da query seguido 
    dos 10 doc_id do top-10 final reordenado. Amostra de 100 
    queries por simulação, para conferência por amostragem pelos analistas 
    da sistema de atendimento."
    
    Args:
        queries: Lista de queries
        search_results: Dicionário query_nr -> top_10_docs
        documents: Dicionário doc_nr -> documento
        output_path: Caminho do arquivo .xlsx
        sample_size: Número de queries na amostra
        seed: Semente para reprodutibilidade
        
    Returns:
        Caminho do arquivo gerado
    """
    try:
        import pandas as pd
    except ImportError:
        print("   ⚠️ pandas não instalado. Pulando geração de planilha.")
        return None
    
    np.random.seed(seed)
    
    # Amostrar queries
    if len(queries) > sample_size:
        sample_indices = np.random.choice(len(queries), size=sample_size, replace=False)
        sample_queries = [queries[i] for i in sample_indices]
    else:
        sample_queries = queries
    
    # Construir dados da planilha
    rows = []
    
    for query in sample_queries:
        query_nr = str(query.get('doc_id', query.get('id', '')))
        query_tipo = documents.get(query_nr, {}).get('category', 'UNKNOWN')
        
        top_10 = search_results.get(query_nr, [''] * 10)
        
        row = {
            'query_doc_id': query_nr,
            'query_category': query_tipo
        }
        
        for i, doc_nr in enumerate(top_10[:10], 1):
            row[f'top10_{i:02d}_doc_id'] = doc_nr
            row[f'top10_{i:02d}_category'] = documents.get(doc_nr, {}).get('category', 'UNKNOWN')
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Salvar
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_excel(output_path, index=False)
    
    print(f"   📊 Planilha de conferência gerada: {output_path}")
    print(f"      Queries na amostra: {len(rows)}")
    
    return str(output_path)


def load_documents_with_tipo_servico(data_path: str) -> Dict[str, Dict]:
    """
    Carrega documentos com campo category.
    
    Args:
        data_path: Caminho para o arquivo de documentos
        
    Returns:
        Dicionário doc_id -> documento
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
                'category': row.get('category', 'UNKNOWN'),
                'region': row.get('region', 'UNKNOWN'),
                'category_l1': row.get('category_l1', 'UNKNOWN'),
                'category_l3': row.get('category_l3', 'UNKNOWN'),
            }
    
    return documents


def evaluate_gt_relevance(
    queries_path: str,
    search_results_path: str,
    documents_path: str,
    output_dir: str,
    sample_size: int = 100,
    seed: int = 42
) -> GTRelevanceReport:
    """
    Função principal para avaliar GT-Relevância.
    
    Args:
        queries_path: Caminho para arquivo de queries
        search_results_path: Caminho para arquivo de resultados de busca
        documents_path: Caminho para arquivo de documentos
        output_dir: Diretório de saída
        sample_size: Tamanho da amostra para conferência
        seed: Semente para reprodutibilidade
        
    Returns:
        GTRelevanceReport
    """
    print("\n" + "=" * 80)
    print("GT-RELEVÂNCIA POR category")
    print("=" * 80)
    
    # Carregar dados
    print("\n   Carregando dados...")
    
    with open(queries_path) as f:
        queries = json.load(f)
    
    with open(search_results_path) as f:
        search_results = json.load(f)
    
    documents = load_documents_with_tipo_servico(documents_path)
    
    print(f"   Queries: {len(queries)}")
    print(f"   Resultados de busca: {len(search_results)}")
    print(f"   Documentos: {len(documents)}")
    
    # Criar avaliador
    evaluator = GTRelevanceEvaluator(documents)
    
    # Avaliar
    report = evaluator.evaluate_all(queries, search_results)
    
    # Gerar planilha de conferência
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    spreadsheet_path = output_dir / f"conferencia_gt_relevance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    generate_conference_spreadsheet(
        queries=queries,
        search_results=search_results,
        documents=documents,
        output_path=str(spreadsheet_path),
        sample_size=sample_size,
        seed=seed
    )
    
    # Salvar relatório
    report_path = output_dir / f"gt_relevance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report.to_json(str(report_path))
    
    print(report.summary())
    
    return report


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("GT-RELEVÂNCIA POR category")
    print("=" * 80)
    print("\nConforme benchmark specification Seção 5.1.2:")
    print("\nRelevância binária: um item do top-10 é relevante (1) se tem")
    print("o mesmo category da query; senão, 0.")
    print("\nMétricas:")
    print("  • Precision@10 = (nº de itens com mesmo nível_2) / 10")
    print("  • NDCG@10 = qualidade da ordenação do ColBERT")
    print()
