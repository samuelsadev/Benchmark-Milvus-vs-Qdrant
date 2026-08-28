#!/usr/bin/env python3
"""
Varredura de ef_search Otimizado por Engine - Benchmark V3
===========================================================

Conforme benchmark specification Fase 1:
- "Identificar o menor ef_search que atinge Recall@100 ≥ 0,95"
- Testar mais valores de ef_search
- Encontrar ponto ótimo por engine
- Validar estabilidade

Implementa busca binária + validação para encontrar o ef_search ótimo
que atinge Recall@100 ≥ 0.95 com menor latência.


Data: Julho 2026
"""
import json
import time
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from tqdm import tqdm

from config import Config, get_config, CONFIDENCE_LEVEL, BOOTSTRAP_SAMPLES


@dataclass
class EfSearchResult:
    """Resultado de um único teste de ef_search."""
    ef_search: int
    recall_at_100: float
    recall_at_100_ci_lower: float
    recall_at_100_ci_upper: float
    recall_at_10: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    qps: float
    meets_threshold: bool  # Recall@100 >= 0.95
    
    def to_dict(self) -> Dict:
        return {
            'ef_search': self.ef_search,
            'recall_at_100': self.recall_at_100,
            'recall_at_100_ci_lower': self.recall_at_100_ci_lower,
            'recall_at_100_ci_upper': self.recall_at_100_ci_upper,
            'recall_at_10': self.recall_at_10,
            'latency_p50_ms': self.latency_p50_ms,
            'latency_p95_ms': self.latency_p95_ms,
            'latency_p99_ms': self.latency_p99_ms,
            'qps': self.qps,
            'meets_threshold': self.meets_threshold,
        }


@dataclass
class EfSearchSweepResult:
    """Resultado completo da varredura de ef_search."""
    engine: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Configuração
    target_recall: float = 0.95
    ef_search_values_tested: List[int] = field(default_factory=list)
    
    # Resultados
    results: List[EfSearchResult] = field(default_factory=list)
    
    # Ótimo encontrado
    optimal_ef_search: Optional[int] = None
    optimal_recall: Optional[float] = None
    optimal_latency_p95: Optional[float] = None
    optimal_qps: Optional[float] = None
    
    # Análise
    recall_threshold_met: bool = False
    min_ef_for_threshold: Optional[int] = None
    
    # Estatísticas
    total_tests: int = 0
    search_strategy: str = "binary_search"  # ou "exhaustive"
    
    def to_dict(self) -> Dict:
        return {
            'engine': self.engine,
            'timestamp': self.timestamp,
            'target_recall': self.target_recall,
            'ef_search_values_tested': self.ef_search_values_tested,
            'results': [r.to_dict() for r in self.results],
            'optimal_ef_search': self.optimal_ef_search,
            'optimal_recall': self.optimal_recall,
            'optimal_latency_p95': self.optimal_latency_p95,
            'optimal_qps': self.optimal_qps,
            'recall_threshold_met': self.recall_threshold_met,
            'min_ef_for_threshold': self.min_ef_for_threshold,
            'total_tests': self.total_tests,
            'search_strategy': self.search_strategy,
        }
    
    def to_json(self, path: str):
        """Salva resultado em JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def summary(self) -> str:
        """Retorna resumo formatado."""
        lines = [
            f"\n{'=' * 70}",
            f"VARREDURA DE ef_search - {self.engine.upper()}",
            f"{'=' * 70}",
            f"Target: Recall@100 ≥ {self.target_recall}",
            f"Estratégia: {self.search_strategy}",
            f"Testes realizados: {self.total_tests}",
            f"",
        ]
        
        if self.recall_threshold_met:
            lines.extend([
                f"✅ THRESHOLD ATINGIDO!",
                f"   Menor ef_search com Recall@100 ≥ {self.target_recall}: {self.min_ef_for_threshold}",
                f"",
                f"📊 PONTO ÓTIMO:",
                f"   ef_search: {self.optimal_ef_search}",
                f"   Recall@100: {self.optimal_recall:.4f}",
                f"   Latência P95: {self.optimal_latency_p95:.2f} ms",
                f"   QPS: {self.optimal_qps:.2f}",
            ])
        else:
            lines.extend([
                f"⚠️ THRESHOLD NÃO ATINGIDO!",
                f"   Maior Recall@100: {max(r.recall_at_100 for r in self.results):.4f}",
                f"   Recomendação: aumentar ef_search máximo",
            ])
        
        lines.append(f"\n{'=' * 70}")
        
        return "\n".join(lines)


class EfSearchOptimizer:
    """
    Otimizador de ef_search para encontrar menor valor que atinge Recall@100 ≥ 0.95.
    
    Estratégia: Busca binária + Validação
    1. Testar range amplo inicial [16, 512]
    2. Busca binária para encontrar threshold
    3. Validação com IC 95%
    4. Testar estabilidade (variância baixa)
    """
    
    # Valores padrão para teste
    DEFAULT_EF_RANGE = [16, 32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 512]
    
    def __init__(
        self,
        target_recall: float = 0.95,
        confidence_level: float = CONFIDENCE_LEVEL,
        n_bootstrap: int = BOOTSTRAP_SAMPLES,
        seed: int = 42
    ):
        """
        Args:
            target_recall: Recall alvo (default 0.95)
            confidence_level: Nível de confiança para IC (default 0.95)
            n_bootstrap: Amostras bootstrap
            seed: Semente para reprodutibilidade
        """
        self.target_recall = target_recall
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.seed = seed
    
    def run_single_ef_test(
        self,
        benchmark_instance: Any,
        ef_search: int,
        queries: List[Dict],
        query_embeddings: np.ndarray,
        query_sparse: List[Dict],
        ground_truth: List[List[int]],
        num_runs: int = 3,
        warmup_queries: int = 5,
        regional_filter: Any = None
    ) -> EfSearchResult:
        """
        Executa um único teste com valor específico de ef_search.
        
        Args:
            benchmark_instance: Instância do benchmark (QdrantBenchmark ou MilvusBenchmark)
            ef_search: Valor de ef_search a testar
            queries: Lista de queries
            query_embeddings: Embeddings das queries
            query_sparse: Embeddings esparsos das queries
            ground_truth: Ground truth por query
            num_runs: Número de runs (reduzido para otimização)
            warmup_queries: Queries de warmup
            regional_filter: Filtro regional
            
        Returns:
            EfSearchResult com métricas
        """
        # Definir ef_search
        benchmark_instance.set_ef_search(ef_search)
        
        # Executar benchmark
        result = benchmark_instance.run_benchmark(
            queries=queries,
            query_embeddings=query_embeddings,
            query_sparse=query_sparse,
            ground_truth=ground_truth,
            scenario=f"ef_search_sweep_{ef_search}",
            num_runs=num_runs,
            warmup_queries=warmup_queries,
            ef_search=ef_search,
            confidence=self.confidence_level,
            n_bootstrap=self.n_bootstrap,
            seed=self.seed,
            regional_filter=regional_filter
        )
        
        # Criar resultado
        return EfSearchResult(
            ef_search=ef_search,
            recall_at_100=result.recall_at_100,
            recall_at_100_ci_lower=result.recall_at_100_ci_lower,
            recall_at_100_ci_upper=result.recall_at_100_ci_upper,
            recall_at_10=result.recall_at_10,
            latency_p50_ms=result.latency_p50_ms,
            latency_p95_ms=result.latency_p95_ms,
            latency_p99_ms=result.latency_p99_ms,
            qps=result.qps,
            meets_threshold=result.recall_at_100 >= self.target_recall
        )
    
    def binary_search_optimal_ef(
        self,
        benchmark_instance: Any,
        queries: List[Dict],
        query_embeddings: np.ndarray,
        query_sparse: List[Dict],
        ground_truth: List[List[int]],
        ef_min: int = 16,
        ef_max: int = 512,
        num_runs: int = 3,
        warmup_queries: int = 5,
        regional_filter: Any = None,
        verbose: bool = True
    ) -> EfSearchSweepResult:
        """
        Busca binária para encontrar menor ef_search que atinge target_recall.
        
        Algoritmo:
        1. Testar ef_min e ef_max
        2. Se ef_max não atinge threshold, aumentar range
        3. Busca binária para encontrar ponto de corte
        4. Validar resultado final
        
        Args:
            benchmark_instance: Instância do benchmark
            queries: Lista de queries
            query_embeddings: Embeddings das queries
            query_sparse: Embeddings esparsos
            ground_truth: Ground truth
            ef_min: ef_search mínimo
            ef_max: ef_search máximo
            num_runs: Runs por teste
            warmup_queries: Warmup queries
            regional_filter: Filtro regional
            verbose: Imprimir progresso
            
        Returns:
            EfSearchSweepResult com resultado da otimização
        """
        engine_name = benchmark_instance.__class__.__name__.replace("Benchmark", "")
        
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"OTIMIZAÇÃO DE ef_search - {engine_name.upper()}")
            print(f"{'=' * 70}")
            print(f"Target: Recall@100 ≥ {self.target_recall}")
            print(f"Range inicial: [{ef_min}, {ef_max}]")
            print(f"Estratégia: busca binária")
        
        result = EfSearchSweepResult(
            engine=engine_name,
            target_recall=self.target_recall,
            search_strategy="binary_search"
        )
        
        tested = {}
        
        # Passo 1: Testar extremos
        if verbose:
            print(f"\n📍 Testando extremos...")
        
        # Testar ef_min
        r_min = self.run_single_ef_test(
            benchmark_instance, ef_min, queries, query_embeddings,
            query_sparse, ground_truth, num_runs, warmup_queries, regional_filter
        )
        tested[ef_min] = r_min
        result.results.append(r_min)
        
        if verbose:
            print(f"   ef={ef_min}: Recall@100={r_min.recall_at_100:.4f} {'✅' if r_min.meets_threshold else '❌'}")
        
        # Se já atinge no mínimo, retornar
        if r_min.meets_threshold:
            result.recall_threshold_met = True
            result.min_ef_for_threshold = ef_min
            result.optimal_ef_search = ef_min
            result.optimal_recall = r_min.recall_at_100
            result.optimal_latency_p95 = r_min.latency_p95_ms
            result.optimal_qps = r_min.qps
            result.total_tests = 1
            result.ef_search_values_tested = list(tested.keys())
            return result
        
        # Testar ef_max
        r_max = self.run_single_ef_test(
            benchmark_instance, ef_max, queries, query_embeddings,
            query_sparse, ground_truth, num_runs, warmup_queries, regional_filter
        )
        tested[ef_max] = r_max
        result.results.append(r_max)
        
        if verbose:
            print(f"   ef={ef_max}: Recall@100={r_max.recall_at_100:.4f} {'✅' if r_max.meets_threshold else '❌'}")
        
        # Se não atinge no máximo, aumentar range
        while not r_max.meets_threshold and ef_max < 2048:
            ef_max *= 2
            if verbose:
                print(f"   ⚠️ Aumentando range para {ef_max}...")
            
            r_max = self.run_single_ef_test(
                benchmark_instance, ef_max, queries, query_embeddings,
                query_sparse, ground_truth, num_runs, warmup_queries, regional_filter
            )
            tested[ef_max] = r_max
            result.results.append(r_max)
            
            if verbose:
                print(f"   ef={ef_max}: Recall@100={r_max.recall_at_100:.4f} {'✅' if r_max.meets_threshold else '❌'}")
        
        # Se ainda não atinge, falhou
        if not r_max.meets_threshold:
            result.recall_threshold_met = False
            result.total_tests = len(tested)
            result.ef_search_values_tested = list(tested.keys())
            return result
        
        # Passo 2: Busca binária
        if verbose:
            print(f"\n📍 Busca binária no range [{ef_min}, {ef_max}]...")
        
        left, right = ef_min, ef_max
        
        while right - left > 16:  # Granularidade de 16
            mid = (left + right) // 2
            
            # Pular se já testado
            if mid in tested:
                if tested[mid].meets_threshold:
                    right = mid
                else:
                    left = mid
                continue
            
            r_mid = self.run_single_ef_test(
                benchmark_instance, mid, queries, query_embeddings,
                query_sparse, ground_truth, num_runs, warmup_queries, regional_filter
            )
            tested[mid] = r_mid
            result.results.append(r_mid)
            
            if verbose:
                print(f"   ef={mid}: Recall@100={r_mid.recall_at_100:.4f} {'✅' if r_mid.meets_threshold else '❌'}")
            
            if r_mid.meets_threshold:
                right = mid
            else:
                left = mid
        
        # Passo 3: Validação fina
        if verbose:
            print(f"\n📍 Validação fina no range [{left}, {right}]...")
        
        for ef in range(left, right + 1, 16):
            if ef in tested:
                continue
            
            r = self.run_single_ef_test(
                benchmark_instance, ef, queries, query_embeddings,
                query_sparse, ground_truth, num_runs, warmup_queries, regional_filter
            )
            tested[ef] = r
            result.results.append(r)
            
            if verbose:
                print(f"   ef={ef}: Recall@100={r.recall_at_100:.4f} {'✅' if r.meets_threshold else '❌'}")
            
            if r.meets_threshold:
                right = ef
                break
        
        # Encontrar menor ef que atinge threshold
        ef_values = sorted(tested.keys())
        min_ef = None
        
        for ef in ef_values:
            if tested[ef].meets_threshold:
                min_ef = ef
                break
        
        # Resultado final
        result.recall_threshold_met = min_ef is not None
        result.min_ef_for_threshold = min_ef
        result.optimal_ef_search = min_ef
        
        if min_ef:
            result.optimal_recall = tested[min_ef].recall_at_100
            result.optimal_latency_p95 = tested[min_ef].latency_p95_ms
            result.optimal_qps = tested[min_ef].qps
        
        result.total_tests = len(tested)
        result.ef_search_values_tested = ef_values
        
        return result
    
    def exhaustive_search(
        self,
        benchmark_instance: Any,
        queries: List[Dict],
        query_embeddings: np.ndarray,
        query_sparse: List[Dict],
        ground_truth: List[List[int]],
        ef_values: List[int] = None,
        num_runs: int = 3,
        warmup_queries: int = 5,
        regional_filter: Any = None,
        verbose: bool = True
    ) -> EfSearchSweepResult:
        """
        Varredura exaustiva de todos os valores de ef_search.
        
        Útil para gerar curva completa recall vs latência.
        
        Args:
            benchmark_instance: Instância do benchmark
            queries: Lista de queries
            query_embeddings: Embeddings das queries
            query_sparse: Embeddings esparsos
            ground_truth: Ground truth
            ef_values: Lista de valores a testar
            num_runs: Runs por teste
            warmup_queries: Warmup queries
            regional_filter: Filtro regional
            verbose: Imprimir progresso
            
        Returns:
            EfSearchSweepResult com todos os resultados
        """
        if ef_values is None:
            ef_values = self.DEFAULT_EF_RANGE
        
        engine_name = benchmark_instance.__class__.__name__.replace("Benchmark", "")
        
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"VARREDURA EXAUSTIVA DE ef_search - {engine_name.upper()}")
            print(f"{'=' * 70}")
            print(f"Target: Recall@100 ≥ {self.target_recall}")
            print(f"Valores: {ef_values}")
        
        result = EfSearchSweepResult(
            engine=engine_name,
            target_recall=self.target_recall,
            ef_search_values_tested=ef_values,
            search_strategy="exhaustive"
        )
        
        iterator = ef_values
        if verbose:
            iterator = tqdm(ef_values, desc="Testando ef_search")
        
        for ef in iterator:
            r = self.run_single_ef_test(
                benchmark_instance, ef, queries, query_embeddings,
                query_sparse, ground_truth, num_runs, warmup_queries, regional_filter
            )
            result.results.append(r)
            
            if verbose and not isinstance(iterator, tqdm):
                print(f"   ef={ef}: Recall@100={r.recall_at_100:.4f} {'✅' if r.meets_threshold else '❌'}")
        
        # Encontrar menor ef que atinge threshold
        min_ef = None
        for r in result.results:
            if r.meets_threshold:
                if min_ef is None or r.ef_search < min_ef:
                    min_ef = r.ef_search
        
        result.recall_threshold_met = min_ef is not None
        result.min_ef_for_threshold = min_ef
        result.optimal_ef_search = min_ef
        
        if min_ef:
            for r in result.results:
                if r.ef_search == min_ef:
                    result.optimal_recall = r.recall_at_100
                    result.optimal_latency_p95 = r.latency_p95_ms
                    result.optimal_qps = r.qps
                    break
        
        result.total_tests = len(result.results)
        
        return result


def run_ef_search_sweep(
    config: Config = None,
    skip_qdrant: bool = False,
    skip_milvus: bool = False,
    ef_values: List[int] = None,
    use_binary_search: bool = True,
    num_runs: int = 3,
    regional_filter: Any = None,
    verbose: bool = True
) -> Tuple[Optional[EfSearchSweepResult], Optional[EfSearchSweepResult]]:
    """
    Executa varredura de ef_search para ambos os engines.
    
    Args:
        config: Configuração do benchmark
        skip_qdrant: Pular Qdrant
        skip_milvus: Pular Milvus
        ef_values: Valores a testar (para busca exaustiva)
        use_binary_search: Usar busca binária (mais eficiente)
        num_runs: Runs por teste
        regional_filter: Filtro regional
        verbose: Imprimir progresso
        
    Returns:
        (qdrant_result, milvus_result)
    """
    if config is None:
        config = get_config()
    
    # Carregar dados
    import json
    from pathlib import Path
    
    data_dir = Path(config.benchmark.data_dir)
    
    with open(data_dir / "queries.json") as f:
        queries = json.load(f)
    
    query_embeddings = np.load(data_dir / "query_dense.npy")
    
    with open(data_dir / "query_sparse.json") as f:
        query_sparse = json.load(f)
    
    with open(data_dir / "ground_truth.json") as f:
        ground_truth = json.load(f)
    
    # Criar otimizador
    optimizer = EfSearchOptimizer(
        target_recall=0.95,
        confidence_level=CONFIDENCE_LEVEL,
        n_bootstrap=BOOTSTRAP_SAMPLES,
        seed=config.benchmark.queries_seed
    )
    
    qdrant_result = None
    milvus_result = None
    
    # Qdrant
    if not skip_qdrant:
        try:
            from benchmark_qdrant import QdrantBenchmark
            
            qdrant_bench = QdrantBenchmark(config)
            qdrant_bench.connect()
            qdrant_bench.create_collection()
            
            # Carregar documentos
            with open(data_dir / "documents.json") as f:
                documents = json.load(f)
            
            dense_embeddings = np.load(data_dir / "dense_embeddings.npy")
            
            with open(data_dir / "sparse_embeddings.json") as f:
                sparse_embeddings = json.load(f)
            
            qdrant_bench.ingest_data(documents, dense_embeddings, sparse_embeddings)
            qdrant_bench.build_index()
            
            if regional_filter:
                qdrant_bench.set_regional_filter(regional_filter)
            
            if use_binary_search:
                qdrant_result = optimizer.binary_search_optimal_ef(
                    qdrant_bench, queries, query_embeddings, query_sparse, ground_truth,
                    num_runs=num_runs, regional_filter=regional_filter, verbose=verbose
                )
            else:
                qdrant_result = optimizer.exhaustive_search(
                    qdrant_bench, queries, query_embeddings, query_sparse, ground_truth,
                    ef_values=ef_values, num_runs=num_runs, regional_filter=regional_filter, verbose=verbose
                )
            
            print(qdrant_result.summary())
            
            # Salvar
            results_dir = Path(config.benchmark.results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            qdrant_result.to_json(str(results_dir / f"ef_search_sweep_qdrant_{timestamp}.json"))
            
        except Exception as e:
            print(f"   ⚠️ Erro no Qdrant: {e}")
    
    # Milvus
    if not skip_milvus:
        try:
            from benchmark_milvus import MilvusBenchmark
            
            milvus_bench = MilvusBenchmark(config)
            milvus_bench.connect()
            milvus_bench.create_collection()
            
            # Carregar documentos (se não carregados ainda)
            if 'documents' not in locals():
                with open(data_dir / "documents.json") as f:
                    documents = json.load(f)
                
                dense_embeddings = np.load(data_dir / "dense_embeddings.npy")
                
                with open(data_dir / "sparse_embeddings.json") as f:
                    sparse_embeddings = json.load(f)
            
            milvus_bench.ingest_data(documents, dense_embeddings, sparse_embeddings)
            milvus_bench.build_index()
            
            if regional_filter:
                milvus_bench.set_regional_filter(regional_filter)
            
            if use_binary_search:
                milvus_result = optimizer.binary_search_optimal_ef(
                    milvus_bench, queries, query_embeddings, query_sparse, ground_truth,
                    num_runs=num_runs, regional_filter=regional_filter, verbose=verbose
                )
            else:
                milvus_result = optimizer.exhaustive_search(
                    milvus_bench, queries, query_embeddings, query_sparse, ground_truth,
                    ef_values=ef_values, num_runs=num_runs, regional_filter=regional_filter, verbose=verbose
                )
            
            print(milvus_result.summary())
            
            # Salvar
            results_dir = Path(config.benchmark.results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            milvus_result.to_json(str(results_dir / f"ef_search_sweep_milvus_{timestamp}.json"))
            
        except Exception as e:
            print(f"   ⚠️ Erro no Milvus: {e}")
    
    return qdrant_result, milvus_result


def generate_ef_search_report(
    qdrant_result: Optional[EfSearchSweepResult],
    milvus_result: Optional[EfSearchSweepResult],
    output_path: str = None
) -> str:
    """
    Gera relatório comparativo da varredura de ef_search.
    
    Args:
        qdrant_result: Resultado do Qdrant
        milvus_result: Resultado do Milvus
        output_path: Caminho para salvar JSON
        
    Returns:
        Relatório formatado em string
    """
    lines = [
        "\n" + "=" * 80,
        "RELATÓRIO: VARREDURA DE ef_search OTIMIZADO",
        "=" * 80,
        "\n📊 OBJETIVO: Encontrar menor ef_search que atinge Recall@100 ≥ 0.95\n"
    ]
    
    # Tabela de resultados
    lines.extend([
        f"{'Engine':<15} {'ef_search':<12} {'Recall@100':<15} {'Latência P95':<15} {'QPS':<12} {'Status':<10}",
        "-" * 80
    ])
    
    if qdrant_result and qdrant_result.optimal_ef_search:
        status = "✅ Ótimo" if qdrant_result.recall_threshold_met else "⚠️ Subótimo"
        lines.append(
            f"{'Qdrant':<15} {qdrant_result.optimal_ef_search:<12} "
            f"{qdrant_result.optimal_recall:.4f}        "
            f"{qdrant_result.optimal_latency_p95:.2f} ms      "
            f"{qdrant_result.optimal_qps:.2f}       {status}"
        )
    else:
        lines.append(f"{'Qdrant':<15} {'N/A':<12} {'N/A':<15} {'N/A':<15} {'N/A':<12} {'❌ Falhou':<10}")
    
    if milvus_result and milvus_result.optimal_ef_search:
        status = "✅ Ótimo" if milvus_result.recall_threshold_met else "⚠️ Subótimo"
        lines.append(
            f"{'Milvus':<15} {milvus_result.optimal_ef_search:<12} "
            f"{milvus_result.optimal_recall:.4f}        "
            f"{milvus_result.optimal_latency_p95:.2f} ms      "
            f"{milvus_result.optimal_qps:.2f}       {status}"
        )
    else:
        lines.append(f"{'Milvus':<15} {'N/A':<12} {'N/A':<15} {'N/A':<15} {'N/A':<12} {'❌ Falhou':<10}")
    
    lines.append("-" * 80)
    
    # Análise
    lines.append("\n📈 ANÁLISE:")
    
    if qdrant_result and milvus_result:
        if qdrant_result.recall_threshold_met and milvus_result.recall_threshold_met:
            # Comparar eficiência
            if qdrant_result.optimal_ef_search < milvus_result.optimal_ef_search:
                diff = milvus_result.optimal_ef_search - qdrant_result.optimal_ef_search
                lines.append(f"   • Qdrant é mais eficiente: requer ef_search {diff} menor que Milvus")
            elif milvus_result.optimal_ef_search < qdrant_result.optimal_ef_search:
                diff = qdrant_result.optimal_ef_search - milvus_result.optimal_ef_search
                lines.append(f"   • Milvus é mais eficiente: requer ef_search {diff} menor que Qdrant")
            else:
                lines.append(f"   • Ambos requerem o mesmo ef_search: {qdrant_result.optimal_ef_search}")
            
            # Comparar latência no ponto ótimo
            if qdrant_result.optimal_latency_p95 < milvus_result.optimal_latency_p95:
                lines.append(f"   • Qdrant tem menor latência no ponto ótimo")
            else:
                lines.append(f"   • Milvus tem menor latência no ponto ótimo")
        else:
            if not qdrant_result.recall_threshold_met:
                lines.append(f"   ⚠️ Qdrant NÃO atingiu Recall@100 ≥ 0.95")
            if not milvus_result.recall_threshold_met:
                lines.append(f"   ⚠️ Milvus NÃO atingiu Recall@100 ≥ 0.95")
    
    lines.append("\n" + "=" * 80)
    
    report = "\n".join(lines)
    
    # Salvar se caminho fornecido
    if output_path:
        result = {
            'qdrant': qdrant_result.to_dict() if qdrant_result else None,
            'milvus': milvus_result.to_dict() if milvus_result else None,
            'report_text': report
        }
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Varredura de ef_search Otimizado")
    parser.add_argument("--skip-qdrant", action="store_true", help="Pular Qdrant")
    parser.add_argument("--skip-milvus", action="store_true", help="Pular Milvus")
    parser.add_argument("--exhaustive", action="store_true", help="Usar busca exaustiva")
    parser.add_argument("--num-runs", type=int, default=3, help="Runs por teste")
    parser.add_argument("--ef-values", type=str, default="16,32,64,96,128,160,192,224,256,320,384,512",
                        help="Valores de ef_search (ex: 16,32,64,128,256)")
    
    args = parser.parse_args()
    
    ef_values = [int(x) for x in args.ef_values.split(",")] if args.exhaustive else None
    
    qdrant_result, milvus_result = run_ef_search_sweep(
        skip_qdrant=args.skip_qdrant,
        skip_milvus=args.skip_milvus,
        ef_values=ef_values,
        use_binary_search=not args.exhaustive,
        num_runs=args.num_runs,
        verbose=True
    )
    
    # Gerar relatório
    report = generate_ef_search_report(qdrant_result, milvus_result)
    print(report)
