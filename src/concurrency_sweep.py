#!/usr/bin/env python3
"""
Varredura de Concorrência para QPS de Saturação - Benchmark V3
===============================================================

Implementa teste de estresse com clientes concorrentes para identificar
o ponto de saturação do banco vetorial (QPS máximo).

Conforme especificação V3 2.2.3:
- Testar com 1, 2, 4, 8, 16, 32, 64 clientes concorrentes
- Medir QPS de saturação real
- Identificar ponto onde QPS para de crescer ou latência explode

Metodologia:
- ThreadPoolExecutor para simular clientes concorrentes
- Cada "cliente" executa queries sequencialmente
- Coleta QPS, latência P50, P95, P99 por nível de concorrência
- Detecta saturação automaticamente
"""
import json
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any
from statistics import mean, stdev
import threading

from config import Config, get_config


@dataclass
class ConcurrencyResult:
    """Resultado de um único nível de concorrência."""
    num_clients: int
    total_queries: int
    total_time_seconds: float
    qps: float
    qps_std: float = 0.0
    latency_mean_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    errors: int = 0
    error_rate: float = 0.0
    
    # IC 95% para QPS
    qps_ci_lower: float = 0.0
    qps_ci_upper: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def summary(self) -> str:
        """Retorna resumo formatado."""
        return f"""
  {self.num_clients:2d} clientes: QPS={self.qps:8.2f} ± {self.qps_std:6.2f} | 
  Latência: {self.latency_mean_ms:6.2f}ms (P95={self.latency_p95_ms:6.2f}ms) | 
  Erros: {self.error_rate:.2%}"""


@dataclass
class ConcurrencySweepResult:
    """Resultado completo da varredura de concorrência."""
    engine: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Níveis de concorrência testados
    client_levels: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64])
    
    # Resultados por nível
    results: List[ConcurrencyResult] = field(default_factory=list)
    
    # Análise de saturação
    optimal_clients: int = 0
    optimal_qps: float = 0.0
    saturation_detected: bool = False
    saturation_clients: Optional[int] = None
    saturation_reason: str = ""
    
    # Configuração
    queries_per_level: int = 100
    warmup_queries: int = 10
    cooldown_seconds: int = 5
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'results': [r.to_dict() for r in self.results]
        }
    
    def to_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def summary(self) -> str:
        """Retorna resumo completo da varredura."""
        lines = [
            f"\n{'='*70}",
            f"VARREDURA DE CONCORRÊNCIA - {self.engine.upper()}",
            f"{'='*70}",
            f"Timestamp: {self.timestamp}",
            f"Níveis testados: {self.client_levels}",
            f"Queries por nível: {self.queries_per_level}",
            f"",
            f"RESULTADOS POR NÍVEL:",
            f"{'-'*70}",
        ]
        
        for r in self.results:
            lines.append(f"  {r.num_clients:2d} clientes: QPS={r.qps:8.2f} ± {r.qps_std:6.2f} | "
                        f"Latência: {r.latency_mean_ms:6.2f}ms (P95={r.latency_p95_ms:6.2f}ms) | "
                        f"Erros: {r.error_rate:.2%}")
        
        lines.extend([
            f"{'-'*70}",
            f"",
            f"ANÁLISE DE SATURAÇÃO:",
            f"  Cliente ótimo: {self.optimal_clients}",
            f"  QPS ótimo: {self.optimal_qps:.2f}",
            f"  Saturação detectada: {self.saturation_detected}",
        ])
        
        if self.saturation_detected:
            lines.append(f"  Ponto de saturação: {self.saturation_clients} clientes")
            lines.append(f"  Razão: {self.saturation_reason}")
        
        lines.append(f"{'='*70}")
        
        return '\n'.join(lines)


class ConcurrencySweep:
    """
    Executor de varredura de concorrência.
    
    Testa múltiplos níveis de concorrência para encontrar o QPS máximo
    sustentável pelo banco vetorial.
    """
    
    def __init__(
        self,
        search_func: Callable,
        engine_name: str = "unknown",
        config: Config = None
    ):
        """
        Inicializa a varredura.
        
        Args:
            search_func: Função de busca que recebe (query_embedding, query_sparse, top_k)
                        e retorna (ids, latency_ms)
            engine_name: Nome do engine para relatório
            config: Configuração do benchmark
        """
        self.search_func = search_func
        self.engine_name = engine_name
        self.config = config or get_config()
        
        # Lock para operações thread-safe
        self._lock = threading.Lock()
        
        # Métricas coletadas durante execução
        self._latencies: List[float] = []
        self._errors: int = 0
        self._queries_completed: int = 0
    
    def _execute_single_query(
        self,
        query_embedding: np.ndarray,
        query_sparse: Optional[Dict],
        top_k: int
    ) -> Tuple[Optional[List[int]], float]:
        """
        Executa uma única query e retorna (ids, latency_ms).
        
        Trata erros internamente para não afetar outras threads.
        """
        try:
            ids, latency_ms = self.search_func(
                query_embedding,
                query_sparse,
                top_k
            )
            return ids, latency_ms
        except Exception as e:
            with self._lock:
                self._errors += 1
            return None, 0.0
    
    def _run_single_client(
        self,
        query_embeddings: np.ndarray,
        query_sparse: List[Optional[Dict]],
        top_k: int,
        num_queries: int,
        client_id: int
    ) -> List[float]:
        """
        Executa queries sequencialmente como um único cliente.
        
        Args:
            query_embeddings: Embeddings das queries
            query_sparse: Embeddings esparsos das queries
            top_k: Número de resultados
            num_queries: Número de queries a executar
            client_id: ID  (para debugging)
            
        Returns:
            Lista de latências em ms
        """
        latencies = []
        
        for i in range(num_queries):
            idx = i % len(query_embeddings)
            
            ids, latency_ms = self._execute_single_query(
                query_embeddings[idx],
                query_sparse[idx] if query_sparse else None,
                top_k
            )
            
            if ids is not None:
                latencies.append(latency_ms)
        
        return latencies
    
    def _run_concurrent_clients(
        self,
        query_embeddings: np.ndarray,
        query_sparse: List[Optional[Dict]],
        top_k: int,
        num_queries_per_client: int,
        num_clients: int
    ) -> Tuple[List[float], int, int]:
        """
        Executa múltiplos clientes concorrentes.
        
        Args:
            query_embeddings: Embeddings das queries
            query_sparse: Embeddings esparsos das queries
            top_k: Número de resultados
            num_queries_per_client: Queries por cliente
            num_clients: Número de clientes concorrentes
            
        Returns:
            (latências, total_queries, erros)
        """
        # Resetar métricas
        with self._lock:
            self._latencies = []
            self._errors = 0
            self._queries_completed = 0
        
        all_latencies = []
        total_queries = num_clients * num_queries_per_client
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_clients) as executor:
            futures = []
            
            for client_id in range(num_clients):
                future = executor.submit(
                    self._run_single_client,
                    query_embeddings,
                    query_sparse,
                    top_k,
                    num_queries_per_client,
                    client_id
                )
                futures.append(future)
            
            # Coletar resultados
            for future in as_completed(futures):
                try:
                    client_latencies = future.result()
                    all_latencies.extend(client_latencies)
                except Exception as e:
                    pass
        
        total_time = time.time() - start_time
        
        return all_latencies, total_queries, total_time
    
    def measure_level(
        self,
        query_embeddings: np.ndarray,
        query_sparse: List[Optional[Dict]],
        top_k: int,
        num_queries_per_client: int,
        num_clients: int,
        num_runs: int = 3,
        warmup: bool = True
    ) -> ConcurrencyResult:
        """
        Medir QPS e latência para um nível específico de concorrência.
        
        Args:
            query_embeddings: Embeddings das queries
            query_sparse: Embeddings esparsos das queries
            top_k: Número de resultados
            num_queries_per_client: Queries por cliente
            num_clients: Número de clientes concorrentes
            num_runs: Número de runs para média
            warmup: Se deve executar warmup antes
            
        Returns:
            ConcurrencyResult com métricas
        """
        print(f"   Testando {num_clients} cliente(s)...")
        
        # Warmup
        if warmup:
            warmup_queries = min(10, num_queries_per_client)
            self._run_concurrent_clients(
                query_embeddings,
                query_sparse,
                top_k,
                warmup_queries,
                num_clients
            )
            time.sleep(1)
        
        # Múltiplas runs
        all_qps = []
        all_latencies = []
        total_errors = 0
        
        for run in range(num_runs):
            latencies, total_queries, total_time = self._run_concurrent_clients(
                query_embeddings,
                query_sparse,
                top_k,
                num_queries_per_client,
                num_clients
            )
            
            with self._lock:
                errors = self._errors
                self._errors = 0
            
            qps = total_queries / total_time if total_time > 0 else 0
            all_qps.append(qps)
            all_latencies.extend(latencies)
            total_errors += errors
            
            time.sleep(self.config.benchmark.cooldown_seconds)
        
        # Calcular estatísticas
        if not all_latencies:
            return ConcurrencyResult(
                num_clients=num_clients,
                total_queries=0,
                total_time_seconds=0,
                qps=0,
                errors=total_errors,
                error_rate=1.0
            )
        
        latencies_arr = np.array(all_latencies)
        qps_arr = np.array(all_qps)
        
        # IC 95% para QPS
        if len(qps_arr) >= 3:
            qps_ci_lower = np.percentile(qps_arr, 2.5)
            qps_ci_upper = np.percentile(qps_arr, 97.5)
        else:
            qps_ci_lower = qps_arr.min()
            qps_ci_upper = qps_arr.max()
        
        error_rate = total_errors / (num_runs * num_clients * num_queries_per_client)
        
        return ConcurrencyResult(
            num_clients=num_clients,
            total_queries=len(all_latencies),
            total_time_seconds=len(all_latencies) / mean(all_qps),
            qps=mean(all_qps),
            qps_std=stdev(all_qps) if len(all_qps) > 1 else 0,
            latency_mean_ms=mean(all_latencies),
            latency_p50_ms=np.percentile(latencies_arr, 50),
            latency_p95_ms=np.percentile(latencies_arr, 95),
            latency_p99_ms=np.percentile(latencies_arr, 99),
            errors=total_errors,
            error_rate=error_rate,
            qps_ci_lower=qps_ci_lower,
            qps_ci_upper=qps_ci_upper
        )
    
    def detect_saturation(
        self,
        results: List[ConcurrencyResult]
    ) -> Tuple[int, float, bool, Optional[int], str]:
        """
        Detecta ponto de saturação baseado nos resultados.
        
        Critérios de saturação:
        1. QPS para de crescer (incremento < 5%)
        2. Latência explode (aumenta > 50%)
        3. Taxa de erro aumenta significativamente (> 1%)
        
        Args:
            results: Resultados por nível de concorrência
            
        Returns:
            (optimal_clients, optimal_qps, saturation_detected, 
             saturation_clients, saturation_reason)
        """
        if not results:
            return 0, 0, False, None, "Sem resultados"
        
        # Encontrar QPS máximo
        max_qps = 0
        max_qps_idx = 0
        
        for i, r in enumerate(results):
            if r.qps > max_qps:
                max_qps = r.qps
                max_qps_idx = i
        
        optimal_result = results[max_qps_idx]
        optimal_clients = optimal_result.num_clients
        optimal_qps = optimal_result.qps
        
        # Detectar saturação
        saturation_detected = False
        saturation_clients = None
        saturation_reason = ""
        
        # Verificar se QPS parou de crescer
        for i in range(1, len(results)):
            prev = results[i-1]
            curr = results[i]
            
            # Taxa de erro alta
            if curr.error_rate > 0.05:  # > 5% erros
                saturation_detected = True
                saturation_clients = curr.num_clients
                saturation_reason = f"Taxa de erro alta ({curr.error_rate:.1%})"
                break
            
            # Latência explodiu
            if prev.latency_mean_ms > 0 and curr.latency_mean_ms > prev.latency_mean_ms * 2:
                saturation_detected = True
                saturation_clients = curr.num_clients
                saturation_reason = f"Latência dobrou ({prev.latency_mean_ms:.1f}ms -> {curr.latency_mean_ms:.1f}ms)"
                break
            
            # QPS parou de crescer ou diminuiu
            if curr.qps < prev.qps * 1.05:  # Crescimento < 5%
                # Verificar se não é apenas variabilidade
                if curr.qps < prev.qps * 0.95:  # QPS caiu > 5%
                    saturation_detected = True
                    saturation_clients = curr.num_clients
                    saturation_reason = f"QPS diminuiu ({prev.qps:.1f} -> {curr.qps:.1f})"
                    break
        
        return (
            optimal_clients,
            optimal_qps,
            saturation_detected,
            saturation_clients,
            saturation_reason
        )
    
    def run_sweep(
        self,
        query_embeddings: np.ndarray,
        query_sparse: List[Optional[Dict]],
        top_k: int = 100,
        client_levels: List[int] = None,
        queries_per_client: int = 100,
        num_runs_per_level: int = 3,
        warmup: bool = True,
        ef_search: Optional[int] = None
    ) -> ConcurrencySweepResult:
        """
        Executa varredura completa de concorrência.
        
        Args:
            query_embeddings: Embeddings das queries
            query_sparse: Embeddings esparsos das queries
            top_k: Número de resultados
            client_levels: Níveis de concorrência a testar
            queries_per_client: Queries por cliente em cada nível
            num_runs_per_level: Runs por nível para média
            warmup: Se deve executar warmup antes de cada nível
            
        Returns:
            ConcurrencySweepResult completo
        """
        if client_levels is None:
            # Conforme benchmark specification Fase 5: 1, 4, 8, 16, 32, 64
            # (não inclui 2 clientes)
            client_levels = [1, 4, 8, 16, 32, 64]
        
        print(f"\n{'='*70}")
        print(f"VARREDURA DE CONCORRÊNCIA - {self.engine_name.upper()}")
        print(f"{'='*70}")
        print(f"Níveis: {client_levels}")
        print(f"Queries/cliente: {queries_per_client}")
        print(f"Runs/nível: {num_runs_per_level}")
        if ef_search:
            print(f"ef_search: {ef_search}")
        print(f"{'='*70}")
        
        results = []
        
        for num_clients in client_levels:
            result = self.measure_level(
                query_embeddings=query_embeddings,
                query_sparse=query_sparse,
                top_k=top_k,
                num_queries_per_client=queries_per_client,
                num_clients=num_clients,
                num_runs=num_runs_per_level,
                warmup=warmup
            )
            results.append(result)
            print(f"   {result.summary()}")
        
        # Detectar saturação
        optimal_clients, optimal_qps, saturation_detected, saturation_clients, saturation_reason = \
            self.detect_saturation(results)
        
        sweep_result = ConcurrencySweepResult(
            engine=self.engine_name,
            client_levels=client_levels,
            results=results,
            optimal_clients=optimal_clients,
            optimal_qps=optimal_qps,
            saturation_detected=saturation_detected,
            saturation_clients=saturation_clients,
            saturation_reason=saturation_reason,
            queries_per_level=queries_per_client,
            warmup_queries=10 if warmup else 0,
            cooldown_seconds=self.config.benchmark.cooldown_seconds
        )
        
        print(sweep_result.summary())
        
        return sweep_result


def run_concurrency_sweep_for_engine(
    benchmark_instance,
    query_embeddings: np.ndarray,
    query_sparse: List[Optional[Dict]],
    client_levels: List[int] = None,
    queries_per_client: int = 100,
    output_dir: str = None
) -> ConcurrencySweepResult:
    """
    Função auxiliar para executar varredura de concorrência para um engine.
    
    Args:
        benchmark_instance: Instância do benchmark (QdrantBenchmark ou MilvusBenchmark)
        query_embeddings: Embeddings das queries
        query_sparse: Embeddings esparsos das queries
        client_levels: Níveis de concorrência
        queries_per_client: Queries por cliente
        output_dir: Diretório para salvar resultados
        
    Returns:
        ConcurrencySweepResult
    """
    engine_name = benchmark_instance.__class__.__name__.replace("Benchmark", "").lower()
    
    # Criar função de busca wrapper
    def search_func(query_emb, query_sp, top_k):
        return benchmark_instance.hybrid_search(
            dense_embedding=query_emb,
            sparse_embedding=query_sp,
            top_k=top_k
        )
    
    sweep = ConcurrencySweep(
        search_func=search_func,
        engine_name=engine_name,
        config=benchmark_instance.config
    )
    
    result = sweep.run_sweep(
        query_embeddings=query_embeddings,
        query_sparse=query_sparse,
        top_k=benchmark_instance.config.hybrid.top_k_retrieval,
        client_levels=client_levels,
        queries_per_client=queries_per_client
    )
    
    # Salvar resultado
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = output_path / f"concurrency_sweep_{engine_name}_{timestamp}.json"
        result.to_json(str(result_file))
        print(f"\n💾 Resultado salvo em: {result_file}")
    
    return result


# =============================================================================
# TESTE
# =============================================================================

def test_concurrency_sweep():
    """Teste básico do módulo de concurrency sweep."""
    print("\n" + "="*70)
    print("TESTE: Concurrency Sweep")
    print("="*70)
    
    # Mock da função de busca
    def mock_search(query_emb, query_sp, top_k):
        # Simular latência variável
        latency = np.random.uniform(5, 15)  # 5-15ms
        time.sleep(latency / 1000)
        return list(range(top_k)), latency
    
    # Criar dados de teste
    num_queries = 50
    dense_dim = 1024
    
    query_embeddings = np.random.randn(num_queries, dense_dim).astype(np.float32)
    query_sparse = [{"indices": [1, 2, 3], "values": [0.5, 0.3, 0.2]}] * num_queries
    
    # Executar sweep
    sweep = ConcurrencySweep(
        search_func=mock_search,
        engine_name="mock"
    )
    
    result = sweep.run_sweep(
        query_embeddings=query_embeddings,
        query_sparse=query_sparse,
        top_k=10,
        client_levels=[1, 2, 4, 8],  # Reduzido para teste
        queries_per_client=20,  # Reduzido para teste
        num_runs_per_level=2,
        warmup=False
    )
    
    # Verificações
    assert len(result.results) == 4, "Deve ter 4 níveis"
    assert result.optimal_qps > 0, "QPS ótimo deve ser > 0"
    
    print("\n✅ Teste passou!")
    print(f"   Cliente ótimo: {result.optimal_clients}")
    print(f"   QPS ótimo: {result.optimal_qps:.2f}")
    print(f"   Saturação detectada: {result.saturation_detected}")
    
    return result


if __name__ == "__main__":
    test_concurrency_sweep()
