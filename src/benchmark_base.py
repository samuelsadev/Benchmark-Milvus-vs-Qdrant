#!/usr/bin/env python3
"""
Classes base para o Benchmark V3 - Hybrid Search Benchmark

Contém:
- BenchmarkResult: Estrutura para resultados com média ± desvio padrão
- BaseBenchmark: Classe base para benchmarks
- Métricas: recall, MRR, NDCG, latência, QPS
- FootprintCollector: Coleta de métricas de RAM/disco (V3)
"""
import json
import time
import numpy as np
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from statistics import mean, stdev

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """
    Resultado do benchmark com estatísticas completas - V3.

    MÉTRICA PRIMÁRIA: Recall@100
    Critério #9: Média ± desvio padrão, ≥5 runs
    V3: IC 95% por bootstrap
    """
    engine: str
    scenario: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    num_runs: int = 5

    # Tempos (Critério #10: separar ingestão de build)
    ingest_time_seconds: float = 0.0
    build_index_time_seconds: float = 0.0

    # Latência (ms)
    latency_mean_ms: float = 0.0
    latency_std_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0

    # Throughput
    qps: float = 0.0
    qps_std: float = 0.0

    # *** MÉTRICA PRIMÁRIA: Recall@100 ***
    recall_at_100: float = 0.0
    recall_at_100_std: float = 0.0
    recall_at_100_ci_lower: float = 0.0  # IC 95%
    recall_at_100_ci_upper: float = 0.0  # IC 95%

    # Métricas secundárias com IC 95%
    recall_at_10: float = 0.0
    recall_at_10_std: float = 0.0
    recall_at_10_ci_lower: float = 0.0
    recall_at_10_ci_upper: float = 0.0

    mrr_at_10: float = 0.0
    mrr_at_10_std: float = 0.0
    mrr_at_10_ci_lower: float = 0.0
    mrr_at_10_ci_upper: float = 0.0

    ndcg_at_10: float = 0.0
    ndcg_at_10_std: float = 0.0
    ndcg_at_10_ci_lower: float = 0.0
    ndcg_at_10_ci_upper: float = 0.0

    # Precision@10 para GT-Relevância (V3) com IC 95%
    precision_at_10: float = 0.0
    precision_at_10_std: float = 0.0
    precision_at_10_ci_lower: float = 0.0
    precision_at_10_ci_upper: float = 0.0

    # Footprint (V3)
    ram_gb: float = 0.0
    ram_peak_gb: float = 0.0
    disk_gb: float = 0.0
    disk_peak_gb: float = 0.0
    index_size_gb: float = 0.0  # Tamanho específico do índice

    # Metadata
    num_documents: int = 0
    num_queries: int = 0
    dense_dim: int = 0
    config: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def summary(self) -> str:
        """Retorna resumo formatado do resultado - V3 com IC 95% para todas as métricas"""
        return f"""
=== {self.engine.upper()} - {self.scenario} ===
Timestamp: {self.timestamp}
Runs: {self.num_runs}

*** MÉTRICA PRIMÁRIA ***
  Recall@100: {self.recall_at_100:.4f} ± {self.recall_at_100_std:.4f}
  IC 95%: [{self.recall_at_100_ci_lower:.4f}, {self.recall_at_100_ci_upper:.4f}]

📊 Performance:
  - QPS: {self.qps:.2f} ± {self.qps_std:.2f}
  - Latência média: {self.latency_mean_ms:.2f} ± {self.latency_std_ms:.2f} ms
  - Latência P95: {self.latency_p95_ms:.2f} ms

🎯 Qualidade (com IC 95%):
  - Recall@10: {self.recall_at_10:.4f} IC [{self.recall_at_10_ci_lower:.4f}, {self.recall_at_10_ci_upper:.4f}]
  - Precision@10: {self.precision_at_10:.4f} IC [{self.precision_at_10_ci_lower:.4f}, {self.precision_at_10_ci_upper:.4f}]
  - NDCG@10: {self.ndcg_at_10:.4f} IC [{self.ndcg_at_10_ci_lower:.4f}, {self.ndcg_at_10_ci_upper:.4f}]
  - MRR@10: {self.mrr_at_10:.4f} IC [{self.mrr_at_10_ci_lower:.4f}, {self.mrr_at_10_ci_upper:.4f}]

⏱️ Tempos:
  - Ingestão: {self.ingest_time_seconds:.2f}s
  - Build Index: {self.build_index_time_seconds:.2f}s

💾 Footprint:
  - RAM: {self.ram_gb:.2f} GB (pico: {self.ram_peak_gb:.2f} GB)
  - Disco: {self.disk_gb:.2f} GB (pico: {self.disk_peak_gb:.2f} GB)
  - Índice: {self.index_size_gb:.2f} GB
"""


class FootprintCollector:
    """
    Coletor de métricas de footprint (RAM/disco) - V3.

    Mede consumo de recursos do banco vetorial durante a execução.
    """

    def __init__(self, engine_name: str = "unknown"):
        self.engine_name = engine_name
        self.measurements_ram = []
        self.measurements_disk = []

    def get_process_memory_gb(self, process_name: str = None) -> float:
        """Obtém uso de memória RAM em GB para um processo específico."""
        if not PSUTIL_AVAILABLE:
            return 0.0

        try:
            if process_name:
                for proc in psutil.process_iter(['name', 'memory_info']):
                    try:
                        if process_name.lower() in proc.info['name'].lower():
                            return proc.info['memory_info'].rss / (1024 ** 3)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                return 0.0
            else:
                return psutil.virtual_memory().used / (1024 ** 3)
        except Exception as e:
            print(f"   ⚠️ Erro ao coletar memória: {e}")
            return 0.0

    def get_container_memory_gb(self, container_name: str = None) -> float:
        """Obtém uso de memória RAM em GB para containers Docker/Kubernetes."""
        try:
            result = subprocess.run(
                ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container_name],
                capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0 and result.stdout:
                mem_str = result.stdout.strip().split("/")[0].strip()
                return self._parse_memory_string(mem_str)
        except (subprocess.TimeoutExpired, FileNotFoundError, IndexError):
            pass

        process_names = {
            "qdrant": ["qdrant"],
            "milvus": ["milvus", "standalone", "proxy", "querynode", "datanode"]
        }

        for proc_name in process_names.get(container_name, [container_name]):
            mem = self.get_process_memory_gb(proc_name)
            if mem > 0:
                return mem

        return 0.0

    def _parse_memory_string(self, mem_str: str) -> float:
        """Converte string de memória para GB"""
        mem_str = mem_str.strip().upper()

        try:
            if "GIB" in mem_str or "GB" in mem_str:
                return float(mem_str.replace("GIB", "").replace("GB", "").strip())
            elif "MIB" in mem_str or "MB" in mem_str:
                return float(mem_str.replace("MIB", "").replace("MB", "").strip()) / 1024
            elif "KIB" in mem_str or "KB" in mem_str:
                return float(mem_str.replace("KIB", "").replace("KB", "").strip()) / (1024 ** 2)
            elif "B" in mem_str:
                return float(mem_str.replace("B", "").strip()) / (1024 ** 3)
        except ValueError:
            pass

        return 0.0

    def get_disk_usage_gb(self, path: str = None) -> float:
        """Obtém uso de disco em GB para um caminho específico."""
        if not PSUTIL_AVAILABLE:
            return 0.0

        try:
            if path and Path(path).exists():
                usage = psutil.disk_usage(path)
                return usage.used / (1024 ** 3)
            else:
                usage = psutil.disk_usage("/")
                return usage.used / (1024 ** 3)
        except Exception as e:
            print(f"   ⚠️ Erro ao coletar disco: {e}")
            return 0.0

    def get_directory_size_gb(self, directory: str) -> float:
        """Calcula tamanho de um diretório específico em GB."""
        try:
            total_size = 0
            dir_path = Path(directory)

            if not dir_path.exists():
                return 0.0

            for file_path in dir_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size

            return total_size / (1024 ** 3)
        except Exception as e:
            print(f"   ⚠️ Erro ao calcular tamanho do diretório: {e}")
            return 0.0

    def collect(self, ram_path: str = None, disk_path: str = None) -> Tuple[float, float]:
        """Coleta métricas de footprint no momento atual."""
        ram_gb = self.get_process_memory_gb(ram_path)
        disk_gb = self.get_disk_usage_gb(disk_path)

        self.measurements_ram.append(ram_gb)
        self.measurements_disk.append(disk_gb)

        return ram_gb, disk_gb

    def get_summary(self) -> Dict[str, float]:
        """Retorna resumo das métricas de footprint."""
        return {
            'ram_gb': mean(self.measurements_ram) if self.measurements_ram else 0.0,
            'disk_gb': mean(self.measurements_disk) if self.measurements_disk else 0.0,
            'ram_peak_gb': max(self.measurements_ram) if self.measurements_ram else 0.0,
            'disk_peak_gb': max(self.measurements_disk) if self.measurements_disk else 0.0,
        }


class MetricsCalculator:
    """Calculadora de métricas de retrieval"""

    @staticmethod
    def bootstrap_ci(values: List[float], confidence: float = 0.95, n_samples: int = 1000, seed: int = 42) -> Tuple[float, float]:
        """
        Calcula intervalo de confiança por bootstrap (V3).

        Args:
            values: Lista de valores
            confidence: Nível de confiança (default 0.95 para IC 95%)
            n_samples: Número de amostras bootstrap
            seed: Semente para reprodutibilidade

        Returns:
            (limite_inferior, limite_superior)
        """
        if not values:
            return (0.0, 0.0)

        np.random.seed(seed)
        values = np.array(values)
        n = len(values)

        bootstrap_means = []
        for _ in range(n_samples):
            sample = np.random.choice(values, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))

        bootstrap_means = np.array(bootstrap_means)

        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, alpha / 2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

        return (lower, upper)

    @staticmethod
    def recall_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
        """Calcula Recall@k"""
        retrieved_k = set(retrieved[:k])
        relevant_set = set(relevant[:k])

        if not relevant_set:
            return 0.0

        return len(retrieved_k & relevant_set) / len(relevant_set)

    @staticmethod
    def precision_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
        """Calcula Precision@k"""
        retrieved_k = set(retrieved[:k])
        relevant_set = set(relevant)

        return len(retrieved_k & relevant_set) / k

    @staticmethod
    def mrr_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
        """Calcula Mean Reciprocal Rank@k"""
        relevant_set = set(relevant)

        for i, doc_id in enumerate(retrieved[:k]):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)

        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
        """Calcula Normalized Discounted Cumulative Gain@k"""
        def dcg(scores: List[float]) -> float:
            return sum(
                score / np.log2(i + 2)
                for i, score in enumerate(scores)
            )

        relevant_set = set(relevant)

        retrieved_scores = [1.0 if doc_id in relevant_set else 0.0 for doc_id in retrieved[:k]]
        ideal_scores = [1.0] * min(k, len(relevant_set))

        dcg_value = dcg(retrieved_scores)
        idcg_value = dcg(ideal_scores)

        if idcg_value == 0:
            return 0.0

        return dcg_value / idcg_value

    @staticmethod
    def calculate_all(
        all_retrieved: List[List[int]],
        all_relevant: List[List[int]],
    ) -> Dict[str, float]:
        """Calcula todas as métricas para um conjunto de queries"""
        recall_10 = []
        recall_100 = []
        mrr_10 = []
        ndcg_10 = []

        for retrieved, relevant in zip(all_retrieved, all_relevant):
            recall_10.append(MetricsCalculator.recall_at_k(retrieved, relevant, 10))
            recall_100.append(MetricsCalculator.recall_at_k(retrieved, relevant, 100))
            mrr_10.append(MetricsCalculator.mrr_at_k(retrieved, relevant, 10))
            ndcg_10.append(MetricsCalculator.ndcg_at_k(retrieved, relevant, 10))

        return {
            'recall_at_10': mean(recall_10),
            'recall_at_100': mean(recall_100),
            'mrr_at_10': mean(mrr_10),
            'ndcg_at_10': mean(ndcg_10),
        }

    @staticmethod
    def calculate_all_per_query(
        all_retrieved: List[List[int]],
        all_relevant: List[List[int]],
        k_recall: int = 100,
        k_precision: int = 10
    ) -> Dict[str, List[float]]:
        """
        Calcula todas as métricas POR QUERY (para IC 95% via bootstrap).

        V3: Retorna valores por query, não agregados.
        """
        recall_100 = []
        recall_10 = []
        precision_10 = []
        mrr_10 = []
        ndcg_10 = []

        for retrieved, relevant in zip(all_retrieved, all_relevant):
            recall_100.append(MetricsCalculator.recall_at_k(retrieved, relevant, k_recall))
            recall_10.append(MetricsCalculator.recall_at_k(retrieved, relevant, k_precision))
            precision_10.append(MetricsCalculator.precision_at_k(retrieved, relevant, k_precision))
            mrr_10.append(MetricsCalculator.mrr_at_k(retrieved, relevant, k_precision))
            ndcg_10.append(MetricsCalculator.ndcg_at_k(retrieved, relevant, k_precision))

        return {
            'recall_at_100': recall_100,
            'recall_at_10': recall_10,
            'precision_at_10': precision_10,
            'mrr_at_10': mrr_10,
            'ndcg_at_10': ndcg_10,
        }

    @staticmethod
    def calculate_metrics_with_ci(
        all_retrieved: List[List[int]],
        all_relevant: List[List[int]],
        confidence: float = 0.95,
        n_bootstrap: int = 1000,
        seed: int = 42
    ) -> Dict[str, Dict[str, float]]:
        """
        Calcula todas as métricas com IC 95% via bootstrap (V3).

        Esta é a função principal para cálculo de métricas com rigor estatístico.
        """
        per_query = MetricsCalculator.calculate_all_per_query(
            all_retrieved, all_relevant, k_recall=100, k_precision=10
        )

        results = {}

        for metric_name, values in per_query.items():
            if not values:
                results[metric_name] = {
                    'mean': 0.0, 'std': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0
                }
                continue

            values_arr = np.array(values)

            metric_mean = np.mean(values_arr)
            metric_std = np.std(values_arr, ddof=1) if len(values_arr) > 1 else 0.0

            ci_lower, ci_upper = MetricsCalculator.bootstrap_ci(
                values, confidence=confidence, n_samples=n_bootstrap, seed=seed
            )

            results[metric_name] = {
                'mean': float(metric_mean),
                'std': float(metric_std),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
            }

        return results


class BaseBenchmark(ABC):
    """Classe base para benchmarks"""

    def __init__(self, config):
        self.config = config
        self.metrics_calc = MetricsCalculator()
        self.footprint_collector = FootprintCollector(engine_name=self.__class__.__name__)

    @abstractmethod
    def connect(self):
        """Conecta ao banco de dados"""
        pass

    @abstractmethod
    def create_collection(self):
        """Cria a collection/índice"""
        pass

    @abstractmethod
    def ingest_data(self, documents: List[Dict], embeddings: np.ndarray, sparse_embeddings: List[Dict]) -> float:
        """Ingere dados e retorna tempo de ingestão em segundos."""
        pass

    @abstractmethod
    def build_index(self) -> float:
        """Constrói o índice e retorna tempo de build em segundos."""
        pass

    @abstractmethod
    def get_storage_path(self) -> str:
        """Retorna caminho de armazenamento do índice (para medir footprint)."""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        query_sparse: Optional[Dict],
        top_k: int,
        query_region: Optional[str] = None
    ) -> Tuple[List[int], float]:
        """
        Executa busca e retorna (IDs, latência em ms)

        V3: query_region é obrigatório quando filtro regional está habilitado.
        """
        pass

    @abstractmethod
    def hybrid_search(
        self,
        dense_embedding: np.ndarray,
        sparse_embedding: Dict,
        top_k: int,
        fusion_method: str = "rrf",
        query_region: Optional[str] = None
    ) -> Tuple[List[int], float]:
        """
        Executa busca híbrida (dense + sparse) e retorna (IDs, latência em ms)
        """
        pass

    def run_benchmark(
        self,
        queries: List[Dict],
        query_embeddings: np.ndarray,
        query_sparse: List[Dict],
        ground_truth: List[List[int]],
        scenario: str = "hybrid_search",
        num_runs: int = 5,
        warmup_queries: int = 10,
        ef_search: Optional[int] = None,
        confidence: float = 0.95,
        n_bootstrap: int = 1000,
        seed: int = 42,
        regional_filter: Optional['RegionalFilterManager'] = None
    ) -> BenchmarkResult:
        """
        Executa benchmark completo com múltiplas runs.

        V3: IC 95% via bootstrap para todas as métricas.
        V3: Filtro regional OBRIGATÓRIO - todas as buscas filtradas por região.
        """
        ef_display = ef_search if ef_search is not None else self.config.hnsw.ef_search_values[0]
        print(f"\n🔄 Executando benchmark: {scenario} (ef_search={ef_display}, {num_runs} runs)")
        print(f"   Calculando IC 95% via bootstrap ({n_bootstrap} amostras)")

        # V3: Verificar filtro regional
        if regional_filter and self.config.data.regional_filter_enabled:
            print(f"   ✅ Filtro regional ATIVO - {len(regional_filter.documents):,} documentos mapeados")

        all_qps = []
        all_latencies = []

        retrieved_final = None

        # Coletar footprint inicial
        print(f"   Coletando métricas de footprint...")
        self.footprint_collector.collect()

        for run in range(num_runs):
            print(f"   Run {run + 1}/{num_runs}...")

            # Warmup
            for i in range(min(warmup_queries, len(queries))):
                query_region = None
                if regional_filter and self.config.data.regional_filter_enabled:
                    query_nr = queries[i].get('doc_id', '')
                    query_region = regional_filter.get_region(query_nr)

                self.hybrid_search(
                    query_embeddings[i],
                    query_sparse[i] if query_sparse else None,
                    top_k=self.config.hybrid.top_k_retrieval,
                    ef_search=ef_search,
                    query_region=query_region
                )

            # Benchmark
            latencies = []
            retrieved_all = []

            start_time = time.time()

            for i, query in enumerate(queries):
                query_region = None
                if regional_filter and self.config.data.regional_filter_enabled:
                    query_nr = query.get('doc_id', '')
                    query_region = regional_filter.get_region(query_nr)

                ids, latency = self.hybrid_search(
                    query_embeddings[i],
                    query_sparse[i] if query_sparse else None,
                    top_k=self.config.hybrid.top_k_retrieval,
                    ef_search=ef_search,
                    query_region=query_region
                )
                latencies.append(latency)
                retrieved_all.append(ids)

            total_time = time.time() - start_time

            retrieved_final = retrieved_all

            qps = len(queries) / total_time
            all_qps.append(qps)
            all_latencies.extend(latencies)

            # Coletar footprint durante execução
            self.footprint_collector.collect()

            # Cooldown entre runs
            time.sleep(self.config.benchmark.cooldown_seconds)

        # Coletar footprint final
        self.footprint_collector.collect()
        footprint = self.footprint_collector.get_summary()

        # Tentar obter tamanho do índice
        try:
            storage_path = self.get_storage_path()
            index_size = self.footprint_collector.get_directory_size_gb(storage_path)
        except:
            index_size = 0.0

        # Calcular métricas com IC 95% via bootstrap
        print(f"   Calculando métricas com IC 95%...")
        metrics_with_ci = self.metrics_calc.calculate_metrics_with_ci(
            all_retrieved=retrieved_final,
            all_relevant=ground_truth,
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            seed=seed
        )

        # Calcular estatísticas finais
        config_dict = asdict(self.config.hybrid)
        config_dict['ef_search'] = ef_display
        config_dict['regional_filter_enabled'] = self.config.data.regional_filter_enabled

        r100 = metrics_with_ci['recall_at_100']
        r10 = metrics_with_ci['recall_at_10']
        p10 = metrics_with_ci['precision_at_10']
        mrr = metrics_with_ci['mrr_at_10']
        ndcg = metrics_with_ci['ndcg_at_10']

        result = BenchmarkResult(
            engine=self.__class__.__name__.replace("Benchmark", "").lower(),
            scenario=scenario,
            num_runs=num_runs,

            # Latência
            latency_mean_ms=mean(all_latencies),
            latency_std_ms=stdev(all_latencies) if len(all_latencies) > 1 else 0,
            latency_p50_ms=np.percentile(all_latencies, 50),
            latency_p95_ms=np.percentile(all_latencies, 95),
            latency_p99_ms=np.percentile(all_latencies, 99),

            # QPS
            qps=mean(all_qps),
            qps_std=stdev(all_qps) if len(all_qps) > 1 else 0,

            # *** MÉTRICA PRIMÁRIA: Recall@100 com IC 95% ***
            recall_at_100=r100['mean'],
            recall_at_100_std=r100['std'],
            recall_at_100_ci_lower=r100['ci_lower'],
            recall_at_100_ci_upper=r100['ci_upper'],

            # Métricas secundárias com IC 95%
            recall_at_10=r10['mean'],
            recall_at_10_std=r10['std'],
            recall_at_10_ci_lower=r10['ci_lower'],
            recall_at_10_ci_upper=r10['ci_upper'],

            precision_at_10=p10['mean'],
            precision_at_10_std=p10['std'],
            precision_at_10_ci_lower=p10['ci_lower'],
            precision_at_10_ci_upper=p10['ci_upper'],

            mrr_at_10=mrr['mean'],
            mrr_at_10_std=mrr['std'],
            mrr_at_10_ci_lower=mrr['ci_lower'],
            mrr_at_10_ci_upper=mrr['ci_upper'],

            ndcg_at_10=ndcg['mean'],
            ndcg_at_10_std=ndcg['std'],
            ndcg_at_10_ci_lower=ndcg['ci_lower'],
            ndcg_at_10_ci_upper=ndcg['ci_upper'],

            # Footprint (V3)
            ram_gb=footprint['ram_gb'],
            ram_peak_gb=footprint['ram_peak_gb'],
            disk_gb=footprint['disk_gb'],
            disk_peak_gb=footprint['disk_peak_gb'],
            index_size_gb=index_size,

            # Metadata
            num_queries=len(queries),
            config=config_dict,
        )

        return result
