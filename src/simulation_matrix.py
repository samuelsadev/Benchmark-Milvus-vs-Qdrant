#!/usr/bin/env python3
"""
Matriz de Simulações (Produto Cruzado) - Benchmark V3
======================================================

Conforme benchmark specification Seção 5.2 e texto-pdf:
- 2 fontes densas × 3 pesos RRF × 2 engines = 12 execuções
- Produto cruzado completo executado em todas as fases

Fontes Densas:
  - vetor_denso_sumario_ia (sumário)
  - vetor_denso_topico_ia (tópico)

Pesos RRF:
  - 60/40 (denso-dominante)
  - 50/50 (balanceado)
  - 40/60 (esparso-dominante)

Engines:
  - Qdrant
  - Milvus

Fonte Esparsa (fixa):
  - vetor_esparso_palavras_chave_ia

Filtro Prévio (fixo):
  - region

Reranking (fixo):
  - ColBERT do sumário
"""
import json
import time
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from itertools import product

from config import Config, get_config, RRF_WEIGHTS


# =============================================================================
# CONFIGURAÇÕES DA MATRIZ
# =============================================================================

# Fontes densas disponíveis
DENSE_SOURCES = {
    "sumario": {
        "name": "vetor_denso_sumario_ia",
        "file": "df_base_hist_topicos_vetor_denso_sumario_ia_bge-m3_v1.0_2026-06-17.parquet",
        "description": "Embeddings densos do sumário"
    },
    "topico": {
        "name": "vetor_denso_topico_ia",
        "file": "df_base_hist_topicos_vetor_denso_topico_ia_bge-m3_v1.0_2026-06-17.parquet",
        "description": "Embeddings densos do tópico"
    }
}

# Pesos RRF per specification
RRF_WEIGHT_CONFIGS = {
    "denso_dominante": (0.6, 0.4),
    "balanceado": (0.5, 0.5),
    "esparso_dominante": (0.4, 0.6)
}

# Engines disponíveis
ENGINES = ["qdrant", "milvus"]


@dataclass
class SimulationConfig:
    """Configuração de uma simulação individual."""
    simulation_id: int
    engine: str
    dense_source: str
    rrf_weight_name: str
    rrf_weight: Tuple[float, float]
    
    def __str__(self):
        return f"Sim#{self.simulation_id:02d}: {self.engine} | {self.dense_source} | {self.rrf_weight_name}"


@dataclass
class SimulationResult:
    """Resultado de uma simulação individual."""
    config: SimulationConfig
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Métricas de qualidade (primária: Recall@100)
    recall_at_100: float = 0.0
    recall_at_100_ci_lower: float = 0.0
    recall_at_100_ci_upper: float = 0.0
    recall_at_10: float = 0.0
    precision_at_10: float = 0.0
    ndcg_at_10: float = 0.0
    mrr_at_10: float = 0.0
    
    # Performance
    qps: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    
    # Footprint
    ram_gb: float = 0.0
    disk_gb: float = 0.0
    index_size_gb: float = 0.0
    
    # Erro
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "config": asdict(self.config),
            "timestamp": self.timestamp,
            "metrics": {
                "recall_at_100": self.recall_at_100,
                "recall_at_100_ci": [self.recall_at_100_ci_lower, self.recall_at_100_ci_upper],
                "recall_at_10": self.recall_at_10,
                "precision_at_10": self.precision_at_10,
                "ndcg_at_10": self.ndcg_at_10,
                "mrr_at_10": self.mrr_at_10,
            },
            "performance": {
                "qps": self.qps,
                "latency_p50_ms": self.latency_p50_ms,
                "latency_p95_ms": self.latency_p95_ms,
                "latency_p99_ms": self.latency_p99_ms,
            },
            "footprint": {
                "ram_gb": self.ram_gb,
                "disk_gb": self.disk_gb,
                "index_size_gb": self.index_size_gb,
            },
            "error": self.error
        }


@dataclass
class SimulationMatrixResult:
    """Resultado completo da matriz de simulações."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    total_simulations: int = 12
    completed_simulations: int = 0
    failed_simulations: int = 0
    
    # Resultados por simulação
    results: List[SimulationResult] = field(default_factory=list)
    
    # Melhor combinação (por Recall@100)
    best_config: Optional[SimulationConfig] = None
    best_recall_100: float = 0.0
    
    # Comparação por engine
    qdrant_best: Optional[SimulationResult] = None
    milvus_best: Optional[SimulationResult] = None
    
    # Comparação por fonte densa
    sumario_best: Optional[SimulationResult] = None
    topico_best: Optional[SimulationResult] = None
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "total_simulations": self.total_simulations,
            "completed_simulations": self.completed_simulations,
            "failed_simulations": self.failed_simulations,
            "results": [r.to_dict() for r in self.results],
            "best_config": asdict(self.best_config) if self.best_config else None,
            "best_recall_100": self.best_recall_100,
            "qdrant_best": self.qdrant_best.to_dict() if self.qdrant_best else None,
            "milvus_best": self.milvus_best.to_dict() if self.milvus_best else None,
            "sumario_best": self.sumario_best.to_dict() if self.sumario_best else None,
            "topico_best": self.topico_best.to_dict() if self.topico_best else None,
        }
    
    def to_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def summary(self) -> str:
        """Retorna resumo formatado da matriz."""
        lines = [
            "",
            "=" * 80,
            "MATRIZ DE SIMULAÇÕES - RESULTADO",
            "=" * 80,
            f"Timestamp: {self.timestamp}",
            f"Simulações: {self.completed_simulations}/{self.total_simulations} completas",
            f"Falhas: {self.failed_simulations}",
            "",
        ]
        
        if self.best_config:
            lines.append("=" * 80)
            lines.append("MELHOR COMBINAÇÃO (Recall@100)")
            lines.append("=" * 80)
            lines.append(f"  Engine: {self.best_config.engine}")
            lines.append(f"  Fonte densa: {self.best_config.dense_source}")
            lines.append(f"  Peso RRF: {self.best_config.rrf_weight_name} ({self.best_config.rrf_weight})")
            lines.append(f"  Recall@100: {self.best_recall_100:.4f}")
            lines.append("")
        
        # Comparação por engine
        lines.append("=" * 80)
        lines.append("COMPARAÇÃO POR ENGINE")
        lines.append("=" * 80)
        
        if self.qdrant_best:
            lines.append(f"  Qdrant:")
            lines.append(f"    Config: {self.qdrant_best.config}")
            lines.append(f"    Recall@100: {self.qdrant_best.recall_at_100:.4f}")
            lines.append(f"    QPS: {self.qdrant_best.qps:.2f}")
        
        if self.milvus_best:
            lines.append(f"  Milvus:")
            lines.append(f"    Config: {self.milvus_best.config}")
            lines.append(f"    Recall@100: {self.milvus_best.recall_at_100:.4f}")
            lines.append(f"    QPS: {self.milvus_best.qps:.2f}")
        
        lines.append("")
        
        # Comparação por fonte
        lines.append("=" * 80)
        lines.append("COMPARAÇÃO POR FONTE DENSA")
        lines.append("=" * 80)
        
        if self.sumario_best:
            lines.append(f"  Sumário:")
            lines.append(f"    Recall@100: {self.sumario_best.recall_at_100:.4f}")
        
        if self.topico_best:
            lines.append(f"  Tópico:")
            lines.append(f"    Recall@100: {self.topico_best.recall_at_100:.4f}")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)


class SimulationMatrix:
    """
    Executor da matriz de simulações.
    
    Gera e executa todas as combinações do produto cruzado:
    2 fontes densas × 3 pesos RRF × 2 engines = 12 execuções
    """
    
    def __init__(self, config: Config = None):
        self.config = config or get_config()
        self.simulations: List[SimulationConfig] = []
        self._generate_simulations()
    
    def _generate_simulations(self):
        """Gera todas as combinações de simulação."""
        sim_id = 0
        
        for engine, dense_source, (weight_name, weight) in product(
            ENGINES,
            DENSE_SOURCES.keys(),
            RRF_WEIGHT_CONFIGS.items()
        ):
            sim_id += 1
            self.simulations.append(SimulationConfig(
                simulation_id=sim_id,
                engine=engine,
                dense_source=dense_source,
                rrf_weight_name=weight_name,
                rrf_weight=weight
            ))
        
        print(f"   Matriz gerada: {len(self.simulations)} simulações")
    
    def list_simulations(self) -> List[str]:
        """Lista todas as simulações em formato legível."""
        return [str(sim) for sim in self.simulations]
    
    def run_single_simulation(
        self,
        sim_config: SimulationConfig,
        benchmark_runner
    ) -> SimulationResult:
        """
        Executa uma simulação individual.
        
        Args:
            sim_config: Configuração da simulação
            benchmark_runner: Função para executar o benchmark
            
        Returns:
            SimulationResult com métricas
        """
        print(f"\n   {'─'*60}")
        print(f"   {sim_config}")
        print(f"   {'─'*60}")
        
        result = SimulationResult(config=sim_config)
        
        try:
            # Executar benchmark com configuração específica
            bench_result = benchmark_runner(
                engine=sim_config.engine,
                dense_source=sim_config.dense_source,
                rrf_weights=sim_config.rrf_weight,
                config=self.config
            )
            
            if bench_result:
                # Extrair métricas
                result.recall_at_100 = getattr(bench_result, 'recall_at_100', 0)
                result.recall_at_100_ci_lower = getattr(bench_result, 'recall_at_100_ci_lower', 0)
                result.recall_at_100_ci_upper = getattr(bench_result, 'recall_at_100_ci_upper', 0)
                result.recall_at_10 = getattr(bench_result, 'recall_at_10', 0)
                result.precision_at_10 = getattr(bench_result, 'precision_at_10', 0)
                result.ndcg_at_10 = getattr(bench_result, 'ndcg_at_10', 0)
                result.mrr_at_10 = getattr(bench_result, 'mrr_at_10', 0)
                result.qps = getattr(bench_result, 'qps', 0)
                result.latency_p50_ms = getattr(bench_result, 'latency_p50_ms', 0)
                result.latency_p95_ms = getattr(bench_result, 'latency_p95_ms', 0)
                result.latency_p99_ms = getattr(bench_result, 'latency_p99_ms', 0)
                result.ram_gb = getattr(bench_result, 'ram_gb', 0)
                result.disk_gb = getattr(bench_result, 'disk_gb', 0)
                result.index_size_gb = getattr(bench_result, 'index_size_gb', 0)
                
                print(f"      Recall@100: {result.recall_at_100:.4f} [{result.recall_at_100_ci_lower:.4f}, {result.recall_at_100_ci_upper:.4f}]")
                print(f"      QPS: {result.qps:.2f} | Latência P95: {result.latency_p95_ms:.2f}ms")
            else:
                result.error = "Benchmark retornou None"
                
        except Exception as e:
            result.error = str(e)
            print(f"      ❌ Erro: {e}")
        
        return result
    
    def run_matrix(
        self,
        benchmark_runner,
        engines: List[str] = None,
        dense_sources: List[str] = None,
        rrf_weights: List[str] = None
    ) -> SimulationMatrixResult:
        """
        Executa a matriz de simulações completa.
        
        Args:
            benchmark_runner: Função para executar benchmark
            engines: Subset de engines (default: todas)
            dense_sources: Subset de fontes densas (default: todas)
            rrf_weights: Subset de pesos RRF (default: todos)
            
        Returns:
            SimulationMatrixResult com todos os resultados
        """
        print("\n" + "=" * 80)
        print("MATRIZ DE SIMULAÇÕES (PRODUTO CRUZADO)")
        print("=" * 80)
        print("   2 fontes densas × 3 pesos RRF × 2 engines = 12 execuções")
        print("=" * 80)
        
        matrix_result = SimulationMatrixResult()
        
        # Filtrar simulações se necessário
        sims_to_run = self.simulations
        
        if engines:
            sims_to_run = [s for s in sims_to_run if s.engine in engines]
        if dense_sources:
            sims_to_run = [s for s in sims_to_run if s.dense_source in dense_sources]
        if rrf_weights:
            sims_to_run = [s for s in sims_to_run if s.rrf_weight_name in rrf_weights]
        
        matrix_result.total_simulations = len(sims_to_run)
        
        print(f"\n   Simulações a executar: {len(sims_to_run)}")
        for sim in sims_to_run:
            print(f"      {sim}")
        
        # Executar cada simulação
        start_time = time.time()
        
        for sim in sims_to_run:
            result = self.run_single_simulation(sim, benchmark_runner)
            matrix_result.results.append(result)
            
            if result.error:
                matrix_result.failed_simulations += 1
            else:
                matrix_result.completed_simulations += 1
                
                # Atualizar melhores
                if result.recall_at_100 > matrix_result.best_recall_100:
                    matrix_result.best_recall_100 = result.recall_at_100
                    matrix_result.best_config = result.config
                
                # Melhor por engine
                if sim.engine == "qdrant":
                    if not matrix_result.qdrant_best or result.recall_at_100 > matrix_result.qdrant_best.recall_at_100:
                        matrix_result.qdrant_best = result
                else:
                    if not matrix_result.milvus_best or result.recall_at_100 > matrix_result.milvus_best.recall_at_100:
                        matrix_result.milvus_best = result
                
                # Melhor por fonte
                if sim.dense_source == "sumario":
                    if not matrix_result.sumario_best or result.recall_at_100 > matrix_result.sumario_best.recall_at_100:
                        matrix_result.sumario_best = result
                else:
                    if not matrix_result.topico_best or result.recall_at_100 > matrix_result.topico_best.recall_at_100:
                        matrix_result.topico_best = result
        
        total_time = time.time() - start_time
        
        # Resumo
        print("\n" + "=" * 80)
        print("RESUMO DA MATRIZ")
        print("=" * 80)
        print(f"   Tempo total: {total_time:.1f}s")
        print(f"   Simulações completas: {matrix_result.completed_simulations}/{matrix_result.total_simulations}")
        print(f"   Falhas: {matrix_result.failed_simulations}")
        
        if matrix_result.best_config:
            print(f"\n   🏆 MELHOR COMBINAÇÃO:")
            print(f"      Engine: {matrix_result.best_config.engine}")
            print(f"      Fonte: {matrix_result.best_config.dense_source}")
            print(f"      Peso: {matrix_result.best_config.rrf_weight_name}")
            print(f"      Recall@100: {matrix_result.best_recall_100:.4f}")
        
        print("=" * 80)
        
        return matrix_result


def get_simulation_matrix(config: Config = None) -> SimulationMatrix:
    """Retorna instância da matriz de simulações."""
    return SimulationMatrix(config)


def print_matrix_config():
    """Imprime configuração da matriz."""
    print("\n" + "=" * 80)
    print("CONFIGURAÇÃO DA MATRIZ DE SIMULAÇÕES")
    print("=" * 80)
    
    print("\n📐 DIMENSÕES:")
    print(f"   Fontes densas: {len(DENSE_SOURCES)}")
    for name, info in DENSE_SOURCES.items():
        print(f"      • {name}: {info['description']}")
    
    print(f"\n   Pesos RRF: {len(RRF_WEIGHT_CONFIGS)}")
    for name, weight in RRF_WEIGHT_CONFIGS.items():
        print(f"      • {name}: {weight[0]:.1f}/{weight[1]:.1f}")
    
    print(f"\n   Engines: {len(ENGINES)}")
    for engine in ENGINES:
        print(f"      • {engine}")
    
    print(f"\n📊 PRODUTO CRUZADO:")
    print(f"   {len(DENSE_SOURCES)} × {len(RRF_WEIGHT_CONFIGS)} × {len(ENGINES)} = {len(DENSE_SOURCES) * len(RRF_WEIGHT_CONFIGS) * len(ENGINES)} simulações")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    print_matrix_config()
    
    # Exemplo de uso
    matrix = SimulationMatrix()
    print("\n📋 SIMULAÇÕES:")
    for sim in matrix.list_simulations():
        print(f"   {sim}")
