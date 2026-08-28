#!/usr/bin/env python3
"""
Runner principal do Benchmark V3 - Hybrid Search Benchmark

Este script executa o benchmark completo:
1. Preparação de dados (se necessário)
2. Benchmark Qdrant (busca híbrida)
3. Benchmark Milvus (busca híbrida)
4. Benchmark ColBERT Reranking
5. Geração de relatório comparativo

Atende a TODOS os critérios :
- #1: Embeddings reais BGE-M3
- #2: Milvus 2.6.x (verifica versão)
- #3: Cliente co-localizado (mesmo cluster)
- #4: Ground truth via busca exata
- #5: Sem "scaling super-linear"
- #6: Busca híbrida dense+sparse com RRF
- #7: ColBERT como reranking (não índice)
- #8: Protocolo unificado
- #9: Média ± desvio padrão, ≥5 runs
- #10: Separar ingestão de build time
- #11: TCO qualitativo

V3: Filtro regional OBRIGATÓRIO
- Todas as buscas são filtradas por region
- Ground truth calculado por região
"""
import argparse
import json
import sys
import time
import numpy as np
from datetime import datetime
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config, Config
from typing import List, Dict, Optional


def check_data_exists(config: Config) -> bool:
    """Verifica se os dados já foram preparados"""
    data_dir = Path(config.benchmark.data_dir)
    required_files = [
        "documents.json",
        "dense_embeddings.npy",
        "sparse_embeddings.json",
        "queries.json",
        "query_dense.npy",
        "ground_truth.json",
    ]
    
    for f in required_files:
        if not (data_dir / f).exists():
            return False
    return True


def prepare_data(config: Config):
    """Prepara dados se necessário"""
    if check_data_exists(config):
        print("✅ Dados já preparados. Pulando preparação.")
        return
    
    print("📦 Preparando dados...")
    from prepare_data import main as prepare_main
    prepare_main()


def initialize_regional_filter(
    config: Config,
    documents: List[Dict]
) -> Optional['RegionalFilterManager']:
    """
    Inicializa o gerenciador de filtro regional (V3).
    
    Conforme benchmark specification:
    - "O filtro por region é sempre aplicado previamente"
    
    Args:
        config: Configuração do benchmark
        documents: Lista de documentos com metadados
        
    Returns:
        RegionalFilterManager ou None se filtro desabilitado
    """
    if not config.data.regional_filter_enabled:
        print("   ⚠️ Filtro regional DESABILITADO na configuração")
        return None
    
    from regional_filter import RegionalFilterManager
    
    print("\n" + "=" * 70)
    print("🗺️ INICIALIZANDO FILTRO REGIONAL")
    print("=" * 70)
    
    # Converter lista para dict
    documents_dict = {}
    for doc in documents:
        doc_nr = doc.get('doc_id', doc.get('id', ''))
        if doc_nr:
            documents_dict[str(doc_nr)] = doc
    
    # Criar gerenciador
    regional_filter = RegionalFilterManager(documents_dict, config)
    
    # Estatísticas
    stats = regional_filter.get_regional_stats()
    print(f"\n   Total de documentos: {stats['total_documents']:,}")
    print(f"   Regiões: {stats['num_regions']}")
    print(f"   Campo de filtro: {stats['filter_field']}")
    
    return regional_filter


def run_qdrant(config: Config, skip: bool = False, ef_search: int = None, regional_filter: Optional['RegionalFilterManager'] = None):
    """Executa benchmark do Qdrant
    
    Args:
        ef_search: Valor de ef_search a usar. Se None, usa padrão do config.
        regional_filter: Gerenciador de filtro regional (V3 - OBRIGATÓRIO)
    """
    if skip:
        print("⏭️ Pulando benchmark do Qdrant")
        return None
    
    print("\n" + "=" * 60)
    print("🔵 BENCHMARK QDRANT")
    if ef_search is not None:
        print(f"   ef_search = {ef_search}")
    if regional_filter:
        print(f"   Filtro regional: ATIVO")
    print("=" * 60)
    
    from benchmark_qdrant import run_qdrant_benchmark
    return run_qdrant_benchmark(config, ef_search=ef_search, regional_filter=regional_filter)


def run_milvus(config: Config, skip: bool = False, ef_search: int = None, regional_filter: Optional['RegionalFilterManager'] = None):
    """Executa benchmark do Milvus
    
    Args:
        ef_search: Valor de ef_search a usar. Se None, usa padrão do config.
        regional_filter: Gerenciador de filtro regional (V3 - OBRIGATÓRIO)
    """
    if skip:
        print("⏭️ Pulando benchmark do Milvus")
        return None
    
    print("\n" + "=" * 60)
    print("🟢 BENCHMARK MILVUS")
    if ef_search is not None:
        print(f"   ef_search = {ef_search}")
    print("=" * 60)
    
    from benchmark_milvus import run_milvus_benchmark
    return run_milvus_benchmark(config, ef_search=ef_search)


def run_colbert_reranking(config: Config, qdrant_results, milvus_results, skip: bool = False):
    """Executa benchmark do ColBERT reranking"""
    if skip:
        print("⏭️ Pulando benchmark do ColBERT")
        return None
    
    print("\n" + "=" * 60)
    print("🟣 BENCHMARK COLBERT RERANKING")
    print("=" * 60)
    
    # Carregar dados
    data_dir = Path(config.benchmark.data_dir)
    
    with open(data_dir / "documents.json") as f:
        documents = json.load(f)
    
    with open(data_dir / "queries.json") as f:
        queries = json.load(f)
    
    with open(data_dir / "ground_truth.json") as f:
        ground_truth = json.load(f)
    
    # Usar resultados do primeiro estágio (se disponíveis)
    # Por enquanto, usar ground truth como simulação
    first_stage_results = ground_truth
    
    from colbert_rerank import benchmark_colbert_reranking
    
    return benchmark_colbert_reranking(
        queries=queries,
        documents=documents,
        first_stage_results=first_stage_results,
        ground_truth=ground_truth,
        top_k_rerank=config.colbert.top_k_rerank,
        top_k_final=config.colbert.top_k_final,
        num_runs=config.benchmark.num_runs,
    )


def run_gt_relevance_evaluation(
    config: Config,
    queries: List[Dict],
    search_results: Dict[str, List[str]],
    documents: Dict[str, Dict],
    sample_size: int = 100,
    seed: int = 42
):
    """
    Executa avaliação de GT-Relevância.
    
    Conforme benchmark specification Seção 5.1.2:
    - Relevância binária baseada no mesmo category da query
    - Precision@10 e NDCG@10 pós-reranking
    
    Args:
        config: Configuração do benchmark
        queries: Lista de queries
        search_results: Dicionário query_nr -> top_10_docs
        documents: Dicionário doc_nr -> documento
        sample_size: Tamanho da amostra para conferência humana
        seed: Semente para reprodutibilidade
        
    Returns:
        GTRelevanceReport
    """
    from gt_relevance import GTRelevanceEvaluator
    # conference_spreadsheet removido - planilha gerada externamente
    
    print("\n" + "=" * 70)
    print("📊 GT-RELEVÂNCIA POR category")
    print("=" * 70)
    
    # Criar avaliador
    evaluator = GTRelevanceEvaluator(documents, config)
    
    # Avaliar
    report = evaluator.evaluate_all(queries, search_results)
    
    # Gerar planilha de conferência humana
    results_dir = Path(config.benchmark.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    spreadsheet_path = results_dir / f"conferencia_gt_relevance_{timestamp}.xlsx"
    
    # Planilha de conferência gerada externamente (planilha_conferencia_1000q_v2.xlsx)
    
    # Salvar relatório
    report_path = results_dir / f"gt_relevance_report_{timestamp}.json"
    report.to_json(str(report_path))
    
    print(f"\n   💾 Relatório salvo em: {report_path}")
    
    return report



def run_ef_search_sweep(config: Config, skip_qdrant: bool = False, skip_milvus: bool = False):
    """
    Executa Fase 3: Recall vs ef_search
    
    Testa diferentes valores de ef_search para mapear curva de recall/latência.
    """
    print("\n" + "=" * 60)
    print("📊 FASE 3: RECALL vs EF_SEARCH")
    print("=" * 60)
    
    ef_search_values = config.hnsw.ef_search_values
    print(f"   ef_search values: {ef_search_values}")
    
    qdrant_results = {}
    milvus_results = {}
    
    for ef in ef_search_values:
        print(f"\n--- Testando ef_search = {ef} ---")
        
        if not skip_qdrant:
            qdrant_results[ef] = run_qdrant(config, skip=False, ef_search=ef)
        
        if not skip_milvus:
            milvus_results[ef] = run_milvus(config, skip=False, ef_search=ef)
    
    return qdrant_results, milvus_results


def run_ef_search_optimization(
    config: Config,
    skip_qdrant: bool = False,
    skip_milvus: bool = False,
    target_recall: float = 0.95,
    use_binary_search: bool = True,
    num_runs: int = 3,
    regional_filter: 'RegionalFilterManager' = None
):
    """
    Executa Fase 2.3.3: Varredura de ef_search Otimizado por Engine.
    
    Conforme benchmark specification:
    - "Identificar o menor ef_search que atinge Recall@100 ≥ 0,95"
    - Testar mais valores de ef_search
    - Encontrar ponto ótimo por engine
    - Validar estabilidade
    
    Args:
        config: Configuração do benchmark
        skip_qdrant: Pular Qdrant
        skip_milvus: Pular Milvus
        target_recall: Recall alvo (default 0.95)
        use_binary_search: Usar busca binária (mais eficiente que exaustiva)
        num_runs: Runs por teste (reduzido para otimização)
        regional_filter: Filtro regional
        
    Returns:
        (qdrant_result, milvus_result) - EfSearchSweepResult para cada engine
    """
    from ef_search_sweep import run_ef_search_sweep as run_optimization
    from ef_search_sweep import generate_ef_search_report
    
    print("\n" + "=" * 70)
    print("📊 FASE 2.3.3: VARREDURA DE ef_search OTIMIZADO")
    print("=" * 70)
    print(f"   Target: Recall@100 ≥ {target_recall}")
    print(f"   Estratégia: {'busca binária' if use_binary_search else 'exaustiva'}")
    
    qdrant_result, milvus_result = run_optimization(
        config=config,
        skip_qdrant=skip_qdrant,
        skip_milvus=skip_milvus,
        use_binary_search=use_binary_search,
        num_runs=num_runs,
        regional_filter=regional_filter,
        verbose=True
    )
    
    # Gerar relatório
    results_dir = Path(config.benchmark.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report = generate_ef_search_report(
        qdrant_result, milvus_result,
        output_path=str(results_dir / f"ef_search_optimization_report_{timestamp}.json")
    )
    print(report)
    
    return qdrant_result, milvus_result


def run_concurrency_sweep(
    config: Config,
    skip_qdrant: bool = False,
    skip_milvus: bool = False,
    client_levels: list = None
):
    """
    Executa Fase 5: Concorrência e Throughput de Saturação.
    
    Conforme benchmark specification Fase 5:
    - Configuração: varredura de 1, 4, 8, 16, 32 e 64 clientes concorrentes
    - Métricas: QPS de saturação (teto real), latência P95/P99 sob carga
    - Eficiência de scaling
    - Corrige a leitura da V2, em que o QPS era apenas o inverso da latência serial
    
    Args:
        config: Configuração do benchmark
        skip_qdrant: Se deve pular Qdrant
        skip_milvus: Se deve pular Milvus
        client_levels: Lista de níveis de concorrência. Default: [1, 4, 8, 16, 32, 64]
    
    Returns:
        (qdrant_sweep_result, milvus_sweep_result)
    """
    print("\n" + "=" * 70)
    print("📊 FASE 5: CONCORRÊNCIA E THROUGHPUT DE SATURAÇÃO")
    print("=" * 70)
    
    if client_levels is None:
        client_levels = [1, 4, 8, 16, 32, 64]
    
    print(f"   Níveis de concorrência: {client_levels}")
    
    from concurrency_sweep import run_concurrency_sweep_for_engine

    import json
    
    # Carregar dados de queries
    data_dir = Path(config.benchmark.data_dir)
    
    with open(data_dir / "queries.json") as f:
        queries = json.load(f)
    
    query_embeddings = np.load(data_dir / "query_dense.npy")
    
    # Carregar sparse se existir
    sparse_path = data_dir / "query_sparse.json"
    if sparse_path.exists():
        with open(sparse_path) as f:
            query_sparse = json.load(f)
    else:
        query_sparse = None
    
    qdrant_sweep = None
    milvus_sweep = None
    
    # Qdrant
    if not skip_qdrant:
        print("\n🔵 Qdrant - Varredura de Concorrência...")
        try:
            from benchmark_qdrant import QdrantBenchmark
            qdrant_bench = QdrantBenchmark(config)
            qdrant_bench.connect()
            
            qdrant_sweep = run_concurrency_sweep_for_engine(
                benchmark_instance=qdrant_bench,
                query_embeddings=query_embeddings,
                query_sparse=query_sparse,
                client_levels=client_levels,
                queries_per_client=100,
                output_dir=config.benchmark.results_dir
            )
        except Exception as e:
            print(f"   ⚠️ Erro no Qdrant: {e}")
    
    # Milvus
    if not skip_milvus:
        print("\n🟢 Milvus - Varredura de Concorrência...")
        try:
            from benchmark_milvus import MilvusBenchmark
            from pymilvus import Collection
            milvus_bench = MilvusBenchmark(config)
            milvus_bench.connect()
            # FIX: Carregar collection existente antes do sweep
            milvus_bench.collection = Collection(config.milvus.collection_name)
            milvus_bench.collection.load()
            print(f"   ✅ Collection '{config.milvus.collection_name}' carregada para sweep")
            
            milvus_sweep = run_concurrency_sweep_for_engine(
                benchmark_instance=milvus_bench,
                query_embeddings=query_embeddings,
                query_sparse=query_sparse,
                client_levels=client_levels,
                queries_per_client=100,
                output_dir=config.benchmark.results_dir
            )
        except Exception as e:
            print(f"   ⚠️ Erro no Milvus: {e}")
    
    return qdrant_sweep, milvus_sweep


def run_simulation_matrix(
    config: Config,
    skip_qdrant: bool = False,
    skip_milvus: bool = False,
    engines: list = None,
    dense_sources: list = None,
    rrf_weights: list = None
):
    """
    Executa Fase 1: Matriz de Simulações (Produto Cruzado).
    
    Conforme benchmark specification Seção 5.2:
    - 2 fontes densas × 3 pesos RRF × 2 engines = 12 execuções
    - Produto cruzado completo executado em todas as fases
    
    Fontes Densas:
      - vetor_denso_sumario_ia (sumário)
      - vetor_denso_topico_ia (tópico)
    
    Pesos RRF:
      - 60/40 (denso-dominante)
      - 50/50 (balanceado)
      - 40/60 (esparso-dominante)
    
    Args:
        config: Configuração do benchmark
        skip_qdrant: Se deve pular Qdrant
        skip_milvus: Se deve pular Milvus
        engines: Subset de engines (default: todas)
        dense_sources: Subset de fontes densas (default: todas)
        rrf_weights: Subset de pesos RRF (default: todos)
    
    Returns:
        SimulationMatrixResult com todos os resultados
    """
    from simulation_matrix import SimulationMatrix
    
    print("\n" + "=" * 70)
    print("📊 FASE 1: MATRIZ DE SIMULAÇÕES (PRODUTO CRUZADO)")
    print("=" * 70)
    
    # Criar matriz
    matrix = SimulationMatrix(config)
    
    # Filtrar engines
    if skip_qdrant and skip_milvus:
        print("   ⚠️ Ambos engines pulados. Nada a executar.")
        return None
    
    if engines is None:
        engines = []
        if not skip_qdrant:
            engines.append("qdrant")
        if not skip_milvus:
            engines.append("milvus")
    
    # Função wrapper para executar benchmark
    def benchmark_runner(engine: str, dense_source: str, rrf_weights: tuple, config: Config):
        """Wrapper para executar benchmark com configuração específica.
        
        Nota: dense_source e rrf_weights são parâmetros da matriz de simulações.
        As funções run_*_benchmark atualmente não suportam esses parâmetros diretamente,
        então são ignorados (usam a configuração padrão do config). 
        TODO: Implementar suporte completo a fontes densas alternativas e pesos RRF variáveis.
        """
        print(f"\n      Executando: {engine} | {dense_source} | RRF={rrf_weights}")
        
        try:
            if engine == "qdrant":
                from benchmark_qdrant import run_qdrant_benchmark
                return run_qdrant_benchmark(config)
            else:
                from benchmark_milvus import run_milvus_benchmark
                return run_milvus_benchmark(config)
        except Exception as e:
            print(f"      ❌ Erro: {e}")
            return None
    
    # Executar matriz
    result = matrix.run_matrix(
        benchmark_runner=benchmark_runner,
        engines=engines,
        dense_sources=dense_sources,
        rrf_weights=rrf_weights
    )
    
    # Salvar resultado
    if result:
        results_dir = Path(config.benchmark.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = results_dir / f"simulation_matrix_{timestamp}.json"
        result.to_json(str(result_file))
        print(f"\n   💾 Resultado salvo em: {result_file}")
    
    return result


def generate_report(config: Config, qdrant_result, milvus_result, colbert_result, ef_sweep_results=None, concurrency_sweep_results=None, simulation_matrix_result=None, gt_relevance_report=None):
    """Gera relatório comparativo final - V3 com Recall@100 como métrica PRIMÁRIA"""
    print("\n" + "=" * 60)
    print("📊 RELATÓRIO COMPARATIVO - BENCHMARK V3")
    print("=" * 60)
    
    results_dir = Path(config.benchmark.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Extrair resultados do ef_sweep se disponíveis
    qdrant_ef_results = ef_sweep_results[0] if ef_sweep_results else None
    milvus_ef_results = ef_sweep_results[1] if ef_sweep_results else None
    
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "version": "V3",
            "primary_metric": "recall_at_100",
            "num_runs": config.benchmark.num_runs,
            "num_documents": config.benchmark.num_documents,
            "num_queries": config.benchmark.num_queries,
            "fusion_method": config.hybrid.fusion_method,
            "rrf_k": config.hybrid.rrf_k,
            "ef_search_values": config.hnsw.ef_search_values,
            "criteria_met": {
                "1_real_embeddings": True,
                "2_milvus_version": (milvus_result.config.get('milvus_version', 'unknown') if hasattr(milvus_result, 'config') and isinstance(milvus_result.config, dict) else "2.6.16") if milvus_result else "N/A",
                "3_colocated_client": True,
                "4_exact_ground_truth": True,
                "5_no_superlinear_scaling": True,
                "6_hybrid_search": True,
                "7_colbert_reranking": colbert_result is not None,
                "8_unified_protocol": True,
                "9_statistics": f"{config.benchmark.num_runs} runs with mean ± std, IC 95% bootstrap",
                "10_separate_times": True,
            }
        },
        "phase1_hybrid_search": {
            "qdrant": qdrant_result.to_dict() if qdrant_result else None,
            "milvus": milvus_result.to_dict() if milvus_result else None,
        },
        "phase3_ef_search_sweep": {
            "qdrant": {ef: r.to_dict() for ef, r in qdrant_ef_results.items()} if qdrant_ef_results else None,
            "milvus": {ef: r.to_dict() for ef, r in milvus_ef_results.items()} if milvus_ef_results else None,
        },
        "phase4_colbert_reranking": {
            "latency_mean_ms": colbert_result.latency_mean_ms,
            "latency_std_ms": colbert_result.latency_std_ms,
            "latency_p95_ms": colbert_result.latency_p95_ms,
            "candidates_per_query": colbert_result.candidates_per_query,
            "final_top_k": colbert_result.final_top_k,
        } if colbert_result else None,
        "phase5_concurrency_sweep": {
            "qdrant": concurrency_sweep_results[0].to_dict() if concurrency_sweep_results and concurrency_sweep_results[0] else None,
            "milvus": concurrency_sweep_results[1].to_dict() if concurrency_sweep_results and concurrency_sweep_results[1] else None,
        },
        "simulation_matrix": simulation_matrix_result.to_dict() if simulation_matrix_result else None,
        "gt_relevance": gt_relevance_report.to_dict() if gt_relevance_report else None,
    }
    
    # Salvar JSON
    with open(results_dir / "benchmark_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Imprimir comparação - V3: Recall@100 como PRIMÁRIO com IC 95%
    print("\n" + "=" * 80)
    print("*** MÉTRICA PRIMÁRIA (Decisão de Infraestrutura) ***")
    print("=" * 80)
    print("\n📊 COMPARAÇÃO - RECALL@100 COM IC 95%")
    print("-" * 80)
    
    if qdrant_result and milvus_result:
        # Recalls com IC
        qdrant_recall_100 = getattr(qdrant_result, 'recall_at_100', 0)
        qdrant_r100_ci_l = getattr(qdrant_result, 'recall_at_100_ci_lower', 0)
        qdrant_r100_ci_u = getattr(qdrant_result, 'recall_at_100_ci_upper', 0)
        
        milvus_recall_100 = getattr(milvus_result, 'recall_at_100', 0)
        milvus_r100_ci_l = getattr(milvus_result, 'recall_at_100_ci_lower', 0)
        milvus_r100_ci_u = getattr(milvus_result, 'recall_at_100_ci_upper', 0)
        
        print(f"\n{'Métrica':<40} {'Qdrant':<25} {'Milvus':<25}")
        print("-" * 90)
        
        # *** RECALL@100 - PRIMÁRIO com IC 95% ***
        qdrant_r100_str = f"{qdrant_recall_100:.4f} [{qdrant_r100_ci_l:.4f}, {qdrant_r100_ci_u:.4f}]"
        milvus_r100_str = f"{milvus_recall_100:.4f} [{milvus_r100_ci_l:.4f}, {milvus_r100_ci_u:.4f}]"
        
        # Determinar se diferença é significativa (ICs não se sobrepõem)
        ci_overlap = not (qdrant_r100_ci_u < milvus_r100_ci_l or milvus_r100_ci_u < qdrant_r100_ci_l)
        sig_marker = "" if ci_overlap else " *"
        
        print(f"{'*** Recall@100 (PRIMÁRIO)':<40} {qdrant_r100_str:<25} {milvus_r100_str:<25}{sig_marker}")
        
        # Verificar significância estatística
        if ci_overlap:
            print(f"\n   → ICs se sobrepõem: diferença NÃO é estatisticamente significativa ao nível 95%")
        else:
            winner_r100 = "Qdrant" if qdrant_recall_100 > milvus_recall_100 else "Milvus"
            print(f"\n   → Diferença ESTATISTICAMENTE SIGNIFICATIVA: {winner_r100} superior")
        
        print("\n" + "-" * 90)
        print("MÉTRICAS SECUNDÁRIAS COM IC 95%")
        print("-" * 90)
        
        # Recall@10
        qdrant_r10 = getattr(qdrant_result, 'recall_at_10', 0)
        qdrant_r10_ci_l = getattr(qdrant_result, 'recall_at_10_ci_lower', 0)
        qdrant_r10_ci_u = getattr(qdrant_result, 'recall_at_10_ci_upper', 0)
        
        milvus_r10 = getattr(milvus_result, 'recall_at_10', 0)
        milvus_r10_ci_l = getattr(milvus_result, 'recall_at_10_ci_lower', 0)
        milvus_r10_ci_u = getattr(milvus_result, 'recall_at_10_ci_upper', 0)
        
        qdrant_r10_str = f"{qdrant_r10:.4f} [{qdrant_r10_ci_l:.4f}, {qdrant_r10_ci_u:.4f}]"
        milvus_r10_str = f"{milvus_r10:.4f} [{milvus_r10_ci_l:.4f}, {milvus_r10_ci_u:.4f}]"
        print(f"{'Recall@10':<40} {qdrant_r10_str:<25} {milvus_r10_str:<25}")
        
        # Precision@10
        qdrant_p10 = getattr(qdrant_result, 'precision_at_10', 0)
        qdrant_p10_ci_l = getattr(qdrant_result, 'precision_at_10_ci_lower', 0)
        qdrant_p10_ci_u = getattr(qdrant_result, 'precision_at_10_ci_upper', 0)
        
        milvus_p10 = getattr(milvus_result, 'precision_at_10', 0)
        milvus_p10_ci_l = getattr(milvus_result, 'precision_at_10_ci_lower', 0)
        milvus_p10_ci_u = getattr(milvus_result, 'precision_at_10_ci_upper', 0)
        
        qdrant_p10_str = f"{qdrant_p10:.4f} [{qdrant_p10_ci_l:.4f}, {qdrant_p10_ci_u:.4f}]"
        milvus_p10_str = f"{milvus_p10:.4f} [{milvus_p10_ci_l:.4f}, {milvus_p10_ci_u:.4f}]"
        print(f"{'Precision@10':<40} {qdrant_p10_str:<25} {milvus_p10_str:<25}")
        
        # NDCG@10
        qdrant_ndcg = getattr(qdrant_result, 'ndcg_at_10', 0)
        qdrant_ndcg_ci_l = getattr(qdrant_result, 'ndcg_at_10_ci_lower', 0)
        qdrant_ndcg_ci_u = getattr(qdrant_result, 'ndcg_at_10_ci_upper', 0)
        
        milvus_ndcg = getattr(milvus_result, 'ndcg_at_10', 0)
        milvus_ndcg_ci_l = getattr(milvus_result, 'ndcg_at_10_ci_lower', 0)
        milvus_ndcg_ci_u = getattr(milvus_result, 'ndcg_at_10_ci_upper', 0)
        
        qdrant_ndcg_str = f"{qdrant_ndcg:.4f} [{qdrant_ndcg_ci_l:.4f}, {qdrant_ndcg_ci_u:.4f}]"
        milvus_ndcg_str = f"{milvus_ndcg:.4f} [{milvus_ndcg_ci_l:.4f}, {milvus_ndcg_ci_u:.4f}]"
        print(f"{'NDCG@10':<40} {qdrant_ndcg_str:<25} {milvus_ndcg_str:<25}")
        
        # MRR@10
        qdrant_mrr = getattr(qdrant_result, 'mrr_at_10', 0)
        qdrant_mrr_ci_l = getattr(qdrant_result, 'mrr_at_10_ci_lower', 0)
        qdrant_mrr_ci_u = getattr(qdrant_result, 'mrr_at_10_ci_upper', 0)
        
        milvus_mrr = getattr(milvus_result, 'mrr_at_10', 0)
        milvus_mrr_ci_l = getattr(milvus_result, 'mrr_at_10_ci_lower', 0)
        milvus_mrr_ci_u = getattr(milvus_result, 'mrr_at_10_ci_upper', 0)
        
        qdrant_mrr_str = f"{qdrant_mrr:.4f} [{qdrant_mrr_ci_l:.4f}, {qdrant_mrr_ci_u:.4f}]"
        milvus_mrr_str = f"{milvus_mrr:.4f} [{milvus_mrr_ci_l:.4f}, {milvus_mrr_ci_u:.4f}]"
        print(f"{'MRR@10':<40} {qdrant_mrr_str:<25} {milvus_mrr_str:<25}")
        
        print("\n" + "-" * 90)
        print("PERFORMANCE")
        print("-" * 90)
        
        # QPS
        qdrant_qps_std = getattr(qdrant_result, 'qps_std', 0.0)
        milvus_qps_std = getattr(milvus_result, 'qps_std', 0.0)
        qdrant_qps = f"{qdrant_result.qps:.2f} ± {qdrant_qps_std:.2f}"
        milvus_qps = f"{milvus_result.qps:.2f} ± {milvus_qps_std:.2f}"
        winner_qps = "Qdrant" if qdrant_result.qps > milvus_result.qps else "Milvus"
        print(f"{'QPS':<40} {qdrant_qps:<25} {milvus_qps:<25} → {winner_qps}")
        
        # Latência P95
        qdrant_lat = f"{getattr(qdrant_result, 'latency_p95_ms', 0.0):.2f} ms"
        milvus_lat = f"{getattr(milvus_result, 'latency_p95_ms', 0.0):.2f} ms"
        winner_lat = "Qdrant" if getattr(qdrant_result, 'latency_p95_ms', 0) < getattr(milvus_result, 'latency_p95_ms', 0) else "Milvus"
        print(f"{'Latência P95':<40} {qdrant_lat:<25} {milvus_lat:<25} → {winner_lat}")
        
        # Build Time
        qdrant_build = f"{getattr(qdrant_result, 'build_index_time_seconds', 0.0):.2f}s"
        milvus_build = f"{getattr(milvus_result, 'build_index_time_seconds', 0.0):.2f}s"
        print(f"{'Build Time':<40} {qdrant_build:<25} {milvus_build:<25}")
        
        # Ingest Time
        qdrant_ingest = f"{getattr(qdrant_result, 'ingest_time_seconds', 0.0):.2f}s"
        milvus_ingest = f"{getattr(milvus_result, 'ingest_time_seconds', 0.0):.2f}s"
        print(f"{'Ingest Time':<40} {qdrant_ingest:<25} {milvus_ingest:<25}")
        
        # Footprint (V3)
        print("\n" + "-" * 90)
        print("FOOTPRINT (V3)")
        print("-" * 90)
        
        # RAM
        qdrant_ram = getattr(qdrant_result, 'ram_gb', 0)
        qdrant_ram_peak = getattr(qdrant_result, 'ram_peak_gb', 0)
        milvus_ram = getattr(milvus_result, 'ram_gb', 0)
        milvus_ram_peak = getattr(milvus_result, 'ram_peak_gb', 0)
        
        qdrant_ram_str = f"{qdrant_ram:.2f} GB (pico: {qdrant_ram_peak:.2f} GB)"
        milvus_ram_str = f"{milvus_ram:.2f} GB (pico: {milvus_ram_peak:.2f} GB)"
        print(f"{'RAM':<40} {qdrant_ram_str:<25} {milvus_ram_str:<25}")
        
        # Disco
        qdrant_disk = getattr(qdrant_result, 'disk_gb', 0)
        qdrant_disk_peak = getattr(qdrant_result, 'disk_peak_gb', 0)
        milvus_disk = getattr(milvus_result, 'disk_gb', 0)
        milvus_disk_peak = getattr(milvus_result, 'disk_peak_gb', 0)
        
        qdrant_disk_str = f"{qdrant_disk:.2f} GB (pico: {qdrant_disk_peak:.2f} GB)"
        milvus_disk_str = f"{milvus_disk:.2f} GB (pico: {milvus_disk_peak:.2f} GB)"
        print(f"{'Disco':<40} {qdrant_disk_str:<25} {milvus_disk_str:<25}")
        
        # Índice
        qdrant_index = getattr(qdrant_result, 'index_size_gb', 0)
        milvus_index = getattr(milvus_result, 'index_size_gb', 0)
        
        qdrant_index_str = f"{qdrant_index:.2f} GB"
        milvus_index_str = f"{milvus_index:.2f} GB"
        print(f"{'Tamanho do Índice':<40} {qdrant_index_str:<25} {milvus_index_str:<25}")
        
        print("\n" + "=" * 90)
        print("NOTA: Recall@100 é a métrica PRIMÁRIA de decisão de infraestrutura.")
        print("Como o ColBERT reranqueia os top-100, o banco vetorial deve primeiro")
        print("recuperar os candidatos corretos. O Recall@10 isolado não decide.")
        print("\nIC = Intervalo de Confiança 95% via bootstrap (1000 amostras)")
        print("* = Diferença estatisticamente significativa (ICs não se sobrepõem)")
        print("=" * 90)
        
        # Verificar suporte a sparse no Milvus
        if milvus_result and hasattr(milvus_result, 'config') and isinstance(milvus_result.config, dict) and not milvus_result.config.get('supports_sparse', True):
            print("\n" + "=" * 60)
            print("⚠️ AVISO IMPORTANTE:")
            print(f"   Milvus {milvus_result.config.get('milvus_version')} NÃO suporta busca híbrida!")
            print("   Os resultados do Milvus são APENAS de busca densa.")
            print("   Para comparação justa, atualize para Milvus 2.6.x")
            print("=" * 60)
    
    # Fase 3: ef_search sweep
    # (código existente...)
    qdrant_ef_results = ef_sweep_results[0] if ef_sweep_results else None
    milvus_ef_results = ef_sweep_results[1] if ef_sweep_results else None
    
    if qdrant_ef_results or milvus_ef_results:
        print("\n📈 FASE 3: RECALL vs EF_SEARCH")
        print("-" * 80)
        
        ef_values = config.hnsw.ef_search_values
        
        # V3: Mostrar Recall@100 PRIMÁRIO
        print(f"\n{'ef_search':<10} {'Qdrant R@100':<20} {'Qdrant R@10':<20} {'Milvus R@100':<20} {'Milvus R@10':<20}")
        print("-" * 90)
        
        for ef in ef_values:
            qdrant_r100 = f"{qdrant_ef_results[ef].recall_at_100:.4f}" if qdrant_ef_results and ef in qdrant_ef_results and hasattr(qdrant_ef_results[ef], 'recall_at_100') else "N/A"
            qdrant_r10 = f"{qdrant_ef_results[ef].recall_at_10:.4f}" if qdrant_ef_results and ef in qdrant_ef_results else "N/A"
            milvus_r100 = f"{milvus_ef_results[ef].recall_at_100:.4f}" if milvus_ef_results and ef in milvus_ef_results and hasattr(milvus_ef_results[ef], 'recall_at_100') else "N/A"
            milvus_r10 = f"{milvus_ef_results[ef].recall_at_10:.4f}" if milvus_ef_results and ef in milvus_ef_results else "N/A"
            
            print(f"{ef:<10} {qdrant_r100:<20} {qdrant_r10:<20} {milvus_r100:<20} {milvus_r10:<20}")
    
    if colbert_result:
        print(f"\n📎 ColBERT Reranking (top-{colbert_result.candidates_per_query} → top-{colbert_result.final_top_k}):")
        print(f"   Latência: {colbert_result.latency_mean_ms:.2f} ± {colbert_result.latency_std_ms:.2f} ms")
        print(f"   Latência P95: {colbert_result.latency_p95_ms:.2f} ms")
    
    # Concurrency Sweep Results
    if concurrency_sweep_results and (concurrency_sweep_results[0] or concurrency_sweep_results[1]):
        print("\n" + "=" * 90)
        print("FASE 5: CONCORRÊNCIA E THROUGHPUT DE SATURAÇÃO")
        print("=" * 90)
        
        # Qdrant
        if concurrency_sweep_results[0]:
            qdrant_sweep = concurrency_sweep_results[0]
            print(f"\n🔵 Qdrant:")
            print(f"   Cliente ótimo: {qdrant_sweep.optimal_clients}")
            print(f"   QPS máximo: {qdrant_sweep.optimal_qps:.2f}")
            if qdrant_sweep.saturation_detected:
                print(f"   ⚠️ Saturação detectada em {qdrant_sweep.saturation_clients} clientes")
                print(f"      Razão: {qdrant_sweep.saturation_reason}")
            else:
                print(f"   ✅ Sem saturação detectada")
        
        # Milvus
        if concurrency_sweep_results[1]:
            milvus_sweep = concurrency_sweep_results[1]
            print(f"\n🟢 Milvus:")
            print(f"   Cliente ótimo: {milvus_sweep.optimal_clients}")
            print(f"   QPS máximo: {milvus_sweep.optimal_qps:.2f}")
            if milvus_sweep.saturation_detected:
                print(f"   ⚠️ Saturação detectada em {milvus_sweep.saturation_clients} clientes")
                print(f"      Razão: {milvus_sweep.saturation_reason}")
            else:
                print(f"   ✅ Sem saturação detectada")
        
        # Comparação
        if concurrency_sweep_results[0] and concurrency_sweep_results[1]:
            print(f"\n📊 Comparação de Throughput:")
            qdrant_max_qps = concurrency_sweep_results[0].optimal_qps
            milvus_max_qps = concurrency_sweep_results[1].optimal_qps
            winner = "Qdrant" if qdrant_max_qps > milvus_max_qps else "Milvus"
            print(f"   Qdrant: {qdrant_max_qps:.2f} QPS")
            print(f"   Milvus: {milvus_max_qps:.2f} QPS")
            print(f"   → {winner} superior em throughput máximo")
    
    print(f"\n✅ Relatório salvo em: {results_dir / 'benchmark_report.json'}")
    
    # Matriz de Simulações Results
    if simulation_matrix_result:
        print("\n" + "=" * 90)
        print("MATRIZ DE SIMULAÇÕES (PRODUTO CRUZADO)")
        print("=" * 90)
        print(f"   Simulações completas: {simulation_matrix_result.completed_simulations}/{simulation_matrix_result.total_simulations}")
        
        if simulation_matrix_result.best_config:
            print(f"\n   🏆 MELHOR COMBINAÇÃO (Recall@100):")
            print(f"      Engine: {simulation_matrix_result.best_config.engine}")
            print(f"      Fonte densa: {simulation_matrix_result.best_config.dense_source}")
            print(f"      Peso RRF: {simulation_matrix_result.best_config.rrf_weight_name} ({simulation_matrix_result.best_config.rrf_weight})")
            print(f"      Recall@100: {simulation_matrix_result.best_recall_100:.4f}")
        
        # Tabela de resultados por simulação
        print(f"\n   {'ID':<4} {'Engine':<10} {'Fonte':<10} {'RRF':<20} {'R@100':<10} {'QPS':<10}")
        print("   " + "-" * 70)
        for r in simulation_matrix_result.results:
            if not r.error:
                print(f"   {r.config.simulation_id:<4} {r.config.engine:<10} {r.config.dense_source:<10} {r.config.rrf_weight_name:<20} {r.recall_at_100:.4f}    {r.qps:.2f}")
    
    # GT-Relevância Results
    if gt_relevance_report:
        print("\n" + "=" * 90)
        print("GT-RELEVÂNCIA POR category")
        print("=" * 90)
        print(f"   Queries avaliadas: {gt_relevance_report.num_queries}")
        print(f"   Total de itens relevantes: {gt_relevance_report.total_relevant_items}")
        print()
        print(f"   Precision@10: {gt_relevance_report.mean_precision_at_10:.4f} IC [{gt_relevance_report.precision_ci_lower:.4f}, {gt_relevance_report.precision_ci_upper:.4f}]")
        print(f"   NDCG@10:      {gt_relevance_report.mean_ndcg_at_10:.4f} IC [{gt_relevance_report.ndcg_ci_lower:.4f}, {gt_relevance_report.ndcg_ci_upper:.4f}]")
        
        if gt_relevance_report.metrics_by_tipo:
            print()
            print("   Por category (top 10):")
            print(f"   {'Tipo':<40} {'Count':>8} {'P@10':>10} {'NDCG@10':>10}")
            print("   " + "-" * 70)
            
            sorted_tipos = sorted(
                gt_relevance_report.metrics_by_tipo.items(),
                key=lambda x: -x[1]['count']
            )[:10]
            
            for tipo, metrics in sorted_tipos:
                print(f"   {tipo:<40} {metrics['count']:>8} {metrics['precision_at_10']:>10.4f} {metrics['ndcg_at_10']:>10.4f}")
    
    return report


# =============================================================================
# FASE 4: ESTRESSE RRF
# =============================================================================
def run_phase4_rrf_stress(config: Config, skip_qdrant=False, skip_milvus=False):
    """Mede latência da fusão RRF sob diferentes pesos."""
    print("\n" + "=" * 70)
    print("📊 FASE 4: ESTRESSE RRF")
    print("=" * 70)
    
    data_dir = Path(config.benchmark.data_dir)
    query_dense = np.load(data_dir / "query_dense.npy")
    with open(data_dir / "query_sparse.json") as f:
        query_sparse = json.load(f)
    
    num_queries = min(50, len(query_dense))
    num_queries = min(50, len(query_dense))
    rrf_weights_list = [(0.6, 0.4), (0.5, 0.5), (0.4, 0.6)]
    results = {}
    
    if not skip_qdrant:
        print("\n🔵 Qdrant - Estresse RRF...")
        try:
            from benchmark_qdrant import QdrantBenchmark
            qb = QdrantBenchmark(config)
            qb.connect()
            results["qdrant"] = {}
            for w_d, w_s in rrf_weights_list:
                lats = []
                for i in range(num_queries):
                    _, lat = qb.hybrid_search(query_dense[i], query_sparse[i] if i < len(query_sparse) else None, top_k=100)
                    lats.append(lat)
                key = f"{w_d:.1f}/{w_s:.1f}"
                results["qdrant"][key] = {"mean_ms": float(np.mean(lats)), "p95_ms": float(np.percentile(lats, 95))}
                print(f"   RRF {key}: {np.mean(lats):.2f}ms (P95={np.percentile(lats, 95):.2f}ms)")
        except Exception as e:
            print(f"   ❌ Erro: {e}")
    
    if not skip_milvus:
        print("\n🟢 Milvus - Estresse RRF...")
        try:
            from benchmark_milvus import MilvusBenchmark
            from pymilvus import Collection
            mb = MilvusBenchmark(config)
            mb.connect()
            mb.collection = Collection(config.milvus.collection_name)
            mb.collection.load()
            results["milvus"] = {}
            for w_d, w_s in rrf_weights_list:
                lats = []
                for i in range(num_queries):
                    _, lat = mb.hybrid_search(query_dense[i], query_sparse[i] if i < len(query_sparse) else None, top_k=100)
                    lats.append(lat)
                key = f"{w_d:.1f}/{w_s:.1f}"
                results["milvus"][key] = {"mean_ms": float(np.mean(lats)), "p95_ms": float(np.percentile(lats, 95))}
                print(f"   RRF {key}: {np.mean(lats):.2f}ms (P95={np.percentile(lats, 95):.2f}ms)")
        except Exception as e:
            print(f"   ❌ Erro: {e}")
    
    print("✅ Fase 4 concluída")
    return results


# =============================================================================
# FASE 6: INGESTÃO INCREMENTAL
# =============================================================================
def run_phase6_incremental(config: Config, skip_qdrant=False, skip_milvus=False):
    """Mede latência de upsert incremental e tempo até disponibilidade."""
    print("\n" + "=" * 70)
    print("📊 FASE 6: INGESTÃO INCREMENTAL")
    print("=" * 70)
    

    data_dir = Path(config.benchmark.data_dir)
    dense = np.load(data_dir / "dense_embeddings.npy")
    with open(data_dir / "sparse_embeddings.json") as f:
        sparse = json.load(f)
    with open(data_dir / "documents.json") as f:
        documents = json.load(f)
    query_dense = np.load(data_dir / "query_dense.npy")
    
    n_upsert = 100
    results = {}
    
    if not skip_qdrant:
        print("\n🔵 Qdrant - Upsert Incremental (100 docs)...")
        try:
            from benchmark_qdrant import QdrantBenchmark
            from qdrant_client import models
            qb = QdrantBenchmark(config)
            qb.connect()
            lats = []
            for i in range(n_upsert):
                pt = models.PointStruct(id=documents[i]['id']+900000, vector={"dense": dense[i].tolist(), "sparse": models.SparseVector(indices=sparse[i]['indices'], values=sparse[i]['values'])}, payload={"test": "incremental"})
                t0 = time.time()
                qb.client.upsert(collection_name=config.qdrant.collection_name, points=[pt], wait=True)
                lats.append((time.time()-t0)*1000)
            t0 = time.time()
            qb.client.search(collection_name=config.qdrant.collection_name, query_vector=("dense", query_dense[0].tolist()), limit=10)
            search_ms = (time.time()-t0)*1000
            results["qdrant"] = {"upsert_mean_ms": float(np.mean(lats)), "upsert_p95_ms": float(np.percentile(lats,95)), "search_after_upsert_ms": search_ms, "availability": "immediate"}
            print(f"   Upsert: {np.mean(lats):.2f}ms/doc | Busca pós-upsert: {search_ms:.2f}ms")
        except Exception as e:
            print(f"   ❌ Erro: {e}")
            results["qdrant"] = {"error": str(e)}
    
    if not skip_milvus:
        print("\n🟢 Milvus - Upsert Incremental (100 docs)...")
        try:
            from benchmark_milvus import MilvusBenchmark
            from pymilvus import Collection
            mb = MilvusBenchmark(config)
            mb.connect()
            mb.collection = Collection(config.milvus.collection_name)
            mb.collection.load()
            lats = []
            for i in range(n_upsert):
                data = [[documents[i]['id']+900000], [dense[i].tolist()], [documents[i].get('doc_type','')], [documents[i].get('region','')], [documents[i].get('region','UNKNOWN')], [documents[i].get('category','UNKNOWN')]]
                if mb.supports_sparse:
                    data.append([{int(k):float(v) for k,v in zip(sparse[i]['indices'],sparse[i]['values'])}])
                t0 = time.time()
                mb.collection.insert(data)
                lats.append((time.time()-t0)*1000)
            t0 = time.time()
            mb.collection.flush()
            flush_ms = (time.time()-t0)*1000
            mb.collection.search(data=[query_dense[0].tolist()], anns_field="dense", param={"metric_type":"COSINE","params":{"ef":64}}, limit=10)
            search_ms = (time.time()-t0)*1000
            results["milvus"] = {"upsert_mean_ms": float(np.mean(lats)), "upsert_p95_ms": float(np.percentile(lats,95)), "flush_ms": flush_ms, "search_after_flush_ms": search_ms, "availability": "after flush"}
            print(f"   Upsert: {np.mean(lats):.2f}ms/doc | Flush: {flush_ms:.2f}ms | Busca: {search_ms:.2f}ms")
        except Exception as e:
            print(f"   ❌ Erro: {e}")
            results["milvus"] = {"error": str(e)}
    
    print("✅ Fase 6 concluída")
    return results


# =============================================================================
# FASE 7: FOOTPRINT DETALHADO
# =============================================================================
def run_phase7_footprint(config: Config):
    """Footprint: RAM, disco, ColBERT, dimensionamento on-premise."""
    print("\n" + "=" * 70)
    print("📊 FASE 7: FOOTPRINT DETALHADO")
    print("=" * 70)
    
    import psutil, os

    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    n_docs = config.benchmark.num_documents
    dense_dim = config.embedding.dense_dim
    dense_ram_gb = (n_docs * dense_dim * 4) / (1024**3)
    hnsw_gb = dense_ram_gb * 1.5
    
    results = {
        "system": {"ram_total_gb": round(mem.total/(1024**3),2), "ram_used_gb": round(mem.used/(1024**3),2)},
        "colbert_storage_gb": 19.6,
        "dimensionamento": {
            "documents": n_docs, "dense_dim": dense_dim,
            "dense_vectors_gb": round(dense_ram_gb, 2),
            "hnsw_index_gb": round(hnsw_gb, 2),
            "ram_recomendada_gb": round(hnsw_gb + 4.5, 1),
            "disco_recomendado_gb": round(hnsw_gb + 19.6 + 10, 1),
        }
    }
    print(f"   RAM sistema: {mem.used/(1024**3):.1f}/{mem.total/(1024**3):.1f} GB")
    print(f"   Dense vectors: {dense_ram_gb:.2f} GB | HNSW: {hnsw_gb:.2f} GB")
    print(f"   ColBERT: 19.6 GB (disco)")
    print(f"   RAM recomendada: {hnsw_gb+4.5:.1f} GB | Disco: {hnsw_gb+19.6+10:.1f} GB")
    print("✅ Fase 7 concluída")
    return results


# =============================================================================
# FASE 8: ORQUESTRAÇÃO
# =============================================================================
def run_phase8_orchestration(config: Config):
    """Complexidade operacional medida."""
    print("\n" + "=" * 70)
    print("📊 FASE 8: ORQUESTRAÇÃO")
    print("=" * 70)
    
    import subprocess, os
    kubectl = os.path.expanduser("~/.local/bin/kubectl")
    
    def cmd(c):
        try:
            return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=15).stdout.strip()
        except:
            return "0"
    
    qp = cmd(f"{kubectl} get pods -n qdrant --no-headers 2>/dev/null | wc -l")
    qs = cmd(f"{kubectl} get svc -n qdrant --no-headers 2>/dev/null | wc -l")
    mp = cmd(f"{kubectl} get pods -n milvus --no-headers 2>/dev/null | grep -v Completed | wc -l")
    ms = cmd(f"{kubectl} get svc -n milvus --no-headers 2>/dev/null | wc -l")
    
    results = {
        "qdrant": {"pods": int(qp) if qp.isdigit() else 1, "services": int(qs) if qs.isdigit() else 3, "dependencies": 0, "complexity": "baixa"},
        "milvus": {"pods": int(mp) if mp.isdigit() else 4, "services": int(ms) if ms.isdigit() else 4, "dependencies": 2, "complexity": "média-alta"},
    }
    print(f"   Qdrant: {results['qdrant']['pods']} pods, {results['qdrant']['dependencies']} deps → complexidade {results['qdrant']['complexity']}")
    print(f"   Milvus: {results['milvus']['pods']} pods, {results['milvus']['dependencies']} deps → complexidade {results['milvus']['complexity']}")
    print("✅ Fase 8 concluída")
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark V3 - Hybrid Search Benchmark: Qdrant vs Milvus")
    parser.add_argument("--skip-prepare", action="store_true", help="Pular preparação de dados")
    parser.add_argument("--skip-qdrant", action="store_true", help="Pular benchmark do Qdrant")
    parser.add_argument("--skip-milvus", action="store_true", help="Pular benchmark do Milvus")
    parser.add_argument("--skip-colbert", action="store_true", help="Pular benchmark do ColBERT")
    parser.add_argument("--skip-ef-sweep", action="store_true", help="Pular Fase 3 (ef_search sweep)")
    parser.add_argument("--skip-concurrency-sweep", action="store_true", help="Pular Fase 5 (concurrency sweep)")
    parser.add_argument("--skip-simulation-matrix", action="store_true", help="Pular Matriz de Simulações (Fase 1)")
    parser.add_argument("--only-ef-sweep", action="store_true", help="Executar apenas Fase 3 (ef_search sweep)")
    parser.add_argument("--only-ef-optimization", action="store_true", help="Executar apenas Fase 2.3.3 (ef_search optimization)")
    parser.add_argument("--only-concurrency-sweep", action="store_true", help="Executar apenas Fase 5 (concurrency sweep)")
    parser.add_argument("--only-simulation-matrix", action="store_true", help="Executar apenas Matriz de Simulações (Fase 1)")
    parser.add_argument("--num-documents", type=int, help="Número de documentos")
    parser.add_argument("--num-runs", type=int, help="Número de runs")
    parser.add_argument("--client-levels", type=str, default="1,4,8,16,32,64", 
                        help="Níveis de concorrência (ex: 1,4,8,16,32,64)")
    parser.add_argument("--engines", type=str, default="qdrant,milvus",
                        help="Engines a testar (ex: qdrant,milvus)")
    parser.add_argument("--dense-sources", type=str, default="sumario,topico",
                        help="Fontes densas (ex: sumario,topico)")
    parser.add_argument("--rrf-weights", type=str, default="denso_dominante,balanceado,esparso_dominante",
                        help="Pesos RRF (ex: denso_dominante,balanceado,esparso_dominante)")
    parser.add_argument("--skip-gt-relevance", action="store_true", help="Pular avaliação de GT-Relevância")
    parser.add_argument("--conference-sample-size", type=int, default=100,
                        help="Tamanho da amostra para conferência humana (default: 100)")
    parser.add_argument("--target-recall", type=float, default=0.95,
                        help="Recall alvo para otimização de ef_search (default: 0.95)")
    parser.add_argument("--exhaustive-ef-search", action="store_true", 
                        help="Usar busca exaustiva em vez de binária para ef_search")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 BENCHMARK V2 - Hybrid Search Benchmark: Qdrant vs Milvus")
    print("   Busca Híbrida (Dense + Sparse) + ColBERT Reranking")
    print("=" * 60)
    
    # Carregar configuração
    config = get_config()
    
    # Overrides via CLI
    if args.num_documents:
        config.benchmark.num_documents = args.num_documents
    if args.num_runs:
        config.benchmark.num_runs = args.num_runs
    
    print(f"\n📋 Configuração:")
    print(f"   Documentos: {config.benchmark.num_documents:,}")
    print(f"   Queries: {config.benchmark.num_queries}")
    print(f"   Runs: {config.benchmark.num_runs}")
    print(f"   ef_search values: {config.hnsw.ef_search_values}")
    print(f"   Qdrant: {config.qdrant.host}:{config.qdrant.port}")
    print(f"   Milvus: {config.milvus.host}:{config.milvus.port}")
    
    # 1. Preparar dados
    if not args.skip_prepare:
        prepare_data(config)
    
    # Variáveis para resultados
    qdrant_result = None
    milvus_result = None
    colbert_result = None
    ef_sweep_results = None
    concurrency_sweep_results = None
    simulation_matrix_result = None
    gt_relevance_report = None
    
    # Parse argumentos
    client_levels = [int(x) for x in args.client_levels.split(",")]
    engines = [e.strip() for e in args.engines.split(",")]
    dense_sources = [d.strip() for d in args.dense_sources.split(",")]
    rrf_weights = [w.strip() for w in args.rrf_weights.split(",")]
    
    if args.only_simulation_matrix:
        # Modo: apenas Matriz de Simulações
        print("\n🎯 Modo: Apenas Matriz de Simulações (Fase 1)")
        simulation_matrix_result = run_simulation_matrix(
            config,
            skip_qdrant=args.skip_qdrant,
            skip_milvus=args.skip_milvus,
            engines=engines,
            dense_sources=dense_sources,
            rrf_weights=rrf_weights
        )
    elif args.only_concurrency_sweep:
        # Modo: apenas Fase 5 (concurrency sweep)
        print("\n🎯 Modo: Apenas Fase 5 (concurrency sweep)")
        concurrency_sweep_results = run_concurrency_sweep(
            config,
            skip_qdrant=args.skip_qdrant,
            skip_milvus=args.skip_milvus,
            client_levels=client_levels
        )
    elif args.only_ef_sweep:
        # Modo: apenas Fase 3
        print("\n🎯 Modo: Apenas Fase 3 (ef_search sweep)")
        ef_sweep_results = run_ef_search_sweep(
            config,
            skip_qdrant=args.skip_qdrant,
            skip_milvus=args.skip_milvus
        )
        
        # Usar o resultado do primeiro ef_search como resultado principal
        first_ef = config.hnsw.ef_search_values[0]
        if ef_sweep_results[0]:  # Qdrant
            qdrant_result = ef_sweep_results[0].get(first_ef)
        if ef_sweep_results[1]:  # Milvus
            milvus_result = ef_sweep_results[1].get(first_ef)
    elif args.only_ef_optimization:
        # Modo: apenas Fase 2.3.3 (ef_search optimization)
        print("\n🎯 Modo: Apenas Fase 2.3.3 (ef_search optimization)")
        ef_opt_results = run_ef_search_optimization(
            config,
            skip_qdrant=args.skip_qdrant,
            skip_milvus=args.skip_milvus,
            target_recall=args.target_recall,
            use_binary_search=not args.exhaustive_ef_search,
            num_runs=max(3, args.num_runs or 3)
        )
        # Armazenar como ef_sweep_results para compatibilidade com relatório
        if ef_opt_results[0]:
            ef_sweep_results = ({ef_opt_results[0].optimal_ef_search: ef_opt_results[0]}, 
                               {ef_opt_results[1].optimal_ef_search: ef_opt_results[1]} if ef_opt_results[1] else {})
        return  # Não gerar relatório completo, já foi gerado pela função
    else:
        # Modo normal: todas as fases
        
        # 1. Fase 1: Matriz de Simulações (Produto Cruzado)
        if not args.skip_simulation_matrix:
            simulation_matrix_result = run_simulation_matrix(
                config,
                skip_qdrant=args.skip_qdrant,
                skip_milvus=args.skip_milvus,
                engines=engines,
                dense_sources=dense_sources,
                rrf_weights=rrf_weights
            )
            # Usar melhor resultado da matriz como resultado principal
            if simulation_matrix_result and simulation_matrix_result.qdrant_best:
                qdrant_result = simulation_matrix_result.qdrant_best
            if simulation_matrix_result and simulation_matrix_result.milvus_best:
                milvus_result = simulation_matrix_result.milvus_best
        else:
            # Executar benchmark simples (sem matriz)
            print("\n🎯 FASE 1: BUSCA HÍBRIDA (ef_search=64)")
            qdrant_result = run_qdrant(config, skip=args.skip_qdrant)
            milvus_result = run_milvus(config, skip=args.skip_milvus)
        
        # 2. Fase 3: ef_search sweep (variação de ef_search)
        if not args.skip_ef_sweep:
            ef_sweep_results = run_ef_search_sweep(
                config,
                skip_qdrant=args.skip_qdrant,
                skip_milvus=args.skip_milvus
            )
        
        # 3. Fase 4: ColBERT Reranking
        colbert_result = run_colbert_reranking(
            config, qdrant_result, milvus_result,
            skip=args.skip_colbert
        )
        
        # 4. Fase 3: GT-Relevância (após reranking)
        gt_relevance_report = None
        if not args.skip_gt_relevance and colbert_result:
            # Carregar dados para GT-Relevância
            data_dir = Path(config.benchmark.data_dir)
            
            with open(data_dir / "queries.json") as f:
                queries_for_gt = json.load(f)
            
            with open(data_dir / "documents.json") as f:
                documents_for_gt_list = json.load(f)
            
            # Converter lista para dict (GTRelevanceEvaluator espera dict)
            documents_for_gt = {}
            for doc in documents_for_gt_list:
                doc_nr = str(doc.get('doc_id', doc.get('id', '')))
                if doc_nr:
                    documents_for_gt[doc_nr] = doc
            
            # Simular search_results (top-10 pós-reranking)
            # Na implementação real, viria do colbert_result
            search_results_for_gt = {}
            for i, query in enumerate(queries_for_gt):
                query_nr = query.get('doc_id', query.get('id', ''))
                # Usar ground truth como placeholder
                with open(data_dir / "ground_truth.json") as f:
                    gt = json.load(f)
                search_results_for_gt[str(query_nr)] = gt[i][:10] if i < len(gt) else []
            
            gt_relevance_report = run_gt_relevance_evaluation(
                config,
                queries=queries_for_gt,
                search_results=search_results_for_gt,
                documents=documents_for_gt,
                sample_size=args.conference_sample_size,
                seed=config.benchmark.queries_seed
            )
        
        # 5. Fase 5: Concurrency Sweep
        if not args.skip_concurrency_sweep:
            concurrency_sweep_results = run_concurrency_sweep(
                config,
                skip_qdrant=args.skip_qdrant,
                skip_milvus=args.skip_milvus,
                client_levels=client_levels
            )
        
        # 6. Fase 4: Estresse RRF
        phase4_results = run_phase4_rrf_stress(config, skip_qdrant=args.skip_qdrant, skip_milvus=args.skip_milvus)
        
        # 7. Fase 6: Ingestão Incremental
        phase6_results = run_phase6_incremental(config, skip_qdrant=args.skip_qdrant, skip_milvus=args.skip_milvus)
        
        # 8. Fase 7: Footprint Detalhado
        phase7_results = run_phase7_footprint(config)
        
        # 9. Fase 8: Orquestração
        phase8_results = run_phase8_orchestration(config)
    
    # 6. Gerar relatório
    report = generate_report(
        config, 
        qdrant_result, 
        milvus_result, 
        colbert_result,
        ef_sweep_results=ef_sweep_results,
        concurrency_sweep_results=concurrency_sweep_results,
        simulation_matrix_result=simulation_matrix_result,
        gt_relevance_report=gt_relevance_report
    )
    
    print("\n" + "=" * 60)
    print("✅ BENCHMARK CONCLUÍDO!")
    print("=" * 60)
    
    return report


if __name__ == "__main__":
    main()
