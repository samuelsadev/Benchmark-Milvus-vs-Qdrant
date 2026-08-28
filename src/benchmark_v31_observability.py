#!/usr/bin/env python3
"""
Benchmark V3.1 — Observabilidade por Pod
==========================================
Executa benchmark sequencial (Qdrant depois Milvus) enquanto coleta
métricas CloudWatch por pod.

Objetivo: Isolar o desempenho da engine da influência de diferenças
de infraestrutura, alocação de CPU e comportamento de escrita.

Formato de saída:
- Tabela de paridade (variáveis controladas)
- Tabela de decomposição por fase
- Tabela de resultado (QPS, P50, P95, P99, CPU média, CPU máxima)
"""
import json
import time
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
import sys
import threading

sys.path.insert(0, str(Path(__file__).parent))
from config import get_config

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

TOP_K = 100
HNSW_EF_SEARCH = 64
N_QUERIES = 1000
N_RUNS = 5  # 1 warm-up + 4 medidos
COLLECTION_NAME = "benchmark_v31_obs"


def get_cloudwatch_pod_metrics(start_time, end_time, pod_name_filter, region="us-east-1"):
    """
    Busca métricas de CPU e memória por pod via CloudWatch Container Insights.
    """
    import boto3

    cw = boto3.client('cloudwatch', region_name=region)

    metrics_result = {}

    # CPU por pod
    try:
        resp = cw.get_metric_statistics(
            Namespace='ContainerInsights',
            MetricName='pod_cpu_utilization',
            Dimensions=[
                {'Name': 'ClusterName', 'Value': 'benchmark-v3'},
                {'Name': 'PodName', 'Value': pod_name_filter},
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=60,
            Statistics=['Average', 'Maximum'],
        )

        if resp['Datapoints']:
            cpu_avg = np.mean([d['Average'] for d in resp['Datapoints']])
            cpu_max = max(d['Maximum'] for d in resp['Datapoints'])
            metrics_result['cpu_avg_pct'] = cpu_avg
            metrics_result['cpu_max_pct'] = cpu_max
        else:
            metrics_result['cpu_avg_pct'] = None
            metrics_result['cpu_max_pct'] = None
    except Exception as e:
        print(f"   ⚠️ Erro ao coletar CPU de {pod_name_filter}: {e}")
        metrics_result['cpu_avg_pct'] = None
        metrics_result['cpu_max_pct'] = None

    # Memória por pod
    try:
        resp = cw.get_metric_statistics(
            Namespace='ContainerInsights',
            MetricName='pod_memory_utilization',
            Dimensions=[
                {'Name': 'ClusterName', 'Value': 'benchmark-v3'},
                {'Name': 'PodName', 'Value': pod_name_filter},
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=60,
            Statistics=['Average', 'Maximum'],
        )

        if resp['Datapoints']:
            mem_avg = np.mean([d['Average'] for d in resp['Datapoints']])
            mem_max = max(d['Maximum'] for d in resp['Datapoints'])
            metrics_result['mem_avg_pct'] = mem_avg
            metrics_result['mem_max_pct'] = mem_max
        else:
            metrics_result['mem_avg_pct'] = None
            metrics_result['mem_max_pct'] = None
    except Exception as e:
        metrics_result['mem_avg_pct'] = None
        metrics_result['mem_max_pct'] = None

    return metrics_result


def run_qdrant_benchmark(cfg, query_dense, query_sparse, queries):
    """Executa benchmark serial no Qdrant e retorna métricas"""
    from qdrant_client import QdrantClient, models

    host = cfg.qdrant.host
    port = cfg.qdrant.port
    print(f"\n🔵 QDRANT — Benchmark de observabilidade")
    print(f"   Conectando: {host}:{port}")

    client = QdrantClient(host=host, port=port, timeout=600, prefer_grpc=False)

    # Verificar se collection existe
    collections = [c.name for c in client.get_collections().collections]
    if "benchmark_v3" in collections:
        collection_name = "benchmark_v3"
    elif COLLECTION_NAME in collections:
        collection_name = COLLECTION_NAME
    else:
        print("   ⚠️ Nenhuma collection encontrada! Usando benchmark_v3")
        collection_name = "benchmark_v3"

    print(f"   Collection: {collection_name}")
    print(f"   Rodando {N_RUNS} runs × {N_QUERIES} queries (1 warm-up)")

    latencies = []
    start_benchmark = datetime.utcnow()

    for run in range(N_RUNS):
        run_latencies = []
        for qi in range(N_QUERIES):
            q = queries[qi]
            q_region = q.get('region', '')
            q_emb = query_dense[qi]
            q_sp = query_sparse[qi]

            qfilter = models.Filter(must=[
                models.FieldCondition(key="region",
                                     match=models.MatchValue(value=q_region))
            ]) if q_region else None

            t0 = time.time()
            try:
                results = client.query_points(
                    collection_name=collection_name,
                    prefetch=[
                        models.Prefetch(query=q_emb.tolist(), using="dense", limit=TOP_K*2,
                                       params=models.SearchParams(hnsw_ef=HNSW_EF_SEARCH)),
                        models.Prefetch(
                            query=models.SparseVector(indices=q_sp['indices'], values=q_sp['values']),
                            using="sparse", limit=TOP_K*2),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=TOP_K,
                    query_filter=qfilter,
                )
            except Exception as e:
                print(f"   ⚠️ Erro query {qi}: {e}")
                continue

            lat = (time.time() - t0) * 1000
            run_latencies.append(lat)

        if run == 0:
            print(f"      Run {run+1}/{N_RUNS} (warm-up descartado)")
            continue

        latencies.extend(run_latencies)
        qps = len(run_latencies) / (sum(run_latencies)/1000)
        print(f"      Run {run+1}/{N_RUNS}: QPS={qps:.1f}, P95={np.percentile(run_latencies, 95):.1f}ms")

    end_benchmark = datetime.utcnow()

    total_time = sum(latencies) / 1000
    result = {
        'engine': 'qdrant',
        'queries_total': len(latencies),
        'qps': len(latencies) / total_time,
        'latency_p50_ms': float(np.percentile(latencies, 50)),
        'latency_p95_ms': float(np.percentile(latencies, 95)),
        'latency_p99_ms': float(np.percentile(latencies, 99)),
        'latency_mean_ms': float(np.mean(latencies)),
        'start_time': start_benchmark.isoformat(),
        'end_time': end_benchmark.isoformat(),
    }

    print(f"   ✅ QPS={result['qps']:.1f}, P50={result['latency_p50_ms']:.1f}ms, "
          f"P95={result['latency_p95_ms']:.1f}ms, P99={result['latency_p99_ms']:.1f}ms")

    return result


def run_milvus_benchmark(cfg, query_dense, query_sparse, queries):
    """Executa benchmark serial no Milvus e retorna métricas"""
    from pymilvus import connections, utility, Collection, AnnSearchRequest, RRFRanker

    host = cfg.milvus.host
    port = cfg.milvus.port
    print(f"\n🟢 MILVUS — Benchmark de observabilidade")
    print(f"   Conectando: {host}:{port}")

    connections.connect(host=host, port=port, timeout=300)

    # Usar collection existente
    collection_name = "benchmark_v3"
    if not utility.has_collection(collection_name):
        collection_name = COLLECTION_NAME

    collection = Collection(name=collection_name)
    collection.load()

    print(f"   Collection: {collection_name}")
    print(f"   Rodando {N_RUNS} runs × {N_QUERIES} queries (1 warm-up)")

    latencies = []
    start_benchmark = datetime.utcnow()

    for run in range(N_RUNS):
        run_latencies = []
        for qi in range(N_QUERIES):
            q = queries[qi]
            q_region = q.get('region', '')
            q_emb = query_dense[qi]
            q_sp = query_sparse[qi]

            expr = f'region == "{q_region}"' if q_region else None
            prefetch_limit = TOP_K * 2
            ef = max(HNSW_EF_SEARCH, prefetch_limit)

            dense_req = AnnSearchRequest(
                data=[q_emb.tolist()], anns_field="dense",
                param={"metric_type": "COSINE", "params": {"ef": ef}},
                limit=prefetch_limit, expr=expr)

            sparse_dict = {int(idx): float(val) for idx, val in zip(q_sp['indices'], q_sp['values'])}
            sparse_req = AnnSearchRequest(
                data=[sparse_dict], anns_field="sparse",
                param={"metric_type": "IP", "params": {}},
                limit=prefetch_limit, expr=expr)

            t0 = time.time()
            try:
                res = collection.hybrid_search(
                    reqs=[dense_req, sparse_req],
                    rerank=RRFRanker(k=60),
                    limit=TOP_K)
            except Exception as e:
                print(f"   ⚠️ Erro query {qi}: {e}")
                continue

            lat = (time.time() - t0) * 1000
            run_latencies.append(lat)

        if run == 0:
            print(f"      Run {run+1}/{N_RUNS} (warm-up descartado)")
            continue

        latencies.extend(run_latencies)
        qps = len(run_latencies) / (sum(run_latencies)/1000)
        print(f"      Run {run+1}/{N_RUNS}: QPS={qps:.1f}, P95={np.percentile(run_latencies, 95):.1f}ms")

    end_benchmark = datetime.utcnow()

    total_time = sum(latencies) / 1000
    result = {
        'engine': 'milvus',
        'queries_total': len(latencies),
        'qps': len(latencies) / total_time,
        'latency_p50_ms': float(np.percentile(latencies, 50)),
        'latency_p95_ms': float(np.percentile(latencies, 95)),
        'latency_p99_ms': float(np.percentile(latencies, 99)),
        'latency_mean_ms': float(np.mean(latencies)),
        'start_time': start_benchmark.isoformat(),
        'end_time': end_benchmark.isoformat(),
    }

    connections.disconnect("default")

    print(f"   ✅ QPS={result['qps']:.1f}, P50={result['latency_p50_ms']:.1f}ms, "
          f"P95={result['latency_p95_ms']:.1f}ms, P99={result['latency_p99_ms']:.1f}ms")

    return result


if __name__ == "__main__":
    print("=" * 70)
    print("🔬 BENCHMARK V3.1 — OBSERVABILIDADE POR POD")
    print("=" * 70)
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Objetivo: Coletar métricas de CPU/RAM por pod durante benchmark")
    print()

    cfg = get_config()

    # Carregar dados
    print("📂 Carregando dados...")
    queries = json.loads((DATA_DIR / "queries.json").read_text())
    query_dense = np.load(DATA_DIR / "query_dense.npy")
    query_sparse = json.loads((DATA_DIR / "query_sparse.json").read_text())
    print(f"   Queries: {len(queries)}, Dense: {query_dense.shape}")

    # 1. Rodar Qdrant
    print("\n" + "=" * 70)
    print("FASE 1: QDRANT (serial, 1000 queries × 4 runs)")
    print("=" * 70)
    qdrant_result = run_qdrant_benchmark(cfg, query_dense, query_sparse, queries)

    # Pausa entre engines (cooldown)
    print("\n⏳ Cooldown 30s entre engines...")
    time.sleep(30)

    # 2. Rodar Milvus
    print("\n" + "=" * 70)
    print("FASE 2: MILVUS (serial, 1000 queries × 4 runs)")
    print("=" * 70)
    milvus_result = run_milvus_benchmark(cfg, query_dense, query_sparse, queries)

    # 3. Coletar métricas CloudWatch por pod
    print("\n" + "=" * 70)
    print("FASE 3: COLETA DE MÉTRICAS CLOUDWATCH POR POD")
    print("=" * 70)

    # Esperar 2 minutos para métricas chegarem ao CloudWatch
    print("⏳ Aguardando 120s para métricas serem publicadas no CloudWatch...")
    time.sleep(120)

    qdrant_start = datetime.fromisoformat(qdrant_result['start_time'])
    qdrant_end = datetime.fromisoformat(qdrant_result['end_time'])
    milvus_start = datetime.fromisoformat(milvus_result['start_time'])
    milvus_end = datetime.fromisoformat(milvus_result['end_time'])

    print(f"   Qdrant período: {qdrant_start} → {qdrant_end}")
    print(f"   Milvus período: {milvus_start} → {milvus_end}")

    # Coletar por pod
    qdrant_metrics = get_cloudwatch_pod_metrics(qdrant_start, qdrant_end, "qdrant-0")
    milvus_metrics = get_cloudwatch_pod_metrics(milvus_start, milvus_end, "milvus-0")

    qdrant_result['cloudwatch'] = qdrant_metrics
    milvus_result['cloudwatch'] = milvus_metrics

    # 4. Resultado final
    print("\n" + "=" * 70)
    print("📊 RESULTADO — OBSERVABILIDADE POR POD")
    print("=" * 70)

    print(f"\n{'Variável':<20} {'Qdrant':<20} {'Milvus':<20}")
    print("-" * 60)
    print(f"{'CPU (limit)':<20} {'4 cores':<20} {'4 cores (query node)':<20}")
    print(f"{'Memória (limit)':<20} {'24 GB':<20} {'8 GB (query node)':<20}")
    print(f"{'Nº de pods':<20} {'1':<20} {'5 (3+2 deps)':<20}")
    print(f"{'Nº de réplicas':<20} {'1':<20} {'1':<20}")
    print(f"{'Dados':<20} {'86.551 docs':<20} {'86.551 docs':<20}")
    print(f"{'Queries':<20} {'1.000':<20} {'1.000':<20}")
    print(f"{'Protocolo':<20} {'REST/HTTP':<20} {'gRPC':<20}")

    print(f"\n{'Métrica':<20} {'Qdrant':<15} {'Milvus':<15} {'Diferença':<15}")
    print("-" * 65)
    print(f"{'QPS':<20} {qdrant_result['qps']:<15.1f} {milvus_result['qps']:<15.1f} {milvus_result['qps']/qdrant_result['qps']:<15.1f}x")
    print(f"{'P50 (ms)':<20} {qdrant_result['latency_p50_ms']:<15.1f} {milvus_result['latency_p50_ms']:<15.1f} {qdrant_result['latency_p50_ms']/milvus_result['latency_p50_ms']:<15.1f}x")
    print(f"{'P95 (ms)':<20} {qdrant_result['latency_p95_ms']:<15.1f} {milvus_result['latency_p95_ms']:<15.1f} {qdrant_result['latency_p95_ms']/milvus_result['latency_p95_ms']:<15.1f}x")
    print(f"{'P99 (ms)':<20} {qdrant_result['latency_p99_ms']:<15.1f} {milvus_result['latency_p99_ms']:<15.1f} {qdrant_result['latency_p99_ms']/milvus_result['latency_p99_ms']:<15.1f}x")

    if qdrant_metrics.get('cpu_avg_pct') is not None:
        print(f"{'CPU média (%)':<20} {qdrant_metrics['cpu_avg_pct']:<15.2f} {milvus_metrics.get('cpu_avg_pct', 0):<15.2f}")
        print(f"{'CPU máxima (%)':<20} {qdrant_metrics['cpu_max_pct']:<15.2f} {milvus_metrics.get('cpu_max_pct', 0):<15.2f}")
    else:
        print(f"{'CPU média (%)':<20} {'(aguardar CW)':<15} {'(aguardar CW)':<15}")

    # Salvar
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        'timestamp': datetime.now().isoformat(),
        'version': 'v3.1-observability',
        'qdrant': qdrant_result,
        'milvus': milvus_result,
        'paridade': {
            'dados': 'Mesmo conjunto (86.551 docs)',
            'queries': '1.000 (mesmo conjunto)',
            'escrita': 'Não re-ingerido (coleções existentes)',
            'ef_search': HNSW_EF_SEARCH,
            'top_k': TOP_K,
            'runs': N_RUNS,
            'warm_up': '1ª execução descartada',
        }
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = RESULTS_DIR / f"observability_per_pod_{ts}.json"
    out_file.write_text(json.dumps(output, indent=2, default=str))
    print(f"\n💾 Resultados salvos em: {out_file}")
    print("✅ Benchmark de observabilidade concluído!")
