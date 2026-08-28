#!/usr/bin/env python3
"""
Benchmark V3.1 — Dual Dense Source (Corrigido)
================================================
Corrige a V3: agora alterna efetivamente entre sumário e tópico.

Execução:
  python benchmark_v31_dual_dense.py

Resultado: 4 simulações efetivas (2 engines × 2 dense_source)
RRF: nativo k=60 pesos iguais (limitação documentada)
"""
import json
import time
import math
import re
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent))
from config import get_config

# ============================================================
# CONFIGURAÇÃO
# ============================================================
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

COLLECTION_NAME = "benchmark_v31"
DENSE_DIM = 1024
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 128
HNSW_EF_SEARCH = 64
TOP_K = 100
RRF_K = 60
BATCH_SIZE = 500  # Reduzido para evitar timeout com dual dense
N_RUNS = 5


# ============================================================
# MÉTRICAS
# ============================================================
def normalize_label(label):
    if not label:
        return ''
    return re.sub(r'\s+', ' ', label).strip()

def recall_at_k(retrieved, ground_truth, k):
    return len(set(retrieved[:k]) & set(ground_truth[:k])) / k

def precision_at_10_relevance(retrieved_ids, query_category, id_to_doc):
    qt = normalize_label(query_category)
    matches = sum(1 for did in retrieved_ids[:10]
                  if normalize_label(id_to_doc.get(did, {}).get('category', '')) == qt)
    return matches / 10

def ndcg_at_10(retrieved_ids, query_category, id_to_doc):
    qt = normalize_label(query_category)
    dcg = 0.0
    for i, did in enumerate(retrieved_ids[:10]):
        dt = normalize_label(id_to_doc.get(did, {}).get('category', ''))
        rel = 1.0 if dt == qt else 0.0
        dcg += rel / math.log2(i + 2)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(10))
    return dcg / idcg if idcg > 0 else 0.0

def mrr_at_10(retrieved_ids, query_category, id_to_doc):
    qt = normalize_label(query_category)
    for i, did in enumerate(retrieved_ids[:10]):
        dt = normalize_label(id_to_doc.get(did, {}).get('category', ''))
        if dt == qt:
            return 1.0 / (i + 1)
    return 0.0

def bootstrap_ci(values, n=1000):
    np.random.seed(42)
    means = [np.mean(np.random.choice(values, len(values), replace=True)) for _ in range(n)]
    return [np.percentile(means, 2.5), np.percentile(means, 97.5)]


# ============================================================
# QDRANT ENGINE
# ============================================================
def run_qdrant(cfg, documents, dense_sumario, dense_topico, sparse_embeddings,
               queries, query_dense_sum, query_dense_top, query_sparse,
               ground_truth, id_to_doc):
    """Executa benchmark no Qdrant com dual dense"""
    from qdrant_client import QdrantClient, models

    host = cfg.qdrant.host
    port = cfg.qdrant.port
    print(f"\n{'='*70}")
    print(f"🔵 QDRANT — Ingestão com dual dense")
    print(f"{'='*70}")
    print(f"🔗 Conectando: {host}:{port}")

    client = QdrantClient(host=host, port=port, timeout=600, prefer_grpc=False)

    # Criar collection com 2 campos densos
    print(f"📦 Criando collection: {COLLECTION_NAME}")
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense_sumario": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE),
            "dense_topico": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
        },
        hnsw_config=models.HnswConfigDiff(m=HNSW_M, ef_construct=HNSW_EF_CONSTRUCTION),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
    )
    print(f"✅ Collection criada com dense_sumario + dense_topico + sparse")

    # Ingestão
    print(f"📥 Ingerindo {len(documents):,} documentos...")
    start = time.time()
    for i in range(0, len(documents), BATCH_SIZE):
        batch_docs = documents[i:i+BATCH_SIZE]
        batch_sum = dense_sumario[i:i+BATCH_SIZE]
        batch_top = dense_topico[i:i+BATCH_SIZE]
        batch_sp = sparse_embeddings[i:i+BATCH_SIZE]

        points = []
        for j, (doc, ds, dt, sp) in enumerate(zip(batch_docs, batch_sum, batch_top, batch_sp)):
            points.append(models.PointStruct(
                id=doc['id'],
                vector={
                    "dense_sumario": ds.tolist(),
                    "dense_topico": dt.tolist(),
                    "sparse": models.SparseVector(indices=sp['indices'], values=sp['values']),
                },
                payload={
                    'region': doc.get('region', 'UNKNOWN'),
                    'category': doc.get('category', 'UNKNOWN'),
                    'doc_id': doc.get('doc_id', ''),
                }
            ))
        client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
        if (i + BATCH_SIZE) % 10000 == 0:
            print(f"   {i+BATCH_SIZE:,} inseridos...")

    ingest_time = time.time() - start
    print(f"✅ Ingestão: {ingest_time:.1f}s")

    # Aguardar index
    print("🔨 Aguardando índice...")
    time.sleep(5)
    idx_start = time.time()
    while True:
        info = client.get_collection(COLLECTION_NAME)
        if info.status.name == "GREEN":
            break
        time.sleep(2)
    idx_time = time.time() - idx_start
    print(f"✅ Índice pronto: {idx_time:.1f}s")

    # Rodar buscas para cada dense_source
    results = {}
    for source_name, query_embs in [("sumario", query_dense_sum), ("topico", query_dense_top)]:
        print(f"\n   📊 Executando: Qdrant + {source_name}")
        dense_field = f"dense_{source_name}"

        all_recall100 = []
        all_recall10 = []
        all_p10 = []
        all_ndcg = []
        all_mrr = []
        latencies = []

        for run in range(N_RUNS):
            run_recall100 = []
            run_recall10 = []
            run_p10 = []
            run_ndcg = []
            run_mrr = []

            for qi in range(len(queries)):
                q = queries[qi]
                q_region = q.get('region', '')
                q_category = q.get('category', '')
                q_emb = query_embs[qi]
                q_sp = query_sparse[qi]

                # Filtro regional
                qfilter = models.Filter(must=[
                    models.FieldCondition(key="region",
                                         match=models.MatchValue(value=q_region))
                ]) if q_region else None

                t0 = time.time()
                results_q = client.query_points(
                    collection_name=COLLECTION_NAME,
                    prefetch=[
                        models.Prefetch(query=q_emb.tolist(), using=dense_field, limit=TOP_K*2,
                                       params=models.SearchParams(hnsw_ef=HNSW_EF_SEARCH)),
                        models.Prefetch(
                            query=models.SparseVector(indices=q_sp['indices'], values=q_sp['values']),
                            using="sparse", limit=TOP_K*2),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=TOP_K,
                    query_filter=qfilter,
                )
                lat = (time.time() - t0) * 1000
                latencies.append(lat)

                ids = [p.id for p in results_q.points]
                gt = ground_truth[qi]

                run_recall100.append(recall_at_k(ids, gt, 100))
                run_recall10.append(recall_at_k(ids, gt, 10))
                run_p10.append(precision_at_10_relevance(ids, q_category, id_to_doc))
                run_ndcg.append(ndcg_at_10(ids, q_category, id_to_doc))
                run_mrr.append(mrr_at_10(ids, q_category, id_to_doc))

            if run == 0:
                print(f"      Run {run+1}/{N_RUNS} (warm-up, descartado)")
                latencies = []  # descartar warm-up
                continue

            all_recall100.extend(run_recall100)
            all_recall10.extend(run_recall10)
            all_p10.extend(run_p10)
            all_ndcg.extend(run_ndcg)
            all_mrr.extend(run_mrr)

            r100 = np.mean(run_recall100)
            p10 = np.mean(run_p10)
            print(f"      Run {run+1}/{N_RUNS}: R@100={r100:.4f}, P@10={p10:.4f}")

        # Calcular métricas finais (sobre todas as queries de todos os runs válidos)
        per_query_r100 = [np.mean(all_recall100[i::len(queries)]) for i in range(len(queries))]
        per_query_p10 = [np.mean(all_p10[i::len(queries)]) for i in range(len(queries))]
        per_query_ndcg = [np.mean(all_ndcg[i::len(queries)]) for i in range(len(queries))]
        per_query_mrr = [np.mean(all_mrr[i::len(queries)]) for i in range(len(queries))]

        total_time = sum(latencies) / 1000  # seconds
        qps = len(latencies) / total_time if total_time > 0 else 0

        results[source_name] = {
            'engine': 'qdrant',
            'dense_source': source_name,
            'recall_at_100': float(np.mean(per_query_r100)),
            'recall_at_100_ci': bootstrap_ci(per_query_r100),
            'recall_at_10': float(np.mean(all_recall10)) / (N_RUNS-1),
            'precision_at_10': float(np.mean(per_query_p10)),
            'precision_at_10_ci': bootstrap_ci(per_query_p10),
            'ndcg_at_10': float(np.mean(per_query_ndcg)),
            'mrr_at_10': float(np.mean(per_query_mrr)),
            'qps': qps,
            'latency_p50_ms': float(np.percentile(latencies, 50)),
            'latency_p95_ms': float(np.percentile(latencies, 95)),
            'latency_p99_ms': float(np.percentile(latencies, 99)),
            'ingest_time_s': ingest_time,
        }
        print(f"      ✅ R@100={results[source_name]['recall_at_100']:.4f}, "
              f"P@10={results[source_name]['precision_at_10']:.4f}, "
              f"QPS={qps:.1f}")

    # Cleanup
    client.delete_collection(COLLECTION_NAME)
    return results


# ============================================================
# MILVUS ENGINE
# ============================================================
def run_milvus(cfg, documents, dense_sumario, dense_topico, sparse_embeddings,
               queries, query_dense_sum, query_dense_top, query_sparse,
               ground_truth, id_to_doc):
    """Executa benchmark no Milvus com dual dense"""
    from pymilvus import (connections, utility, Collection, CollectionSchema,
                          FieldSchema, DataType, AnnSearchRequest, RRFRanker)

    host = cfg.milvus.host
    port = cfg.milvus.port
    print(f"\n{'='*70}")
    print(f"🟢 MILVUS — Ingestão com dual dense")
    print(f"{'='*70}")
    print(f"🔗 Conectando: {host}:{port}")

    connections.connect(host=host, port=port, timeout=300)
    print(f"✅ Conectado")

    # Criar collection
    print(f"📦 Criando collection: {COLLECTION_NAME}")
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="dense_sumario", dtype=DataType.FLOAT_VECTOR, dim=DENSE_DIM),
        FieldSchema(name="dense_topico", dtype=DataType.FLOAT_VECTOR, dim=DENSE_DIM),
        FieldSchema(name="sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="region", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=200),
    ]
    schema = CollectionSchema(fields=fields, description="Benchmark V3.1 Dual Dense")
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    print(f"✅ Collection criada")

    # Ingestão
    print(f"📥 Ingerindo {len(documents):,} documentos...")
    start = time.time()
    for i in range(0, len(documents), BATCH_SIZE):
        batch_docs = documents[i:i+BATCH_SIZE]
        batch_sum = dense_sumario[i:i+BATCH_SIZE]
        batch_top = dense_topico[i:i+BATCH_SIZE]
        batch_sp = sparse_embeddings[i:i+BATCH_SIZE]

        ids = [d['id'] for d in batch_docs]
        vecs_sum = batch_sum.tolist()
        vecs_top = batch_top.tolist()
        sparse_vecs = [{int(idx): float(val) for idx, val in zip(s['indices'], s['values'])} for s in batch_sp]
        regions = [d.get('region', 'UNKNOWN') for d in batch_docs]
        categories = [d.get('category', 'UNKNOWN') for d in batch_docs]

        collection.insert([ids, vecs_sum, vecs_top, sparse_vecs, regions, categories])
        if (i + BATCH_SIZE) % 10000 == 0:
            print(f"   {i+BATCH_SIZE:,} inseridos...")

    collection.flush()
    ingest_time = time.time() - start
    print(f"✅ Ingestão: {ingest_time:.1f}s")

    # Build indexes
    print("🔨 Construindo índices HNSW...")
    idx_start = time.time()
    for field_name in ["dense_sumario", "dense_topico"]:
        collection.create_index(field_name, {
            "metric_type": "COSINE", "index_type": "HNSW",
            "params": {"M": HNSW_M, "efConstruction": HNSW_EF_CONSTRUCTION}
        })
    collection.create_index("sparse", {"metric_type": "IP", "index_type": "SPARSE_INVERTED_INDEX", "params": {}})
    collection.load()
    idx_time = time.time() - idx_start
    print(f"✅ Índices prontos: {idx_time:.1f}s")

    # Rodar buscas
    results = {}
    for source_name, query_embs in [("sumario", query_dense_sum), ("topico", query_dense_top)]:
        print(f"\n   📊 Executando: Milvus + {source_name}")
        dense_field = f"dense_{source_name}"

        all_recall100 = []
        all_p10 = []
        all_ndcg = []
        all_mrr = []
        latencies = []

        for run in range(N_RUNS):
            run_recall100 = []
            run_p10 = []
            run_ndcg = []
            run_mrr = []

            for qi in range(len(queries)):
                q = queries[qi]
                q_region = q.get('region', '')
                q_category = q.get('category', '')
                q_emb = query_embs[qi]
                q_sp = query_sparse[qi]

                expr = f'region == "{q_region}"' if q_region else None
                prefetch_limit = TOP_K * 2
                ef = max(HNSW_EF_SEARCH, prefetch_limit)

                dense_req = AnnSearchRequest(
                    data=[q_emb.tolist()], anns_field=dense_field,
                    param={"metric_type": "COSINE", "params": {"ef": ef}},
                    limit=prefetch_limit, expr=expr)

                sparse_dict = {int(idx): float(val) for idx, val in zip(q_sp['indices'], q_sp['values'])}
                sparse_req = AnnSearchRequest(
                    data=[sparse_dict], anns_field="sparse",
                    param={"metric_type": "IP", "params": {}},
                    limit=prefetch_limit, expr=expr)

                t0 = time.time()
                res = collection.hybrid_search(
                    reqs=[dense_req, sparse_req],
                    rerank=RRFRanker(k=RRF_K),
                    limit=TOP_K)
                lat = (time.time() - t0) * 1000
                latencies.append(lat)

                ids = [hit.id for hit in res[0]]
                gt = ground_truth[qi]

                run_recall100.append(recall_at_k(ids, gt, 100))
                run_p10.append(precision_at_10_relevance(ids, q_category, id_to_doc))
                run_ndcg.append(ndcg_at_10(ids, q_category, id_to_doc))
                run_mrr.append(mrr_at_10(ids, q_category, id_to_doc))

            if run == 0:
                print(f"      Run {run+1}/{N_RUNS} (warm-up, descartado)")
                latencies = []
                continue

            r100 = np.mean(run_recall100)
            p10 = np.mean(run_p10)
            print(f"      Run {run+1}/{N_RUNS}: R@100={r100:.4f}, P@10={p10:.4f}")

            all_recall100.extend(run_recall100)
            all_p10.extend(run_p10)
            all_ndcg.extend(run_ndcg)
            all_mrr.extend(run_mrr)

        per_query_r100 = [np.mean(all_recall100[i::len(queries)]) for i in range(len(queries))]
        per_query_p10 = [np.mean(all_p10[i::len(queries)]) for i in range(len(queries))]

        total_time = sum(latencies) / 1000
        qps = len(latencies) / total_time if total_time > 0 else 0

        results[source_name] = {
            'engine': 'milvus',
            'dense_source': source_name,
            'recall_at_100': float(np.mean(per_query_r100)),
            'recall_at_100_ci': bootstrap_ci(per_query_r100),
            'recall_at_10': float(np.mean(all_recall100)) / (N_RUNS-1),
            'precision_at_10': float(np.mean(per_query_p10)),
            'precision_at_10_ci': bootstrap_ci(per_query_p10),
            'ndcg_at_10': float(np.mean(all_ndcg)) / (N_RUNS-1),
            'mrr_at_10': float(np.mean(all_mrr)) / (N_RUNS-1),
            'qps': qps,
            'latency_p50_ms': float(np.percentile(latencies, 50)),
            'latency_p95_ms': float(np.percentile(latencies, 95)),
            'latency_p99_ms': float(np.percentile(latencies, 99)),
            'ingest_time_s': ingest_time,
        }
        print(f"      ✅ R@100={results[source_name]['recall_at_100']:.4f}, "
              f"P@10={results[source_name]['precision_at_10']:.4f}, "
              f"QPS={qps:.1f}")

    # Cleanup
    utility.drop_collection(COLLECTION_NAME)
    connections.disconnect("default")
    return results


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🚀 BENCHMARK V3.1 — DUAL DENSE SOURCE")
    print("=" * 70)
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Simulações: 2 engines × 2 dense_source = 4 efetivas")
    print(f"RRF: nativo k=60 pesos iguais (limitação das engines)")
    print()

    cfg = get_config()

    # Carregar dados
    print("📂 Carregando dados...")
    documents = json.loads((DATA_DIR / "documents.json").read_text())
    queries = json.loads((DATA_DIR / "queries.json").read_text())
    ground_truth = json.loads((DATA_DIR / "ground_truth.json").read_text())

    dense_sumario = np.load(DATA_DIR / "dense_embeddings_sumario.npy")
    dense_topico = np.load(DATA_DIR / "dense_embeddings_topico.npy")
    sparse_embeddings = json.loads((DATA_DIR / "sparse_embeddings.json").read_text())

    query_dense_sum = np.load(DATA_DIR / "query_dense.npy")
    query_dense_top = np.load(DATA_DIR / "query_embeddings_topico.npy")
    query_sparse = json.loads((DATA_DIR / "query_sparse.json").read_text())

    id_to_doc = {d['id']: d for d in documents}

    print(f"   Docs: {len(documents):,}, Queries: {len(queries)}")
    print(f"   Dense sumário: {dense_sumario.shape}")
    print(f"   Dense tópico: {dense_topico.shape}")
    print(f"   Query sumário: {query_dense_sum.shape}")
    print(f"   Query tópico: {query_dense_top.shape}")

    # Executar
    all_results = {}

    print("\n" + "=" * 70)
    print("FASE 1: QDRANT")
    print("=" * 70)
    qdrant_results = run_qdrant(cfg, documents, dense_sumario, dense_topico,
                                sparse_embeddings, queries, query_dense_sum,
                                query_dense_top, query_sparse, ground_truth, id_to_doc)
    all_results['qdrant'] = qdrant_results

    print("\n" + "=" * 70)
    print("FASE 2: MILVUS")
    print("=" * 70)
    milvus_results = run_milvus(cfg, documents, dense_sumario, dense_topico,
                                sparse_embeddings, queries, query_dense_sum,
                                query_dense_top, query_sparse, ground_truth, id_to_doc)
    all_results['milvus'] = milvus_results

    # Salvar resultados
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': datetime.now().isoformat(),
        'version': 'v3.1',
        'description': 'Dual dense source (sumario vs topico) - corrigido',
        'rrf_config': 'nativo k=60, pesos iguais (limitação das engines)',
        'n_runs': N_RUNS,
        'results': all_results,
    }

    output_file = RESULTS_DIR / f"benchmark_v31_dual_dense_{timestamp}.json"
    output_file.write_text(json.dumps(output, indent=2, default=str))

    # Resumo
    print("\n" + "=" * 70)
    print("📊 RESUMO V3.1")
    print("=" * 70)
    print(f"{'Engine':<10} {'Dense':<10} {'R@100':>8} {'P@10':>8} {'NDCG@10':>8} {'MRR@10':>8} {'QPS':>8}")
    print("-" * 70)
    for engine, sources in all_results.items():
        for source, metrics in sources.items():
            print(f"{engine:<10} {source:<10} {metrics['recall_at_100']:>8.4f} "
                  f"{metrics['precision_at_10']:>8.4f} {metrics['ndcg_at_10']:>8.4f} "
                  f"{metrics['mrr_at_10']:>8.4f} {metrics['qps']:>8.1f}")

    print(f"\n💾 Resultados salvos em: {output_file}")
    print("✅ Benchmark V3.1 concluído!")
