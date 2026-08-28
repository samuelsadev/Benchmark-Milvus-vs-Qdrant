#!/usr/bin/env python3
"""
Script para gerar Ground Truth por Fusão Exata - V3

Fase 2 do Plano de Testes:
- Para cada query, calcular top-100 exato (denso + esparso + RRF) dentro da região
- Salvar top-100 exato por query para reuso nas 12 configurações
- Tempo estimado: ~2-4 horas (depende de paralelização)
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from datetime import datetime

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent))

# from ground_truth_hybrid import GroundTruthGenerator, calculate_recall_against_gt  # Módulo integrado abaixo
from config import get_config, RRF_WEIGHTS, BOOTSTRAP_SAMPLES


def load_client_embeddings(
    embeddings_dir: str,
    dense_source: str = "denso_sumario"
) -> Tuple[pd.DataFrame, np.ndarray, List[Dict]]:
    """
    Carrega embeddings.
    
    Args:
        embeddings_dir: Diretório com os arquivos parquet
        dense_source: "denso_sumario" ou "denso_topico"
    
    Returns:
        DataFrame com metadados, array de embeddings densos, lista de embeddings esparsos
    """
    print(f"\n📦 Carregando embeddings...")
    
    # Carregar embeddings densos
    if dense_source == "denso_sumario":
        dense_file = "df_base_hist_topicos_vetor_denso_sumario_ia_bge-m3_v1.0_2026-06-17.parquet"
    else:
        dense_file = "df_base_hist_topicos_vetor_denso_topico_ia_bge-m3_v1.0_2026-06-17.parquet"
    
    dense_path = os.path.join(embeddings_dir, dense_file)
    print(f"   Carregando {dense_file}...")
    df_dense = pq.read_table(dense_path).to_pandas()
    
    # Carregar embeddings esparsos
    sparse_file = "df_base_hist_topicos_vetor_esparso_palavras_chave_ia_bge-m3_v1.0_2026-06-17.parquet"
    sparse_path = os.path.join(embeddings_dir, sparse_file)
    print(f"   Carregando {sparse_file}...")
    df_sparse = pq.read_table(sparse_path).to_pandas()
    
    # Extrair arrays
    doc_ids = df_dense["doc_id"].tolist()
    doc_dense = np.vstack(df_dense["embedding_denso"].values)
    doc_sparses = df_sparse["embedding_esparso"].tolist()
    regions = df_dense["region"].tolist()
    
    # Metadados
    metadata = df_dense[[
        "doc_id",
        "region",
        "category_l1",
        "category",
        "category_l3",
        "doc_type",
    ]].copy()
    
    print(f"   ✓ {len(doc_ids):,} documentos carregados")
    print(f"   ✓ Dimensão embeddings: {doc_dense.shape[1]}")
    
    return metadata, doc_ids, doc_dense, doc_sparses, regions


def generate_stratified_queries(
    metadata: pd.DataFrame,
    n_queries: int = 1000,
    seed: int = 42
) -> List[str]:
    """
    Gera queries held-out estratificadas por category.
    
    Per specification:
    - Amostra aleatória estratificada por category
    - Exclusão do próprio documento dos resultados (implementado no GT)
    """
    print(f"\n📊 Gerando {n_queries} queries estratificadas...")
    
    np.random.seed(seed)
    
    # Estratificar por category
    tipo_nivel_2_counts = metadata["category"].value_counts()
    
    # Calcular quantidade proporcional por tipo
    queries_per_tipo = (tipo_nivel_2_counts / len(metadata) * n_queries).astype(int)
    
    # Ajustar para totalizar exatamente n_queries
    diff = n_queries - queries_per_tipo.sum()
    if diff > 0:
        # Adicionar aos tipos mais frequentes
        for i in range(diff):
            tipo = tipo_nivel_2_counts.index[i % len(tipo_nivel_2_counts)]
            queries_per_tipo[tipo] += 1
    
    # Amostrar queries
    query_ids = []
    for tipo, n in queries_per_tipo.items():
        if n > 0:
            tipo_docs = metadata[metadata["category"] == tipo]["doc_id"].tolist()
            sampled = np.random.choice(tipo_docs, size=min(n, len(tipo_docs)), replace=False)
            query_ids.extend(sampled)
    
    print(f"   ✓ {len(query_ids):,} queries geradas")
    
    return query_ids


def generate_stratified_queries_double(
    metadata: pd.DataFrame,
    n_queries: int = 1000,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[List[str], pd.DataFrame]:
    """
    Gera queries held-out com estratificação dupla (V3).
    
    Per specification:
    - 1.000 queries (vs 100 na V2)
    - Estratificação por região E category
    - Exclusão do próprio documento dos resultados (self-match exclusion)
    - Semente fixa (seed=42) para reprodutibilidade
    
    Args:
        metadata: DataFrame com metadados dos documentos
        n_queries: Número de queries a gerar (default 1000)
        seed: Semente para reprodutibilidade (default 42)
        verbose: Mostrar progresso
    
    Returns:
        Tuple de (lista de query_ids, DataFrame com info das queries)
    """
    if verbose:
        print(f"\n📊 Gerando {n_queries:,} queries held-out estratificadas...")
        print(f"   Estratificação: região + category")
        print(f"   Seed: {seed}")
    
    np.random.seed(seed)
    
    # Criar grupos estratificados (RF + tipo_nivel_2)
    metadata_copy = metadata.copy()
    metadata_copy['strata'] = (
        metadata_copy['region'].astype(str) + '_' + 
        metadata_copy['category'].astype(str)
    )
    
    # Contagem por grupo
    strata_counts = metadata_copy['strata'].value_counts()
    n_strata = len(strata_counts)
    
    if verbose:
        print(f"   Grupos estratificados: {n_strata}")
    
    # Calcular quantidade proporcional por grupo
    queries_per_strata = (strata_counts / len(metadata_copy) * n_queries).astype(int)
    
    # Garantir mínimo de 1 query por grupo (se possível)
    min_queries = 1
    for strata in queries_per_strata.index:
        if queries_per_strata[strata] < min_queries and strata_counts[strata] >= min_queries:
            queries_per_strata[strata] = min_queries
    
    # Ajustar para totalizar exatamente n_queries
    total_allocated = queries_per_strata.sum()
    diff = n_queries - total_allocated
    
    if diff > 0:
        # Adicionar aos grupos mais frequentes
        sorted_strata = queries_per_strata.sort_values(ascending=False)
        for i in range(diff):
            strata = sorted_strata.index[i % len(sorted_strata)]
            queries_per_strata[strata] += 1
    elif diff < 0:
        # Remover dos grupos com mais queries
        sorted_strata = queries_per_strata.sort_values(ascending=False)
        for i in range(abs(diff)):
            strata = sorted_strata.index[i % len(sorted_strata)]
            if queries_per_strata[strata] > 1:  # Manter pelo menos 1
                queries_per_strata[strata] -= 1
    
    # Amostrar queries por grupo
    query_data = []
    
    for strata, n in queries_per_strata.items():
        if n > 0:
            strata_docs = metadata_copy[metadata_copy['strata'] == strata]
            
            if len(strata_docs) == 0:
                continue
            
            # Amostrar sem reposição
            n_sample = min(n, len(strata_docs))
            sampled_idx = np.random.choice(
                strata_docs.index, 
                size=n_sample, 
                replace=False
            )
            
            for idx in sampled_idx:
                row = strata_docs.loc[idx]
                query_data.append({
                    'doc_id': row['doc_id'],
                    'region': row['region'],
                    'category': row['category'],
                    'strata': strata
                })
    
    # Criar DataFrame das queries
    queries_df = pd.DataFrame(query_data)
    query_ids = queries_df['doc_id'].tolist()
    
    if verbose:
        print(f"\n   ✓ {len(query_ids):,} queries geradas")
        
        # Mostrar distribuição por região
        print(f"\n   Distribuição por região:")
        rf_dist = queries_df['region'].value_counts().sort_index()
        for rf, count in rf_dist.items():
            pct = count / len(queries_df) * 100
            print(f"      {rf}: {count:,} ({pct:.1f}%)")
        
        # Mostrar distribuição por category (top 10)
        print(f"\n   Top 10 tipos de serviço:")
        tipo_dist = queries_df['category'].value_counts().head(10)
        for tipo, count in tipo_dist.items():
            pct = count / len(queries_df) * 100
            print(f"      {tipo[:50]}: {count:,} ({pct:.1f}%)")
    
    return query_ids, queries_df


def save_queries_held_out(
    query_ids: List[str],
    queries_df: pd.DataFrame,
    output_dir: str,
    seed: int = 42,
    exclude_self: bool = True
):
    """
    Salva queries held-out em arquivos.
    
    Args:
        query_ids: Lista de IDs das queries
        queries_df: DataFrame com info das queries
        output_dir: Diretório de saída
        seed: Semente usada
        exclude_self: Se houve exclusão de self-match
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Salvar lista de IDs
    queries_json = {
        "query_ids": query_ids,
        "n_queries": len(query_ids),
        "seed": seed,
        "exclude_self_match": exclude_self,
        "stratified_by": ["region", "category"],
        "generated_at": datetime.now().isoformat()
    }
    
    with open(output_dir / "queries_held_out.json", 'w') as f:
        json.dump(queries_json, f, indent=2)
    
    # Salvar DataFrame com metadados
    queries_df.to_parquet(output_dir / "queries_held_out_metadata.parquet", index=False)
    
    print(f"\n💾 Queries salvas em:")
    print(f"   - {output_dir / 'queries_held_out.json'}")
    print(f"   - {output_dir / 'queries_held_out_metadata.parquet'}")


def generate_gt_by_region(
    query_ids: List[str],
    metadata: pd.DataFrame,
    doc_ids: List[str],
    doc_dense: np.ndarray,
    doc_sparses: List[Dict],
    regions: List[str],
    rrf_weights: List[Tuple[float, float]],
    top_k: int = 100
) -> Dict[Tuple[float, float], Dict[str, List[str]]]:
    """
    Gera Ground Truth por região.
    
    Per specification:
    "a busca exata é feita dentro do subconjunto da região da query"
    """
    print(f"\n🔍 Gerando Ground Truth por região...")
    
    # Criar mapeamentos
    doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    doc_id_to_rf = {doc_id: rf for doc_id, rf in zip(doc_ids, regions)}
    
    # Agrupar queries por região
    queries_by_rf = {}
    for qid in query_ids:
        rf = doc_id_to_rf.get(qid)
        if rf:
            if rf not in queries_by_rf:
                queries_by_rf[rf] = []
            queries_by_rf[rf].append(qid)
    
    print(f"   Regiões fiscais encontradas: {len(queries_by_rf)}")
    for rf, qids in queries_by_rf.items():
        print(f"      {rf}: {len(qids)} queries")
    
    # Gerador de GT
    generator = GroundTruthGenerator(
        rrf_weights=rrf_weights,
        exclude_self=True
    )
    
    # Resultados por peso RRF
    gt_all_weights = {weights: {} for weights in rrf_weights}
    
    start_time = time.time()
    
    for rf, rf_query_ids in queries_by_rf.items():
        print(f"\n   Processando {rf} ({len(rf_query_ids)} queries)...")
        
        # Filtrar documentos desta região
        rf_doc_indices = [i for i, d_rf in enumerate(regions) if d_rf == rf]
        rf_doc_ids = [doc_ids[i] for i in rf_doc_indices]
        rf_doc_dense = doc_dense[rf_doc_indices]
        rf_doc_sparses = [doc_sparses[i] for i in rf_doc_indices]
        
        # Filtrar queries
        rf_query_indices = [doc_id_to_idx[qid] for qid in rf_query_ids if qid in doc_id_to_idx]
        rf_query_dense = doc_dense[rf_query_indices]
        rf_query_sparse = [doc_sparses[i] for i in rf_query_indices]
        
        # Gerar GT para esta região
        rf_gt = generator.generate_gt_all_weights(
            query_ids=rf_query_ids,
            query_dense=rf_query_dense,
            query_sparse=rf_query_sparse,
            doc_ids=rf_doc_ids,
            doc_dense=rf_doc_dense,
            doc_sparses=rf_doc_sparses,
            top_k=top_k,
            show_progress=True
        )
        
        # Consolidar resultados
        for weights, gt in rf_gt.items():
            gt_all_weights[weights].update(gt)
    
    elapsed = time.time() - start_time
    print(f"\n   ✓ GT gerado em {elapsed:.2f} segundos")
    
    return gt_all_weights


def main():
    print("=" * 80)
    print("GERAÇÃO DE GROUND TRUTH POR FUSÃO EXATA - V3")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Configuração
    config = get_config()
    
    # Paths - usar embeddings do S3 bucket
    # Verificar qual diretório tem os arquivos
    from download_embeddings import get_embeddings_dir
    
    try:
        embeddings_dir = get_embeddings_dir(prefer_s3_download=False)
    except FileNotFoundError:
        print("⚠️ Embeddings não encontrados. Baixando do S3 bucket 'your-benchmark-bucket'...")
        from download_embeddings import download_all_embeddings
        embeddings_dir = "./data"
        download_all_embeddings(embeddings_dir, include_colbert=False)
    
    output_dir = Path(config.benchmark.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parâmetros
    n_queries = 1000  # Per specification V3
    top_k = 100  # GT-Fidelidade
    rrf_weights = [(0.6, 0.4), (0.5, 0.5), (0.4, 0.6)]
    
    print(f"\n📋 Parâmetros:")
    print(f"   Queries: {n_queries}")
    print(f"   Top-k (GT-Fidelidade): {top_k}")
    print(f"   Pesos RRF: {rrf_weights}")
    
    # 1. Carregar dados
    metadata, doc_ids, doc_dense, doc_sparses, regions = load_client_embeddings(
        embeddings_dir,
        dense_source="denso_sumario"
    )
    
    # 2. Gerar queries estratificadas
    query_ids = generate_stratified_queries(metadata, n_queries=n_queries)
    
    # Salvar queries
    queries_path = output_dir / "queries_held_out.json"
    with open(queries_path, 'w') as f:
        json.dump({
            "query_ids": query_ids,
            "n_queries": len(query_ids),
            "seed": 42,
            "stratified_by": "category"
        }, f, indent=2)
    print(f"   ✓ Queries salvas em: {queries_path}")
    
    # 3. Gerar Ground Truth por região
    gt_all_weights = generate_gt_by_region(
        query_ids=query_ids,
        metadata=metadata,
        doc_ids=doc_ids,
        doc_dense=doc_dense,
        doc_sparses=doc_sparses,
        regions=regions,
        rrf_weights=rrf_weights,
        top_k=top_k
    )
    
    # 4. Salvar Ground Truth
    print(f"\n💾 Salvando Ground Truth...")
    for weights, gt in gt_all_weights.items():
        filename = f"ground_truth_fidelidade_{weights[0]:.1f}_{weights[1]:.1f}.json"
        gt_path = output_dir / filename
        with open(gt_path, 'w') as f:
            json.dump(gt, f, indent=2)
        print(f"   ✓ {filename} ({len(gt)} queries)")
    
    # 5. Salvar metadados das queries
    query_metadata = metadata[metadata["doc_id"].isin(query_ids)]
    query_metadata_path = output_dir / "query_metadata.parquet"
    query_metadata.to_parquet(query_metadata_path, index=False)
    print(f"   ✓ Metadados das queries salvos")
    
    print("\n" + "=" * 80)
    print("✅ GROUND TRUTH GERADO COM SUCESSO!")
    print("=" * 80)
    
    return gt_all_weights


if __name__ == "__main__":
    main()
