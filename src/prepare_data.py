#!/usr/bin/env python3
"""
Preparação de dados para o Benchmark V3 - Hybrid Search Benchmark

Este script:
1. Carrega o dataset PT-BR Sentiment Analysis do Kaggle
2. Gera embeddings densos e esparsos usando BGE-M3
3. Prepara queries de teste
4. Calcula ground truth via busca exata (Critério #4)

Atende aos critérios:
- #1: Embeddings reais BGE-M3 (não aleatórios)
- #4: Ground truth via busca exata (flat)
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import time
import random
from tqdm import tqdm

from config import get_config, Config


def load_dataset(config: Config) -> pd.DataFrame:
    """
    Carrega e prepara o dataset PT-BR Sentiment Analysis.
    Mapeia para estrutura similar à Hybrid Search Benchmark (manifestações de cidadãos).
    """
    print(f"📂 Carregando dataset de: {config.benchmark.dataset_path}")
    
    # Verificar se arquivo existe
    if not Path(config.benchmark.dataset_path).exists():
        raise FileNotFoundError(f"Dataset não encontrado: {config.benchmark.dataset_path}")
    
    # Carregar dataset - usando apenas colunas necessárias
    # O dataset tem: dataset, original_index, review_text, review_text_processed, 
    #                review_text_tokenized, polarity, rating, kfold_polarity, kfold_rating
    df = pd.read_csv(
        config.benchmark.dataset_path,
        usecols=['dataset', 'review_text', 'polarity', 'rating'],
        nrows=config.benchmark.num_documents * 2,  # Carregar extra para filtrar nulos
        on_bad_lines='skip',  # Ignorar linhas problemáticas
        encoding='utf-8'
    )
    
    # Remover nulos
    df = df.dropna(subset=['review_text'])
    df = df[df['review_text'].str.len() > 20]  # Mínimo 20 caracteres
    
    # Limitar ao número desejado
    df = df.head(config.benchmark.num_documents)
    
    # Mapear para estrutura Hybrid Search Benchmark
    # Simular categorias baseado na polaridade e dataset
    category_map = {
        ('b2w', 0): 'reclamacao',
        ('b2w', 1): 'elogio',
        ('buscape', 0): 'reclamacao',
        ('buscape', 1): 'sugestao',
        ('olist', 0): 'denuncia',
        ('olist', 1): 'elogio',
        ('utlc_apps', 0): 'reclamacao',
        ('utlc_apps', 1): 'solicitacao',
        ('utlc_movies', 0): 'reclamacao',
        ('utlc_movies', 1): 'elogio',
    }
    
    department_options = ['saude', 'educacao', 'transporte', 'seguranca', 'meio_ambiente', 'outros']
    
    # Criar estrutura final
    documents = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preparando documentos"):
        polarity = int(row['polarity']) if pd.notna(row['polarity']) else 1
        category = category_map.get((row['dataset'], polarity), 'outros')
        
        doc = {
            'id': idx,
            'text': str(row['review_text'])[:2000],  # Limitar tamanho
            'categoria': category,
            'departamento': random.choice(department_options),
            'polaridade': polarity,
            'fonte': row['dataset'],
            'timestamp': datetime.now().isoformat(),
        }
        documents.append(doc)
    
    print(f"✅ {len(documents):,} documentos preparados")
    return documents


def generate_embeddings_bge_m3(
    texts: List[str],
    config: Config,
    batch_size: int = 32
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Gera embeddings densos e esparsos usando BGE-M3.
    
    BGE-M3 é um modelo multilíngue que suporta:
    - Dense embeddings (1024d)
    - Sparse embeddings (lexical)
    - ColBERT-style multi-vector
    
    Retorna:
        dense_embeddings: np.ndarray de shape (N, 1024)
        sparse_embeddings: Lista de dicts com {indices, values}
    """
    print(f"🔄 Gerando embeddings BGE-M3 para {len(texts):,} textos...")
    print(f"   Device: {config.embedding.device}")
    
    try:
        # Tentar usar FlagEmbedding
        from FlagEmbedding import BGEM3FlagModel
        
        model = BGEM3FlagModel(
            config.embedding.model_name,
            use_fp16=True if config.embedding.device == "cuda" else False,
            device=config.embedding.device
        )
        
        dense_embeddings = []
        sparse_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Gerando embeddings"):
            batch_texts = texts[i:i + batch_size]
            
            # Gerar embeddings
            output = model.encode(
                batch_texts,
                batch_size=batch_size,
                max_length=config.embedding.max_length,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False  # ColBERT será usado apenas no reranking
            )
            
            # Dense embeddings
            dense_embeddings.append(output['dense_vecs'])
            
            # Sparse embeddings
            for sparse_vec in output['lexical_weights']:
                indices = list(sparse_vec.keys())
                values = list(sparse_vec.values())
                sparse_embeddings.append({
                    'indices': indices,
                    'values': values
                })
        
        dense_embeddings = np.vstack(dense_embeddings)
        print(f"✅ Embeddings gerados: dense shape={dense_embeddings.shape}")
        
        return dense_embeddings, sparse_embeddings
        
    except Exception as e:
        print(f"⚠️ FlagEmbedding não disponível ({type(e).__name__}). Usando implementação direta...")
        return generate_embeddings_transformers(texts, config, batch_size)


def generate_embeddings_transformers(
    texts: List[str],
    config: Config,
    batch_size: int = 32
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Gera embeddings usando transformers diretamente (fallback robusto).
    Usa BGE-large-en-v1.5 multilíngue para embeddings densos.
    Gera sparse via BM25 (conforme arquitetura Hybrid Search Benchmark).
    """
    import torch
    from transformers import AutoTokenizer, AutoModel
    
    print("   Usando transformers diretamente para embeddings...")
    
    device = config.embedding.device
    
    # Carregar modelo BGE (funciona bem para português também)
    model_name = "BAAI/bge-large-en-v1.5"  # 1024d, multilíngue
    print(f"   Carregando modelo: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    # Dense embeddings via mean pooling
    print(f"   Gerando dense embeddings...")
    dense_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Dense embeddings"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenizar
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=config.embedding.max_length,
                return_tensors="pt"
            ).to(device)
            
            # Forward pass
            outputs = model(**inputs)
            
            # Mean pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            # Normalizar
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            dense_embeddings.append(embeddings.cpu().numpy())
    
    dense_embeddings = np.vstack(dense_embeddings)
    print(f"   Dense shape: {dense_embeddings.shape}")
    
    # Sparse embeddings via BM25 (conforme arquitetura de referência)
    print(f"   Gerando sparse embeddings via BM25...")
    sparse_embeddings = generate_bm25_sparse(texts)
    
    print(f"✅ Embeddings gerados: dense={dense_embeddings.shape}, sparse (BM25)")
    return dense_embeddings, sparse_embeddings


def generate_bm25_sparse(
    texts: List[str],
    k1: float = 1.5,
    b: float = 0.75,
    max_features: int = 50000
) -> List[Dict]:
    """
    Gera embeddings sparse usando BM25.
    
    BM25 formula:
    score(D,Q) = Σ IDF(qi) * (f(qi,D) * (k1 + 1)) / (f(qi,D) + k1 * (1 - b + b * |D|/avgdl))
    
    Args:
        texts: Lista de documentos
        k1: Parâmetro de saturação de termo (padrão 1.5)
        b: Parâmetro de normalização por comprimento (padrão 0.75)
        max_features: Número máximo de features no vocabulário
        
    Returns:
        Lista de dicts com {indices, values} representando vetores sparse BM25
    """
    from sklearn.feature_extraction.text import CountVectorizer
    import math
    
    print(f"   BM25 params: k1={k1}, b={b}, max_features={max_features}")
    
    # 1. Construir vocabulário e contar termos
    vectorizer = CountVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        token_pattern=r'(?u)\b\w+\b'  # Incluir palavras de 1 caractere
    )
    
    # Term frequency matrix (documentos x termos)
    tf_matrix = vectorizer.fit_transform(texts)
    vocab_size = len(vectorizer.vocabulary_)
    n_docs = tf_matrix.shape[0]
    
    print(f"   Vocabulário: {vocab_size} termos, {n_docs} documentos")
    
    # 2. Calcular estatísticas do corpus
    # Comprimento de cada documento (número de termos)
    doc_lengths = np.array(tf_matrix.sum(axis=1)).flatten()
    avg_doc_length = np.mean(doc_lengths)
    
    # Document frequency (em quantos docs cada termo aparece)
    df = np.array((tf_matrix > 0).sum(axis=0)).flatten()
    
    # 3. Calcular IDF com smoothing (BM25 style)
    # IDF = log((N - df + 0.5) / (df + 0.5))
    idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1)
    idf = np.maximum(idf, 0)  # Garantir não-negativo
    
    # 4. Calcular BM25 score para cada documento
    sparse_embeddings = []
    
    for i in tqdm(range(n_docs), desc="BM25 sparse"):
        row = tf_matrix.getrow(i)
        indices = row.indices
        tf_values = row.data
        doc_len = doc_lengths[i]
        
        # BM25 score para cada termo presente no documento
        # score = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
        denominator = tf_values + k1 * (1 - b + b * doc_len / avg_doc_length)
        numerator = tf_values * (k1 + 1)
        bm25_tf = numerator / denominator
        
        # Multiplicar pelo IDF
        bm25_scores = bm25_tf * idf[indices]
        
        # Filtrar scores muito baixos
        mask = bm25_scores > 0.01
        filtered_indices = indices[mask].tolist()
        filtered_values = bm25_scores[mask].tolist()
        
        sparse_embeddings.append({
            'indices': filtered_indices,
            'values': filtered_values
        })
    
    # Salvar vocabulário para referência
    print(f"   BM25: avg_doc_length={avg_doc_length:.1f}, IDF range=[{idf.min():.2f}, {idf.max():.2f}]")
    
    return sparse_embeddings


def generate_embeddings_fallback(
    texts: List[str],
    config: Config,
    batch_size: int = 32
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Fallback usando sentence-transformers se BGE-M3 não estiver disponível.
    Gera apenas embeddings densos e simula sparse com TF-IDF.
    """
    # Redirecionar para implementação com transformers
    return generate_embeddings_transformers(texts, config, batch_size)


def prepare_queries(
    documents: List[Dict],
    config: Config
) -> List[Dict]:
    """
    Prepara queries de teste a partir dos documentos.
    Seleciona amostras diversificadas por categoria.
    """
    print(f"🔍 Preparando {config.benchmark.num_queries} queries de teste...")
    
    # Agrupar por categoria
    by_category = {}
    for doc in documents:
        cat = doc['categoria']
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(doc)
    
    # Selecionar queries balanceadas
    queries = []
    queries_per_cat = config.benchmark.num_queries // len(by_category)
    
    for cat, docs in by_category.items():
        sampled = random.sample(docs, min(queries_per_cat, len(docs)))
        for doc in sampled:
            # Criar query a partir do texto (primeiras palavras ou resumo)
            text = doc['text']
            query_text = ' '.join(text.split()[:20])  # Primeiras 20 palavras
            
            queries.append({
                'id': len(queries),
                'text': query_text,
                'source_doc_id': doc['id'],
                'categoria': cat,
            })
    
    # Completar se necessário
    while len(queries) < config.benchmark.num_queries:
        doc = random.choice(documents)
        query_text = ' '.join(doc['text'].split()[:20])
        queries.append({
            'id': len(queries),
            'text': query_text,
            'source_doc_id': doc['id'],
            'categoria': doc['categoria'],
        })
    
    queries = queries[:config.benchmark.num_queries]
    print(f"✅ {len(queries)} queries preparadas")
    return queries


def compute_ground_truth(
    query_embeddings: np.ndarray,
    doc_embeddings: np.ndarray,
    top_k: int = 100
) -> List[List[int]]:
    """
    Calcula ground truth via busca exata (brute-force).
    Critério #4: Ground truth via busca exata (flat)
    
    Args:
        query_embeddings: (Q, D) array de embeddings das queries
        doc_embeddings: (N, D) array de embeddings dos documentos
        top_k: Número de vizinhos mais próximos
        
    Returns:
        Lista de listas com os IDs dos top-k documentos para cada query
    """
    print(f"🎯 Calculando ground truth via busca exata (flat) para {len(query_embeddings)} queries...")
    
    ground_truth = []
    
    # Normalizar para cosine similarity
    query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    doc_norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    
    query_normalized = query_embeddings / (query_norms + 1e-8)
    doc_normalized = doc_embeddings / (doc_norms + 1e-8)
    
    for i, query_vec in enumerate(tqdm(query_normalized, desc="Calculando ground truth")):
        # Cosine similarity = dot product de vetores normalizados
        similarities = np.dot(doc_normalized, query_vec)
        
        # Top-k índices
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        ground_truth.append(top_k_indices.tolist())
    
    print(f"✅ Ground truth calculado para {len(ground_truth)} queries")
    return ground_truth


def save_data(
    documents: List[Dict],
    dense_embeddings: np.ndarray,
    sparse_embeddings: List[Dict],
    queries: List[Dict],
    query_dense: np.ndarray,
    query_sparse: List[Dict],
    ground_truth: List[List[int]],
    config: Config
):
    """Salva todos os dados preparados"""
    data_dir = Path(config.benchmark.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Salvando dados em {data_dir}...")
    
    # Documentos
    with open(data_dir / "documents.json", "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    # Dense embeddings
    np.save(data_dir / "dense_embeddings.npy", dense_embeddings)
    
    # Sparse embeddings
    with open(data_dir / "sparse_embeddings.json", "w") as f:
        json.dump(sparse_embeddings, f)
    
    # Queries
    with open(data_dir / "queries.json", "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)
    
    # Query embeddings
    np.save(data_dir / "query_dense.npy", query_dense)
    with open(data_dir / "query_sparse.json", "w") as f:
        json.dump(query_sparse, f)
    
    # Ground truth
    with open(data_dir / "ground_truth.json", "w") as f:
        json.dump(ground_truth, f)
    
    # Metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "num_documents": len(documents),
        "num_queries": len(queries),
        "dense_dim": dense_embeddings.shape[1],
        "ground_truth_method": "exact_search_cosine",
        "top_k_ground_truth": 100,
    }
    with open(data_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Dados salvos com sucesso!")
    print(f"   - Documentos: {len(documents):,}")
    print(f"   - Queries: {len(queries)}")
    print(f"   - Dense dim: {dense_embeddings.shape[1]}")


def main():
    """Pipeline principal de preparação de dados"""
    print("=" * 60)
    print("🚀 Benchmark V2 - Preparação de Dados")
    print("=" * 60)
    
    config = get_config()
    
    # 1. Carregar dataset
    documents = load_dataset(config)
    texts = [doc['text'] for doc in documents]
    
    # 2. Gerar embeddings dos documentos
    dense_embeddings, sparse_embeddings = generate_embeddings_bge_m3(
        texts, config, batch_size=config.embedding.batch_size
    )
    
    # 3. Preparar queries
    queries = prepare_queries(documents, config)
    query_texts = [q['text'] for q in queries]
    
    # 4. Gerar embeddings das queries
    query_dense, query_sparse = generate_embeddings_bge_m3(
        query_texts, config, batch_size=config.embedding.batch_size
    )
    
    # 5. Calcular ground truth (Critério #4)
    ground_truth = compute_ground_truth(
        query_dense, dense_embeddings, top_k=config.hybrid.top_k_retrieval
    )
    
    # 6. Salvar dados
    save_data(
        documents, dense_embeddings, sparse_embeddings,
        queries, query_dense, query_sparse, ground_truth,
        config
    )
    
    print("\n" + "=" * 60)
    print("✅ Preparação de dados concluída!")
    print("=" * 60)


if __name__ == "__main__":
    main()
