#!/usr/bin/env python3
"""
ColBERT Reranking para o Benchmark V3 - Hybrid Search Benchmark

Implementação nativa de ColBERT com MaxSim (late interaction) em GPU.

Arquitetura Hybrid Search Benchmark:
Query → Busca Híbrida → top-100 → Reranking ColBERT (MaxSim em GPU) → top-10

Características:
- Detecção automática de GPU (CUDA)
- Fallback para CPU se GPU não disponível
- MaxSim (late interaction) para scoring
- Suporte a token embeddings pré-computados
"""
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from statistics import mean, stdev


@dataclass
class RerankResult:
    """Resultado do reranking ColBERT"""
    latency_mean_ms: float
    latency_std_ms: float
    latency_p95_ms: float
    candidates_per_query: int
    final_top_k: int
    device_used: str = "cpu"


class ColBERTReranker:
    """
    ColBERT nativo com MaxSim (late interaction) em GPU.
    
    MaxSim scoring:
    - Para cada token da query, encontra max similarity com tokens do documento
    - Score final = soma dos max similarities
    
    Suporta:
    - GPU (CUDA) com detecção automática
    - Fallback para CPU se GPU não disponível
    - Token embeddings pré-computados ou gerados on-the-fly
    """
    
    def __init__(self, model_name: str = "colbert-ir/colbertv2.0", device: str = "auto"):
        self.model_name = model_name
        self.device = self._detect_device(device)
        self.model = None
        self.tokenizer = None
        self.use_colbert_native = False
        
        print(f"🖥️ Dispositivo configurado: {self.device.upper()}")
    
    def _detect_device(self, device: str) -> str:
        """Detecta automaticamente GPU se disponível."""
        if device != "auto":
            return device
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"✅ GPU detectada: {gpu_name} ({vram_gb:.1f} GB VRAM)")
                return "cuda"
            else:
                print("⚠️ CUDA não disponível, usando CPU")
                return "cpu"
        except ImportError:
            print("⚠️ PyTorch não instalado, usando CPU")
            return "cpu"
    
    def load_model(self):
        """Carrega modelo ColBERT nativo ou fallback para cross-encoder."""
        print(f"📥 Carregando modelo ColBERT para reranking em {self.device.upper()}...")
        
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            
            # Tentar carregar ColBERT nativo
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
                self.use_colbert_native = True
                
                print(f"✅ ColBERT nativo carregado: {self.model_name}")
                print(f"   Device: {self.device.upper()}")
                
            except Exception as e:
                print(f"⚠️ Erro ao carregar ColBERT nativo: {e}")
                print("   Usando BGE-reranker como fallback...")
                self._load_cross_encoder_fallback()
        
        except ImportError as e:
            print(f"❌ Erro de dependência: {e}")
            raise
    
    def _load_cross_encoder_fallback(self):
        """Fallback para cross-encoder se ColBERT nativo não disponível."""
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        model_name = "BAAI/bge-reranker-base"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.use_colbert_native = False
        
        print(f"✅ Cross-encoder carregado: {model_name}")
        print(f"   Device: {self.device.upper()}")
    
    def encode_query_tokens(self, query_text: str) -> np.ndarray:
        """
        Codifica query em token embeddings para MaxSim.
        
        Returns:
            np.ndarray de shape (num_tokens, dim) para MaxSim
        """
        import torch
        
        inputs = self.tokenizer(
            query_text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            if self.use_colbert_native:
                # ColBERT: usar last_hidden_state (token embeddings)
                token_embeddings = outputs.last_hidden_state.squeeze(0)  # (num_tokens, dim)
            else:
                # Fallback: usar last_hidden_state mesmo assim
                token_embeddings = outputs.last_hidden_state.squeeze(0)
        
        # Normalizar por token
        token_embeddings = torch.nn.functional.normalize(token_embeddings, p=2, dim=1)
        
        return token_embeddings.cpu().numpy()
    
    def encode_document_tokens(self, doc_text: str) -> np.ndarray:
        """
        Codifica documento em token embeddings para MaxSim.
        
        Returns:
            np.ndarray de shape (num_tokens, dim) para MaxSim
        """
        return self.encode_query_tokens(doc_text)  # Mesmo processo
    
    def maxsim_score(self, query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
        """
        Calcula score MaxSim entre query e documento.
        
        MaxSim = Σ_{t∈query} max_{u∈doc} (sim(t, u))
        
        Args:
            query_tokens: (num_query_tokens, dim)
            doc_tokens: (num_doc_tokens, dim)
            
        Returns:
            Score float
        """
        # Matriz de similaridade (Q, D)
        similarities = np.dot(query_tokens, doc_tokens.T)
        
        # Max por linha (para cada token da query, encontrar max com doc)
        max_sims = np.max(similarities, axis=1)
        
        # Score final = soma dos max
        return float(np.sum(max_sims))
    
    def maxsim_score_gpu(self, query_tokens, doc_tokens) -> float:
        """
        Calcula score MaxSim em GPU (mais rápido para batches grandes).
        """
        import torch
        
        # Garantir que estão na GPU
        if not isinstance(query_tokens, torch.Tensor):
            query_tokens = torch.tensor(query_tokens, device=self.device)
        if not isinstance(doc_tokens, torch.Tensor):
            doc_tokens = torch.tensor(doc_tokens, device=self.device)
        
        # Matriz de similaridade
        with torch.no_grad():
            similarities = torch.mm(query_tokens, doc_tokens.T)
            max_sims = torch.max(similarities, dim=1).values
            score = torch.sum(max_sims).item()
        
        return score
    
    def rerank(
        self,
        query_text: str,
        candidate_docs: List[Dict],
        top_k: int = 10,
        use_gpu: bool = True
    ) -> Tuple[List[Dict], float]:
        """
        Rerank candidatos usando MaxSim.
        
        Args:
            query_text: Texto da query
            candidate_docs: Lista de dicts com 'id' e 'text'
            top_k: Número de resultados finais
            use_gpu: Se deve usar GPU para cálculo MaxSim
            
        Returns:
            (docs_reranked, latency_ms)
        """
        import torch
        
        start_time = time.time()
        
        # Codificar query uma única vez
        query_tokens = self.encode_query_tokens(query_text)
        
        scores = []
        
        for doc in candidate_docs:
            doc_text = doc.get('text', '')[:512]  # Limitar tamanho
            
            if self.use_colbert_native or not hasattr(self, 'use_cross_encoder'):
                # MaxSim scoring
                doc_tokens = self.encode_document_tokens(doc_text)
                
                if use_gpu and self.device == "cuda":
                    score = self.maxsim_score_gpu(query_tokens, doc_tokens)
                else:
                    score = self.maxsim_score(query_tokens, doc_tokens)
            else:
                # Fallback: cross-encoder
                inputs = self.tokenizer(
                    query_text,
                    doc_text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    score = outputs.logits.squeeze().item()
            
            scores.append((score, doc))
        
        # Ordenar por score decrescente
        scores.sort(key=lambda x: x[0], reverse=True)
        
        latency = (time.time() - start_time) * 1000  # ms
        
        reranked = [doc for _, doc in scores[:top_k]]
        
        return reranked, latency
    
    def rerank_batch(
        self,
        query_text: str,
        candidate_docs: List[Dict],
        top_k: int = 10
    ) -> Tuple[List[Dict], float]:
        """
        Rerank candidatos em batch (mais eficiente em GPU).
        
        Processa todos os documentos de uma vez para melhor utilização da GPU.
        """
        import torch
        
        start_time = time.time()
        
        # Codificar query
        query_tokens = torch.tensor(
            self.encode_query_tokens(query_text),
            device=self.device
        )
        
        # Codificar todos os documentos em batch
        doc_texts = [doc.get('text', '')[:512] for doc in candidate_docs]
        
        # Batch encoding
        inputs = self.tokenizer(
            doc_texts,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # (batch, num_tokens, dim)
            doc_token_embeddings = outputs.last_hidden_state
            doc_token_embeddings = torch.nn.functional.normalize(doc_token_embeddings, p=2, dim=-1)
        
        # Calcular scores em batch
        # query_tokens: (num_q_tokens, dim)
        # doc_token_embeddings: (batch, num_d_tokens, dim)
        
        scores = []
        for i, doc in enumerate(candidate_docs):
            doc_tokens = doc_token_embeddings[i]  # (num_d_tokens, dim)
            
            # MaxSim
            similarities = torch.mm(query_tokens, doc_tokens.T)
            max_sims = torch.max(similarities, dim=1).values
            score = torch.sum(max_sims).item()
            
            scores.append((score, doc))
        
        # Ordenar
        scores.sort(key=lambda x: x[0], reverse=True)
        
        latency = (time.time() - start_time) * 1000
        
        reranked = [doc for _, doc in scores[:top_k]]
        
        return reranked, latency


def benchmark_colbert_reranking(
    queries: List[Dict],
    documents: List[Dict],
    first_stage_results: List[List[int]],
    ground_truth: List[List[int]],
    top_k_rerank: int = 100,
    top_k_final: int = 10,
    num_runs: int = 5,
    device: str = "auto",
) -> RerankResult:
    """
    Benchmark do reranking ColBERT sobre os resultados do primeiro estágio.
    
    Args:
        queries: Lista de queries com 'text'
        documents: Lista de documentos com 'id' e 'text'
        first_stage_results: IDs recuperados no primeiro estágio (busca híbrida)
        ground_truth: Ground truth para avaliação
        top_k_rerank: Número de candidatos do primeiro estágio
        top_k_final: Número de resultados finais
        num_runs: Número de execuções
        device: "auto", "cuda" ou "cpu"
        
    Returns:
        RerankResult com estatísticas
    """
    print(f"\n{'='*60}")
    print(f"🔄 BENCHMARK COLBERT RERANKING ({num_runs} runs)")
    print(f"{'='*60}")
    print(f"   Candidatos por query: {top_k_rerank}")
    print(f"   Resultados finais: {top_k_final}")
    
    # Criar índice de documentos por ID
    doc_index = {doc['id']: doc for doc in documents}
    
    # Inicializar reranker
    reranker = ColBERTReranker(device=device)
    reranker.load_model()
    
    all_latencies = []
    
    for run in range(num_runs):
        print(f"   Run {run + 1}/{num_runs}...", end="", flush=True)
        run_latencies = []
        
        for i, query in enumerate(queries):
            query_text = query['text']
            
            # Pegar candidatos do primeiro estágio
            candidate_ids = first_stage_results[i][:top_k_rerank]
            candidate_docs = [doc_index.get(cid, {'id': cid, 'text': ''}) for cid in candidate_ids]
            
            # Rerank
            reranked, latency = reranker.rerank(
                query_text,
                candidate_docs,
                top_k=top_k_final
            )
            
            run_latencies.append(latency)
        
        all_latencies.extend(run_latencies)
        avg_latency = mean(run_latencies)
        print(f" done (avg: {avg_latency:.2f}ms)")
    
    # Calcular estatísticas
    result = RerankResult(
        latency_mean_ms=mean(all_latencies),
        latency_std_ms=stdev(all_latencies) if len(all_latencies) > 1 else 0,
        latency_p95_ms=np.percentile(all_latencies, 95),
        candidates_per_query=top_k_rerank,
        final_top_k=top_k_final,
        device_used=reranker.device,
    )
    
    print(f"\n📊 Resultados ColBERT Reranking:")
    print(f"   Device: {result.device_used.upper()}")
    print(f"   Latência média: {result.latency_mean_ms:.2f} ± {result.latency_std_ms:.2f} ms")
    print(f"   Latência P95: {result.latency_p95_ms:.2f} ms")
    
    return result


if __name__ == "__main__":
    # Teste básico
    print("=" * 60)
    print("TESTE COLBERT RERANKER")
    print("=" * 60)
    
    reranker = ColBERTReranker()
    reranker.load_model()
    
    query = "Reclamação sobre atendimento no posto de saúde"
    docs = [
        {"id": 1, "text": "O atendimento no posto foi péssimo, esperei 4 horas"},
        {"id": 2, "text": "Parabéns pelo excelente serviço prestado"},
        {"id": 3, "text": "Solicito informações sobre horário de funcionamento"},
    ]
    
    reranked, latency = reranker.rerank(query, docs, top_k=3)
    
    print(f"\nQuery: {query}")
    print(f"Reranked (latência: {latency:.2f}ms, device: {reranker.device}):")
    for i, doc in enumerate(reranked, 1):
        print(f"  {i}. ID {doc['id']}: {doc['text'][:50]}...")
