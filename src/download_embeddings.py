#!/usr/bin/env python3
"""
Download de embeddings do S3 bucket - V3

Baixa os embeddings BGE-M3 do S3 bucket:
- Embeddings densos (sumário e tópico)
- Embeddings esparsos
- ColBERT por região
"""

import os
import subprocess
from pathlib import Path
from typing import Optional
import boto3


S3_BUCKET = "your-benchmark-bucket"
S3_REGION = "us-east-1"

# Arquivos principais
MAIN_FILES = [
    "df_base_hist_topicos_vetor_denso_sumario_ia_bge-m3_v1.0_2026-06-17.parquet",
    "df_base_hist_topicos_vetor_denso_topico_ia_bge-m3_v1.0_2026-06-17.parquet",
    "df_base_hist_topicos_vetor_esparso_palavras_chave_ia_bge-m3_v1.0_2026-06-17.parquet",
]

# Partições ColBERT por região
COLBERT_PARTITIONS = [f"region_{i:02d}" for i in range(1, 13)]  # 12 regiões


def check_s3_file_exists(bucket: str, key: str) -> bool:
    """Verifica se arquivo existe no S3"""
    s3 = boto3.client('s3', region_name=S3_REGION)
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except:
        return False


def download_from_s3(
    s3_path: str,
    local_path: str,
    show_progress: bool = True
) -> bool:
    """
    Baixa arquivo do S3 usando AWS CLI.
    
    Args:
        s3_path: Caminho S3 (s3://bucket/key)
        local_path: Caminho local
        show_progress: Mostrar progresso
    
    Returns:
        True se sucesso, False caso contrário
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    cmd = ["aws", "s3", "cp", s3_path, local_path]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            if show_progress:
                print(f"   ✓ Baixado: {os.path.basename(local_path)}")
            return True
        else:
            print(f"   ✗ Erro: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"   ✗ Timeout ao baixar {s3_path}")
        return False


def download_all_embeddings(
    output_dir: str,
    include_colbert: bool = False,
    show_progress: bool = True
) -> dict:
    """
    Baixa todos os embeddings do S3 bucket.
    
    Args:
        output_dir: Diretório de destino
        include_colbert: Incluir partições ColBERT (~20GB)
        show_progress: Mostrar progresso
    
    Returns:
        Dict com status dos downloads
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "success": [],
        "failed": [],
        "skipped": []
    }
    
    print("=" * 80)
    print("DOWNLOAD DE EMBEDDINGS DO S3 - BENCHMARK V3")
    print("=" * 80)
    print(f"\nBucket: s3://{S3_BUCKET}")
    print(f"Destino: {output_dir}")
    
    # 1. Baixar arquivos principais
    print(f"\n📦 Baixando embeddings principais...")
    
    for filename in MAIN_FILES:
        local_path = output_dir / filename
        
        if local_path.exists():
            print(f"   ✓ Já existe: {filename}")
            results["skipped"].append(filename)
            continue
        
        s3_path = f"s3://{S3_BUCKET}/{filename}"
        
        if show_progress:
            print(f"   Baixando {filename}...")
        
        if download_from_s3(s3_path, str(local_path), show_progress):
            results["success"].append(filename)
        else:
            results["failed"].append(filename)
    
    # 2. Baixar ColBERT (opcional, ~20GB)
    if include_colbert:
        print(f"\n📦 Baixando ColBERT por região (~20GB)...")
        
        colbert_dir = output_dir / "colbert_sumario"
        colbert_dir.mkdir(parents=True, exist_ok=True)
        
        for partition in COLBERT_PARTITIONS:
            filename = f"df_base_hist_{partition}_colbert_sumario_ia_bge-m3_v1.0_2026-06-17.parquet"
            local_path = colbert_dir / filename
            
            if local_path.exists():
                print(f"   ✓ Já existe: {partition}")
                results["skipped"].append(f"colbert_sumario/{filename}")
                continue
            
            s3_path = f"s3://{S3_BUCKET}/colbert_sumario/{filename}"
            
            if show_progress:
                print(f"   Baixando {partition}...")
            
            if download_from_s3(s3_path, str(local_path), show_progress):
                results["success"].append(f"colbert_sumario/{filename}")
            else:
                results["failed"].append(f"colbert_sumario/{filename}")
    
    # Resumo
    print("\n" + "=" * 80)
    print("RESUMO DO DOWNLOAD")
    print("=" * 80)
    print(f"   Sucesso: {len(results['success'])}")
    print(f"   Falhou: {len(results['failed'])}")
    print(f"   Pulou (já existe): {len(results['skipped'])}")
    
    if results['failed']:
        print(f"\n   Arquivos com falha:")
        for f in results['failed']:
            print(f"      - {f}")
    
    return results


def verify_embeddings_dir(embeddings_dir: str) -> dict:
    """
    Verifica se os embeddings existem no diretório.
    
    Returns:
        Dict com status de cada arquivo
    """
    embeddings_dir = Path(embeddings_dir)
    
    status = {
        "denso_sumario": (embeddings_dir / MAIN_FILES[0]).exists(),
        "denso_topico": (embeddings_dir / MAIN_FILES[1]).exists(),
        "esparso": (embeddings_dir / MAIN_FILES[2]).exists(),
        "colbert_dir": (embeddings_dir / "colbert_sumario").exists(),
    }
    
    # Verificar partições ColBERT
    colbert_dir = embeddings_dir / "colbert_sumario"
    if colbert_dir.exists():
        colbert_files = list(colbert_dir.glob("*.parquet"))
        status["colbert_partitions"] = len(colbert_files)
    else:
        status["colbert_partitions"] = 0
    
    status["all_main_present"] = all([
        status["denso_sumario"],
        status["denso_topico"],
        status["esparso"]
    ])
    
    return status


def get_embeddings_dir(prefer_s3_download: bool = False) -> str:
    """
    Retorna o diretório de embeddings, baixando do S3 se necessário.
    
    Args:
        prefer_s3_download: Se True, força download do S3
    
    Returns:
        Caminho do diretório de embeddings
    """
    # Prioridade de locais
    possible_dirs = [
        "./data",
        "./data",
    ]
    
    # Verificar se já existe em algum local
    for dir_path in possible_dirs:
        status = verify_embeddings_dir(dir_path)
        if status["all_main_present"]:
            print(f"✓ Embeddings encontrados em: {dir_path}")
            return dir_path
    
    # Se não encontrou, baixar do S3
    if prefer_s3_download:
        print("Embeddings não encontrados localmente. Baixando do S3...")
        output_dir = "./data"
        download_all_embeddings(output_dir, include_colbert=False)
        return output_dir
    
    raise FileNotFoundError(
        "Embeddings não encontrados. Execute download_from_s3() ou "
        "baixe manualmente do S3 bucket 'your-benchmark-bucket'"
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download de embeddings do S3")
    parser.add_argument("--output-dir", default="./data",
                        help="Diretório de destino")
    parser.add_argument("--include-colbert", action="store_true",
                        help="Incluir partições ColBERT (~20GB)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Apenas verificar se arquivos existem")
    
    args = parser.parse_args()
    
    if args.verify_only:
        print(f"\nVerificando: {args.output_dir}")
        status = verify_embeddings_dir(args.output_dir)
        
        print("\nStatus dos arquivos:")
        for key, value in status.items():
            print(f"  {key}: {value}")
    else:
        download_all_embeddings(
            args.output_dir,
            include_colbert=args.include_colbert
        )
