#!/usr/bin/env python3
"""
Voice space explorer for interactive voice navigation and analysis.

This script provides an interactive interface for exploring voice spaces,
finding similar voices, and analyzing voice characteristics.
"""

import sys
import os
import numpy as np
import torch
import torchaudio
from pathlib import Path
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Any, Tuple, Optional
import librosa
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from synthesis.enhanced_synthesizer import EnhancedSynthesizer
from voice.voice_analyzer import VoiceAnalyzer
from voice.voice_interpolator import VoiceInterpolator
from utils.voice_utils import VoiceUtils


def main():
    """Main voice space exploration function."""
    print("🗺️ Voice Space Explorer")
    print("=" * 50)
    
    # Configuration
    config = {
        'sample_rate': 22050,
        'n_mels': 80,
        'hop_length': 256,
        'win_length': 1024,
        'embedding_dim': 64,
    }
    
    # Initialize components
    print("Initializing components...")
    synthesizer = EnhancedSynthesizer(config)
    voice_analyzer = VoiceAnalyzer()
    voice_interpolator = VoiceInterpolator()
    voice_utils = VoiceUtils()
    
    # Initialize models
    print("Initializing TTS models...")
    synthesizer.initialize_models()
    
    # Create or load voice database
    print("Setting up voice database...")
    voice_database = create_comprehensive_voice_database()
    synthesizer.load_voice_database(voice_database)
    
    print(f"Loaded {len(voice_database)} voices for exploration")
    
    # Voice space analysis
    print("\n🔍 Analyzing voice space...")
    voice_space_analysis = analyze_voice_space(voice_database, voice_analyzer)
    
    # Create voice space visualizations
    print("\n📊 Creating voice space visualizations...")
    create_voice_space_visualizations(voice_database, voice_space_analysis)
    
    # Voice similarity analysis
    print("\n🔗 Analyzing voice similarities...")
    similarity_analysis = analyze_voice_similarities(voice_database, voice_analyzer)
    
    # Voice clustering
    print("\n🎯 Performing voice clustering...")
    clustering_results = perform_voice_clustering(voice_database, voice_analyzer)
    
    # Voice navigation
    print("\n🧭 Voice navigation and exploration...")
    navigation_results = explore_voice_navigation(voice_database, synthesizer, voice_interpolator)
    
    # Voice recommendations
    print("\n💡 Generating voice recommendations...")
    recommendations = generate_voice_recommendations(voice_database, voice_analyzer)
    
    # Generate exploration report
    print("\n📋 Generating exploration report...")
    generate_exploration_report(voice_space_analysis, similarity_analysis, clustering_results, navigation_results, recommendations)
    
    print("\n✅ Voice space exploration complete!")
    print("Check the 'output' directory for detailed results and visualizations.")


def create_comprehensive_voice_database() -> Dict[str, np.ndarray]:
    """Create a comprehensive voice database for exploration."""
    print("Creating comprehensive voice database...")
    
    voice_database = {}
    
    # Create diverse voice types
    voice_types = [
        # High pitch voices
        {'name': 'high_pitch_bright', 'pitch': 0.8, 'brightness': 0.7, 'energy': 0.6},
        {'name': 'high_pitch_warm', 'pitch': 0.7, 'brightness': 0.4, 'energy': 0.5},
        {'name': 'high_pitch_energetic', 'pitch': 0.9, 'brightness': 0.8, 'energy': 0.9},
        
        # Medium pitch voices
        {'name': 'medium_pitch_balanced', 'pitch': 0.5, 'brightness': 0.5, 'energy': 0.5},
        {'name': 'medium_pitch_warm', 'pitch': 0.4, 'brightness': 0.3, 'energy': 0.4},
        {'name': 'medium_pitch_bright', 'pitch': 0.6, 'brightness': 0.7, 'energy': 0.6},
        
        # Low pitch voices
        {'name': 'low_pitch_warm', 'pitch': 0.3, 'brightness': 0.3, 'energy': 0.4},
        {'name': 'low_pitch_deep', 'pitch': 0.2, 'brightness': 0.2, 'energy': 0.3},
        {'name': 'low_pitch_rich', 'pitch': 0.4, 'brightness': 0.4, 'energy': 0.5},
        
        # Specialized voices
        {'name': 'whisper_voice', 'pitch': 0.3, 'brightness': 0.2, 'energy': 0.1},
        {'name': 'shout_voice', 'pitch': 0.8, 'brightness': 0.9, 'energy': 0.9},
        {'name': 'singing_voice', 'pitch': 0.6, 'brightness': 0.6, 'energy': 0.7},
    ]
    
    for voice_type in voice_types:
        # Create embedding based on voice characteristics
        embedding = np.zeros(64)
        
        # Set primary characteristics
        embedding[0] = voice_type['pitch']  # Pitch
        embedding[1] = np.random.uniform(0.1, 0.5)  # Pitch variation
        embedding[2] = voice_type['brightness'] * 4000 / 4000  # Spectral centroid
        embedding[3] = voice_type['brightness'] * 8000 / 8000  # Spectral rolloff
        embedding[4] = np.random.uniform(0.1, 0.6)  # Zero crossing rate
        embedding[5] = np.random.uniform(0.3, 0.8)  # Duration factor
        embedding[6] = voice_type['energy']  # MFCC mean
        embedding[7] = np.random.uniform(0.1, 0.4)  # MFCC std
        
        # Fill remaining dimensions with random values
        for i in range(8, 64):
            embedding[i] = np.random.normal(0, 0.1)
        
        voice_database[voice_type['name']] = embedding
    
    print(f"Created {len(voice_database)} diverse voices")
    return voice_database


def analyze_voice_space(voice_database: Dict[str, np.ndarray], voice_analyzer: VoiceAnalyzer) -> Dict[str, Any]:
    """Analyze the voice space structure."""
    print("Analyzing voice space structure...")
    
    # Extract embeddings
    voice_ids = list(voice_database.keys())
    embeddings = np.array([voice_database[voice_id] for voice_id in voice_ids])
    
    # Basic statistics
    analysis = {
        'num_voices': len(voice_database),
        'embedding_dim': embeddings.shape[1],
        'voice_ids': voice_ids,
    }
    
    # Statistical analysis
    analysis['statistics'] = {
        'mean_embedding': np.mean(embeddings, axis=0).tolist(),
        'std_embedding': np.std(embeddings, axis=0).tolist(),
        'embedding_range': (np.max(embeddings) - np.min(embeddings)).tolist(),
    }
    
    # Pairwise distances
    pairwise_distances = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            distance = np.linalg.norm(embeddings[i] - embeddings[j])
            pairwise_distances.append(distance)
    
    analysis['distance_analysis'] = {
        'mean_distance': float(np.mean(pairwise_distances)),
        'std_distance': float(np.std(pairwise_distances)),
        'min_distance': float(np.min(pairwise_distances)),
        'max_distance': float(np.max(pairwise_distances)),
    }
    
    # Dimensionality analysis
    pca = PCA()
    pca.fit(embeddings)
    
    analysis['dimensionality'] = {
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
        'effective_dimensions_95': int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1),
        'effective_dimensions_99': int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.99) + 1),
    }
    
    return analysis


def create_voice_space_visualizations(voice_database: Dict[str, np.ndarray], analysis: Dict[str, Any]) -> None:
    """Create comprehensive voice space visualizations."""
    print("Creating voice space visualizations...")
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    voice_ids = list(voice_database.keys())
    embeddings = np.array([voice_database[voice_id] for voice_id in voice_ids])
    
    # 1. PCA visualization
    print("  Creating PCA visualization...")
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=100)
    
    # Add labels
    for i, voice_id in enumerate(voice_ids):
        plt.annotate(voice_id, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.title('Voice Space - PCA Visualization')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "voice_space_pca.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. t-SNE visualization
    print("  Creating t-SNE visualization...")
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], alpha=0.7, s=100)
    
    # Add labels
    for i, voice_id in enumerate(voice_ids):
        plt.annotate(voice_id, (embeddings_tsne[i, 0], embeddings_tsne[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.title('Voice Space - t-SNE Visualization')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "voice_space_tsne.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Similarity heatmap
    print("  Creating similarity heatmap...")
    similarity_matrix = cosine_similarity(embeddings)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, fmt='.2f', 
                xticklabels=voice_ids, yticklabels=voice_ids, cmap='viridis')
    plt.title('Voice Similarity Matrix')
    plt.xlabel('Voice ID')
    plt.ylabel('Voice ID')
    plt.tight_layout()
    plt.savefig(output_dir / "voice_similarity_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Dimensionality analysis
    print("  Creating dimensionality analysis...")
    plt.figure(figsize=(12, 6))
    
    # Explained variance
    plt.subplot(1, 2, 1)
    explained_variance = analysis['dimensionality']['explained_variance_ratio']
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, 'o-')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
    plt.axhline(y=0.99, color='g', linestyle='--', label='99% Variance')
    plt.title('PCA Explained Variance')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Cumulative variance
    plt.subplot(1, 2, 2)
    cumulative_variance = analysis['dimensionality']['cumulative_variance']
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'o-')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
    plt.axhline(y=0.99, color='g', linestyle='--', label='99% Variance')
    plt.title('Cumulative Explained Variance')
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Variance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "dimensionality_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Voice space visualizations saved to output directory")


def analyze_voice_similarities(voice_database: Dict[str, np.ndarray], voice_analyzer: VoiceAnalyzer) -> Dict[str, Any]:
    """Analyze voice similarities and relationships."""
    print("Analyzing voice similarities...")
    
    voice_ids = list(voice_database.keys())
    embeddings = np.array([voice_database[voice_id] for voice_id in voice_ids])
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    # Find most similar pairs
    most_similar_pairs = []
    for i in range(len(voice_ids)):
        for j in range(i+1, len(voice_ids)):
            similarity = similarity_matrix[i, j]
            most_similar_pairs.append((voice_ids[i], voice_ids[j], similarity))
    
    # Sort by similarity
    most_similar_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Find most dissimilar pairs
    most_dissimilar_pairs = []
    for i in range(len(voice_ids)):
        for j in range(i+1, len(voice_ids)):
            similarity = similarity_matrix[i, j]
            most_dissimilar_pairs.append((voice_ids[i], voice_ids[j], similarity))
    
    # Sort by similarity (ascending)
    most_dissimilar_pairs.sort(key=lambda x: x[2])
    
    # Voice centrality analysis
    centrality_scores = {}
    for i, voice_id in enumerate(voice_ids):
        # Calculate average similarity to all other voices
        similarities = [similarity_matrix[i, j] for j in range(len(voice_ids)) if i != j]
        centrality_scores[voice_id] = np.mean(similarities)
    
    # Sort by centrality
    most_central_voices = sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'similarity_matrix': similarity_matrix.tolist(),
        'most_similar_pairs': most_similar_pairs[:5],  # Top 5
        'most_dissimilar_pairs': most_dissimilar_pairs[:5],  # Top 5
        'centrality_scores': centrality_scores,
        'most_central_voices': most_central_voices[:5],  # Top 5
    }


def perform_voice_clustering(voice_database: Dict[str, np.ndarray], voice_analyzer: VoiceAnalyzer) -> Dict[str, Any]:
    """Perform voice clustering analysis."""
    print("Performing voice clustering...")
    
    voice_ids = list(voice_database.keys())
    embeddings = np.array([voice_database[voice_id] for voice_id in voice_ids])
    
    clustering_results = {}
    
    # K-means clustering
    print("  Running K-means clustering...")
    for n_clusters in [3, 5, 7]:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(embeddings)
        
        # Group voices by cluster
        clusters = {}
        for i, voice_id in enumerate(voice_ids):
            cluster_id = f"cluster_{kmeans_labels[i]}"
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(voice_id)
        
        clustering_results[f'kmeans_{n_clusters}'] = {
            'labels': kmeans_labels.tolist(),
            'clusters': clusters,
            'inertia': float(kmeans.inertia_),
        }
    
    # DBSCAN clustering
    print("  Running DBSCAN clustering...")
    dbscan = DBSCAN(eps=0.5, min_samples=2)
    dbscan_labels = dbscan.fit_predict(embeddings)
    
    # Group voices by cluster
    dbscan_clusters = {}
    for i, voice_id in enumerate(voice_ids):
        cluster_id = f"cluster_{dbscan_labels[i]}" if dbscan_labels[i] != -1 else "noise"
        if cluster_id not in dbscan_clusters:
            dbscan_clusters[cluster_id] = []
        dbscan_clusters[cluster_id].append(voice_id)
    
    clustering_results['dbscan'] = {
        'labels': dbscan_labels.tolist(),
        'clusters': dbscan_clusters,
        'n_clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
        'n_noise': list(dbscan_labels).count(-1),
    }
    
    return clustering_results


def explore_voice_navigation(voice_database: Dict[str, np.ndarray], synthesizer: EnhancedSynthesizer, voice_interpolator: VoiceInterpolator) -> Dict[str, Any]:
    """Explore voice navigation and interpolation."""
    print("Exploring voice navigation...")
    
    voice_ids = list(voice_database.keys())
    navigation_results = {}
    
    # Voice continuum exploration
    print("  Creating voice continua...")
    continua = []
    for i in range(len(voice_ids) - 1):
        voice1_id = voice_ids[i]
        voice2_id = voice_ids[i + 1]
        
        try:
            continuum = voice_interpolator.create_voice_continuum(
                voice_database[voice1_id],
                voice_database[voice2_id],
                num_steps=5
            )
            continua.append({
                'voice1': voice1_id,
                'voice2': voice2_id,
                'continuum': continuum,
            })
        except Exception as e:
            print(f"    Error creating continuum {voice1_id} -> {voice2_id}: {e}")
    
    navigation_results['continua'] = continua
    
    # Voice morphing exploration
    print("  Creating voice morphing...")
    morphing_results = []
    for i in range(len(voice_ids) - 2):
        source_voices = [voice_database[voice_ids[i]], voice_database[voice_ids[i + 1]]]
        target_voice = voice_database[voice_ids[i + 2]]
        
        try:
            morphed_voices = voice_interpolator.create_voice_morphing(
                source_voices, target_voice, morphing_steps=5
            )
            morphing_results.append({
                'source_voices': [voice_ids[i], voice_ids[i + 1]],
                'target_voice': voice_ids[i + 2],
                'morphed_voices': morphed_voices,
            })
        except Exception as e:
            print(f"    Error creating morphing: {e}")
    
    navigation_results['morphing'] = morphing_results
    
    # Voice family creation
    print("  Creating voice families...")
    families = []
    for voice_id in voice_ids[:3]:  # Limit for performance
        try:
            family_voices = voice_interpolator.create_voice_family(
                voice_database[voice_id], family_size=3, variation_scale=0.2
            )
            families.append({
                'parent_voice': voice_id,
                'family_voices': family_voices,
            })
        except Exception as e:
            print(f"    Error creating family for {voice_id}: {e}")
    
    navigation_results['families'] = families
    
    return navigation_results


def generate_voice_recommendations(voice_database: Dict[str, np.ndarray], voice_analyzer: VoiceAnalyzer) -> Dict[str, Any]:
    """Generate voice recommendations based on analysis."""
    print("Generating voice recommendations...")
    
    voice_ids = list(voice_database.keys())
    embeddings = np.array([voice_database[voice_id] for voice_id in voice_ids])
    
    recommendations = {}
    
    # For each voice, find similar voices
    for i, voice_id in enumerate(voice_ids):
        query_embedding = embeddings[i]
        similar_voices = voice_analyzer.recommend_similar_voices(
            query_embedding, voice_database, top_k=3, similarity_threshold=0.3
        )
        recommendations[voice_id] = similar_voices
    
    # Find outlier voices
    outliers = voice_analyzer.find_voice_outliers(voice_database, method="isolation_forest")
    
    # Find representative voices (most central)
    centrality_scores = {}
    for i, voice_id in enumerate(voice_ids):
        similarities = [1 - np.linalg.norm(embeddings[i] - embeddings[j]) for j in range(len(voice_ids)) if i != j]
        centrality_scores[voice_id] = np.mean(similarities)
    
    most_representative = sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    
    return {
        'similar_voices': recommendations,
        'outlier_voices': outliers,
        'representative_voices': most_representative,
    }


def generate_exploration_report(voice_space_analysis: Dict[str, Any], similarity_analysis: Dict[str, Any], 
                              clustering_results: Dict[str, Any], navigation_results: Dict[str, Any], 
                              recommendations: Dict[str, Any]) -> None:
    """Generate comprehensive exploration report."""
    print("Generating exploration report...")
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    report_path = output_dir / "voice_space_exploration_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("Voice Space Exploration Report\n")
        f.write("=" * 40 + "\n\n")
        
        # Voice space analysis
        f.write("Voice Space Analysis:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Number of voices: {voice_space_analysis['num_voices']}\n")
        f.write(f"Embedding dimension: {voice_space_analysis['embedding_dim']}\n")
        
        if 'distance_analysis' in voice_space_analysis:
            dist_analysis = voice_space_analysis['distance_analysis']
            f.write(f"Mean pairwise distance: {dist_analysis['mean_distance']:.4f}\n")
            f.write(f"Distance std: {dist_analysis['std_distance']:.4f}\n")
            f.write(f"Min distance: {dist_analysis['min_distance']:.4f}\n")
            f.write(f"Max distance: {dist_analysis['max_distance']:.4f}\n")
        
        if 'dimensionality' in voice_space_analysis:
            dim_analysis = voice_space_analysis['dimensionality']
            f.write(f"Effective dimensions (95% variance): {dim_analysis['effective_dimensions_95']}\n")
            f.write(f"Effective dimensions (99% variance): {dim_analysis['effective_dimensions_99']}\n")
        
        f.write("\n")
        
        # Similarity analysis
        f.write("Similarity Analysis:\n")
        f.write("-" * 20 + "\n")
        f.write("Most similar voice pairs:\n")
        for voice1, voice2, similarity in similarity_analysis['most_similar_pairs']:
            f.write(f"  {voice1} <-> {voice2}: {similarity:.4f}\n")
        
        f.write("\nMost dissimilar voice pairs:\n")
        for voice1, voice2, similarity in similarity_analysis['most_dissimilar_pairs']:
            f.write(f"  {voice1} <-> {voice2}: {similarity:.4f}\n")
        
        f.write("\nMost central voices:\n")
        for voice_id, centrality in similarity_analysis['most_central_voices']:
            f.write(f"  {voice_id}: {centrality:.4f}\n")
        
        f.write("\n")
        
        # Clustering results
        f.write("Clustering Results:\n")
        f.write("-" * 20 + "\n")
        for method, results in clustering_results.items():
            f.write(f"{method}:\n")
            if 'clusters' in results:
                for cluster_id, voices in results['clusters'].items():
                    f.write(f"  {cluster_id}: {voices}\n")
            if 'inertia' in results:
                f.write(f"  Inertia: {results['inertia']:.4f}\n")
            f.write("\n")
        
        # Navigation results
        f.write("Navigation Results:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Voice continua created: {len(navigation_results['continua'])}\n")
        f.write(f"Voice morphing created: {len(navigation_results['morphing'])}\n")
        f.write(f"Voice families created: {len(navigation_results['families'])}\n")
        
        f.write("\n")
        
        # Recommendations
        f.write("Voice Recommendations:\n")
        f.write("-" * 20 + "\n")
        f.write("Representative voices:\n")
        for voice_id, centrality in recommendations['representative_voices']:
            f.write(f"  {voice_id}: {centrality:.4f}\n")
        
        f.write(f"\nOutlier voices: {recommendations['outlier_voices']}\n")
    
    print(f"Exploration report saved to: {report_path}")


if __name__ == "__main__":
    main()









































