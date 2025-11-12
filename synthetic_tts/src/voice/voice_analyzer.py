"""
Voice analyzer for characterizing and analyzing voice embeddings.

This module provides comprehensive voice analysis capabilities including
voice quality assessment, similarity measurement, and voice characterization.
"""

import torch
import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class VoiceAnalyzer:
    """
    Comprehensive voice analyzer for voice embeddings and characteristics.
    
    This class provides various analysis methods for understanding voice
    characteristics, quality assessment, and similarity measurement.
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        sample_rate: int = 22050,
        n_mels: int = 80,
    ):
        """
        Initialize the voice analyzer.
        
        Args:
            embedding_dim: Dimension of voice embeddings
            sample_rate: Audio sample rate
            n_mels: Number of mel-spectrogram channels
        """
        self.embedding_dim = embedding_dim
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        
        # Analysis cache
        self.analysis_cache = {}
        
        # Standard scaler for normalization
        self.scaler = StandardScaler()
        self.scaler_fitted = False
    
    def analyze_voice_characteristics(
        self,
        voice_embedding: np.ndarray,
        audio: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Analyze characteristics of a single voice.
        
        Args:
            voice_embedding: Voice embedding vector
            audio: Optional audio signal for additional analysis
        
        Returns:
            Dictionary containing voice characteristics
        """
        characteristics = {}
        
        # Basic embedding statistics
        characteristics['embedding_stats'] = {
            'mean': float(np.mean(voice_embedding)),
            'std': float(np.std(voice_embedding)),
            'min': float(np.min(voice_embedding)),
            'max': float(np.max(voice_embedding)),
            'norm': float(np.linalg.norm(voice_embedding)),
        }
        
        # Voice characteristics from embedding
        characteristics['voice_traits'] = self._extract_voice_traits(voice_embedding)
        
        # Audio analysis if provided
        if audio is not None:
            characteristics['audio_analysis'] = self._analyze_audio(audio)
        
        # Voice quality assessment
        characteristics['quality_metrics'] = self._assess_voice_quality(voice_embedding)
        
        return characteristics
    
    def _extract_voice_traits(self, voice_embedding: np.ndarray) -> Dict[str, float]:
        """Extract voice traits from embedding."""
        traits = {}
        
        # Assume embedding structure based on our speaker embedding format
        if len(voice_embedding) >= 8:
            traits['pitch_mean'] = float(voice_embedding[0] * 300)  # Denormalize
            traits['pitch_variability'] = float(voice_embedding[1] * 100)
            traits['spectral_centroid'] = float(voice_embedding[2] * 4000)
            traits['spectral_rolloff'] = float(voice_embedding[3] * 8000)
            traits['zero_crossing_rate'] = float(voice_embedding[4])
            traits['duration_factor'] = float(voice_embedding[5] * 10)
            traits['mfcc_mean'] = float(voice_embedding[6])
            traits['mfcc_variability'] = float(voice_embedding[7])
        else:
            # Fallback for different embedding formats
            traits['embedding_mean'] = float(np.mean(voice_embedding))
            traits['embedding_std'] = float(np.std(voice_embedding))
            traits['embedding_range'] = float(np.max(voice_embedding) - np.min(voice_embedding))
        
        # Derived characteristics
        traits['voice_brightness'] = traits.get('spectral_centroid', 0) / 2000.0
        traits['voice_warmth'] = 1.0 - traits.get('zero_crossing_rate', 0)
        traits['voice_energy'] = traits.get('mfcc_mean', 0) + 0.5
        
        return traits
    
    def _analyze_audio(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze audio signal for voice characteristics."""
        analysis = {}
        
        # Basic audio statistics
        analysis['rms_energy'] = float(np.sqrt(np.mean(audio ** 2)))
        analysis['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(audio)[0]))
        analysis['spectral_centroid'] = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]))
        analysis['spectral_rolloff'] = float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]))
        
        # Pitch analysis
        pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if pitch_values:
            analysis['pitch_mean'] = float(np.mean(pitch_values))
            analysis['pitch_std'] = float(np.std(pitch_values))
            analysis['pitch_range'] = float(np.max(pitch_values) - np.min(pitch_values))
        else:
            analysis['pitch_mean'] = 0.0
            analysis['pitch_std'] = 0.0
            analysis['pitch_range'] = 0.0
        
        # MFCC analysis
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        analysis['mfcc_mean'] = float(np.mean(mfcc))
        analysis['mfcc_std'] = float(np.std(mfcc))
        
        # Spectral features
        analysis['spectral_bandwidth'] = float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]))
        analysis['spectral_contrast'] = float(np.mean(librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)[0]))
        
        return analysis
    
    def _assess_voice_quality(self, voice_embedding: np.ndarray) -> Dict[str, float]:
        """Assess voice quality based on embedding."""
        quality_metrics = {}
        
        # Embedding quality indicators
        embedding_norm = np.linalg.norm(voice_embedding)
        embedding_std = np.std(voice_embedding)
        embedding_range = np.max(voice_embedding) - np.min(voice_embedding)
        
        # Quality scores (0-1 scale)
        quality_metrics['embedding_stability'] = min(1.0, embedding_norm / 10.0)  # Normalize
        quality_metrics['voice_consistency'] = min(1.0, embedding_std / 2.0)  # Moderate variation is good
        quality_metrics['voice_diversity'] = min(1.0, embedding_range / 4.0)  # Good range
        
        # Overall quality
        quality_metrics['overall_quality'] = np.mean(list(quality_metrics.values()))
        
        return quality_metrics
    
    def compare_voices(
        self,
        voice1: np.ndarray,
        voice2: np.ndarray,
        metrics: List[str] = None,
    ) -> Dict[str, float]:
        """
        Compare two voice embeddings.
        
        Args:
            voice1: First voice embedding
            voice2: Second voice embedding
            metrics: List of similarity metrics to compute
        
        Returns:
            Dictionary of similarity metrics
        """
        if metrics is None:
            metrics = ['cosine', 'euclidean', 'pearson', 'spearman']
        
        comparison = {}
        
        for metric in metrics:
            if metric == 'cosine':
                comparison['cosine_similarity'] = 1 - cosine(voice1, voice2)
            elif metric == 'euclidean':
                comparison['euclidean_distance'] = euclidean(voice1, voice2)
            elif metric == 'pearson':
                correlation, _ = pearsonr(voice1, voice2)
                comparison['pearson_correlation'] = correlation
            elif metric == 'spearman':
                correlation, _ = spearmanr(voice1, voice2)
                comparison['spearman_correlation'] = correlation
            elif metric == 'dot_product':
                comparison['dot_product'] = np.dot(voice1, voice2)
            elif metric == 'manhattan':
                comparison['manhattan_distance'] = np.sum(np.abs(voice1 - voice2))
        
        return comparison
    
    def analyze_voice_database(
        self,
        voice_database: Dict[str, np.ndarray],
        analysis_type: str = "comprehensive",
    ) -> Dict[str, Any]:
        """
        Analyze a database of voice embeddings.
        
        Args:
            voice_database: Dictionary mapping voice IDs to embeddings
            analysis_type: Type of analysis to perform
        
        Returns:
            Dictionary containing database analysis results
        """
        if not voice_database:
            return {}
        
        voice_ids = list(voice_database.keys())
        embeddings = np.array([voice_database[voice_id] for voice_id in voice_ids])
        
        analysis = {
            'database_info': {
                'num_voices': len(voice_database),
                'embedding_dim': embeddings.shape[1],
                'voice_ids': voice_ids,
            }
        }
        
        if analysis_type == "comprehensive":
            # Statistical analysis
            analysis['statistics'] = self._compute_database_statistics(embeddings)
            
            # Similarity analysis
            analysis['similarity_matrix'] = self._compute_similarity_matrix(embeddings)
            
            # Clustering analysis
            analysis['clusters'] = self._perform_clustering(embeddings, voice_ids)
            
            # Dimensionality analysis
            analysis['dimensionality'] = self._analyze_dimensionality(embeddings)
            
            # Quality assessment
            analysis['quality_scores'] = self._assess_database_quality(embeddings, voice_ids)
        
        return analysis
    
    def _compute_database_statistics(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Compute statistical properties of the voice database."""
        stats = {
            'mean_embedding': np.mean(embeddings, axis=0).tolist(),
            'std_embedding': np.std(embeddings, axis=0).tolist(),
            'embedding_range': (np.max(embeddings) - np.min(embeddings)).tolist(),
            'pairwise_distances': {
                'mean': float(np.mean([euclidean(embeddings[i], embeddings[j]) 
                                     for i in range(len(embeddings)) 
                                     for j in range(i+1, len(embeddings))])),
                'std': float(np.std([euclidean(embeddings[i], embeddings[j]) 
                                   for i in range(len(embeddings)) 
                                   for j in range(i+1, len(embeddings))])),
                'min': float(np.min([euclidean(embeddings[i], embeddings[j]) 
                                   for i in range(len(embeddings)) 
                                   for j in range(i+1, len(embeddings))])),
                'max': float(np.max([euclidean(embeddings[i], embeddings[j]) 
                                   for i in range(len(embeddings)) 
                                   for j in range(i+1, len(embeddings))])),
            }
        }
        
        return stats
    
    def _compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise similarity matrix."""
        return cosine_similarity(embeddings)
    
    def _perform_clustering(self, embeddings: np.ndarray, voice_ids: List[str]) -> Dict[str, Any]:
        """Perform clustering analysis on voice embeddings."""
        # K-means clustering
        kmeans = KMeans(n_clusters=min(5, len(embeddings)), random_state=42)
        kmeans_labels = kmeans.fit_predict(embeddings)
        
        # DBSCAN clustering
        dbscan = DBSCAN(eps=0.5, min_samples=2)
        dbscan_labels = dbscan.fit_predict(embeddings)
        
        # Group voices by clusters
        kmeans_clusters = {}
        for i, voice_id in enumerate(voice_ids):
            cluster_id = f"kmeans_cluster_{kmeans_labels[i]}"
            if cluster_id not in kmeans_clusters:
                kmeans_clusters[cluster_id] = []
            kmeans_clusters[cluster_id].append(voice_id)
        
        dbscan_clusters = {}
        for i, voice_id in enumerate(voice_ids):
            cluster_id = f"dbscan_cluster_{dbscan_labels[i]}"
            if cluster_id not in dbscan_clusters:
                dbscan_clusters[cluster_id] = []
            dbscan_clusters[cluster_id].append(voice_id)
        
        return {
            'kmeans': kmeans_clusters,
            'dbscan': dbscan_clusters,
            'kmeans_labels': kmeans_labels.tolist(),
            'dbscan_labels': dbscan_labels.tolist(),
        }
    
    def _analyze_dimensionality(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Analyze dimensionality of voice embeddings."""
        # PCA analysis
        pca = PCA()
        pca.fit(embeddings)
        
        # Explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Effective dimensionality (95% variance)
        effective_dim = np.argmax(cumulative_variance >= 0.95) + 1
        
        return {
            'original_dimensions': embeddings.shape[1],
            'effective_dimensions': int(effective_dim),
            'explained_variance_ratio': explained_variance_ratio.tolist(),
            'cumulative_variance': cumulative_variance.tolist(),
            'variance_95_percent': float(cumulative_variance[effective_dim - 1]),
        }
    
    def _assess_database_quality(self, embeddings: np.ndarray, voice_ids: List[str]) -> Dict[str, Any]:
        """Assess quality of voice database."""
        quality_scores = {}
        
        # Individual voice quality
        individual_qualities = []
        for i, voice_id in enumerate(voice_ids):
            quality = self._assess_voice_quality(embeddings[i])
            individual_qualities.append(quality['overall_quality'])
        
        quality_scores['individual_qualities'] = {
            'mean': float(np.mean(individual_qualities)),
            'std': float(np.std(individual_qualities)),
            'min': float(np.min(individual_qualities)),
            'max': float(np.max(individual_qualities)),
        }
        
        # Database diversity
        pairwise_similarities = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                similarity = 1 - cosine(embeddings[i], embeddings[j])
                pairwise_similarities.append(similarity)
        
        quality_scores['diversity'] = {
            'mean_similarity': float(np.mean(pairwise_similarities)),
            'similarity_std': float(np.std(pairwise_similarities)),
            'diversity_score': 1.0 - float(np.mean(pairwise_similarities)),
        }
        
        # Overall database quality
        quality_scores['overall_quality'] = (
            quality_scores['individual_qualities']['mean'] * 0.7 +
            quality_scores['diversity']['diversity_score'] * 0.3
        )
        
        return quality_scores
    
    def visualize_voice_analysis(
        self,
        voice_database: Dict[str, np.ndarray],
        analysis_results: Dict[str, Any],
        save_path: Optional[str] = None,
    ) -> None:
        """
        Create visualizations for voice analysis.
        
        Args:
            voice_database: Database of voice embeddings
            analysis_results: Results from database analysis
            save_path: Path to save visualizations
        """
        voice_ids = list(voice_database.keys())
        embeddings = np.array([voice_database[voice_id] for voice_id in voice_ids])
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Similarity heatmap
        if 'similarity_matrix' in analysis_results:
            similarity_matrix = analysis_results['similarity_matrix']
            sns.heatmap(similarity_matrix, annot=True, fmt='.2f', ax=axes[0, 0])
            axes[0, 0].set_title('Voice Similarity Matrix')
            axes[0, 0].set_xlabel('Voice Index')
            axes[0, 0].set_ylabel('Voice Index')
        
        # 2. PCA visualization
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        axes[0, 1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
        for i, voice_id in enumerate(voice_ids):
            axes[0, 1].annotate(voice_id, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
        axes[0, 1].set_title('Voice Space (PCA)')
        axes[0, 1].set_xlabel('PC1')
        axes[0, 1].set_ylabel('PC2')
        
        # 3. Quality scores
        if 'quality_scores' in analysis_results:
            quality_scores = analysis_results['quality_scores']
            individual_qualities = quality_scores['individual_qualities']
            
            axes[1, 0].bar(['Mean', 'Std', 'Min', 'Max'], [
                individual_qualities['mean'],
                individual_qualities['std'],
                individual_qualities['min'],
                individual_qualities['max']
            ])
            axes[1, 0].set_title('Voice Quality Statistics')
            axes[1, 0].set_ylabel('Quality Score')
        
        # 4. Dimensionality analysis
        if 'dimensionality' in analysis_results:
            dim_analysis = analysis_results['dimensionality']
            explained_variance = dim_analysis['explained_variance_ratio']
            
            axes[1, 1].plot(range(1, len(explained_variance) + 1), explained_variance, 'o-')
            axes[1, 1].axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
            axes[1, 1].set_title('PCA Explained Variance')
            axes[1, 1].set_xlabel('Principal Component')
            axes[1, 1].set_ylabel('Explained Variance Ratio')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def find_voice_outliers(
        self,
        voice_database: Dict[str, np.ndarray],
        method: str = "isolation_forest",
        contamination: float = 0.1,
    ) -> List[str]:
        """
        Find outlier voices in the database.
        
        Args:
            voice_database: Database of voice embeddings
            method: Outlier detection method
            contamination: Expected proportion of outliers
        
        Returns:
            List of outlier voice IDs
        """
        from sklearn.ensemble import IsolationForest
        
        voice_ids = list(voice_database.keys())
        embeddings = np.array([voice_database[voice_id] for voice_id in voice_ids])
        
        if method == "isolation_forest":
            detector = IsolationForest(contamination=contamination, random_state=42)
            outlier_labels = detector.fit_predict(embeddings)
            
            outliers = [voice_ids[i] for i, label in enumerate(outlier_labels) if label == -1]
        
        return outliers
    
    def recommend_similar_voices(
        self,
        query_voice: np.ndarray,
        voice_database: Dict[str, np.ndarray],
        top_k: int = 5,
        similarity_threshold: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """
        Recommend similar voices based on a query voice.
        
        Args:
            query_voice: Query voice embedding
            voice_database: Database of voice embeddings
            top_k: Number of recommendations
            similarity_threshold: Minimum similarity threshold
        
        Returns:
            List of (voice_id, similarity_score) tuples
        """
        similarities = []
        
        for voice_id, voice_embedding in voice_database.items():
            similarity = 1 - cosine(query_voice, voice_embedding)
            if similarity >= similarity_threshold:
                similarities.append((voice_id, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def generate_voice_report(
        self,
        voice_database: Dict[str, np.ndarray],
        output_path: str = "voice_analysis_report.txt",
    ) -> str:
        """
        Generate a comprehensive voice analysis report.
        
        Args:
            voice_database: Database of voice embeddings
            output_path: Path to save the report
        
        Returns:
            Report content as string
        """
        # Perform comprehensive analysis
        analysis = self.analyze_voice_database(voice_database, "comprehensive")
        
        # Generate report
        report_lines = [
            "=" * 60,
            "VOICE DATABASE ANALYSIS REPORT",
            "=" * 60,
            "",
            f"Database Information:",
            f"  Number of voices: {analysis['database_info']['num_voices']}",
            f"  Embedding dimension: {analysis['database_info']['embedding_dim']}",
            "",
        ]
        
        # Statistics section
        if 'statistics' in analysis:
            stats = analysis['statistics']
            report_lines.extend([
                "STATISTICS:",
                f"  Mean pairwise distance: {stats['pairwise_distances']['mean']:.4f}",
                f"  Distance standard deviation: {stats['pairwise_distances']['std']:.4f}",
                f"  Distance range: {stats['pairwise_distances']['min']:.4f} - {stats['pairwise_distances']['max']:.4f}",
                "",
            ])
        
        # Quality section
        if 'quality_scores' in analysis:
            quality = analysis['quality_scores']
            report_lines.extend([
                "QUALITY ASSESSMENT:",
                f"  Overall quality: {quality['overall_quality']:.4f}",
                f"  Individual quality mean: {quality['individual_qualities']['mean']:.4f}",
                f"  Diversity score: {quality['diversity']['diversity_score']:.4f}",
                "",
            ])
        
        # Dimensionality section
        if 'dimensionality' in analysis:
            dim = analysis['dimensionality']
            report_lines.extend([
                "DIMENSIONALITY ANALYSIS:",
                f"  Original dimensions: {dim['original_dimensions']}",
                f"  Effective dimensions (95% variance): {dim['effective_dimensions']}",
                f"  Variance explained by 95%: {dim['variance_95_percent']:.4f}",
                "",
            ])
        
        # Clustering section
        if 'clusters' in analysis:
            clusters = analysis['clusters']
            report_lines.extend([
                "CLUSTERING ANALYSIS:",
                f"  K-means clusters: {len(clusters['kmeans'])}",
                f"  DBSCAN clusters: {len(clusters['dbscan'])}",
                "",
            ])
        
        report_content = "\n".join(report_lines)
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(report_content)
        
        return report_content








































