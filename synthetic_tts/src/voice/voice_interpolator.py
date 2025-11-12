"""
Voice interpolation engine for smooth voice transitions and morphing.

This module provides real-time voice interpolation capabilities for creating
smooth transitions between different voices and voice morphing.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
import librosa
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pathlib import Path


class VoiceInterpolator:
    """
    Voice interpolation engine for smooth voice transitions.
    
    This class provides various interpolation methods for creating
    smooth transitions between different voices.
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        interpolation_methods: List[str] = None,
    ):
        """
        Initialize the voice interpolator.
        
        Args:
            embedding_dim: Dimension of voice embeddings
            interpolation_methods: Available interpolation methods
        """
        self.embedding_dim = embedding_dim
        
        if interpolation_methods is None:
            interpolation_methods = [
                "linear",
                "spherical",
                "weighted",
                "pca_based",
                "gaussian",
                "spline"
            ]
        
        self.interpolation_methods = interpolation_methods
        
        # Interpolation cache
        self.cache = {}
        
        # PCA for dimensionality reduction
        self.pca = None
        self.pca_fitted = False
    
    def interpolate_voices(
        self,
        voice1_embedding: np.ndarray,
        voice2_embedding: np.ndarray,
        weight: float,
        method: str = "linear",
        **kwargs
    ) -> np.ndarray:
        """
        Interpolate between two voice embeddings.
        
        Args:
            voice1_embedding: First voice embedding
            voice2_embedding: Second voice embedding
            weight: Interpolation weight (0 = voice1, 1 = voice2)
            method: Interpolation method
            **kwargs: Additional parameters for specific methods
        
        Returns:
            Interpolated voice embedding
        """
        if method not in self.interpolation_methods:
            raise ValueError(f"Unknown interpolation method: {method}")
        
        # Clamp weight to [0, 1]
        weight = np.clip(weight, 0.0, 1.0)
        
        if method == "linear":
            return self._linear_interpolation(voice1_embedding, voice2_embedding, weight)
        elif method == "spherical":
            return self._spherical_interpolation(voice1_embedding, voice2_embedding, weight)
        elif method == "weighted":
            return self._weighted_interpolation(voice1_embedding, voice2_embedding, weight, **kwargs)
        elif method == "pca_based":
            return self._pca_based_interpolation(voice1_embedding, voice2_embedding, weight, **kwargs)
        elif method == "gaussian":
            return self._gaussian_interpolation(voice1_embedding, voice2_embedding, weight, **kwargs)
        elif method == "spline":
            return self._spline_interpolation(voice1_embedding, voice2_embedding, weight, **kwargs)
        else:
            return self._linear_interpolation(voice1_embedding, voice2_embedding, weight)
    
    def _linear_interpolation(
        self,
        voice1: np.ndarray,
        voice2: np.ndarray,
        weight: float
    ) -> np.ndarray:
        """Linear interpolation between two voices."""
        return (1 - weight) * voice1 + weight * voice2
    
    def _spherical_interpolation(
        self,
        voice1: np.ndarray,
        voice2: np.ndarray,
        weight: float
    ) -> np.ndarray:
        """Spherical interpolation on unit sphere."""
        # Normalize to unit sphere
        voice1_norm = voice1 / np.linalg.norm(voice1)
        voice2_norm = voice2 / np.linalg.norm(voice2)
        
        # Compute dot product
        dot_product = np.dot(voice1_norm, voice2_norm)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # Compute angle
        theta = np.arccos(dot_product)
        
        if theta == 0:
            return voice1_norm
        
        # Spherical interpolation
        sin_theta = np.sin(theta)
        interpolated = (np.sin((1 - weight) * theta) * voice1_norm + 
                      np.sin(weight * theta) * voice2_norm) / sin_theta
        
        return interpolated
    
    def _weighted_interpolation(
        self,
        voice1: np.ndarray,
        voice2: np.ndarray,
        weight: float,
        weights: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """Weighted interpolation with custom weights."""
        if weights is None:
            weights = np.ones_like(voice1)
        
        # Apply weights to interpolation
        weighted_voice1 = voice1 * weights
        weighted_voice2 = voice2 * weights
        
        interpolated = (1 - weight) * weighted_voice1 + weight * weighted_voice2
        
        # Normalize by weights
        return interpolated / (weights + 1e-8)
    
    def _pca_based_interpolation(
        self,
        voice1: np.ndarray,
        voice2: np.ndarray,
        weight: float,
        pca_components: int = 10,
        **kwargs
    ) -> np.ndarray:
        """PCA-based interpolation in reduced space."""
        if not self.pca_fitted:
            # Fit PCA on the two voices
            voices = np.vstack([voice1, voice2])
            self.pca = PCA(n_components=min(pca_components, len(voice1)))
            self.pca.fit(voices)
            self.pca_fitted = True
        
        # Transform to PCA space
        voice1_pca = self.pca.transform(voice1.reshape(1, -1))[0]
        voice2_pca = self.pca.transform(voice2.reshape(1, -1))[0]
        
        # Interpolate in PCA space
        interpolated_pca = (1 - weight) * voice1_pca + weight * voice2_pca
        
        # Transform back to original space
        interpolated = self.pca.inverse_transform(interpolated_pca.reshape(1, -1))[0]
        
        return interpolated
    
    def _gaussian_interpolation(
        self,
        voice1: np.ndarray,
        voice2: np.ndarray,
        weight: float,
        sigma: float = 1.0,
        **kwargs
    ) -> np.ndarray:
        """Gaussian-weighted interpolation."""
        # Compute distance between voices
        distance = np.linalg.norm(voice1 - voice2)
        
        # Gaussian weight
        gaussian_weight = np.exp(-(distance ** 2) / (2 * sigma ** 2))
        
        # Adjust interpolation weight
        adjusted_weight = weight * gaussian_weight
        
        return (1 - adjusted_weight) * voice1 + adjusted_weight * voice2
    
    def _spline_interpolation(
        self,
        voice1: np.ndarray,
        voice2: np.ndarray,
        weight: float,
        control_points: int = 5,
        **kwargs
    ) -> np.ndarray:
        """Spline-based interpolation."""
        # Create control points
        t = np.linspace(0, 1, control_points)
        
        # Create spline control points
        control_voices = []
        for i, t_val in enumerate(t):
            if i == 0:
                control_voices.append(voice1)
            elif i == control_points - 1:
                control_voices.append(voice2)
            else:
                # Intermediate control points
                intermediate = self._linear_interpolation(voice1, voice2, t_val)
                control_voices.append(intermediate)
        
        control_voices = np.array(control_voices)
        
        # Interpolate using spline
        from scipy.interpolate import interp1d
        
        interpolated = np.zeros_like(voice1)
        for i in range(len(voice1)):
            f = interp1d(t, control_voices[:, i], kind='cubic')
            interpolated[i] = f(weight)
        
        return interpolated
    
    def create_voice_continuum(
        self,
        voice1_embedding: np.ndarray,
        voice2_embedding: np.ndarray,
        num_steps: int = 10,
        method: str = "linear",
        **kwargs
    ) -> List[np.ndarray]:
        """
        Create a voice continuum between two voices.
        
        Args:
            voice1_embedding: First voice embedding
            voice2_embedding: Second voice embedding
            num_steps: Number of interpolation steps
            method: Interpolation method
            **kwargs: Additional parameters
        
        Returns:
            List of interpolated voice embeddings
        """
        continuum = []
        
        for i in range(num_steps):
            weight = i / (num_steps - 1)  # 0 to 1
            interpolated_voice = self.interpolate_voices(
                voice1_embedding, voice2_embedding, weight, method, **kwargs
            )
            continuum.append(interpolated_voice)
        
        return continuum
    
    def create_voice_morphing(
        self,
        source_voices: List[np.ndarray],
        target_voice: np.ndarray,
        morphing_steps: int = 5,
        method: str = "linear",
    ) -> List[np.ndarray]:
        """
        Create voice morphing from multiple source voices to target.
        
        Args:
            source_voices: List of source voice embeddings
            target_voice: Target voice embedding
            morphing_steps: Number of morphing steps
            method: Interpolation method
        
        Returns:
            List of morphed voice embeddings
        """
        morphed_voices = []
        
        for step in range(morphing_steps):
            weight = step / (morphing_steps - 1)
            
            # Interpolate between all source voices and target
            if len(source_voices) == 1:
                morphed_voice = self.interpolate_voices(
                    source_voices[0], target_voice, weight, method
                )
            else:
                # Multi-source interpolation
                morphed_voice = self._multi_source_interpolation(
                    source_voices, target_voice, weight, method
                )
            
            morphed_voices.append(morphed_voice)
        
        return morphed_voices
    
    def _multi_source_interpolation(
        self,
        source_voices: List[np.ndarray],
        target_voice: np.ndarray,
        weight: float,
        method: str = "linear",
    ) -> np.ndarray:
        """Interpolate between multiple source voices and target."""
        # Average source voices
        avg_source = np.mean(source_voices, axis=0)
        
        # Interpolate between average source and target
        return self.interpolate_voices(avg_source, target_voice, weight, method)
    
    def find_similar_voices(
        self,
        query_voice: np.ndarray,
        voice_database: Dict[str, np.ndarray],
        top_k: int = 5,
        similarity_metric: str = "cosine",
    ) -> List[Tuple[str, float]]:
        """
        Find similar voices in a database.
        
        Args:
            query_voice: Query voice embedding
            voice_database: Database of voice embeddings
            top_k: Number of similar voices to return
            similarity_metric: Similarity metric to use
        
        Returns:
            List of (voice_id, similarity_score) tuples
        """
        similarities = []
        
        for voice_id, voice_embedding in voice_database.items():
            if similarity_metric == "cosine":
                similarity = 1 - cosine(query_voice, voice_embedding)
            elif similarity_metric == "euclidean":
                similarity = 1 / (1 + np.linalg.norm(query_voice - voice_embedding))
            elif similarity_metric == "dot_product":
                similarity = np.dot(query_voice, voice_embedding) / (
                    np.linalg.norm(query_voice) * np.linalg.norm(voice_embedding)
                )
            else:
                similarity = 1 - cosine(query_voice, voice_embedding)
            
            similarities.append((voice_id, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def create_voice_clusters(
        self,
        voice_database: Dict[str, np.ndarray],
        n_clusters: int = 5,
        method: str = "kmeans",
    ) -> Dict[str, List[str]]:
        """
        Create voice clusters from a database.
        
        Args:
            voice_database: Database of voice embeddings
            n_clusters: Number of clusters
            method: Clustering method
        
        Returns:
            Dictionary mapping cluster_id to list of voice_ids
        """
        from sklearn.cluster import KMeans
        
        # Extract embeddings and IDs
        voice_ids = list(voice_database.keys())
        embeddings = np.array([voice_database[voice_id] for voice_id in voice_ids])
        
        # Perform clustering
        if method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(embeddings)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Group voices by cluster
        clusters = {}
        for i, voice_id in enumerate(voice_ids):
            cluster_id = f"cluster_{cluster_labels[i]}"
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(voice_id)
        
        return clusters
    
    def visualize_voice_space(
        self,
        voice_database: Dict[str, np.ndarray],
        method: str = "tsne",
        n_components: int = 2,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Visualize voice embeddings in 2D space.
        
        Args:
            voice_database: Database of voice embeddings
            method: Dimensionality reduction method
            n_components: Number of components for reduction
            save_path: Path to save visualization
        
        Returns:
            Reduced embeddings
        """
        # Extract embeddings and IDs
        voice_ids = list(voice_database.keys())
        embeddings = np.array([voice_database[voice_id] for voice_id in voice_ids])
        
        # Apply dimensionality reduction
        if method == "tsne":
            reducer = TSNE(n_components=n_components, random_state=42)
            reduced_embeddings = reducer.fit_transform(embeddings)
        elif method == "pca":
            reducer = PCA(n_components=n_components)
            reduced_embeddings = reducer.fit_transform(embeddings)
        else:
            raise ValueError(f"Unknown reduction method: {method}")
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        
        if n_components == 2:
            plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7)
            
            # Add labels
            for i, voice_id in enumerate(voice_ids):
                plt.annotate(voice_id, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))
        
        plt.title(f"Voice Space Visualization ({method.upper()})")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return reduced_embeddings
    
    def create_voice_family(
        self,
        parent_voice: np.ndarray,
        family_size: int = 5,
        variation_scale: float = 0.2,
        method: str = "gaussian",
    ) -> List[np.ndarray]:
        """
        Create a family of related voices from a parent voice.
        
        Args:
            parent_voice: Parent voice embedding
            family_size: Number of family members to create
            variation_scale: Scale of variation
            method: Method for generating variations
        
        Returns:
            List of family voice embeddings
        """
        family_voices = [parent_voice]  # Include parent
        
        for _ in range(family_size - 1):
            if method == "gaussian":
                # Add Gaussian noise
                variation = np.random.normal(0, variation_scale, parent_voice.shape)
                family_voice = parent_voice + variation
            elif method == "uniform":
                # Add uniform noise
                variation = np.random.uniform(-variation_scale, variation_scale, parent_voice.shape)
                family_voice = parent_voice + variation
            else:
                # Default to Gaussian
                variation = np.random.normal(0, variation_scale, parent_voice.shape)
                family_voice = parent_voice + variation
            
            # Normalize to prevent extreme values
            family_voice = np.clip(family_voice, -1.0, 1.0)
            family_voices.append(family_voice)
        
        return family_voices
    
    def get_interpolation_quality(
        self,
        voice1: np.ndarray,
        voice2: np.ndarray,
        interpolated_voice: np.ndarray,
        weight: float,
    ) -> Dict[str, float]:
        """
        Assess the quality of voice interpolation.
        
        Args:
            voice1: First voice embedding
            voice2: Second voice embedding
            interpolated_voice: Interpolated voice embedding
            weight: Interpolation weight used
        
        Returns:
            Dictionary of quality metrics
        """
        # Distance metrics
        dist_to_voice1 = np.linalg.norm(interpolated_voice - voice1)
        dist_to_voice2 = np.linalg.norm(interpolated_voice - voice2)
        
        # Expected distance based on weight
        expected_dist_to_voice1 = (1 - weight) * np.linalg.norm(voice2 - voice1)
        expected_dist_to_voice2 = weight * np.linalg.norm(voice2 - voice1)
        
        # Quality metrics
        quality_metrics = {
            'distance_consistency': 1.0 - abs(dist_to_voice1 - expected_dist_to_voice1) / expected_dist_to_voice1,
            'smoothness': 1.0 - abs(dist_to_voice2 - expected_dist_to_voice2) / expected_dist_to_voice2,
            'interpolation_accuracy': 1.0 - np.linalg.norm(interpolated_voice - ((1 - weight) * voice1 + weight * voice2)) / np.linalg.norm(voice2 - voice1),
        }
        
        # Overall quality score
        quality_metrics['overall_quality'] = np.mean(list(quality_metrics.values()))
        
        return quality_metrics
    
    def optimize_interpolation(
        self,
        voice1: np.ndarray,
        voice2: np.ndarray,
        target_voice: np.ndarray,
        method: str = "linear",
        max_iterations: int = 100,
    ) -> Tuple[np.ndarray, float]:
        """
        Optimize interpolation to match a target voice.
        
        Args:
            voice1: First voice embedding
            voice2: Second voice embedding
            target_voice: Target voice to match
            method: Interpolation method
            max_iterations: Maximum optimization iterations
        
        Returns:
            Tuple of (optimized_voice, best_weight)
        """
        best_voice = None
        best_weight = 0.0
        best_distance = float('inf')
        
        # Grid search for optimal weight
        for weight in np.linspace(0, 1, max_iterations):
            interpolated_voice = self.interpolate_voices(voice1, voice2, weight, method)
            distance = np.linalg.norm(interpolated_voice - target_voice)
            
            if distance < best_distance:
                best_distance = distance
                best_voice = interpolated_voice
                best_weight = weight
        
        return best_voice, best_weight








































