#!/usr/bin/env python3
"""
Model comparison script for evaluating different TTS architectures.

This script compares different acoustic models and vocoders to help
select the best architecture for specific use cases.
"""

import sys
import os
import time
import numpy as np
import torch
import torchaudio
from pathlib import Path
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.model_factory import ModelFactory
from synthesis.enhanced_synthesizer import EnhancedSynthesizer
from voice.voice_analyzer import VoiceAnalyzer
from utils.voice_utils import VoiceUtils


def main():
    """Main model comparison function."""
    print("🔬 TTS Model Architecture Comparison")
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
    model_factory = ModelFactory()
    voice_analyzer = VoiceAnalyzer()
    voice_utils = VoiceUtils()
    
    # Test configurations
    acoustic_architectures = ['fastpitch', 'tacotron2']
    vocoder_architectures = ['hifigan', 'waveglow']
    
    print(f"Testing acoustic models: {acoustic_architectures}")
    print(f"Testing vocoders: {vocoder_architectures}")
    
    # Create test data
    print("\n📊 Creating test data...")
    test_data = create_test_data()
    
    # Run model comparisons
    print("\n🧪 Running model comparisons...")
    comparison_results = run_model_comparisons(
        model_factory, acoustic_architectures, vocoder_architectures, test_data
    )
    
    # Analyze results
    print("\n📈 Analyzing results...")
    analysis_results = analyze_comparison_results(comparison_results)
    
    # Generate reports
    print("\n📋 Generating reports...")
    generate_comparison_report(comparison_results, analysis_results)
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    create_comparison_visualizations(comparison_results, analysis_results)
    
    # Performance recommendations
    print("\n💡 Performance Recommendations")
    print("-" * 30)
    provide_recommendations(analysis_results)
    
    print("\n✅ Model comparison complete!")
    print("Check the 'output' directory for detailed results.")


def create_test_data() -> Dict[str, Any]:
    """Create test data for model comparison."""
    print("Creating test data...")
    
    # Test texts of different lengths
    test_texts = [
        "Hello, world!",
        "This is a medium length sentence for testing TTS models.",
        "This is a much longer sentence that will test the model's ability to handle longer sequences and maintain quality throughout the entire text generation process.",
    ]
    
    # Test voice embeddings
    voice_embeddings = []
    for i in range(5):
        # Create diverse voice embeddings
        embedding = np.random.normal(0, 0.5, 64)
        embedding[0] = np.random.uniform(0.2, 0.8)  # Pitch
        embedding[1] = np.random.uniform(0.1, 0.5)  # Pitch variation
        embedding[2] = np.random.uniform(0.3, 0.7)  # Spectral centroid
        embedding[3] = np.random.uniform(0.2, 0.6)  # Spectral rolloff
        voice_embeddings.append(embedding)
    
    # Test emotions
    emotions = ['neutral', 'happy', 'sad', 'angry', 'excited']
    
    # Test prosody controls
    prosody_controls = [
        {'pitch_shift': 0.0, 'rate_scale': 1.0, 'energy_scale': 1.0},
        {'pitch_shift': 0.2, 'rate_scale': 1.1, 'energy_scale': 1.2},
        {'pitch_shift': -0.1, 'rate_scale': 0.9, 'energy_scale': 0.8},
    ]
    
    return {
        'texts': test_texts,
        'voice_embeddings': voice_embeddings,
        'emotions': emotions,
        'prosody_controls': prosody_controls,
    }


def run_model_comparisons(
    model_factory: ModelFactory,
    acoustic_architectures: List[str],
    vocoder_architectures: List[str],
    test_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Run comprehensive model comparisons."""
    print("Running model comparisons...")
    
    results = {}
    
    for acoustic_arch in acoustic_architectures:
        for vocoder_arch in vocoder_architectures:
            print(f"\nTesting {acoustic_arch} + {vocoder_arch}...")
            
            try:
                # Create synthesizer with specific architecture
                synthesizer = EnhancedSynthesizer({
                    'sample_rate': 22050,
                    'n_mels': 80,
                    'embedding_dim': 64,
                })
                
                # Initialize models
                synthesizer.initialize_models(
                    acoustic_architecture=acoustic_arch,
                    vocoder_architecture=vocoder_arch,
                )
                
                # Load test voice database
                voice_database = {f"voice_{i}": emb for i, emb in enumerate(test_data['voice_embeddings'])}
                synthesizer.load_voice_database(voice_database)
                
                # Run tests
                test_results = run_architecture_tests(synthesizer, test_data, acoustic_arch, vocoder_arch)
                
                results[f"{acoustic_arch}_{vocoder_arch}"] = test_results
                
            except Exception as e:
                print(f"Error testing {acoustic_arch} + {vocoder_arch}: {e}")
                results[f"{acoustic_arch}_{vocoder_arch}"] = {
                    'error': str(e),
                    'status': 'failed'
                }
    
    return results


def run_architecture_tests(
    synthesizer: EnhancedSynthesizer,
    test_data: Dict[str, Any],
    acoustic_arch: str,
    vocoder_arch: str,
) -> Dict[str, Any]:
    """Run tests for a specific architecture."""
    test_results = {
        'acoustic_model': acoustic_arch,
        'vocoder': vocoder_arch,
        'tests': {},
        'performance_metrics': {},
    }
    
    # Test 1: Basic synthesis
    print("  Testing basic synthesis...")
    basic_test = test_basic_synthesis(synthesizer, test_data)
    test_results['tests']['basic_synthesis'] = basic_test
    
    # Test 2: Voice interpolation
    print("  Testing voice interpolation...")
    interpolation_test = test_voice_interpolation(synthesizer, test_data)
    test_results['tests']['voice_interpolation'] = interpolation_test
    
    # Test 3: Emotion control
    print("  Testing emotion control...")
    emotion_test = test_emotion_control(synthesizer, test_data)
    test_results['tests']['emotion_control'] = emotion_test
    
    # Test 4: Prosody control
    print("  Testing prosody control...")
    prosody_test = test_prosody_control(synthesizer, test_data)
    test_results['tests']['prosody_control'] = prosody_test
    
    # Test 5: Performance metrics
    print("  Testing performance...")
    performance_test = test_performance(synthesizer, test_data)
    test_results['tests']['performance'] = performance_test
    
    # Test 6: Quality assessment
    print("  Testing quality...")
    quality_test = test_quality(synthesizer, test_data)
    test_results['tests']['quality'] = quality_test
    
    return test_results


def test_basic_synthesis(synthesizer: EnhancedSynthesizer, test_data: Dict[str, Any]) -> Dict[str, Any]:
    """Test basic speech synthesis."""
    results = {
        'success_rate': 0,
        'synthesis_times': [],
        'audio_qualities': [],
    }
    
    successful_syntheses = 0
    total_syntheses = 0
    
    for text in test_data['texts']:
        for i, voice_embedding in enumerate(test_data['voice_embeddings']):
            try:
                start_time = time.time()
                
                result = synthesizer.synthesize_speech(
                    text=text,
                    voice_embedding=voice_embedding,
                )
                
                synthesis_time = time.time() - start_time
                results['synthesis_times'].append(synthesis_time)
                
                # Assess audio quality
                audio_quality = assess_audio_quality(result['audio'])
                results['audio_qualities'].append(audio_quality)
                
                successful_syntheses += 1
                total_syntheses += 1
                
            except Exception as e:
                print(f"    Error in basic synthesis: {e}")
                total_syntheses += 1
    
    results['success_rate'] = successful_syntheses / total_syntheses if total_syntheses > 0 else 0
    results['avg_synthesis_time'] = np.mean(results['synthesis_times']) if results['synthesis_times'] else 0
    results['avg_audio_quality'] = np.mean(results['audio_qualities']) if results['audio_qualities'] else 0
    
    return results


def test_voice_interpolation(synthesizer: EnhancedSynthesizer, test_data: Dict[str, Any]) -> Dict[str, Any]:
    """Test voice interpolation capabilities."""
    results = {
        'interpolation_success_rate': 0,
        'interpolation_times': [],
        'interpolation_qualities': [],
    }
    
    successful_interpolations = 0
    total_interpolations = 0
    
    # Test voice continuum creation
    voice_ids = list(synthesizer.voice_database.keys())
    
    for i in range(len(voice_ids) - 1):
        try:
            start_time = time.time()
            
            continuum_results = synthesizer.create_voice_continuum(
                voice_ids[i], voice_ids[i+1], test_data['texts'][0], num_steps=3
            )
            
            interpolation_time = time.time() - start_time
            results['interpolation_times'].append(interpolation_time)
            
            # Assess interpolation quality
            interpolation_quality = assess_interpolation_quality(continuum_results)
            results['interpolation_qualities'].append(interpolation_quality)
            
            successful_interpolations += 1
            total_interpolations += 1
            
        except Exception as e:
            print(f"    Error in voice interpolation: {e}")
            total_interpolations += 1
    
    results['interpolation_success_rate'] = successful_interpolations / total_interpolations if total_interpolations > 0 else 0
    results['avg_interpolation_time'] = np.mean(results['interpolation_times']) if results['interpolation_times'] else 0
    results['avg_interpolation_quality'] = np.mean(results['interpolation_qualities']) if results['interpolation_qualities'] else 0
    
    return results


def test_emotion_control(synthesizer: EnhancedSynthesizer, test_data: Dict[str, Any]) -> Dict[str, Any]:
    """Test emotion control capabilities."""
    results = {
        'emotion_success_rate': 0,
        'emotion_times': [],
        'emotion_qualities': [],
    }
    
    successful_emotions = 0
    total_emotions = 0
    
    for emotion in test_data['emotions']:
        try:
            start_time = time.time()
            
            result = synthesizer.synthesize_speech(
                text=test_data['texts'][0],
                voice_id=list(synthesizer.voice_database.keys())[0],
                emotion=emotion,
            )
            
            emotion_time = time.time() - start_time
            results['emotion_times'].append(emotion_time)
            
            # Assess emotion quality
            emotion_quality = assess_emotion_quality(result, emotion)
            results['emotion_qualities'].append(emotion_quality)
            
            successful_emotions += 1
            total_emotions += 1
            
        except Exception as e:
            print(f"    Error in emotion control: {e}")
            total_emotions += 1
    
    results['emotion_success_rate'] = successful_emotions / total_emotions if total_emotions > 0 else 0
    results['avg_emotion_time'] = np.mean(results['emotion_times']) if results['emotion_times'] else 0
    results['avg_emotion_quality'] = np.mean(results['emotion_qualities']) if results['emotion_qualities'] else 0
    
    return results


def test_prosody_control(synthesizer: EnhancedSynthesizer, test_data: Dict[str, Any]) -> Dict[str, Any]:
    """Test prosody control capabilities."""
    results = {
        'prosody_success_rate': 0,
        'prosody_times': [],
        'prosody_qualities': [],
    }
    
    successful_prosody = 0
    total_prosody = 0
    
    for prosody_control in test_data['prosody_controls']:
        try:
            start_time = time.time()
            
            result = synthesizer.synthesize_speech(
                text=test_data['texts'][0],
                voice_id=list(synthesizer.voice_database.keys())[0],
                prosody_control=prosody_control,
            )
            
            prosody_time = time.time() - start_time
            results['prosody_times'].append(prosody_time)
            
            # Assess prosody quality
            prosody_quality = assess_prosody_quality(result, prosody_control)
            results['prosody_qualities'].append(prosody_quality)
            
            successful_prosody += 1
            total_prosody += 1
            
        except Exception as e:
            print(f"    Error in prosody control: {e}")
            total_prosody += 1
    
    results['prosody_success_rate'] = successful_prosody / total_prosody if total_prosody > 0 else 0
    results['avg_prosody_time'] = np.mean(results['prosody_times']) if results['prosody_times'] else 0
    results['avg_prosody_quality'] = np.mean(results['prosody_qualities']) if results['prosody_qualities'] else 0
    
    return results


def test_performance(synthesizer: EnhancedSynthesizer, test_data: Dict[str, Any]) -> Dict[str, Any]:
    """Test performance metrics."""
    results = {
        'memory_usage': 0,
        'cpu_usage': 0,
        'gpu_usage': 0,
    }
    
    # Get performance metrics from synthesizer
    metrics = synthesizer.get_performance_metrics()
    
    results.update(metrics)
    
    return results


def test_quality(synthesizer: EnhancedSynthesizer, test_data: Dict[str, Any]) -> Dict[str, Any]:
    """Test audio quality."""
    results = {
        'overall_quality': 0,
        'clarity_score': 0,
        'naturalness_score': 0,
        'consistency_score': 0,
    }
    
    # Synthesize test samples
    test_samples = []
    for text in test_data['texts'][:2]:  # Limit for performance
        for voice_id in list(synthesizer.voice_database.keys())[:2]:
            try:
                result = synthesizer.synthesize_speech(text, voice_id=voice_id)
                test_samples.append(result)
            except Exception as e:
                print(f"    Error in quality test: {e}")
    
    if test_samples:
        # Assess overall quality
        quality_scores = [assess_audio_quality(sample['audio']) for sample in test_samples]
        results['overall_quality'] = np.mean(quality_scores)
        
        # Assess specific quality aspects
        results['clarity_score'] = assess_clarity(test_samples)
        results['naturalness_score'] = assess_naturalness(test_samples)
        results['consistency_score'] = assess_consistency(test_samples)
    
    return results


def assess_audio_quality(audio: np.ndarray) -> float:
    """Assess audio quality based on various metrics."""
    if len(audio) == 0:
        return 0.0
    
    # Basic quality metrics
    rms_energy = np.sqrt(np.mean(audio ** 2))
    peak_amplitude = np.max(np.abs(audio))
    zero_crossing_rate = np.mean(np.diff(np.sign(audio)) != 0)
    
    # Quality score (0-1)
    quality_score = min(1.0, rms_energy * 10)  # Normalize RMS energy
    quality_score *= (1.0 - zero_crossing_rate)  # Penalize high zero crossing rate
    
    return quality_score


def assess_interpolation_quality(continuum_results: List[Dict]) -> float:
    """Assess the quality of voice interpolation."""
    if len(continuum_results) < 2:
        return 0.0
    
    # Check smoothness of interpolation
    qualities = []
    for i in range(len(continuum_results) - 1):
        current_audio = continuum_results[i]['audio']
        next_audio = continuum_results[i + 1]['audio']
        
        # Compare audio characteristics
        current_energy = np.sqrt(np.mean(current_audio ** 2))
        next_energy = np.sqrt(np.mean(next_audio ** 2))
        
        # Smoothness score
        smoothness = 1.0 - abs(current_energy - next_energy) / (current_energy + next_energy + 1e-8)
        qualities.append(smoothness)
    
    return np.mean(qualities) if qualities else 0.0


def assess_emotion_quality(result: Dict, emotion: str) -> float:
    """Assess the quality of emotion expression."""
    # This is a simplified assessment
    # In practice, you'd use more sophisticated emotion recognition
    
    audio = result['audio']
    if len(audio) == 0:
        return 0.0
    
    # Basic emotion-specific quality metrics
    rms_energy = np.sqrt(np.mean(audio ** 2))
    
    if emotion == 'happy' or emotion == 'excited':
        # High energy expected
        quality = min(1.0, rms_energy * 15)
    elif emotion == 'sad' or emotion == 'calm':
        # Lower energy expected
        quality = min(1.0, (1.0 - rms_energy) * 15)
    else:
        # Neutral quality
        quality = min(1.0, rms_energy * 10)
    
    return quality


def assess_prosody_quality(result: Dict, prosody_control: Dict) -> float:
    """Assess the quality of prosody control."""
    audio = result['audio']
    if len(audio) == 0:
        return 0.0
    
    # Assess prosody control effectiveness
    rms_energy = np.sqrt(np.mean(audio ** 2))
    
    # Check if energy control is working
    if 'energy_scale' in prosody_control:
        expected_energy = prosody_control['energy_scale']
        energy_quality = 1.0 - abs(rms_energy - expected_energy) / (expected_energy + 1e-8)
    else:
        energy_quality = 1.0
    
    return energy_quality


def assess_clarity(test_samples: List[Dict]) -> float:
    """Assess audio clarity."""
    if not test_samples:
        return 0.0
    
    clarity_scores = []
    for sample in test_samples:
        audio = sample['audio']
        if len(audio) == 0:
            continue
        
        # Clarity based on signal-to-noise ratio
        signal_power = np.mean(audio ** 2)
        noise_power = np.var(audio - np.mean(audio))
        snr = signal_power / (noise_power + 1e-8)
        clarity = min(1.0, snr / 100.0)  # Normalize SNR
        clarity_scores.append(clarity)
    
    return np.mean(clarity_scores) if clarity_scores else 0.0


def assess_naturalness(test_samples: List[Dict]) -> float:
    """Assess audio naturalness."""
    if not test_samples:
        return 0.0
    
    naturalness_scores = []
    for sample in test_samples:
        audio = sample['audio']
        if len(audio) == 0:
            continue
        
        # Naturalness based on spectral characteristics
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=22050)[0])
        naturalness = min(1.0, spectral_centroid / 2000.0)  # Normalize spectral centroid
        naturalness_scores.append(naturalness)
    
    return np.mean(naturalness_scores) if naturalness_scores else 0.0


def assess_consistency(test_samples: List[Dict]) -> float:
    """Assess consistency across samples."""
    if len(test_samples) < 2:
        return 1.0
    
    # Measure consistency of audio characteristics
    energies = [np.sqrt(np.mean(sample['audio'] ** 2)) for sample in test_samples]
    energy_std = np.std(energies)
    energy_mean = np.mean(energies)
    
    # Consistency score (lower std = higher consistency)
    consistency = 1.0 - (energy_std / (energy_mean + 1e-8))
    return max(0.0, consistency)


def analyze_comparison_results(comparison_results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze comparison results and generate insights."""
    print("Analyzing comparison results...")
    
    analysis = {
        'overall_scores': {},
        'best_architectures': {},
        'performance_rankings': {},
        'quality_rankings': {},
        'recommendations': {},
    }
    
    # Calculate overall scores for each architecture
    for arch_name, results in comparison_results.items():
        if 'error' in results:
            continue
        
        overall_score = 0
        weights = {
            'basic_synthesis': 0.3,
            'voice_interpolation': 0.2,
            'emotion_control': 0.2,
            'prosody_control': 0.1,
            'performance': 0.1,
            'quality': 0.1,
        }
        
        for test_name, weight in weights.items():
            if test_name in results['tests']:
                test_results = results['tests'][test_name]
                if 'success_rate' in test_results:
                    overall_score += test_results['success_rate'] * weight
                elif 'overall_quality' in test_results:
                    overall_score += test_results['overall_quality'] * weight
        
        analysis['overall_scores'][arch_name] = overall_score
    
    # Find best architectures
    if analysis['overall_scores']:
        best_arch = max(analysis['overall_scores'], key=analysis['overall_scores'].get)
        analysis['best_architectures']['overall'] = best_arch
    
    # Performance rankings
    performance_scores = {}
    for arch_name, results in comparison_results.items():
        if 'error' in results:
            continue
        
        if 'tests' in results and 'performance' in results['tests']:
            perf_results = results['tests']['performance']
            if 'avg_synthesis_time' in perf_results:
                performance_scores[arch_name] = 1.0 / (perf_results['avg_synthesis_time'] + 1e-8)
    
    if performance_scores:
        analysis['performance_rankings'] = dict(sorted(performance_scores.items(), key=lambda x: x[1], reverse=True))
    
    # Quality rankings
    quality_scores = {}
    for arch_name, results in comparison_results.items():
        if 'error' in results:
            continue
        
        if 'tests' in results and 'quality' in results['tests']:
            quality_results = results['tests']['quality']
            if 'overall_quality' in quality_results:
                quality_scores[arch_name] = quality_results['overall_quality']
    
    if quality_scores:
        analysis['quality_rankings'] = dict(sorted(quality_scores.items(), key=lambda x: x[1], reverse=True))
    
    return analysis


def generate_comparison_report(comparison_results: Dict[str, Any], analysis_results: Dict[str, Any]) -> None:
    """Generate a detailed comparison report."""
    print("Generating comparison report...")
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    report_path = output_dir / "model_comparison_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("TTS Model Architecture Comparison Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall scores
        f.write("Overall Scores:\n")
        f.write("-" * 20 + "\n")
        for arch_name, score in analysis_results['overall_scores'].items():
            f.write(f"{arch_name}: {score:.4f}\n")
        f.write("\n")
        
        # Best architectures
        f.write("Best Architectures:\n")
        f.write("-" * 20 + "\n")
        for category, arch in analysis_results['best_architectures'].items():
            f.write(f"{category}: {arch}\n")
        f.write("\n")
        
        # Detailed results
        f.write("Detailed Results:\n")
        f.write("-" * 20 + "\n")
        for arch_name, results in comparison_results.items():
            f.write(f"\n{arch_name}:\n")
            if 'error' in results:
                f.write(f"  Error: {results['error']}\n")
            else:
                for test_name, test_results in results['tests'].items():
                    f.write(f"  {test_name}:\n")
                    for metric, value in test_results.items():
                        if isinstance(value, (int, float)):
                            f.write(f"    {metric}: {value:.4f}\n")
                        else:
                            f.write(f"    {metric}: {value}\n")
    
    print(f"Comparison report saved to: {report_path}")


def create_comparison_visualizations(comparison_results: Dict[str, Any], analysis_results: Dict[str, Any]) -> None:
    """Create visualizations for comparison results."""
    print("Creating comparison visualizations...")
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Overall scores bar chart
    plt.figure(figsize=(12, 8))
    
    architectures = list(analysis_results['overall_scores'].keys())
    scores = list(analysis_results['overall_scores'].values())
    
    plt.bar(architectures, scores)
    plt.title('Overall Architecture Scores')
    plt.xlabel('Architecture')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "overall_scores.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Performance comparison
    if analysis_results['performance_rankings']:
        plt.figure(figsize=(10, 6))
        
        perf_archs = list(analysis_results['performance_rankings'].keys())
        perf_scores = list(analysis_results['performance_rankings'].values())
        
        plt.bar(perf_archs, perf_scores)
        plt.title('Performance Rankings')
        plt.xlabel('Architecture')
        plt.ylabel('Performance Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / "performance_rankings.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Quality comparison
    if analysis_results['quality_rankings']:
        plt.figure(figsize=(10, 6))
        
        quality_archs = list(analysis_results['quality_rankings'].keys())
        quality_scores = list(analysis_results['quality_rankings'].values())
        
        plt.bar(quality_archs, quality_scores)
        plt.title('Quality Rankings')
        plt.xlabel('Architecture')
        plt.ylabel('Quality Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / "quality_rankings.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Visualizations saved to output directory")


def provide_recommendations(analysis_results: Dict[str, Any]) -> None:
    """Provide recommendations based on analysis results."""
    print("Performance Recommendations:")
    print("-" * 30)
    
    if analysis_results['overall_scores']:
        best_arch = max(analysis_results['overall_scores'], key=analysis_results['overall_scores'].get)
        print(f"🏆 Best overall architecture: {best_arch}")
    
    if analysis_results['performance_rankings']:
        fastest_arch = list(analysis_results['performance_rankings'].keys())[0]
        print(f"⚡ Fastest architecture: {fastest_arch}")
    
    if analysis_results['quality_rankings']:
        highest_quality_arch = list(analysis_results['quality_rankings'].keys())[0]
        print(f"🎯 Highest quality architecture: {highest_quality_arch}")
    
    print("\nRecommendations:")
    print("- For real-time applications: Choose the fastest architecture")
    print("- For high-quality synthesis: Choose the highest quality architecture")
    print("- For balanced performance: Choose the best overall architecture")
    print("- Consider your specific use case requirements when making the final decision")


if __name__ == "__main__":
    main()








































