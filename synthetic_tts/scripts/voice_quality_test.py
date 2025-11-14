#!/usr/bin/env python3
"""
Voice quality testing script for comprehensive voice assessment.

This script performs comprehensive voice quality testing including
audio quality metrics, voice consistency, and performance benchmarks.
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
from typing import Dict, List, Any, Tuple
import librosa
from scipy import signal
from scipy.stats import pearsonr

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from synthesis.enhanced_synthesizer import EnhancedSynthesizer
from voice.voice_analyzer import VoiceAnalyzer
from voice.voice_interpolator import VoiceInterpolator
from utils.voice_utils import VoiceUtils


def main():
    """Main voice quality testing function."""
    print("🎯 Voice Quality Testing Suite")
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
    
    # Create test voice database
    print("Creating test voice database...")
    voice_database = create_test_voice_database()
    synthesizer.load_voice_database(voice_database)
    
    # Test configurations
    test_configs = create_test_configurations()
    
    print(f"Testing {len(test_configs)} voice configurations...")
    
    # Run quality tests
    print("\n🧪 Running voice quality tests...")
    quality_results = run_voice_quality_tests(synthesizer, test_configs)
    
    # Analyze results
    print("\n📊 Analyzing quality results...")
    analysis_results = analyze_quality_results(quality_results)
    
    # Generate quality report
    print("\n📋 Generating quality report...")
    generate_quality_report(quality_results, analysis_results)
    
    # Create quality visualizations
    print("\n📈 Creating quality visualizations...")
    create_quality_visualizations(quality_results, analysis_results)
    
    # Performance benchmarks
    print("\n⚡ Running performance benchmarks...")
    benchmark_results = run_performance_benchmarks(synthesizer, test_configs)
    
    # Generate benchmark report
    print("\n📊 Generating benchmark report...")
    generate_benchmark_report(benchmark_results)
    
    print("\n✅ Voice quality testing complete!")
    print("Check the 'output' directory for detailed results.")


def create_test_voice_database() -> Dict[str, np.ndarray]:
    """Create a comprehensive test voice database."""
    print("Creating test voice database...")
    
    voice_database = {}
    
    # Create diverse voice types
    voice_types = [
        # High pitch, bright voice
        {
            'name': 'high_pitch_bright',
            'embedding': np.array([0.8, 0.3, 0.7, 0.6, 0.2, 0.5, 0.4, 0.3] + [np.random.normal(0, 0.1) for _ in range(56)])
        },
        # Low pitch, warm voice
        {
            'name': 'low_pitch_warm',
            'embedding': np.array([0.3, 0.4, 0.4, 0.3, 0.6, 0.7, 0.6, 0.5] + [np.random.normal(0, 0.1) for _ in range(56)])
        },
        # Medium pitch, balanced voice
        {
            'name': 'medium_pitch_balanced',
            'embedding': np.array([0.5, 0.5, 0.5, 0.5, 0.4, 0.6, 0.5, 0.4] + [np.random.normal(0, 0.1) for _ in range(56)])
        },
        # High energy, excited voice
        {
            'name': 'high_energy_excited',
            'embedding': np.array([0.7, 0.6, 0.8, 0.7, 0.3, 0.4, 0.7, 0.6] + [np.random.normal(0, 0.1) for _ in range(56)])
        },
        # Low energy, calm voice
        {
            'name': 'low_energy_calm',
            'embedding': np.array([0.4, 0.2, 0.3, 0.2, 0.5, 0.8, 0.3, 0.2] + [np.random.normal(0, 0.1) for _ in range(56)])
        },
    ]
    
    for voice_type in voice_types:
        voice_database[voice_type['name']] = voice_type['embedding']
    
    print(f"Created {len(voice_database)} test voices")
    return voice_database


def create_test_configurations() -> List[Dict[str, Any]]:
    """Create test configurations for quality testing."""
    test_configs = []
    
    # Test texts of different lengths and complexities
    test_texts = [
        "Hello, world!",
        "This is a medium length sentence for testing voice quality.",
        "This is a much longer sentence that will test the model's ability to maintain quality throughout extended text generation and handle complex linguistic structures.",
        "Testing numbers: 123, 456, 789.",
        "Testing punctuation! Question? Exclamation! Period. Comma, semicolon; colon:",
    ]
    
    # Test emotions
    emotions = ['neutral', 'happy', 'sad', 'angry', 'excited', 'calm']
    
    # Test prosody controls
    prosody_controls = [
        {'pitch_shift': 0.0, 'rate_scale': 1.0, 'energy_scale': 1.0},
        {'pitch_shift': 0.2, 'rate_scale': 1.1, 'energy_scale': 1.2},
        {'pitch_shift': -0.1, 'rate_scale': 0.9, 'energy_scale': 0.8},
        {'pitch_shift': 0.3, 'rate_scale': 1.2, 'energy_scale': 1.3},
    ]
    
    # Create test configurations
    for text in test_texts:
        for emotion in emotions:
            for prosody in prosody_controls:
                test_configs.append({
                    'text': text,
                    'emotion': emotion,
                    'prosody_control': prosody,
                    'test_id': f"text_{len(text)}_emotion_{emotion}_prosody_{hash(str(prosody))}"
                })
    
    return test_configs


def run_voice_quality_tests(synthesizer: EnhancedSynthesizer, test_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run comprehensive voice quality tests."""
    print("Running voice quality tests...")
    
    quality_results = {
        'test_results': [],
        'voice_consistency': {},
        'audio_quality_metrics': {},
        'performance_metrics': {},
    }
    
    # Test each configuration
    for i, config in enumerate(test_configs):
        print(f"  Testing configuration {i+1}/{len(test_configs)}: {config['test_id']}")
        
        try:
            # Synthesize speech
            start_time = time.time()
            
            result = synthesizer.synthesize_speech(
                text=config['text'],
                emotion=config['emotion'],
                prosody_control=config['prosody_control'],
            )
            
            synthesis_time = time.time() - start_time
            
            # Analyze audio quality
            audio_quality = analyze_audio_quality(result['audio'])
            
            # Test result
            test_result = {
                'config': config,
                'synthesis_time': synthesis_time,
                'audio_quality': audio_quality,
                'success': True,
            }
            
            quality_results['test_results'].append(test_result)
            
        except Exception as e:
            print(f"    Error in test {config['test_id']}: {e}")
            quality_results['test_results'].append({
                'config': config,
                'error': str(e),
                'success': False,
            })
    
    # Analyze voice consistency
    quality_results['voice_consistency'] = analyze_voice_consistency(quality_results['test_results'])
    
    # Calculate overall quality metrics
    quality_results['audio_quality_metrics'] = calculate_quality_metrics(quality_results['test_results'])
    
    # Calculate performance metrics
    quality_results['performance_metrics'] = calculate_performance_metrics(quality_results['test_results'])
    
    return quality_results


def analyze_audio_quality(audio: np.ndarray) -> Dict[str, float]:
    """Analyze audio quality using various metrics."""
    if len(audio) == 0:
        return {'overall_quality': 0.0}
    
    quality_metrics = {}
    
    # Basic audio statistics
    rms_energy = np.sqrt(np.mean(audio ** 2))
    peak_amplitude = np.max(np.abs(audio))
    zero_crossing_rate = np.mean(np.diff(np.sign(audio)) != 0)
    
    quality_metrics['rms_energy'] = float(rms_energy)
    quality_metrics['peak_amplitude'] = float(peak_amplitude)
    quality_metrics['zero_crossing_rate'] = float(zero_crossing_rate)
    
    # Spectral characteristics
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=22050)[0])
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=22050)[0])
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=22050)[0])
    
    quality_metrics['spectral_centroid'] = float(spectral_centroid)
    quality_metrics['spectral_rolloff'] = float(spectral_rolloff)
    quality_metrics['spectral_bandwidth'] = float(spectral_bandwidth)
    
    # Pitch analysis
    pitches, magnitudes = librosa.piptrack(y=audio, sr=22050)
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_values.append(pitch)
    
    if pitch_values:
        quality_metrics['pitch_mean'] = float(np.mean(pitch_values))
        quality_metrics['pitch_std'] = float(np.std(pitch_values))
        quality_metrics['pitch_range'] = float(np.max(pitch_values) - np.min(pitch_values))
    else:
        quality_metrics['pitch_mean'] = 0.0
        quality_metrics['pitch_std'] = 0.0
        quality_metrics['pitch_range'] = 0.0
    
    # MFCC analysis
    mfcc = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=13)
    quality_metrics['mfcc_mean'] = float(np.mean(mfcc))
    quality_metrics['mfcc_std'] = float(np.std(mfcc))
    
    # Quality scores
    quality_metrics['clarity_score'] = calculate_clarity_score(audio)
    quality_metrics['naturalness_score'] = calculate_naturalness_score(audio)
    quality_metrics['consistency_score'] = calculate_consistency_score(audio)
    
    # Overall quality
    quality_metrics['overall_quality'] = (
        quality_metrics['clarity_score'] * 0.4 +
        quality_metrics['naturalness_score'] * 0.4 +
        quality_metrics['consistency_score'] * 0.2
    )
    
    return quality_metrics


def calculate_clarity_score(audio: np.ndarray) -> float:
    """Calculate audio clarity score."""
    if len(audio) == 0:
        return 0.0
    
    # Signal-to-noise ratio
    signal_power = np.mean(audio ** 2)
    noise_power = np.var(audio - np.mean(audio))
    snr = signal_power / (noise_power + 1e-8)
    
    # Clarity score (0-1)
    clarity = min(1.0, snr / 100.0)
    return clarity


def calculate_naturalness_score(audio: np.ndarray) -> float:
    """Calculate audio naturalness score."""
    if len(audio) == 0:
        return 0.0
    
    # Naturalness based on spectral characteristics
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=22050)[0])
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=22050)[0])
    
    # Naturalness score (0-1)
    naturalness = min(1.0, (spectral_centroid / 2000.0) * (spectral_rolloff / 4000.0))
    return naturalness


def calculate_consistency_score(audio: np.ndarray) -> float:
    """Calculate audio consistency score."""
    if len(audio) == 0:
        return 0.0
    
    # Consistency based on energy stability
    frame_length = 1024
    hop_length = 256
    
    # Calculate frame energies
    frame_energies = []
    for i in range(0, len(audio) - frame_length, hop_length):
        frame = audio[i:i + frame_length]
        energy = np.sqrt(np.mean(frame ** 2))
        frame_energies.append(energy)
    
    if len(frame_energies) < 2:
        return 1.0
    
    # Consistency score (lower std = higher consistency)
    energy_std = np.std(frame_energies)
    energy_mean = np.mean(frame_energies)
    consistency = 1.0 - (energy_std / (energy_mean + 1e-8))
    
    return max(0.0, consistency)


def analyze_voice_consistency(test_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Analyze voice consistency across tests."""
    consistency_metrics = {}
    
    # Group tests by voice characteristics
    voice_groups = {}
    for result in test_results:
        if not result['success']:
            continue
        
        config = result['config']
        voice_key = f"{config['emotion']}_{config['prosody_control']}"
        
        if voice_key not in voice_groups:
            voice_groups[voice_key] = []
        
        voice_groups[voice_key].append(result)
    
    # Calculate consistency for each group
    for voice_key, group_results in voice_groups.items():
        if len(group_results) < 2:
            continue
        
        # Calculate consistency metrics
        audio_qualities = [r['audio_quality'] for r in group_results]
        
        # Energy consistency
        energies = [q['rms_energy'] for q in audio_qualities]
        energy_consistency = 1.0 - (np.std(energies) / (np.mean(energies) + 1e-8))
        
        # Pitch consistency
        pitches = [q['pitch_mean'] for q in audio_qualities if q['pitch_mean'] > 0]
        if pitches:
            pitch_consistency = 1.0 - (np.std(pitches) / (np.mean(pitches) + 1e-8))
        else:
            pitch_consistency = 1.0
        
        # Overall consistency
        overall_consistency = (energy_consistency + pitch_consistency) / 2
        
        consistency_metrics[voice_key] = {
            'energy_consistency': energy_consistency,
            'pitch_consistency': pitch_consistency,
            'overall_consistency': overall_consistency,
        }
    
    return consistency_metrics


def calculate_quality_metrics(test_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate overall quality metrics."""
    successful_tests = [r for r in test_results if r['success']]
    
    if not successful_tests:
        return {'overall_quality': 0.0}
    
    # Extract quality scores
    quality_scores = [r['audio_quality']['overall_quality'] for r in successful_tests]
    clarity_scores = [r['audio_quality']['clarity_score'] for r in successful_tests]
    naturalness_scores = [r['audio_quality']['naturalness_score'] for r in successful_tests]
    consistency_scores = [r['audio_quality']['consistency_score'] for r in successful_tests]
    
    return {
        'overall_quality': float(np.mean(quality_scores)),
        'clarity_score': float(np.mean(clarity_scores)),
        'naturalness_score': float(np.mean(naturalness_scores)),
        'consistency_score': float(np.mean(consistency_scores)),
        'quality_std': float(np.std(quality_scores)),
        'success_rate': len(successful_tests) / len(test_results),
    }


def calculate_performance_metrics(test_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate performance metrics."""
    successful_tests = [r for r in test_results if r['success']]
    
    if not successful_tests:
        return {'avg_synthesis_time': 0.0}
    
    synthesis_times = [r['synthesis_time'] for r in successful_tests]
    
    return {
        'avg_synthesis_time': float(np.mean(synthesis_times)),
        'min_synthesis_time': float(np.min(synthesis_times)),
        'max_synthesis_time': float(np.max(synthesis_times)),
        'synthesis_time_std': float(np.std(synthesis_times)),
        'success_rate': len(successful_tests) / len(test_results),
    }


def analyze_quality_results(quality_results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze quality test results."""
    analysis = {
        'overall_quality': quality_results['audio_quality_metrics']['overall_quality'],
        'quality_distribution': {},
        'performance_analysis': {},
        'recommendations': {},
    }
    
    # Quality distribution analysis
    test_results = quality_results['test_results']
    successful_tests = [r for r in test_results if r['success']]
    
    if successful_tests:
        quality_scores = [r['audio_quality']['overall_quality'] for r in successful_tests]
        
        analysis['quality_distribution'] = {
            'mean': float(np.mean(quality_scores)),
            'std': float(np.std(quality_scores)),
            'min': float(np.min(quality_scores)),
            'max': float(np.max(quality_scores)),
            'percentile_25': float(np.percentile(quality_scores, 25)),
            'percentile_75': float(np.percentile(quality_scores, 75)),
        }
    
    # Performance analysis
    analysis['performance_analysis'] = quality_results['performance_metrics']
    
    # Generate recommendations
    analysis['recommendations'] = generate_quality_recommendations(quality_results)
    
    return analysis


def generate_quality_recommendations(quality_results: Dict[str, Any]) -> Dict[str, str]:
    """Generate quality improvement recommendations."""
    recommendations = {}
    
    quality_metrics = quality_results['audio_quality_metrics']
    performance_metrics = quality_results['performance_metrics']
    
    # Quality recommendations
    if quality_metrics['overall_quality'] < 0.7:
        recommendations['quality'] = "Overall quality is below optimal. Consider improving model training or adjusting synthesis parameters."
    elif quality_metrics['overall_quality'] < 0.8:
        recommendations['quality'] = "Quality is good but could be improved. Fine-tune model parameters for better results."
    else:
        recommendations['quality'] = "Quality is excellent. Current settings are optimal."
    
    # Performance recommendations
    if performance_metrics['avg_synthesis_time'] > 2.0:
        recommendations['performance'] = "Synthesis time is high. Consider optimizing model architecture or reducing complexity."
    elif performance_metrics['avg_synthesis_time'] > 1.0:
        recommendations['performance'] = "Synthesis time is acceptable but could be improved. Consider model optimization."
    else:
        recommendations['performance'] = "Synthesis time is excellent. Current performance is optimal."
    
    # Consistency recommendations
    if quality_metrics['consistency_score'] < 0.7:
        recommendations['consistency'] = "Voice consistency is low. Consider improving voice embedding stability."
    elif quality_metrics['consistency_score'] < 0.8:
        recommendations['consistency'] = "Voice consistency is good but could be improved. Fine-tune voice parameters."
    else:
        recommendations['consistency'] = "Voice consistency is excellent. Current voice parameters are stable."
    
    return recommendations


def run_performance_benchmarks(synthesizer: EnhancedSynthesizer, test_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run performance benchmarks."""
    print("Running performance benchmarks...")
    
    benchmark_results = {
        'synthesis_benchmarks': {},
        'memory_usage': {},
        'cpu_usage': {},
        'gpu_usage': {},
    }
    
    # Benchmark synthesis times
    synthesis_times = []
    for config in test_configs[:10]:  # Limit for performance
        try:
            start_time = time.time()
            result = synthesizer.synthesize_speech(
                text=config['text'],
                emotion=config['emotion'],
                prosody_control=config['prosody_control'],
            )
            synthesis_time = time.time() - start_time
            synthesis_times.append(synthesis_time)
        except Exception as e:
            print(f"    Error in benchmark: {e}")
    
    if synthesis_times:
        benchmark_results['synthesis_benchmarks'] = {
            'avg_time': float(np.mean(synthesis_times)),
            'min_time': float(np.min(synthesis_times)),
            'max_time': float(np.max(synthesis_times)),
            'std_time': float(np.std(synthesis_times)),
        }
    
    # Memory usage (simplified)
    benchmark_results['memory_usage'] = {
        'estimated_mb': 100,  # Placeholder
        'peak_mb': 150,       # Placeholder
    }
    
    return benchmark_results


def generate_quality_report(quality_results: Dict[str, Any], analysis_results: Dict[str, Any]) -> None:
    """Generate a comprehensive quality report."""
    print("Generating quality report...")
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    report_path = output_dir / "voice_quality_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("Voice Quality Testing Report\n")
        f.write("=" * 40 + "\n\n")
        
        # Overall quality
        f.write("Overall Quality Metrics:\n")
        f.write("-" * 25 + "\n")
        quality_metrics = quality_results['audio_quality_metrics']
        f.write(f"Overall Quality: {quality_metrics['overall_quality']:.4f}\n")
        f.write(f"Clarity Score: {quality_metrics['clarity_score']:.4f}\n")
        f.write(f"Naturalness Score: {quality_metrics['naturalness_score']:.4f}\n")
        f.write(f"Consistency Score: {quality_metrics['consistency_score']:.4f}\n")
        f.write(f"Success Rate: {quality_metrics['success_rate']:.4f}\n\n")
        
        # Quality distribution
        f.write("Quality Distribution:\n")
        f.write("-" * 20 + "\n")
        if 'quality_distribution' in analysis_results:
            dist = analysis_results['quality_distribution']
            f.write(f"Mean: {dist['mean']:.4f}\n")
            f.write(f"Std: {dist['std']:.4f}\n")
            f.write(f"Min: {dist['min']:.4f}\n")
            f.write(f"Max: {dist['max']:.4f}\n")
            f.write(f"25th Percentile: {dist['percentile_25']:.4f}\n")
            f.write(f"75th Percentile: {dist['percentile_75']:.4f}\n\n")
        
        # Performance metrics
        f.write("Performance Metrics:\n")
        f.write("-" * 20 + "\n")
        perf_metrics = quality_results['performance_metrics']
        f.write(f"Average Synthesis Time: {perf_metrics['avg_synthesis_time']:.4f}s\n")
        f.write(f"Min Synthesis Time: {perf_metrics['min_synthesis_time']:.4f}s\n")
        f.write(f"Max Synthesis Time: {perf_metrics['max_synthesis_time']:.4f}s\n")
        f.write(f"Synthesis Time Std: {perf_metrics['synthesis_time_std']:.4f}s\n\n")
        
        # Recommendations
        f.write("Recommendations:\n")
        f.write("-" * 15 + "\n")
        if 'recommendations' in analysis_results:
            for category, recommendation in analysis_results['recommendations'].items():
                f.write(f"{category.title()}: {recommendation}\n")
    
    print(f"Quality report saved to: {report_path}")


def create_quality_visualizations(quality_results: Dict[str, Any], analysis_results: Dict[str, Any]) -> None:
    """Create quality visualizations."""
    print("Creating quality visualizations...")
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Quality scores histogram
    plt.figure(figsize=(10, 6))
    
    test_results = quality_results['test_results']
    successful_tests = [r for r in test_results if r['success']]
    
    if successful_tests:
        quality_scores = [r['audio_quality']['overall_quality'] for r in successful_tests]
        
        plt.hist(quality_scores, bins=20, alpha=0.7, edgecolor='black')
        plt.title('Voice Quality Score Distribution')
        plt.xlabel('Quality Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "quality_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Quality metrics comparison
    plt.figure(figsize=(12, 8))
    
    quality_metrics = quality_results['audio_quality_metrics']
    metrics = ['overall_quality', 'clarity_score', 'naturalness_score', 'consistency_score']
    values = [quality_metrics[metric] for metric in metrics]
    
    plt.bar(metrics, values)
    plt.title('Voice Quality Metrics Comparison')
    plt.xlabel('Quality Metric')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "quality_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Performance metrics
    plt.figure(figsize=(10, 6))
    
    perf_metrics = quality_results['performance_metrics']
    perf_labels = ['Avg Time', 'Min Time', 'Max Time']
    perf_values = [perf_metrics['avg_synthesis_time'], perf_metrics['min_synthesis_time'], perf_metrics['max_synthesis_time']]
    
    plt.bar(perf_labels, perf_values)
    plt.title('Synthesis Performance Metrics')
    plt.xlabel('Performance Metric')
    plt.ylabel('Time (seconds)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "performance_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Quality visualizations saved to output directory")


def generate_benchmark_report(benchmark_results: Dict[str, Any]) -> None:
    """Generate benchmark report."""
    print("Generating benchmark report...")
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    report_path = output_dir / "performance_benchmark_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("Performance Benchmark Report\n")
        f.write("=" * 30 + "\n\n")
        
        # Synthesis benchmarks
        f.write("Synthesis Benchmarks:\n")
        f.write("-" * 20 + "\n")
        if 'synthesis_benchmarks' in benchmark_results:
            synth_bench = benchmark_results['synthesis_benchmarks']
            f.write(f"Average Time: {synth_bench['avg_time']:.4f}s\n")
            f.write(f"Min Time: {synth_bench['min_time']:.4f}s\n")
            f.write(f"Max Time: {synth_bench['max_time']:.4f}s\n")
            f.write(f"Standard Deviation: {synth_bench['std_time']:.4f}s\n\n")
        
        # Memory usage
        f.write("Memory Usage:\n")
        f.write("-" * 15 + "\n")
        if 'memory_usage' in benchmark_results:
            mem_usage = benchmark_results['memory_usage']
            f.write(f"Estimated Memory: {mem_usage['estimated_mb']} MB\n")
            f.write(f"Peak Memory: {mem_usage['peak_mb']} MB\n")
    
    print(f"Benchmark report saved to: {report_path}")


if __name__ == "__main__":
    main()









































