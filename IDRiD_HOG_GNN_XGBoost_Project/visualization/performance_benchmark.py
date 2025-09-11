"""
Performance Benchmark Script for IDRiD DR-Vision Project
Measures computational power, timing, memory usage, and throughput
"""

import time
import psutil
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import json
from datetime import datetime
import cv2

# Add project paths - FIXED PATH HANDLING
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent  # Go up one level from visualization to project root
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

print(f"📁 Script directory: {script_dir}")
print(f"📁 Project root: {project_root}")
print(f"📁 Config file path: {project_root / 'configs/main_config.yaml'}")

class PerformanceBenchmark:
    def __init__(self):
        self.project_root = project_root  # Store for use in methods
        self.results = {
            'system_info': self.get_system_info(),
            'training_times': {},
            'inference_times': {},
            'memory_usage': {},
            'model_sizes': {},
            'throughput': {}
        }
        self.start_memory = psutil.virtual_memory().used / (1024 ** 3)  # GB

    def get_system_info(self):
        """Get system hardware information"""
        return {
            'cpu_count': psutil.cpu_count(logical=False),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'cpu_freq': psutil.cpu_freq().max if psutil.cpu_freq() else 0,
            'total_memory_gb': psutil.virtual_memory().total / (1024 ** 3),
            'available_memory_gb': psutil.virtual_memory().available / (1024 ** 3),
            'python_version': sys.version,
            'platform': sys.platform
        }

    def measure_time_memory(self, func, *args, **kwargs):
        """Measure execution time and peak memory usage"""
        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 ** 3)  # GB

        # Measure execution time
        start_time = time.time()
        start_cpu = time.process_time()

        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)

        end_time = time.time()
        end_cpu = time.process_time()

        # Get final memory
        final_memory = process.memory_info().rss / (1024 ** 3)  # GB
        peak_memory = max(initial_memory, final_memory)

        return {
            'result': result,
            'success': success,
            'error': error,
            'wall_time': end_time - start_time,
            'cpu_time': end_cpu - start_cpu,
            'memory_used_gb': peak_memory - initial_memory,
            'peak_memory_gb': peak_memory
        }

    def benchmark_data_loading(self, sample_image_path, num_samples=10):
        """Benchmark data loading and preprocessing"""
        print("📊 Benchmarking Data Loading...")

        def load_and_preprocess():
            image = cv2.imread(sample_image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image_rgb, (1024, 1024))
            return image_resized

        times = []
        for i in range(num_samples):
            start = time.time()
            _ = load_and_preprocess()
            times.append(time.time() - start)

        self.results['inference_times']['data_loading'] = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times)
        }
        print(f"   ✅ Data loading: {np.mean(times):.4f}s ± {np.std(times):.4f}s")

    def benchmark_hog_extraction(self, sample_image_path):
        """Benchmark HOG feature extraction"""
        print("🔍 Benchmarking HOG Feature Extraction...")

        try:
            # FIXED: Use correct config path
            config_path = self.project_root / 'configs/main_config.yaml'
            print(f"   📁 Looking for config at: {config_path}")

            if not config_path.exists():
                print(f"   ❌ Config file not found: {config_path}")
                print("   💡 Make sure configs/main_config.yaml exists in project root")
                return

            from data_processing.hog_extractor import IDRiDHOGExtractor
            from utils.config import load_config

            config = load_config(str(config_path))  # Use absolute path
            hog_extractor = IDRiDHOGExtractor(config)

            # Load sample image
            image = cv2.imread(sample_image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Benchmark HOG extraction
            def extract_hog():
                features, coords = hog_extractor.extract_patch_hog_features(image_rgb)
                return features, coords

            result = self.measure_time_memory(extract_hog)

            if result['success']:
                features, coords = result['result']
                self.results['inference_times']['hog_extraction'] = {
                    'time': result['wall_time'],
                    'memory_gb': result['memory_used_gb'],
                    'num_patches': len(features),
                    'feature_dim': features.shape[1] if len(features) > 0 else 0
                }
                print(f"   ✅ HOG extraction: {result['wall_time']:.4f}s, {len(features)} patches")
            else:
                print(f"   ❌ HOG extraction failed: {result['error']}")

        except Exception as e:
            print(f"   ❌ HOG benchmark failed: {e}")
            import traceback
            traceback.print_exc()

    def benchmark_graph_construction(self, sample_image_path):
        """Benchmark graph construction"""
        print("🕸️ Benchmarking Graph Construction...")

        try:
            # FIXED: Use correct config path
            config_path = self.project_root / 'configs/main_config.yaml'

            if not config_path.exists():
                print(f"   ❌ Config file not found: {config_path}")
                return

            from data_processing.hog_extractor import IDRiDHOGExtractor
            from data_processing.graph_builder import IDRiDGraphBuilder
            from utils.config import load_config

            config = load_config(str(config_path))
            hog_extractor = IDRiDHOGExtractor(config)
            graph_builder = IDRiDGraphBuilder(config)

            # Get HOG features first
            image = cv2.imread(sample_image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            features, coords = hog_extractor.extract_patch_hog_features(image_rgb)

            # Benchmark graph construction
            def build_graph():
                graph = graph_builder.build_spatial_graph(features, coords, "benchmark")
                return graph

            result = self.measure_time_memory(build_graph)

            if result['success']:
                graph = result['result']
                self.results['inference_times']['graph_construction'] = {
                    'time': result['wall_time'],
                    'memory_gb': result['memory_used_gb'],
                    'num_nodes': graph.num_nodes,
                    'num_edges': graph.num_edges
                }
                print(
                    f"   ✅ Graph construction: {result['wall_time']:.4f}s, {graph.num_nodes} nodes, {graph.num_edges} edges")
            else:
                print(f"   ❌ Graph construction failed: {result['error']}")

        except Exception as e:
            print(f"   ❌ Graph benchmark failed: {e}")

    def benchmark_gnn_inference(self, sample_image_path):
        """Benchmark GNN inference"""
        print("🧠 Benchmarking GNN Inference...")

        try:
            # FIXED: Use correct config path
            config_path = self.project_root / 'configs/main_config.yaml'

            if not config_path.exists():
                print(f"   ❌ Config file not found: {config_path}")
                return

            from data_processing.hog_extractor import IDRiDHOGExtractor
            from data_processing.graph_builder import IDRiDGraphBuilder
            from models.gnn_model import GNNWithFeatureExtraction
            from utils.config import load_config

            config = load_config(str(config_path))
            hog_extractor = IDRiDHOGExtractor(config)
            graph_builder = IDRiDGraphBuilder(config)
            gnn = GNNWithFeatureExtraction(config)
            gnn.eval()

            # Prepare data
            image = cv2.imread(sample_image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            features, coords = hog_extractor.extract_patch_hog_features(image_rgb)
            graph = graph_builder.build_spatial_graph(features, coords, "benchmark")

            # Benchmark GNN inference
            def gnn_inference():
                import torch
                with torch.no_grad():
                    embedding = gnn.extract_features(graph)
                return embedding

            result = self.measure_time_memory(gnn_inference)

            if result['success']:
                embedding = result['result']
                embedding_size = embedding.numel() if hasattr(embedding, 'numel') else len(embedding.flatten())

                self.results['inference_times']['gnn_inference'] = {
                    'time': result['wall_time'],
                    'memory_gb': result['memory_used_gb'],
                    'embedding_size': embedding_size
                }
                print(f"   ✅ GNN inference: {result['wall_time']:.4f}s, {embedding_size}D embedding")
            else:
                print(f"   ❌ GNN inference failed: {result['error']}")

        except Exception as e:
            print(f"   ❌ GNN benchmark failed: {e}")

    def benchmark_xgboost_inference(self):
        """Benchmark XGBoost inference with dummy data"""
        print("⚡ Benchmarking XGBoost Inference...")

        try:
            import joblib

            # Try to load trained models - FIXED: Use correct path
            model_dirs = list((self.project_root / "results/models").glob("hybrid_training_*"))
            if not model_dirs:
                print("   ❌ No trained models found")
                print(f"   📁 Looking in: {self.project_root / 'results/models'}")
                return

            latest_model_dir = max(model_dirs, key=os.path.getctime)
            print(f"   📁 Using model directory: {latest_model_dir}")

            # Load models
            dr_model_path = latest_model_dir / "xgb_dr_model.pkl"
            dme_model_path = latest_model_dir / "xgb_dme_model.pkl"

            if not (dr_model_path.exists() and dme_model_path.exists()):
                print("   ❌ Model files not found")
                print(f"   📄 DR model: {dr_model_path} - {'✅' if dr_model_path.exists() else '❌'}")
                print(f"   📄 DME model: {dme_model_path} - {'✅' if dme_model_path.exists() else '❌'}")
                return

            dr_bundle = joblib.load(dr_model_path)
            dme_bundle = joblib.load(dme_model_path)

            # Create dummy 64D embedding
            dummy_embedding = np.random.randn(1, 64)

            # Benchmark DR prediction
            def predict_dr():
                X_scaled = dr_bundle['scaler'].transform(dummy_embedding)
                pred = dr_bundle['model'].predict(X_scaled)
                proba = dr_bundle['model'].predict_proba(X_scaled)
                return pred, proba

            # Benchmark DME prediction
            def predict_dme():
                X_scaled = dme_bundle['scaler'].transform(dummy_embedding)
                pred = dme_bundle['model'].predict(X_scaled)
                proba = dme_bundle['model'].predict_proba(X_scaled)
                return pred, proba

            dr_result = self.measure_time_memory(predict_dr)
            dme_result = self.measure_time_memory(predict_dme)

            if dr_result['success'] and dme_result['success']:
                self.results['inference_times']['xgboost_dr'] = {
                    'time': dr_result['wall_time'],
                    'memory_gb': dr_result['memory_used_gb']
                }
                self.results['inference_times']['xgboost_dme'] = {
                    'time': dme_result['wall_time'],
                    'memory_gb': dme_result['memory_used_gb']
                }
                print(f"   ✅ XGBoost DR: {dr_result['wall_time']:.4f}s")
                print(f"   ✅ XGBoost DME: {dme_result['wall_time']:.4f}s")
            else:
                print("   ❌ XGBoost inference failed")

        except Exception as e:
            print(f"   ❌ XGBoost benchmark failed: {e}")

    def benchmark_end_to_end(self, sample_image_path, num_runs=3):
        """Benchmark complete end-to-end inference pipeline"""
        print("🚀 Benchmarking End-to-End Pipeline...")

        try:
            # FIXED: Use correct config path
            config_path = self.project_root / 'configs/main_config.yaml'

            if not config_path.exists():
                print(f"   ❌ Config file not found: {config_path}")
                return

            # Import all components
            from data_processing.hog_extractor import IDRiDHOGExtractor
            from data_processing.graph_builder import IDRiDGraphBuilder
            from models.gnn_model import GNNWithFeatureExtraction
            from utils.config import load_config
            import joblib
            import torch

            # Load config and models
            config = load_config(str(config_path))
            hog_extractor = IDRiDHOGExtractor(config)
            graph_builder = IDRiDGraphBuilder(config)
            gnn = GNNWithFeatureExtraction(config)
            gnn.eval()

            # Load XGBoost models
            model_dirs = list((self.project_root / "results/models").glob("hybrid_training_*"))
            if not model_dirs:
                print("   ❌ No trained models found")
                return

            latest_model_dir = max(model_dirs, key=os.path.getctime)
            dr_bundle = joblib.load(latest_model_dir / "xgb_dr_model.pkl")
            dme_bundle = joblib.load(latest_model_dir / "xgb_dme_model.pkl")

            def full_pipeline():
                # Load and preprocess image
                image = cv2.imread(sample_image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Extract HOG features
                features, coords = hog_extractor.extract_patch_hog_features(image_rgb)

                # Build graph
                graph = graph_builder.build_spatial_graph(features, coords, "benchmark")

                # GNN inference
                with torch.no_grad():
                    embedding = gnn.extract_features(graph)
                    if hasattr(embedding, 'cpu'):
                        embedding = embedding.cpu().numpy()
                    if embedding.ndim == 1:
                        embedding = embedding.reshape(1, -1)

                # XGBoost predictions
                X_dr = dr_bundle['scaler'].transform(embedding)
                X_dme = dme_bundle['scaler'].transform(embedding)

                dr_pred = dr_bundle['model'].predict(X_dr)[0]
                dr_prob = dr_bundle['model'].predict_proba(X_dr)[0]

                dme_pred = dme_bundle['model'].predict(X_dme)[0]
                dme_prob = dme_bundle['model'].predict_proba(X_dme)[0]

                return {
                    'dr_grade': int(dr_pred),
                    'dr_prob': dr_prob,
                    'dme_risk': int(dme_pred),
                    'dme_prob': dme_prob
                }

            # Run multiple times for statistics
            times = []
            for i in range(num_runs):
                print(f"     Run {i+1}/{num_runs}...")
                result = self.measure_time_memory(full_pipeline)
                if result['success']:
                    times.append(result['wall_time'])
                    print(f"     ✅ Run {i+1}: {result['wall_time']:.4f}s")
                else:
                    print(f"     ❌ Run {i+1} failed: {result['error']}")

            if times:
                self.results['inference_times']['end_to_end'] = {
                    'mean_time': np.mean(times),
                    'std_time': np.std(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'throughput_images_per_second': 1.0 / np.mean(times)
                }
                print(f"   ✅ End-to-end: {np.mean(times):.4f}s ± {np.std(times):.4f}s")
                print(f"   📊 Throughput: {1.0 / np.mean(times):.2f} images/second")
            else:
                print("   ❌ End-to-end benchmark failed")

        except Exception as e:
            print(f"   ❌ End-to-end benchmark failed: {e}")
            import traceback
            traceback.print_exc()

    def measure_model_sizes(self):
        """Measure disk usage of trained models"""
        print("💾 Measuring Model Sizes...")

        try:
            model_dirs = list((self.project_root / "results/models").glob("hybrid_training_*"))
            if not model_dirs:
                print("   ❌ No trained models found")
                return

            latest_model_dir = max(model_dirs, key=os.path.getctime)

            sizes = {}
            total_size = 0

            for model_file in latest_model_dir.glob("*.pkl"):
                size_mb = model_file.stat().st_size / (1024 ** 2)
                sizes[model_file.name] = size_mb
                total_size += size_mb

            self.results['model_sizes'] = sizes
            self.results['model_sizes']['total_mb'] = total_size

            print(f"   📦 Total model size: {total_size:.2f} MB")
            for name, size in sizes.items():
                if name != 'total_mb':
                    print(f"   📄 {name}: {size:.2f} MB")

        except Exception as e:
            print(f"   ❌ Model size measurement failed: {e}")

    def generate_report(self, save_path="performance_report.json"):
        """Generate comprehensive performance report"""

        # Calculate total pipeline time
        pipeline_time = 0
        pipeline_components = ['data_loading', 'hog_extraction', 'graph_construction', 'gnn_inference', 'xgboost_dr',
                               'xgboost_dme']

        for component in pipeline_components:
            if component in self.results['inference_times']:
                time_key = 'time' if 'time' in self.results['inference_times'][component] else 'mean_time'
                pipeline_time += self.results['inference_times'][component].get(time_key, 0)

        # Add summary statistics
        self.results['summary'] = {
            'total_pipeline_time': pipeline_time,
            'theoretical_throughput': 1.0 / pipeline_time if pipeline_time > 0 else 0,
            'timestamp': datetime.now().isoformat(),
            'total_memory_peak_gb': max([
                                            comp.get('memory_gb', 0) for comp in
                                            self.results['inference_times'].values()
                                            if isinstance(comp, dict)
                                        ] + [0])
        }

        # Save detailed report
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"📄 Detailed report saved: {save_path}")
        return self.results

    def print_summary(self):
        """Print performance summary"""
        print("\n" + "=" * 60)
        print("⚡ PERFORMANCE BENCHMARK SUMMARY")
        print("=" * 60)

        # System info
        sys_info = self.results['system_info']
        print(f"🖥️  System: {sys_info['cpu_count']} cores ({sys_info['cpu_count_logical']} logical)")
        print(
            f"💾 Memory: {sys_info['total_memory_gb']:.1f} GB total, {sys_info['available_memory_gb']:.1f} GB available")

        # Inference times
        print(f"\n⏱️  INFERENCE PERFORMANCE:")
        inference_times = self.results['inference_times']

        component_names = {
            'data_loading': 'Data Loading',
            'hog_extraction': 'HOG Extraction',
            'graph_construction': 'Graph Construction',
            'gnn_inference': 'GNN Inference',
            'xgboost_dr': 'XGBoost DR',
            'xgboost_dme': 'XGBoost DME',
            'end_to_end': 'End-to-End Pipeline'
        }

        for key, name in component_names.items():
            if key in inference_times:
                comp = inference_times[key]
                if 'mean_time' in comp:
                    print(f"   {name:20s}: {comp['mean_time']:.4f}s ± {comp.get('std_time', 0):.4f}s")
                elif 'time' in comp:
                    print(f"   {name:20s}: {comp['time']:.4f}s")

        # Throughput
        if 'end_to_end' in inference_times and 'throughput_images_per_second' in inference_times['end_to_end']:
            throughput = inference_times['end_to_end']['throughput_images_per_second']
            print(f"\n📊 THROUGHPUT: {throughput:.2f} images/second")
            print(f"   Time per image: {1 / throughput:.4f}s")
            print(f"   Images per minute: {throughput * 60:.1f}")
            print(f"   Images per hour: {throughput * 3600:.0f}")

        # Memory usage
        print(f"\n💾 MEMORY USAGE:")
        peak_memory = self.results['summary'].get('total_memory_peak_gb', 0)
        if peak_memory > 0:
            print(f"   Peak memory usage: {peak_memory:.2f} GB")

        # Model sizes
        if 'model_sizes' in self.results and 'total_mb' in self.results['model_sizes']:
            total_size = self.results['model_sizes']['total_mb']
            print(f"   Model storage: {total_size:.1f} MB")

        print("=" * 60)


def find_sample_image():
    """Find a sample image for benchmarking"""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    possible_paths = [
        script_dir / "sample_images" / "IDRiD_031.jpg",
        project_root / "data/raw/B. Disease Grading/B. Disease Grading/1. Original Images/a. Training Set/IDRiD_001.jpg",
        project_root / "data/raw/B. Disease Grading/1. Original Images/a. Training Set/IDRiD_001.jpg",
    ]

    print("🔍 Searching for sample image...")
    for path in possible_paths:
        print(f"   Checking: {path} - {'✅' if path.exists() else '❌'}")
        if path.exists():
            return str(path)

    return None


def main():
    """Run complete performance benchmark"""
    print("⚡ IDRiD DR-Vision Performance Benchmark")
    print("=" * 60)

    # Find sample image
    sample_image = find_sample_image()
    if not sample_image:
        print("❌ No sample image found for benchmarking")
        print("💡 Place a sample IDRiD image in visualization/sample_images/")
        return

    print(f"📸 Using sample image: {Path(sample_image).name}")

    # Check if config file exists
    config_path = project_root / 'configs/main_config.yaml'
    print(f"📄 Config file: {config_path} - {'✅' if config_path.exists() else '❌'}")

    if not config_path.exists():
        print("❌ Config file missing! Please ensure configs/main_config.yaml exists")
        print("💡 The script needs the config file to load model parameters")
        return

    # Initialize benchmark
    benchmark = PerformanceBenchmark()

    # Run all benchmarks
    benchmark.benchmark_data_loading(sample_image)
    benchmark.benchmark_hog_extraction(sample_image)
    benchmark.benchmark_graph_construction(sample_image)
    benchmark.benchmark_gnn_inference(sample_image)
    benchmark.benchmark_xgboost_inference()
    benchmark.benchmark_end_to_end(sample_image)
    benchmark.measure_model_sizes()

    # Generate report
    benchmark.generate_report("performance_benchmark_results.json")
    benchmark.print_summary()


if __name__ == "__main__":
    main()
