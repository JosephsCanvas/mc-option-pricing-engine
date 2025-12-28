"""
Tests for experiment infrastructure.
"""

import json

import pytest

from mc_pricer.experiments import (
    ExperimentConfig,
    load_results,
    run_experiment,
    save_results,
)


class TestExperimentConfig:
    """Test experiment configuration."""

    def test_config_creation(self):
        """Test basic config creation."""
        config = ExperimentConfig(
            name="test",
            model="gbm",
            option_type="call",
            style="european",
            S0=100.0,
            K=100.0,
            r=0.05,
            T=1.0,
            sigma=0.2,
            n_paths_list=[1000],
            seeds=[42]
        )

        assert config.name == "test"
        assert config.model == "gbm"
        assert config.sigma == 0.2

    def test_config_heston_parameters(self):
        """Test Heston config with all parameters."""
        config = ExperimentConfig(
            name="test_heston",
            model="heston",
            option_type="call",
            style="european",
            S0=100.0,
            K=100.0,
            r=0.05,
            T=1.0,
            kappa=2.0,
            theta=0.04,
            xi=0.3,
            rho=-0.7,
            v0=0.04,
            n_paths_list=[1000],
            n_steps=100,
            seeds=[42]
        )

        assert config.kappa == 2.0
        assert config.theta == 0.04


class TestExperimentRunner:
    """Test experiment runner."""

    def test_deterministic_output(self):
        """Test that same seed produces identical results."""
        config = ExperimentConfig(
            name="determinism_test",
            model="gbm",
            option_type="call",
            style="european",
            S0=100.0,
            K=100.0,
            r=0.05,
            T=1.0,
            sigma=0.2,
            n_paths_list=[10000],
            seeds=[42]
        )

        results1 = run_experiment(config)
        results2 = run_experiment(config)

        assert len(results1) == len(results2) == 1
        assert abs(results1[0].price - results2[0].price) < 1e-10
        assert abs(results1[0].stderr - results2[0].stderr) < 1e-10

    def test_grid_shape(self):
        """Test that grid produces expected number of results."""
        config = ExperimentConfig(
            name="grid_test",
            model="gbm",
            option_type="call",
            style="european",
            S0=100.0,
            K=100.0,
            r=0.05,
            T=1.0,
            sigma=0.2,
            n_paths_list=[1000, 5000, 10000],
            seeds=[42, 123, 456]
        )

        results = run_experiment(config)

        # Should have 3 n_paths × 3 seeds = 9 results
        assert len(results) == 9

        # Check all combinations present
        n_paths_seen = set()
        seeds_seen = set()
        for r in results:
            n_paths_seen.add(r.n_paths)
            seeds_seen.add(r.metadata.seed)

        assert n_paths_seen == {1000, 5000, 10000}
        assert seeds_seen == {42, 123, 456}

    def test_metadata_fields(self):
        """Test that results contain required metadata."""
        config = ExperimentConfig(
            name="metadata_test",
            model="gbm",
            option_type="call",
            style="european",
            S0=100.0,
            K=100.0,
            r=0.05,
            T=1.0,
            sigma=0.2,
            n_paths_list=[1000],
            seeds=[42]
        )

        results = run_experiment(config)
        result = results[0]

        # Check metadata fields
        assert result.metadata.timestamp is not None
        assert result.metadata.python_version is not None
        assert result.metadata.numpy_version is not None
        assert result.metadata.os_platform is not None
        assert result.metadata.seed == 42
        assert result.metadata.model == "gbm"
        assert result.metadata.option_type == "call"
        assert result.metadata.n_paths == 1000

    def test_runtime_measurement(self):
        """Test that runtime is measured."""
        config = ExperimentConfig(
            name="runtime_test",
            model="gbm",
            option_type="call",
            style="european",
            S0=100.0,
            K=100.0,
            r=0.05,
            T=1.0,
            sigma=0.2,
            n_paths_list=[10000],
            seeds=[42]
        )

        results = run_experiment(config)
        result = results[0]

        assert result.runtime_seconds > 0
        assert result.runtime_seconds < 10  # Should be fast

    def test_control_variate_beta(self):
        """Test that control variate beta is captured."""
        config = ExperimentConfig(
            name="cv_test",
            model="gbm",
            option_type="call",
            style="european",
            S0=100.0,
            K=100.0,
            r=0.05,
            T=1.0,
            sigma=0.2,
            n_paths_list=[10000],
            seeds=[42],
            control_variate=True
        )

        results = run_experiment(config)
        result = results[0]

        assert result.control_variate_beta is not None
        assert abs(result.control_variate_beta) > 0

    def test_heston_experiment(self):
        """Test Heston model experiment."""
        config = ExperimentConfig(
            name="heston_test",
            model="heston",
            option_type="call",
            style="european",
            S0=100.0,
            K=100.0,
            r=0.05,
            T=1.0,
            kappa=2.0,
            theta=0.04,
            xi=0.3,
            rho=-0.7,
            v0=0.04,
            n_paths_list=[10000],
            n_steps=100,
            seeds=[42]
        )

        results = run_experiment(config)
        result = results[0]

        assert result.price > 0
        assert result.stderr > 0
        assert result.n_steps == 100

    def test_american_option_experiment(self):
        """Test American option experiment."""
        config = ExperimentConfig(
            name="american_test",
            model="gbm",
            option_type="put",
            style="american",
            S0=100.0,
            K=110.0,
            r=0.05,
            T=1.0,
            sigma=0.2,
            n_paths_list=[10000],
            n_steps=50,
            seeds=[42],
            lsm_basis="poly2"
        )

        results = run_experiment(config)
        result = results[0]

        assert result.price > 0
        assert result.n_steps == 50

    def test_invalid_config_raises_error(self):
        """Test that invalid config raises error."""
        # GBM without sigma
        config = ExperimentConfig(
            name="invalid_test",
            model="gbm",
            option_type="call",
            style="european",
            S0=100.0,
            K=100.0,
            r=0.05,
            T=1.0,
            n_paths_list=[1000],
            seeds=[42]
        )

        with pytest.raises(ValueError, match="sigma"):
            run_experiment(config)


class TestExperimentIO:
    """Test experiment I/O."""

    def test_save_and_load_results(self, tmp_path):
        """Test saving and loading results."""
        config = ExperimentConfig(
            name="io_test",
            model="gbm",
            option_type="call",
            style="european",
            S0=100.0,
            K=100.0,
            r=0.05,
            T=1.0,
            sigma=0.2,
            n_paths_list=[1000],
            seeds=[42]
        )

        results = run_experiment(config)

        # Save results
        out_dir = tmp_path / "test_results"
        save_results(results, out_dir, "test_experiment")

        # Check files exist
        assert (out_dir / "results.json").exists()
        assert (out_dir / "summary.txt").exists()

        # Load and verify
        loaded = load_results(out_dir)
        assert loaded["experiment_name"] == "test_experiment"
        assert loaded["n_results"] == 1
        assert len(loaded["results"]) == 1

        # Check JSON structure
        result_dict = loaded["results"][0]
        assert "price" in result_dict
        assert "stderr" in result_dict
        assert "metadata" in result_dict

    def test_json_serialization(self, tmp_path):
        """Test that results serialize to valid JSON."""
        config = ExperimentConfig(
            name="json_test",
            model="gbm",
            option_type="call",
            style="european",
            S0=100.0,
            K=100.0,
            r=0.05,
            T=1.0,
            sigma=0.2,
            n_paths_list=[1000, 5000],
            seeds=[42, 123]
        )

        results = run_experiment(config)

        # Save and parse JSON
        out_dir = tmp_path / "json_test"
        save_results(results, out_dir, "json_test")

        json_path = out_dir / "results.json"
        with open(json_path) as f:
            data = json.load(f)

        # Verify structure
        assert isinstance(data, dict)
        assert isinstance(data["results"], list)
        assert len(data["results"]) == 4  # 2 n_paths × 2 seeds

    def test_summary_table_format(self, tmp_path):
        """Test that summary table is properly formatted."""
        config = ExperimentConfig(
            name="table_test",
            model="gbm",
            option_type="call",
            style="european",
            S0=100.0,
            K=100.0,
            r=0.05,
            T=1.0,
            sigma=0.2,
            n_paths_list=[1000],
            seeds=[42],
            antithetic=True
        )

        results = run_experiment(config)
        out_dir = tmp_path / "table_test"
        save_results(results, out_dir, "table_test")

        # Read summary
        summary_path = out_dir / "summary.txt"
        with open(summary_path) as f:
            content = f.read()

        # Check content
        assert "Experiment: table_test" in content
        assert "Price" in content
        assert "Stderr" in content
        assert "Runtime" in content
        assert "GBM" in content  # Model name
