"""Keep the shipped node preset on the validated six-cell graph contract."""

from pathlib import Path

from vllm_omni.config.stage_config import resolve_deploy_yaml


def test_h100_node_graph_configuration() -> None:
    root = Path(__file__).resolve().parents[4]
    config = resolve_deploy_yaml(root / "vllm_omni/deploy/duplexio_opd_h100_8gpu.yaml")
    assert config["active_stream_window"] == 64
    assert config["duplex_session"]["max_sessions"] == 64
    stage, = config["stages"]
    assert stage["devices"] == "0,1,2,3,4,5,6,7"
    assert stage["num_replicas"] == 8
    assert stage["max_num_seqs"] == 8
    assert stage["enforce_eager"] is False
    assert stage["compilation_config"] == {
        "mode": 0,
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [6, 12, 18, 24, 30, 36, 42, 48],
        "cudagraph_copy_inputs": True,
    }


def test_h100_actor_graph_configuration() -> None:
    root = Path(__file__).resolve().parents[4]
    config = resolve_deploy_yaml(root / "vllm_omni/deploy/duplexio_opd_h100.yaml")
    stage, = config["stages"]
    assert stage["devices"] == "0"
    assert stage["max_num_seqs"] == 8
    assert config["active_stream_window"] == 8
    assert config["duplex_session"]["max_sessions"] == 8
    assert stage["compilation_config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
