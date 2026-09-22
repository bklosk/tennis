import subprocess

import pytest

from tennis_pipeline import cloud


def test_budget_caps_runtime_and_bills_a_minimum_minute():
    b = cloud.Budget(price_hourly=1.57, cap_usd=4.0, created_at=0.0)
    assert b.max_seconds == pytest.approx(4.0 / 1.57 * 3600)
    assert b.spent(now=10.0) == pytest.approx(60 / 3600 * 1.57)
    assert b.spent(now=3600.0) == pytest.approx(1.57)
    assert b.remaining_seconds(now=b.max_seconds) == pytest.approx(0.0)


def test_self_destruct_script_is_valid_bash_and_targets_this_droplet():
    script = cloud.self_destruct_script("tok123", 900)
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    assert "sleep 900" in script
    assert "metadata/v1/id" in script
    assert "Bearer tok123" in script and "X DELETE" in script.replace("-X DELETE", "X DELETE")


class FakeDO:
    def __init__(self, sizes):
        self.token, self._sizes = "tok", sizes

    def size(self, slug):
        return self._sizes[slug]


def test_droplet_refuses_unavailable_region():
    do = FakeDO({"gpu-l40sx1-48gb": {"regions": ["tor1"], "available": True, "price_hourly": 1.57}})
    with pytest.raises(RuntimeError, match="not available"):
        cloud.Droplet(do, "gpu-l40sx1-48gb", "nyc1", cap_usd=1.0)
    d = cloud.Droplet(do, "gpu-l40sx1-48gb", "tor1", cap_usd=1.0)
    assert d.image == cloud.GPU_IMAGE and d.price == 1.57


def test_failed_create_destroys_the_droplet(monkeypatch):
    do = FakeDO({"s-1vcpu-1gb": {"regions": ["tor1"], "available": True, "price_hourly": 0.009}})
    d = cloud.Droplet(do, "s-1vcpu-1gb", "tor1", cap_usd=0.05)
    destroyed = []

    def boom():
        d.id = 42
        raise RuntimeError("ssh never came up")

    monkeypatch.setattr(d, "create", boom)
    monkeypatch.setattr(d, "destroy", lambda: destroyed.append(d.id))
    with pytest.raises(RuntimeError):
        with d:
            pass
    assert destroyed == [42]
