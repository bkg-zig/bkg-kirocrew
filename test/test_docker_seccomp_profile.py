import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "docker/seccomp/kirocrew-seccomp.json"
GENERATOR = ROOT / "docker/seccomp/gen_profile.py"


def _unconditional_allow(profile: dict, syscall: str) -> bool:
    return any(
        rule.get("names") == [syscall]
        and rule.get("action") == "SCMP_ACT_ALLOW"
        and "includes" not in rule
        and "excludes" not in rule
        and "args" not in rule
        for rule in profile.get("syscalls", [])
    )


def test_checked_in_profile_allows_umount2_for_userns_sandbox() -> None:
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    assert _unconditional_allow(profile, "unshare")
    assert _unconditional_allow(profile, "mount")
    assert _unconditional_allow(profile, "umount2")


def test_profile_generator_emits_umount2_allow_rule() -> None:
    spec = importlib.util.spec_from_file_location("kirocrew_seccomp_gen", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generated = module.patch_profile({"syscalls": []})
    assert _unconditional_allow(generated, "umount2")
