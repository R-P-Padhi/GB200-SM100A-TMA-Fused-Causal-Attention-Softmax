"""Failure-oriented fixtures for the PTX and ptxas static gates."""
import unittest

from check_ptx import KERNELS, check_ptx, entries
from check_resources import check_resources, occupancy, parse_ptxas

TMA_BODY = """
mbarrier.init.shared.b64;
mbarrier.arrive.expect_tx.shared.b64;
cp.async.bulk.tensor.2d.shared::cta.global;
cp.async.bulk.tensor.2d.global.shared::cta;
mbarrier.try_wait.shared.b64;
fence.proxy.async.shared::cta;
cp.async.bulk.commit_group;
cp.async.bulk.wait_group;
ex2.approx.f32;
"""
ONLINE_BODY = TMA_BODY + """
mbarrier.try_wait.parity.shared.b64;
shfl.sync.bfly.b32;
cvt.f32.f16;
cvt.rn.f16.f32;
"""


def ptx_fixture() -> str:
    bodies = {
        "probe": "ret;",
        "causal_softmax_baseline": "ex2.approx.f32;",
        "causal_softmax_warp": "ex2.approx.f32;",
        "causal_softmax_tma": TMA_BODY,
        "causal_softmax_tma_pipelined": TMA_BODY,
        "causal_softmax_tma_online": ONLINE_BODY,
    }
    return ".target sm_100a\n" + "\n".join(
        f".visible .entry {name}() {{ {body} }}" for name, body in bodies.items()
    )


def entry_log(name: str, regs=29, stores=0, loads=0, stack=0, arch="sm_100a") -> str:
    return f"""ptxas info : Compiling entry function '{name}' for '{arch}'
ptxas info : Function properties for {name}
  {stack} bytes stack frame, {stores} bytes spill stores, {loads} bytes spill loads
ptxas info : Used {regs} registers, 6160 bytes smem
"""


def full_log() -> str:
    return "".join(entry_log(name) for name in KERNELS)


class ValidationTests(unittest.TestCase):
    def test_ptx_accepts_all_entries(self):
        self.assertEqual(check_ptx(ptx_fixture()), [])

    def test_ptx_rejects_empty(self):
        self.assertTrue(check_ptx(""))

    def test_ptx_rejects_wrong_target(self):
        self.assertTrue(any("target" in p for p in check_ptx(ptx_fixture().replace("sm_100a", "sm_103a"))))

    def test_ptx_rejects_missing_kernel(self):
        for name in KERNELS:
            self.assertTrue(any(name in p for p in check_ptx(ptx_fixture().replace(f".entry {name}()", f".entry missing_{name}()"))))

    def test_ptx_rejects_comments_as_evidence(self):
        bad = ptx_fixture().replace("mbarrier.init.shared.b64;", "// mbarrier.init.shared.b64;", 1)
        self.assertTrue(any("mbarrier init" in p for p in check_ptx(bad)))

    def test_ptx_rejects_instructions_in_wrong_kernel(self):
        bad = ptx_fixture().replace(f".entry causal_softmax_tma() {{ {TMA_BODY} }}", ".entry causal_softmax_tma() { ex2.approx.f32; }")
        self.assertTrue(any("causal_softmax_tma" in p and "2D TMA" in p for p in check_ptx(bad)))

    def test_ptx_rejects_missing_online_parity(self):
        bad = ptx_fixture().replace("mbarrier.try_wait.parity.shared.b64;", "")
        self.assertTrue(any("parity wait" in p for p in check_ptx(bad)))

    def test_ptx_rejects_tma_copy_before_expected_bytes(self):
        bad = ptx_fixture().replace(
            "mbarrier.arrive.expect_tx.shared.b64;\ncp.async.bulk.tensor.2d.shared::cta.global;",
            "cp.async.bulk.tensor.2d.shared::cta.global;\nmbarrier.arrive.expect_tx.shared.b64;",
        )
        self.assertTrue(any("before expected bytes" in p for p in check_ptx(bad)))

    def test_ptx_allows_local_scratch_with_spill_check_elsewhere(self):
        with_scratch = ptx_fixture().replace(".entry probe() { ret; }", ".entry probe() { ld.local.u32; }")
        self.assertEqual(check_ptx(with_scratch), [])

    def test_ptx_parses_nested_braces(self):
        self.assertIn("probe", entries(ptx_fixture().replace("ret;", "{ ret; }", 1)))

    def test_ptx_rejects_unclosed_entry(self):
        self.assertTrue(check_ptx(".target sm_100a\n.entry probe() {"))

    def test_resources_accepts_complete_report(self):
        self.assertEqual(check_resources(full_log()), [])
        self.assertEqual(parse_ptxas(full_log()), ([29] * len(KERNELS), 0, 0))

    def test_resources_rejects_missing_entry(self):
        self.assertTrue(any("missing ptxas" in p for p in check_resources(entry_log("probe"))))

    def test_resources_rejects_wrong_arch(self):
        self.assertTrue(check_resources(full_log().replace("'sm_100a'", "'sm_103a'", 1)))

    def test_resources_rejects_spill_store(self):
        self.assertTrue(check_resources(full_log() + entry_log("bonus", stores=4)))

    def test_resources_rejects_spill_load(self):
        self.assertTrue(check_resources(full_log() + entry_log("bonus", loads=4)))

    def test_resources_rejects_large_register_count(self):
        self.assertTrue(check_resources(full_log() + entry_log("bonus", regs=129)))

    def test_resources_reports_but_allows_explicit_non_spill_stack(self):
        self.assertEqual(check_resources(full_log() + entry_log("bonus", stack=40)), [])

    def test_resources_rejects_incomplete_report(self):
        self.assertTrue(check_resources("Compiling entry function 'probe' for 'sm_100a'"))

    def test_resources_rejects_assembler_error(self):
        self.assertTrue(check_resources(full_log() + "ptxas fatal : malformed PTX"))

    def test_register_occupancy_rounding(self):
        self.assertEqual(occupancy(128), 0.25)
        self.assertEqual(occupancy(65), 0.375)


if __name__ == "__main__":
    unittest.main()
