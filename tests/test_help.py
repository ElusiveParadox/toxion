import shlex
from pathlib import Path

import pytest

from toxion import cli, help_text
from toxion.training.pipeline import TrainConfig

ROOT = Path(__file__).resolve().parents[1]
parser = cli.build_parser()
COMMAND_NAMES = list(help_text._subparsers(parser))


def run(capsys, *argv):
    code = cli.main(list(argv))
    return code, capsys.readouterr().out


def test_every_command_is_documented_and_nothing_extra():
    assert set(help_text.COMMANDS) == set(COMMAND_NAMES)
    assert "retrain" in COMMAND_NAMES


def test_every_option_of_every_command_is_explained_and_listed():
    full = help_text.full_reference(parser)
    for name in COMMAND_NAMES:
        rows = help_text.option_rows(parser, name)
        sub = help_text._subparsers(parser)[name]
        n_args = len([a for a in sub._actions if a.dest != "help"])
        assert len(rows) == n_args, name
        for flag, note in rows:
            assert note.strip(), f"{name} {flag} has no explanation"
            assert flag.split()[0].rstrip(",") in full


def test_full_reference_lists_every_command_with_its_sections():
    full = help_text.full_reference(parser)
    for name in COMMAND_NAMES:
        assert f"toxion {name}  -  " in full
    assert full.count("Changes / writes:") == len(COMMAND_NAMES) == full.count("  Reads:")
    for var in ("DATABASE_URL", "TOXION_CHECKPOINT", "TOXION_FPR_CAP", "TOXION_AUTO_RETRAIN"):
        assert var in full


def test_step_guide_mentions_every_command_and_what_it_changes():
    steps = help_text.steps_guide()
    for name in COMMAND_NAMES:
        assert f"toxion {name}" in steps, f"{name} missing from the step guide"
    assert steps.count("Changes:") == len(help_text.STEPS) and "REPLACES results/checkpoints/active.pt" in steps


def _toxion_lines():
    for info in help_text.COMMANDS.values():
        yield from info["usage"]
    for step in help_text.STEPS:
        yield from step["run"]


def test_every_documented_command_line_really_parses():
    seen = 0
    for line in _toxion_lines():
        line = line.split("#")[0].strip()
        if not line.startswith("toxion "):
            continue
        argv = shlex.split(line)[1:]
        parser.parse_args(argv)
        seen += 1
    assert seen >= 25


def test_documented_scripts_and_files_exist():
    for line in _toxion_lines():
        words = shlex.split(line.split("#")[0])
        if words[:1] == ["python"] or words[:2] == ["streamlit", "run"]:
            assert (ROOT / words[-1] if words[0] == "streamlit" else ROOT / words[1]).exists(), line
        if words[:2] == ["cp", ".env.example"]:
            assert (ROOT / ".env.example").exists()


def test_documented_defaults_match_the_code():
    cfg = TrainConfig()
    notes = help_text.OPTION_NOTES
    assert (cfg.epochs, cfg.batch_size, cfg.hidden_dim, cfg.patience, cfg.lr) == (100, 32, 128, 15, 1e-3)
    assert "default 100" in notes[(help_text._TRAIN, "--epochs")] and "default 32" in notes[(help_text._TRAIN, "--batch-size")]
    assert "default 128" in notes[(help_text._TRAIN, "--hidden")] and "default 15" in notes[(help_text._TRAIN, "--patience")]
    assert "0.001" in notes[(help_text._TRAIN, "--lr")] and cfg.monitor == "val_loss" and cfg.schedule == "none"
    assert cfg.node_mask_p == cfg.edge_drop_p == cfg.bit_drop_p == 0.10
    from toxion import config

    assert config.RETRAIN_BATCH_SIZE == 25 and "25" in help_text.steps_guide()


@pytest.mark.parametrize("argv", [["--h"], ["--help"], ["help"]])
def test_full_reference_forms(capsys, argv):
    code, out = run(capsys, *argv)
    assert code == 0 and "complete command reference" in out and "[14/14]" in out


@pytest.mark.parametrize("argv", [["-h", "step"], ["-help", "step"], ["--help", "step"], ["--h", "step"],
                                  ["help", "step"], ["-h", "STEPS"]])
def test_step_guide_forms(capsys, argv):
    code, out = run(capsys, *argv)
    assert code == 0 and "step-by-step guide" in out and "STEP 1:" in out


def test_help_for_one_command_and_unknown_command(capsys):
    code, out = run(capsys, "help", "probe")
    assert code == 0 and out.startswith("toxion probe  -  ") and "complete command reference" not in out
    code, out = run(capsys, "--h", "nonsense")
    assert code == 0 and "no command named 'nonsense'" in out and "probe" in out


@pytest.mark.parametrize("argv", [["-h"], ["-help"]])
def test_plain_short_help_is_still_argparse(capsys, argv):
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    out = capsys.readouterr().out
    assert e.value.code == 0 and out.startswith("usage: toxion") and "probe" in out
    assert "toxion --h" in out and "toxion -h step" in out
    assert "SUPPRESS" not in out and "retrain" in out
    assert "complete command reference" not in out


def test_per_command_help_unchanged(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["probe", "-h"])
    assert e.value.code == 0 and capsys.readouterr().out.startswith("usage: toxion probe")


def test_closed_pipe_is_not_an_error(monkeypatch):
    def boom(*a, **k):
        raise BrokenPipeError

    monkeypatch.setattr(help_text, "full_reference", boom)
    monkeypatch.setattr(cli.sys, "stdout", cli.sys.stdout)
    assert cli.main(["--h"]) == 0
