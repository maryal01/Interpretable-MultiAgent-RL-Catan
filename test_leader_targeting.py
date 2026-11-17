"""Known-outcome checks for RQ1 primary and secondary classification."""

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from summarize_leader_targeting import classify_decision, classify_file, format_summary, load_game_seeds, summarize_decisions


def make_row(scores=None, actor="C", victims=("A", "B"), chosen=0):
    actions = [{"tile": [index, 0, -index], "victim_color": victim} for index, victim in enumerate(victims)]
    return {
        "game_index": "1", "game_seed": "10000", "decision_index": "10",
        "actor_color": actor,
        "visible_vps": json.dumps(scores or {"A": 6, "B": 4, "C": 3, "D": 2}),
        "victim_color": actions[chosen]["victim_color"] or "",
        "selected_tile": json.dumps(actions[chosen]["tile"]),
        "legal_robber_actions": json.dumps(actions),
    }


@pytest.mark.parametrize("chosen,targeted", [(0, True), (1, True), (2, False), (3, False)])
def test_baseline_counts_actions_including_no_victim(chosen, targeted):
    result = classify_decision(make_row(victims=("A", "A", "B", None), chosen=chosen))
    for measure in result.values():
        assert measure == {
            "leader_color": "A", "eligible": True, "exclusion_reason": None,
            "targeted_leader": targeted, "random_legal_action_leader_probability": 0.5,
        }


def test_acting_leader_is_excluded_only_from_primary():
    result = classify_decision(make_row(actor="A", victims=("B", "C")))
    assert result["primary"]["leader_color"] == "A"
    assert result["primary"]["exclusion_reason"] == "actor_is_leader"
    assert result["primary"]["targeted_leader"] is None
    assert result["primary"]["random_legal_action_leader_probability"] is None
    assert result["secondary"]["leader_color"] == "B"
    assert result["secondary"]["eligible"]
    assert result["secondary"]["targeted_leader"] is True


def test_actor_tied_for_first_can_have_unique_opposing_leader():
    result = classify_decision(make_row(scores={"A": 6, "B": 6, "C": 3, "D": 2}, actor="A", victims=("B", "C")))
    assert result["primary"]["exclusion_reason"] == "tied_leaders"
    assert result["secondary"]["eligible"]
    assert result["secondary"]["leader_color"] == "B"


@pytest.mark.parametrize("scores,victims,reason", [
    ({"A": 6, "B": 6, "C": 3, "D": 2}, ("A", "B"), "tied_leaders"),
    ({"A": 6, "B": 4, "C": 3, "D": 2}, ("B", "D"), "leader_unreachable"),
    ({"A": 6, "B": 4, "C": 3, "D": 2}, ("A", "A"), "no_alternative_opponent"),
    ({"A": 6, "B": 4, "C": 3, "D": 2}, ("A", None), "no_alternative_opponent"),
    ({"A": 6, "B": 4, "C": 3, "D": 2}, (None,), "leader_unreachable"),
])
def test_exclusions_are_not_recorded_as_failed_targeting(scores, victims, reason):
    for measure in classify_decision(make_row(scores=scores, victims=victims)).values():
        assert not measure["eligible"]
        assert measure["exclusion_reason"] == reason
        assert measure["targeted_leader"] is None
        assert measure["random_legal_action_leader_probability"] is None


def test_existing_opponent_labels_do_not_override_raw_evidence():
    row = make_row()
    row.update(unique_opponent_leader="D", overall_leaders='["D"]', informative_victim_choice="False")
    assert classify_decision(row)["primary"]["leader_color"] == "A"


@pytest.mark.parametrize("field,value", [
    ("victim_color", "D"),
    ("selected_tile", "[99,0,-99]"),
    ("legal_robber_actions", "[]"),
    ("visible_vps", '{"A":6,"B":4}'),
    ("visible_vps", '{"A":6,"B":4,"C":-1}'),
    ("legal_robber_actions", '[{"tile":[0,0,0],"victim_color":"C"}]'),
])
def test_invalid_evidence_is_rejected(field, value):
    row = make_row()
    row[field] = value
    with pytest.raises(ValueError):
        classify_decision(row)


def test_csv_round_trip_and_duplicate_rejection(tmp_path):
    path = tmp_path / "decisions.csv"
    row = make_row()
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    result = classify_file(path)
    assert len(result) == 1
    assert result[0]["game_seed"] == 10000
    assert result[0]["primary"]["targeted_leader"] is True
    with path.open("a", newline="") as stream:
        csv.DictWriter(stream, fieldnames=list(row)).writerow(row)
    with pytest.raises(ValueError, match="CSV line 3: Duplicate"):
        classify_file(path)


def test_empty_csv_rejected(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("game_index,game_seed,decision_index\n")
    with pytest.raises(ValueError, match="no robber decisions"):
        classify_file(path)


def classified(row, seed=10000):
    return {"game_seed": seed, "actor_color": row["actor_color"], **classify_decision(row)}


def get_group(summary, measure="primary", actor=None):
    return next(row for row in summary["summaries"] if row["measure"] == measure and row["actor_color"] == actor)


def test_summary_averages_probabilities_not_pooled_action_counts():
    decisions = [
        classified(make_row(victims=("A", "B"))),  # target; p=1/2
        classified(make_row(victims=("A", "B", "B", "B"), chosen=1)),  # miss; p=1/4
        classified(make_row(victims=("A", "A"))),  # excluded, even though it targets A
    ]
    row = get_group(summarize_decisions(decisions))
    assert row["total_decisions"] == 3
    assert row["eligible_decisions"] == 2
    assert row["observed_leader_targets"] == 1
    assert row["expected_leader_targets"] == pytest.approx(0.75)
    assert row["observed_targeting_rate"] == pytest.approx(0.5)
    assert row["random_legal_action_baseline_rate"] == pytest.approx(0.375)
    assert row["excess_percentage_points"] == pytest.approx(12.5)
    assert row["exclusion_reasons"] == {"no_alternative_opponent": 1}


def test_all_agents_are_pooled_by_decision_not_equal_agent_weights():
    decisions = [classified(make_row())] + [
        classified(make_row(actor="D", chosen=1), seed=10001) for _ in range(3)
    ]
    summary = summarize_decisions(decisions)
    total = get_group(summary)
    assert total["observed_targeting_rate"] == 0.25  # not mean(100%, 0%)=50%
    assert total["excess_percentage_points"] == -25
    assert total["games_with_logged_decisions"] == 2
    assert total["games_with_eligible_decisions"] == 2
    assert get_group(summary, actor="C")["observed_targeting_rate"] == 1
    assert get_group(summary, actor="D")["observed_targeting_rate"] == 0
    for measure in ("primary", "secondary"):
        groups = [get_group(summary, measure, actor) for actor in ("C", "D")]
        assert sum(group["eligible_decisions"] for group in groups) == get_group(summary, measure)["eligible_decisions"]


def test_zero_eligible_is_missing_rate_not_zero_and_measures_stay_separate():
    decision = classified(make_row(actor="A", victims=("B", "C")))
    summary = summarize_decisions([decision])
    primary = get_group(summary)
    assert primary["eligible_decisions"] == 0
    assert primary["observed_targeting_rate"] is None
    assert primary["random_legal_action_baseline_rate"] is None
    assert primary["excess_percentage_points"] is None
    assert primary["games_with_logged_decisions"] == 1
    assert primary["games_with_eligible_decisions"] == 0
    assert primary["exclusion_reasons"] == {"actor_is_leader": 1}
    assert get_group(summary, "secondary")["observed_targeting_rate"] == 1
    assert "N/A" in format_summary(summary)
    assert "null" in json.dumps(summary, allow_nan=False)


def test_cli_formats_and_invalid_input(tmp_path):
    path = tmp_path / "decisions.csv"
    row = make_row()
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    original = path.read_bytes()
    command = [sys.executable, "-B", str(Path(__file__).with_name("summarize_leader_targeting.py")), str(path)]
    default = subprocess.run(command, capture_output=True, text=True, check=True)
    assert "100.00%" in default.stdout and "50.00%" in default.stdout
    assert "+50.00" in default.stdout
    structured = subprocess.run(command + ["--format", "json"], capture_output=True, text=True, check=True)
    assert get_group(json.loads(structured.stdout))["eligible_decisions"] == 1
    decisions = subprocess.run(command + ["--format", "decisions"], capture_output=True, text=True, check=True)
    assert json.loads(decisions.stdout)["primary"]["targeted_leader"] is True
    assert path.read_bytes() == original
    path.write_text("game_index,game_seed,decision_index\n")
    invalid = subprocess.run(command, capture_output=True, text=True)
    assert invalid.returncode == 2
    assert not invalid.stdout
    assert "no robber decisions" in invalid.stderr


def fixed_game_draws(monkeypatch, draws):
    """Make bootstrap samples explicit so their answers can be checked by hand."""
    class FixedRandom:
        def __init__(self, seed):
            self.draws = iter(draws)

        def choices(self, population, k):
            draw = next(self.draws)
            assert len(draw) == k
            assert all(index in population for index in draw)
            return draw

    monkeypatch.setattr("summarize_leader_targeting.random.Random", FixedRandom)


def test_bootstrap_resamples_games_and_recomputes_ratio(monkeypatch):
    # Game 0: one hit at p=.5. Game 1: three misses, each at p=.5.
    # Resampled excesses are +50, -25, -50 pp. Linear 95% bounds:
    # -50 + .05*25 = -48.75 and -25 + .95*75 = 46.25.
    decisions = [classified(make_row(), seed=0)] + [
        classified(make_row(chosen=1), seed=1) for _ in range(3)
    ]
    fixed_game_draws(monkeypatch, [[0, 0], [0, 1], [1, 1]])
    summary = summarize_decisions(decisions, bootstrap_samples=3)
    for row in summary["summaries"]:
        interval = row["excess_ci_95_pp"]
        assert interval["lower"] == pytest.approx(-48.75)
        assert interval["upper"] == pytest.approx(46.25)
        assert interval["valid_resamples"] == 3
        assert interval["undefined_resamples"] == 0
    assert get_group(summary)["excess_percentage_points"] == -25


def test_zero_eligible_games_remain_in_bootstrap_pool(monkeypatch):
    decisions = [classified(make_row(), seed=0), classified(make_row(chosen=1), seed=1)]
    # Game 2 has no robber records. It must still be sampled from inventory.
    fixed_game_draws(monkeypatch, [[2, 2, 2], [0, 0, 0], [1, 1, 1]])
    summary = summarize_decisions(decisions, game_seeds=[0, 1, 2], bootstrap_samples=3)
    interval = get_group(summary)["excess_ci_95_pp"]
    assert summary["bootstrap"]["game_count"] == 3
    assert interval["valid_resamples"] == 2
    assert interval["undefined_resamples"] == 1
    assert interval["lower"] == pytest.approx(-47.5)
    assert interval["upper"] == pytest.approx(47.5)
    assert "1 resamples omitted" in format_summary(summary)


def test_only_one_informative_game_has_no_interval():
    decisions = [classified(make_row(), seed=0), classified(make_row(victims=("A",)), seed=1)]
    summary = summarize_decisions(decisions, bootstrap_samples=100)
    interval = get_group(summary)["excess_ci_95_pp"]
    assert interval["status"] == "fewer_than_two_eligible_games"
    assert interval["lower"] is None and interval["upper"] is None
    assert interval["valid_resamples"] + interval["undefined_resamples"] == 100
    assert "fewer_than_two_eligible_games" in format_summary(summary)


def test_no_eligible_decisions_never_become_zero_excess():
    summary = summarize_decisions([classified(make_row(victims=("A",)))], bootstrap_samples=20)
    interval = get_group(summary)["excess_ci_95_pp"]
    assert interval["valid_resamples"] == 0
    assert interval["undefined_resamples"] == 20
    assert interval["lower"] is None and interval["upper"] is None


def test_bootstrap_reproducible_and_row_order_independent():
    decisions = [classified(make_row(chosen=i % 2), seed=i) for i in range(5)]
    first = summarize_decisions(decisions, bootstrap_samples=200, bootstrap_seed=9)
    second = summarize_decisions(list(reversed(decisions)), bootstrap_samples=200, bootstrap_seed=9)
    assert first == second
    assert first["bootstrap"]["seed"] == 9
    assert first["bootstrap"]["samples"] == 200


@pytest.mark.parametrize("options", [
    {"bootstrap_samples": 0}, {"bootstrap_samples": 1},
    {"bootstrap_seed": -1}, {"game_seeds": []}, {"game_seeds": [999]},
])
def test_invalid_bootstrap_configuration_rejected(options):
    with pytest.raises(ValueError):
        summarize_decisions([classified(make_row())], **options)


def test_inventory_includes_empty_games_and_checks_counts(tmp_path):
    path = tmp_path / "games.csv"
    decisions = [classified(make_row(), seed=0)]
    path.write_text("game_seed,robber_decision_count\n0,1\n1,0\n")
    assert load_game_seeds(path, decisions) == [0, 1]
    path.write_text("game_seed,robber_decision_count\n0,2\n")
    with pytest.raises(ValueError, match="count disagrees"):
        load_game_seeds(path, decisions)
    path.write_text("game_seed,robber_decision_count\n1,0\n")
    with pytest.raises(ValueError, match="omits games"):
        load_game_seeds(path, decisions)
    path.write_text("game_seed,robber_decision_count\n0,1\n0,1\n")
    with pytest.raises(ValueError, match="Duplicate game_seed"):
        load_game_seeds(path, decisions)


def test_cli_bootstrap_options_and_inventory_discovery(tmp_path):
    path = tmp_path / "robber_decisions.csv"
    row = make_row()
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    games_path = tmp_path / "games.csv"
    games_path.write_text("game_seed,robber_decision_count\n10000,1\n10001,0\n")
    original = (path.read_bytes(), games_path.read_bytes())
    command = [sys.executable, "-B", str(Path(__file__).with_name("summarize_leader_targeting.py")),
               str(path), "--format", "json", "--bootstrap-samples", "25", "--bootstrap-seed", "42"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    summary = json.loads(result.stdout)
    assert summary["bootstrap"]["samples"] == 25
    assert summary["bootstrap"]["seed"] == 42
    assert summary["bootstrap"]["game_count"] == 2
    assert summary["bootstrap"]["inventory_source"] == str(games_path.resolve())
    assert (path.read_bytes(), games_path.read_bytes()) == original
    invalid = subprocess.run(command + ["--bootstrap-samples", "0"], capture_output=True, text=True)
    assert invalid.returncode == 2 and not invalid.stdout
    games_path.write_text("game_seed,robber_decision_count\n10000,2\n")
    invalid = subprocess.run(command, capture_output=True, text=True)
    assert invalid.returncode == 2 and not invalid.stdout
    assert "count disagrees" in invalid.stderr
