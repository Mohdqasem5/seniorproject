"""Functional tests on explicitly synthetic Rihla records, not validation data."""
import copy
import json
import math
import random
import unittest
from datetime import datetime, timedelta, timezone

import fixed_difficulty_irt as fd


def synthetic_fixture():
    rng = random.Random(20261101)
    games, answers = [], []
    for game in range(1, 6):
        completion = datetime(2026, 1, game, 12, tzinfo=timezone.utc)
        games.append({"user_id": 1, "game_id": game, "status": "COMPLETED",
                      "completed_at": completion.isoformat()})
        for question in range(1, 11):
            category = 2 if question <= 5 else 3
            label = fd.LABELS[(question - 1) % 3]
            ability = -0.5 if category == 2 else 0.6
            answers.append({"user_id": 1, "game_id": game, "attempt_id": 100 * game + question,
                            "question_id": question, "question_version": "admin-v1",
                            "category_id": category, "category_name": "Wars" if category == 2 else "Politics",
                            "difficulty": label,
                            "answered_correctly": int(rng.random() < fd.sigmoid(ability - fd.DEFAULT_MAPPING[label])),
                            "answered_at": (completion - timedelta(minutes=20 - question)).isoformat(),
                            "feedback_available_at": completion.isoformat(), "game_status": "COMPLETED"})
    return games, answers


class FixedDifficultyTests(unittest.TestCase):
    def setUp(self):
        self.games, self.answers = synthetic_fixture()
        self.mapping = copy.deepcopy(fd.DEFAULT_MAPPING)
        self.prior = copy.deepcopy(fd.DEFAULT_PRIOR)
        self.context = {"data_context": "synthetic_functional_demo"}

    def one(self, label="Medium", correct=1):
        return dict(self.answers[0], difficulty=label, answered_correctly=correct)

    def theta(self, answer):
        return fd.estimate_user_ability([answer], self.mapping, self.prior)["categories"]["2"]["theta"]

    def test_mapping_is_explicit_ordered_and_provisional(self):
        mapping, prior = fd.normalize_configuration()
        self.assertEqual(mapping, {"Easy": -1.0, "Medium": 0.0, "Hard": 1.0})
        self.assertEqual(prior, {"mean": 0.0, "sd": 1.0})
        self.assertAlmostEqual(fd.sigmoid(0), 0.5)
        self.assertGreater(fd.sigmoid(1), fd.sigmoid(-1))

    def test_invalid_mapping_and_prior_rejected(self):
        for bad in [{"Easy": 0}, {"Easy": 1, "Medium": 0, "Hard": -1},
                    {"Easy": -1, "Medium": 0, "Hard": float("nan")},
                    {"Easy": False, "Medium": 0, "Hard": 1}]:
            with self.assertRaises(ValueError):
                fd.normalize_configuration(bad)
        for bad in [{"mean": 0, "sd": 0}, {"mean": 0, "sd": -1},
                    {"mean": 0, "sd": 1e308}, {"sd": 1}]:
            with self.assertRaises(ValueError):
                fd.normalize_configuration(prior_config=bad)

    def test_invalid_or_missing_difficulty_rejected(self):
        for value in [None, "easy", "Expert", 2]:
            with self.assertRaises(ValueError):
                self.theta(dict(self.answers[0], difficulty=value))

    def test_correct_and_incorrect_update_directions(self):
        self.assertGreater(self.theta(self.one(correct=1)), 0)
        self.assertLess(self.theta(self.one(correct=0)), 0)

    def test_hard_correct_and_easy_incorrect_have_larger_effects(self):
        self.assertGreater(self.theta(self.one("Hard", 1)), self.theta(self.one("Easy", 1)))
        self.assertLess(self.theta(self.one("Easy", 0)), self.theta(self.one("Hard", 0)))

    def test_new_user_and_sparse_history_shrinkage(self):
        state = fd.estimate_user_ability([], user_id=1)
        self.assertEqual(state["applied_count"], 0)
        self.assertEqual(fd.predict_user_response(state, 2, "Medium"), 0.5)
        sparse = fd.estimate_user_ability([self.one()])["categories"]["2"]
        self.assertLess(abs(sparse["theta"]), 1)
        self.assertGreater(sparse["conditional_se"], 0.8)

    def test_input_and_difficulty_mapping_are_immutable(self):
        answer = self.one()
        before = copy.deepcopy((answer, self.mapping, self.prior))
        state = fd.new_user_state(1, self.mapping, self.prior)
        state_before = copy.deepcopy(state)
        fd.update_user_ability(state, answer, self.mapping)
        self.assertEqual((answer, self.mapping, self.prior), before)
        self.assertEqual(state, state_before)

    def test_identical_retry_is_idempotent_including_late_retry(self):
        state = fd.estimate_user_ability(self.answers)
        duplicate = fd.update_user_ability(state, self.answers[0])
        self.assertEqual(duplicate, state)
        self.assertEqual(duplicate["applied_count"], 50)

    def test_conflicting_retry_is_rejected_without_mutation(self):
        state = fd.estimate_user_ability(self.answers)
        before = copy.deepcopy(state)
        with self.assertRaises(ValueError):
            fd.update_user_ability(state, dict(self.answers[0], answered_correctly=1 - self.answers[0]["answered_correctly"]))
        self.assertEqual(state, before)

    def test_chronological_reconstruction_matches_online_updates(self):
        state = fd.new_user_state(1)
        for answer in self.answers:
            state = fd.update_user_ability(state, answer)
        shuffled = list(self.answers)
        random.Random(7).shuffle(shuffled)
        self.assertEqual(state, fd.estimate_user_ability(shuffled))

    def test_late_new_answer_rejected(self):
        state = fd.estimate_user_ability(self.answers[-10:])
        with self.assertRaises(ValueError):
            fd.update_user_ability(state, self.answers[0])

    def test_current_and_future_label_independence(self):
        original = fd.trace_user_ability(self.answers)
        altered = [dict(a, answered_correctly=1 - a["answered_correctly"])
                   if a["game_id"] >= 3 else a for a in self.answers]
        changed = fd.trace_user_ability(altered)
        # Predictions for all answers through game 3 must remain unchanged, even
        # though every answer in game 3 and all subsequent outcomes were flipped.
        before = [r["predicted_correctness_before_feedback"] for r in original["trace"][:30]]
        after = [r["predicted_correctness_before_feedback"] for r in changed["trace"][:30]]
        self.assertEqual(before, after)
        self.assertEqual(original["state"], fd.estimate_user_ability(self.answers))
        for game in range(5):
            rows = original["trace"][10 * game:10 * game + 10]
            for category in ["2", "3"]:
                self.assertEqual(len({r["theta_before_feedback"] for r in rows if r["category_id"] == category}), 1)

    def test_categories_are_estimated_separately(self):
        state = fd.estimate_user_ability(self.answers)
        wars_only = fd.estimate_user_ability([a for a in self.answers if a["category_id"] == 2])
        self.assertEqual(state["categories"]["2"], wars_only["categories"]["2"])
        altered = [dict(a, answered_correctly=1 - a["answered_correctly"]) if a["category_id"] == 3 else a for a in self.answers]
        self.assertEqual(state["categories"]["2"], fd.estimate_user_ability(altered)["categories"]["2"])

    def test_extreme_histories_and_score_residuals_are_finite(self):
        for y in [0, 1]:
            counts = {label: {"attempts": 1000, "correct": 1000 * y} for label in fd.LABELS}
            result = fd.fit_ability_counts(counts)
            self.assertTrue(all(math.isfinite(v) for v in result.values()))
            self.assertLess(result["score_residual"], 1e-7)
        self.assertEqual(fd.sigmoid(-1000), 0)
        self.assertEqual(fd.sigmoid(1000), 1)

    def test_mapping_change_requires_explicit_reconstruction(self):
        state = fd.estimate_user_ability(self.answers)
        other = {"Easy": -0.5, "Medium": 0, "Hard": 0.5}
        with self.assertRaises(ValueError):
            fd.update_user_ability(state, self.answers[0], other)
        reconstructed = fd.estimate_user_ability(self.answers, other)
        self.assertNotEqual(state["configuration_sha256"], reconstructed["configuration_sha256"])

    def test_question_edits_require_a_new_version(self):
        state = fd.estimate_user_ability(self.answers[:10])
        changed = dict(self.answers[10], difficulty="Hard")
        with self.assertRaises(ValueError):
            fd.update_user_ability(state, changed)
        changed["question_version"] = "admin-v2"
        self.assertEqual(fd.update_user_ability(state, changed)["applied_count"], 11)

    def test_five_completed_ten_question_games(self):
        output = fd.analyze_fixed_historical_knowledge(1, self.games, self.answers, self.context)
        self.assertTrue(output["eligible"])
        self.assertEqual(output["completed_games"], 5)
        self.assertFalse(output["shared_parameters_fitted"])
        self.assertEqual(output["configuration_sha256"], fd.new_user_state(1)["configuration_sha256"])
        for row in output["knowledge_assessment"]:
            self.assertEqual(row["total_attempts"], 25)
            matching = [a for a in self.answers if str(a["category_id"]) == row["category_id"]]
            self.assertEqual(row["correct_answers"], sum(a["answered_correctly"] for a in matching))
            self.assertAlmostEqual(row["accuracy_percent"], row["correct_answers"] * 4)
            self.assertAlmostEqual(row["smoothed_accuracy"], (row["correct_answers"] + 5) / 35)
            self.assertIsNotNone(row["estimated_irt_ability"])
            self.assertEqual(row["classification_status"], "STRENGTH_WEAKNESS_THRESHOLDS_NOT_VALIDATED")

    def test_four_games_hidden(self):
        output = fd.analyze_fixed_historical_knowledge(1, self.games[:4], self.answers[:40])
        self.assertFalse(output["eligible"])
        self.assertEqual(output["knowledge_assessment"], [])

    def test_abandoned_incomplete_and_short_games_do_not_count(self):
        for status in ["ABANDONED", "INCOMPLETE", "IN_PROGRESS"]:
            games = [dict(g, status=status) if g["game_id"] == 5 else g for g in self.games]
            answers = [dict(a, game_status=status) if a["game_id"] == 5 else a for a in self.answers]
            self.assertFalse(fd.analyze_fixed_historical_knowledge(1, games, answers)["eligible"])
        self.assertFalse(fd.analyze_fixed_historical_knowledge(1, self.games, self.answers[:-1])["eligible"])

    def test_duplicate_interaction_does_not_inflate_game_or_category_counts(self):
        actual = fd.analyze_fixed_historical_knowledge(1, self.games, self.answers + [self.answers[0]])
        expected = fd.analyze_fixed_historical_knowledge(1, self.games, self.answers)
        self.assertEqual(actual, expected)
        with self.assertRaises(ValueError):
            fd.analyze_fixed_historical_knowledge(1, self.games,
                self.answers + [dict(self.answers[0], answered_correctly=1 - self.answers[0]["answered_correctly"])])

    def test_sparse_category_is_not_mistaken_for_overall_eligibility(self):
        # New question/version because moving an existing item to a different category
        # without a version change should be rejected by the inference ledger.
        sparse = [dict(a, question_id=9999, category_id=99, category_name="Synthetic sparse")
                  if a["attempt_id"] == 101 else a for a in self.answers]
        output = fd.analyze_fixed_historical_knowledge(1, self.games, sparse)
        row = next(r for r in output["knowledge_assessment"] if r["category_id"] == "99")
        self.assertTrue(output["eligible"])
        self.assertEqual(row["assessment_status"], "INSUFFICIENT_EVIDENCE")
        self.assertIsNone(row["estimated_irt_ability"])

    def test_invalid_games_answers_and_foreign_users(self):
        with self.assertRaises(ValueError):
            fd.analyze_fixed_historical_knowledge(1, 5, self.answers)
        for games in [self.games + [self.games[0]],
                      [dict(self.games[0], status="UNKNOWN")] + self.games[1:],
                      [dict(self.games[0], completed_at=None)] + self.games[1:]]:
            with self.assertRaises(ValueError):
                fd.analyze_fixed_historical_knowledge(1, games, self.answers)
        for answer in [dict(self.answers[0], answered_correctly=2),
                       dict(self.answers[0], answered_at="2026-01-01T11:00:00"),
                       dict(self.answers[0], answered_at="2027-01-01T11:00:00Z"),
                       dict(self.answers[0], game_id=999),
                       dict(self.answers[0], game_status="ABANDONED")]:
            with self.assertRaises(ValueError):
                fd.analyze_fixed_historical_knowledge(1, self.games, [answer] + self.answers[1:])
        self.assertFalse(fd.analyze_fixed_historical_knowledge(2, self.games, self.answers)["eligible"])

    def test_repeated_questions_across_games_are_distinct_observations(self):
        state = fd.estimate_user_ability(self.answers)
        self.assertEqual(state["applied_count"], 50)
        output = fd.analyze_fixed_historical_knowledge(1, self.games, self.answers)
        self.assertTrue(all(row["evidence"]["repeat_question_attempts"] == 20 for row in output["knowledge_assessment"]))
        with self.assertRaises(ValueError):
            fd.estimate_user_ability(self.answers[:10] + [dict(self.answers[0], attempt_id=9999)])

    def test_json_state_reload_and_retry_consistency(self):
        state = fd.estimate_user_ability(self.answers[:40])
        loaded = fd.load_state(fd.serialize_state(state))
        self.assertEqual(loaded, state)
        for answer in self.answers[40:]:
            state = fd.update_user_ability(state, answer)
            loaded = fd.update_user_ability(loaded, answer)
        self.assertEqual(loaded, state)
        self.assertEqual(state, fd.estimate_user_ability(self.answers))
        self.assertEqual(fd.update_user_ability(loaded, self.answers[0]), state)
        corrupt = copy.deepcopy(state)
        corrupt["categories"]["2"]["theta"] += 1
        with self.assertRaises(ValueError):
            fd.load_state(json.dumps(corrupt))


if __name__ == "__main__":
    unittest.main(verbosity=2)
