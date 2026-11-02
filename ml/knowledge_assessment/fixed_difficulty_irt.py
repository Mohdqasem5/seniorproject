"""Configured fixed-difficulty 1PL assessment; no shared parameter training.

All difficulty values and evidence rules are provisional. Ability is in logits.
This module uses only the Python standard library and never edits question data.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import datetime, timezone

MODEL_VERSION = "fixed-1pl-map-v1"
LABELS = ("Easy", "Medium", "Hard")
DEFAULT_MAPPING = {"Easy": -1.0, "Medium": 0.0, "Hard": 1.0}
DEFAULT_PRIOR = {"mean": 0.0, "sd": 1.0}
# Inherited Section 14/23 reporting criteria, not newly approved requirements.
DEFAULT_ASSESSMENT = {
    "min_attempts": 10, "min_category_games": 3,
    "smoothing_alpha": 5.0, "smoothing_beta": 5.0,
}


def _identifier(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{field} must be an integer or nonempty string")
    value = str(value).strip()
    if not value:
        raise ValueError(f"Missing {field}")
    return value


def _timestamp(value, field):
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a timezone-aware ISO timestamp")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"Invalid {field}") from error
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{field} requires an explicit timezone")
    return result.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def normalize_configuration(difficulty_mapping=None, prior_config=None):
    source = DEFAULT_MAPPING if difficulty_mapping is None else difficulty_mapping
    if not isinstance(source, dict) or set(source) != set(LABELS):
        raise ValueError("Mapping must contain exactly Easy, Medium and Hard")
    mapping = {label: _number(source[label], "difficulty") for label in LABELS}
    if not mapping["Easy"] < mapping["Medium"] < mapping["Hard"]:
        raise ValueError("Mapping must preserve Easy < Medium < Hard")
    prior_source = DEFAULT_PRIOR if prior_config is None else prior_config
    if not isinstance(prior_source, dict) or set(prior_source) != {"mean", "sd"}:
        raise ValueError("Prior requires mean and sd")
    prior = {key: _number(prior_source[key], "prior " + key) for key in ["mean", "sd"]}
    variance = prior["sd"] * prior["sd"]
    if prior["sd"] <= 0 or not math.isfinite(variance) or variance == 0:
        raise ValueError("Prior variance must be positive and numerically finite")
    if not math.isfinite(1 / variance):
        raise ValueError("Prior precision is not numerically finite")
    return mapping, prior


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def sigmoid(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Nonfinite logistic argument")
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1 + exponential)


def fit_ability_counts(counts, difficulty_mapping=None, prior_config=None):
    """Exact 1D MAP up to solver tolerance, with a fixed Gaussian prior.

    Three binomial counts are sufficient statistics when difficulties are fixed.
    The returned SE is a conditional Laplace approximation, not full uncertainty.
    """
    mapping, prior = normalize_configuration(difficulty_mapping, prior_config)
    if set(counts) != set(LABELS):
        raise ValueError("Counts must cover all three difficulty labels")
    for label in LABELS:
        row = counts[label]
        if set(row) != {"attempts", "correct"}:
            raise ValueError("Invalid count schema")
        if any(isinstance(row[k], bool) or not isinstance(row[k], int) for k in row):
            raise ValueError("Counts must be integers")
        if not 0 <= row["correct"] <= row["attempts"]:
            raise ValueError("Inconsistent counts")
    n = sum(counts[label]["attempts"] for label in LABELS)
    variance = prior["sd"] ** 2
    prior_precision = 1 / variance

    def score(theta):
        probabilities = {label: sigmoid(theta - mapping[label]) for label in LABELS}
        gradient = (theta - prior["mean"]) * prior_precision + math.fsum(
            counts[label]["attempts"] * probabilities[label] - counts[label]["correct"]
            for label in LABELS)
        curvature = prior_precision + math.fsum(
            counts[label]["attempts"] * probabilities[label] * (1 - probabilities[label])
            for label in LABELS)
        return gradient, curvature

    theta = prior["mean"]
    if n:
        radius = variance * n + 1.0
        lower, upper = theta - radius, theta + radius
        if not all(map(math.isfinite, [lower, upper])) or lower == upper:
            raise ValueError("Numerically unsupported prior/history scale")
        for _ in range(100):
            gradient, curvature = score(theta)
            if abs(gradient) <= 1e-9:
                break
            if gradient > 0:
                upper = theta
            else:
                lower = theta
            candidate = theta - gradient / curvature
            if not lower < candidate < upper:
                candidate = (lower + upper) / 2
            theta = candidate
        else:
            raise ArithmeticError("Ability solver did not converge")
    gradient, precision = score(theta)
    if not math.isfinite(theta) or abs(gradient) > 1e-7:
        raise ArithmeticError("Invalid MAP solution")
    return {"theta": float(theta), "conditional_se": float(1 / math.sqrt(precision)),
            "precision": float(precision), "score_residual": float(abs(gradient))}


def new_user_state(user_id, difficulty_mapping=None, prior_config=None):
    mapping, prior = normalize_configuration(difficulty_mapping, prior_config)
    return {"model_version": MODEL_VERSION, "user_id": _identifier(user_id, "user_id"),
            "difficulty_mapping": mapping, "prior_config": prior,
            "configuration_sha256": _hash({"mapping": mapping, "prior": prior}),
            "categories": {}, "attempt_ledger": {}, "game_questions": {},
            "question_versions": {}, "last_feedback_at": None, "applied_count": 0}


def normalize_answer(answer):
    if not isinstance(answer, dict):
        raise ValueError("An answer must be a dictionary")
    required = ["user_id", "attempt_id", "game_id", "question_id", "category_id"]
    result = {key: _identifier(answer.get(key), key) for key in required}
    name = answer.get("category_name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Missing category_name")
    result["category_name"] = name.strip()
    if answer.get("difficulty") not in LABELS:
        raise ValueError("Unknown or missing manual difficulty")
    result["difficulty"] = answer["difficulty"]
    y = answer.get("answered_correctly")
    if not isinstance(y, (int, bool)) or y not in (0, 1):
        raise ValueError("Correctness must be binary, not a score or prediction")
    result["answered_correctly"] = int(y)
    result["answered_at"] = _timestamp(answer.get("answered_at"), "answered_at")
    result["feedback_available_at"] = _timestamp(
        answer.get("feedback_available_at", answer.get("answered_at")), "feedback_available_at")
    if result["feedback_available_at"] < result["answered_at"]:
        raise ValueError("Feedback cannot precede the answer")
    status = answer.get("game_status")
    if status not in {"COMPLETED", "ABANDONED", "IN_PROGRESS", "INCOMPLETE"}:
        raise ValueError("Unknown game status")
    result["game_status"] = status
    result["question_version"] = _identifier(answer.get("question_version", "unversioned"),
                                                 "question_version")
    return result


def predict_user_response(state, category_id, difficulty, difficulty_mapping=None):
    mapping, prior = normalize_configuration(
        state["difficulty_mapping"] if difficulty_mapping is None else difficulty_mapping,
        state["prior_config"])
    if _hash({"mapping": mapping, "prior": prior}) != state["configuration_sha256"]:
        raise ValueError("Mapping changed: reconstruct history under a new version")
    if difficulty not in LABELS:
        raise ValueError("Unknown difficulty")
    category = state["categories"].get(_identifier(category_id, "category_id"))
    theta = prior["mean"] if category is None else category["theta"]
    return sigmoid(theta - mapping[difficulty])


def update_user_ability(current_state, answer, difficulty_mapping=None):
    """Apply one already released, validated completed-game answer exactly once.

    Identical retries return an unchanged copy, including retries older than the
    watermark. Conflicting retries or new out-of-order events are rejected.
    Caller must persist the entire state atomically; this function has no I/O.
    """
    event = normalize_answer(answer)
    if event["game_status"] != "COMPLETED":
        raise ValueError("Only completed-game feedback is accepted by this updater")
    mapping, prior = normalize_configuration(
        current_state["difficulty_mapping"] if difficulty_mapping is None else difficulty_mapping,
        current_state["prior_config"])
    if current_state.get("model_version") != MODEL_VERSION:
        raise ValueError("Unsupported state version")
    if _hash({"mapping": mapping, "prior": prior}) != current_state["configuration_sha256"]:
        raise ValueError("Mapping changed: reconstruct from stored history")
    if event["user_id"] != current_state["user_id"]:
        raise ValueError("State/answer user mismatch")
    signature = _hash(event)
    old_signature = current_state["attempt_ledger"].get(event["attempt_id"])
    if old_signature is not None:
        if signature != old_signature:
            raise ValueError("Conflicting payload for an existing attempt_id")
        return copy.deepcopy(current_state)
    if (current_state["last_feedback_at"] is not None and
            event["feedback_available_at"] < current_state["last_feedback_at"]):
        raise ValueError("Late new event: reconstruct from chronological history")
    game_question = json.dumps([event["game_id"], event["question_id"]], separators=(",", ":"))
    if game_question in current_state["game_questions"]:
        raise ValueError("A question cannot be answered twice in one game")
    question_version = json.dumps([event["question_id"], event["question_version"]], separators=(",", ":"))
    metadata = {key: event[key] for key in ["category_id", "difficulty"]}
    old_metadata = current_state["question_versions"].get(question_version)
    if old_metadata is not None and old_metadata != metadata:
        raise ValueError("Question metadata changed without a new question_version")
    state = copy.deepcopy(current_state)
    category = state["categories"].get(event["category_id"])
    if category is None:
        category = {"category_name": event["category_name"],
                    "counts": {label: {"attempts": 0, "correct": 0} for label in LABELS}}
        state["categories"][event["category_id"]] = category
    if category["category_name"] != event["category_name"]:
        raise ValueError("Category name conflicts with its identifier")
    counts = category["counts"][event["difficulty"]]
    counts["attempts"] += 1
    counts["correct"] += event["answered_correctly"]
    category.update(fit_ability_counts(category["counts"], mapping, prior))
    state["attempt_ledger"][event["attempt_id"]] = signature
    state["game_questions"][game_question] = event["attempt_id"]
    state["question_versions"][question_version] = metadata
    state["last_feedback_at"] = event["feedback_available_at"]
    state["applied_count"] += 1
    return state


def estimate_user_ability(interactions, difficulty_mapping=None, prior_config=None, *, user_id=None):
    events = [normalize_answer(answer) for answer in interactions]
    if user_id is None:
        if not events:
            raise ValueError("user_id is required for an empty history")
        user_id = events[0]["user_id"]
    state = new_user_state(user_id, difficulty_mapping, prior_config)
    events.sort(key=lambda event: (event["feedback_available_at"], event["answered_at"], event["attempt_id"]))
    for event in events:
        state = update_user_ability(state, event)
    return state


def trace_user_ability(interactions, difficulty_mapping=None, prior_config=None, *, user_id=None):
    """Replay released feedback chronologically; freeze predictions within a block.

    Same feedback_available_at means simultaneous feedback. All probabilities in
    that block use the pre-block state, even when multiple categories are present.
    """
    events = [normalize_answer(answer) for answer in interactions]
    if user_id is None:
        if not events:
            raise ValueError("user_id is required for an empty history")
        user_id = events[0]["user_id"]
    events.sort(key=lambda event: (event["feedback_available_at"], event["answered_at"], event["attempt_id"]))
    state = new_user_state(user_id, difficulty_mapping, prior_config)
    trace, index = [], 0
    while index < len(events):
        end = index + 1
        while end < len(events) and events[end]["feedback_available_at"] == events[index]["feedback_available_at"]:
            end += 1
        before = copy.deepcopy(state)
        rows = []
        for event in events[index:end]:
            prediction = predict_user_response(before, event["category_id"], event["difficulty"])
            old_n = state["applied_count"]
            state = update_user_ability(state, event)
            if state["applied_count"] == old_n:
                continue
            previous = before["categories"].get(event["category_id"], {})
            rows.append({"attempt_id": event["attempt_id"], "category_id": event["category_id"],
                         "difficulty": event["difficulty"], "answered_correctly": event["answered_correctly"],
                         "feedback_available_at": event["feedback_available_at"],
                         "predicted_correctness_before_feedback": prediction,
                         "theta_before_feedback": previous.get("theta", before["prior_config"]["mean"])})
        for row in rows:
            row["theta_after_feedback_block"] = state["categories"][row["category_id"]]["theta"]
        trace.extend(rows)
        index = end
    return {"state": state, "trace": trace}


def analyze_fixed_historical_knowledge(user_id, completed_games, interactions, configuration=None):
    """Completed-game retrospective assessment. No HTTP or database writes.

    Short games are excluded under the existing ten-distinct-question rule.
    Exact duplicate attempts are deduplicated before game completeness is tested.
    All IDs in the JSON output are normalized strings.
    """
    configuration = {} if configuration is None else copy.deepcopy(configuration)
    allowed = set(DEFAULT_ASSESSMENT) | {"difficulty_mapping", "prior_config", "data_context"}
    if not isinstance(configuration, dict) or set(configuration) - allowed:
        raise ValueError("Unknown assessment configuration")
    mapping, prior = normalize_configuration(configuration.get("difficulty_mapping"),
                                             configuration.get("prior_config"))
    rules = {key: configuration.get(key, value) for key, value in DEFAULT_ASSESSMENT.items()}
    for key in ["min_attempts", "min_category_games"]:
        if isinstance(rules[key], bool) or not isinstance(rules[key], int) or rules[key] < 1:
            raise ValueError("Evidence minima must be positive integers")
    for key in ["smoothing_alpha", "smoothing_beta"]:
        rules[key] = _number(rules[key], key)
        if rules[key] <= 0:
            raise ValueError("Smoothing prior parameters must be positive")
    if not isinstance(completed_games, (list, tuple)) or not isinstance(interactions, (list, tuple)):
        raise ValueError("Game and interaction records are required, not an integer game count")
    owner = _identifier(user_id, "user_id")
    games = {}
    for raw in completed_games:
        if not isinstance(raw, dict) or _identifier(raw.get("user_id"), "user_id") != owner:
            if not isinstance(raw, dict):
                raise ValueError("Game must be a dictionary")
            continue
        gid = _identifier(raw.get("game_id"), "game_id")
        if gid in games:
            raise ValueError("Duplicate game record")
        status = raw.get("status")
        if status not in {"COMPLETED", "ABANDONED", "IN_PROGRESS", "INCOMPLETE"}:
            raise ValueError("Invalid game status")
        completion = raw.get("completed_at")
        if status == "COMPLETED" or completion is not None:
            completion = _timestamp(completion, "completed_at")
        declared = raw.get("question_count")
        if declared is not None and (isinstance(declared, bool) or not isinstance(declared, int) or declared < 0):
            raise ValueError("Invalid declared question count")
        games[gid] = {"status": status, "completed_at": completion, "declared_count": declared}
    unique_events = {}
    category_names = {}
    per_game = {gid: [] for gid in games}
    for raw in interactions:
        if not isinstance(raw, dict):
            raise ValueError("Interaction must be a dictionary")
        if _identifier(raw.get("user_id"), "user_id") != owner:
            continue
        gid = _identifier(raw.get("game_id"), "game_id")
        if gid not in games:
            raise ValueError("Interaction references an unknown/foreign game")
        game = games[gid]
        enriched = dict(raw)
        if "game_status" in raw and raw["game_status"] != game["status"]:
            raise ValueError("Answer/game status conflict")
        enriched["game_status"] = game["status"]
        if "feedback_available_at" not in enriched and game["status"] == "COMPLETED":
            enriched["feedback_available_at"] = game["completed_at"]
        event = normalize_answer(enriched)
        if game["status"] == "COMPLETED" and event["answered_at"] > game["completed_at"]:
            raise ValueError("Answer occurs after game completion")
        old = unique_events.get(event["attempt_id"])
        if old is not None:
            if old != event:
                raise ValueError("Conflicting duplicate interaction")
            continue
        old_name = category_names.setdefault(event["category_id"], event["category_name"])
        if old_name != event["category_name"]:
            raise ValueError("Inconsistent category names")
        unique_events[event["attempt_id"]] = event
        per_game[gid].append(event)
    valid_games = set()
    for gid, game in games.items():
        answers = per_game[gid]
        unique_questions = {event["question_id"] for event in answers}
        if len(unique_questions) != len(answers):
            raise ValueError("Duplicate question within a game")
        if game["declared_count"] is not None and game["declared_count"] != len(answers):
            raise ValueError("Declared count conflicts with supplied game history")
        if game["status"] == "COMPLETED" and len(answers) == 10:
            valid_games.add(gid)
    output = {"user_id": owner, "eligible": len(valid_games) >= 5,
              "completed_games": len(valid_games), "games_remaining": max(0, 5 - len(valid_games)),
              "model_version": MODEL_VERSION,
              "configuration_sha256": _hash({"mapping": mapping, "prior": prior}),
              "model_type": "configured_fixed_1pl_map",
              "shared_parameters_fitted": False, "mapping_empirically_calibrated": False,
              "data_context": configuration.get("data_context", "caller_supplied_labels_not_independently_verified"),
              "knowledge_assessment": []}
    if not output["eligible"]:
        return output
    accepted = [event for event in unique_events.values() if event["game_id"] in valid_games]
    state = estimate_user_ability(accepted, mapping, prior, user_id=owner)
    for cid, category in sorted(state["categories"].items()):
        history = [event for event in accepted if event["category_id"] == cid]
        n = sum(row["attempts"] for row in category["counts"].values())
        c = sum(row["correct"] for row in category["counts"].values())
        category_games = len({event["game_id"] for event in history})
        sufficient = n >= rules["min_attempts"] and category_games >= rules["min_category_games"]
        output["knowledge_assessment"].append({
            "category_id": cid, "category_name": category["category_name"],
            "total_attempts": n, "correct_answers": c, "accuracy_percent": 100 * c / n,
            "smoothed_accuracy": (c + rules["smoothing_alpha"]) /
                                 (n + rules["smoothing_alpha"] + rules["smoothing_beta"]),
            "smoothed_accuracy_units": "fraction; descriptive statistic, not mastery",
            "estimated_irt_ability": category["theta"] if sufficient else None,
            "ability_units": "logits on the configured provisional scale",
            "completed_games": len(valid_games), "category_games": category_games,
            "assessment_status": "AVAILABLE_PROVISIONAL_MAPPING" if sufficient else "INSUFFICIENT_EVIDENCE",
            "classification_status": "STRENGTH_WEAKNESS_THRESHOLDS_NOT_VALIDATED",
            "evidence": {"sufficient_for_inherited_configuration": sufficient,
                         "min_attempts": rules["min_attempts"], "min_category_games": rules["min_category_games"],
                         "conditional_ability_se": category["conditional_se"],
                         "conditional_uncertainty_ignores_mapping_error": True,
                         "repeat_question_attempts": n - len({event["question_id"] for event in history}),
                         "difficulty_counts": copy.deepcopy(category["counts"])}})
    return output


def serialize_state(state):
    return json.dumps(state, sort_keys=True, allow_nan=False)


def load_state(serialized):
    state = json.loads(serialized)
    mapping, prior = normalize_configuration(state["difficulty_mapping"], state["prior_config"])
    if state["model_version"] != MODEL_VERSION or state["configuration_sha256"] != _hash({"mapping": mapping, "prior": prior}):
        raise ValueError("State configuration/version mismatch")
    n = 0
    for category in state["categories"].values():
        fitted = fit_ability_counts(category["counts"], mapping, prior)
        for key in fitted:
            if not math.isclose(fitted[key], category[key], rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError("Stored ability is inconsistent with response counts")
        n += sum(row["attempts"] for row in category["counts"].values())
    if n != state["applied_count"] or n != len(state["attempt_ledger"]) or n != len(state["game_questions"]):
        raise ValueError("State counts/ledger mismatch")
    return state


# Public assessment name; retain the verified name for compatibility.
analyze_historical_knowledge = analyze_fixed_historical_knowledge
