import pytest

pytest.importorskip("torch")

from modules import prompt_parser


def test_plain_text_keeps_the_default_weight():
    assert prompt_parser.parse_prompt_attention("normal text") == [["normal text", 1.0]]


def test_empty_prompt():
    assert prompt_parser.parse_prompt_attention("") == [["", 1.0]]
    assert prompt_parser.check_prompt_syntax("") == []


def test_round_brackets_increase_the_weight():
    assert prompt_parser.parse_prompt_attention("an (important) word") == [["an ", 1.0], ["important", 1.1], [" word", 1.0]]


def test_escaped_brackets_are_literal_characters():
    assert prompt_parser.parse_prompt_attention(r"\(literal\)") == [["(literal)", 1.0]]


@pytest.mark.parametrize("weight, expected", [
    ("1.5", 1.5),
    (".5", 0.5),
    ("+0.5", 0.5),
    ("-0.5", -0.5),
    ("2.", 2.0),
])
def test_valid_attention_weights(weight, expected):
    assert prompt_parser.parse_prompt_attention(f"(a:{weight})") == [["a", expected]]


@pytest.mark.parametrize("weight", [".", "1.2.3", "..5", "+.", "-.", "1..2"])
def test_malformed_attention_weights_do_not_raise(weight):
    # these used to reach float() and raise ValueError, which aborted the whole generation;
    # a malformed weight is now treated as plain text instead
    assert prompt_parser.parse_prompt_attention(f"a (cat:{weight})") == [["a ", 1.0], [f"cat:{weight}", 1.1]]


@pytest.mark.parametrize("weight", [".", "1.2.3", "..5"])
def test_malformed_attention_weights_are_reported(weight):
    assert prompt_parser.check_prompt_syntax(f"a (cat:{weight})") == [f"invalid weight '{weight}'"]


def test_warnings_argument_does_not_change_the_parsing():
    prompt = "a (cat:1.2) and [a dog:0.5]"
    assert prompt_parser.parse_prompt_attention(prompt, []) == prompt_parser.parse_prompt_attention(prompt)


@pytest.mark.parametrize("prompt", [
    "plain text",
    "a (cat:1.2) in a [box]",
    "an ((important)) thing",
    "[a|b] and (c)",
    r"\(not a bracket\) and \[not one either\]",
])
def test_balanced_prompts_have_no_warnings(prompt):
    assert prompt_parser.check_prompt_syntax(prompt) == []


@pytest.mark.parametrize("prompt, expected", [
    ("a (cat", ["unclosed '(' bracket"]),
    ("a [cat", ["unclosed '[' bracket"]),
    ("a cat)", ["unexpected ')' without a matching '('"]),
    ("a cat]", ["unexpected ']' without a matching '['"]),
    ("a ((cat", ["2 unclosed '(' brackets"]),
    ("a cat) then (dog", ["unexpected ')' without a matching '('", "unclosed '(' bracket"]),
])
def test_bracket_warnings(prompt, expected):
    assert prompt_parser.check_prompt_syntax(prompt) == expected


def test_repeated_warnings_are_reported_once():
    assert prompt_parser.check_prompt_syntax("a ) b ) c") == ["unexpected ')' without a matching '('"]


def test_unbalanced_brackets_still_affect_the_prompt():
    # the warning is only a hint: the existing parsing behaviour has to stay exactly as it was
    assert prompt_parser.parse_prompt_attention("(unbalanced") == [["unbalanced", 1.1]]


def test_prompt_schedules():
    assert prompt_parser.get_learned_conditioning_prompt_schedules(["a [b:3] c"], 10) == [[[3, "a  c"], [10, "a b c"]]]
    assert prompt_parser.get_learned_conditioning_prompt_schedules(["a [b:3] c"], 20) == [[[3, "a  c"], [20, "a b c"]]]


def test_prompt_schedules_are_parsed_once_per_prompt_and_steps():
    first = prompt_parser.get_learned_conditioning_prompt_schedules(["a [b:3] c"], 10)
    second = prompt_parser.get_learned_conditioning_prompt_schedules(["a [b:3] c"], 10)

    assert first == second
    assert first is not second, "cached results must not be shared between calls"

    first[0][0][1] = "modified by the caller"
    assert prompt_parser.get_learned_conditioning_prompt_schedules(["a [b:3] c"], 10)[0][0][1] == "a  c"


def test_prompt_schedule_cache_is_keyed_by_steps():
    ten_steps = prompt_parser.get_learned_conditioning_prompt_schedules(["a [b:3] c"], 10)
    twenty_steps = prompt_parser.get_learned_conditioning_prompt_schedules(["a [b:3] c"], 20)
    hires_steps = prompt_parser.get_learned_conditioning_prompt_schedules(["a [b:3] c"], 10, 20)

    assert ten_steps != twenty_steps
    assert hires_steps != ten_steps
