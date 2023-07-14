import pytest

from config.arg_types.range_list import ParseError, RangeListParser


# Helper to run the parsing with errors
def expect_parse_error(string: str, msg: str):
    with pytest.raises(ParseError) as e:
        RangeListParser(string).parse()
    assert str(e.value) == msg


def test_empty():
    assert RangeListParser("").parse() == []
    assert RangeListParser("    ").parse() == []


def test_single_value():
    assert RangeListParser("123").parse() == [123]


def test_multiple_values():
    assert RangeListParser("1,2,3").parse() == [1, 2, 3]
    assert RangeListParser("1  ,  2,3   ").parse() == [1, 2, 3]


def test_multiple_values_space():
    assert RangeListParser("1 2 3").parse() == [1, 2, 3]
    assert RangeListParser("1       2 3").parse() == [1, 2, 3]


def test_range():
    assert RangeListParser("2-5").parse() == [2, 3, 4, 5]
    assert RangeListParser("2   -5").parse() == [2, 3, 4, 5]
    assert RangeListParser("2-     5").parse() == [2, 3, 4, 5]
    assert RangeListParser("2   -     5").parse() == [2, 3, 4, 5]


def test_range_step():
    assert RangeListParser("0-10:5").parse() == [0, 5, 10]
    assert RangeListParser("0   -10:5").parse() == [0, 5, 10]
    assert RangeListParser("0   -  10:5").parse() == [0, 5, 10]
    assert RangeListParser("0   -  10        :5").parse() == [0, 5, 10]
    assert RangeListParser("0   -  10        :        5").parse() == [0, 5, 10]


def test_combination():
    assert RangeListParser("1,4-6,10-20:5,30").parse() == [1, 4, 5, 6, 10, 15, 20, 30]
    assert RangeListParser("1 4-6,10-20:5 30").parse() == [1, 4, 5, 6, 10, 15, 20, 30]
    assert RangeListParser("30,4-6,1,10-20:5").parse() == [1, 4, 5, 6, 10, 15, 20, 30]
    assert RangeListParser("1,5-8,12-20:4").parse() == [1, 5, 6, 7, 8, 12, 16, 20]


def test_combination_range_overlap():
    assert RangeListParser("0-3,1-4").parse() == [0, 1, 2, 3, 4]
    assert RangeListParser("2,0-3,1-4").parse() == [0, 1, 2, 3, 4]


def test_invalid():
    expect_parse_error(":", "Expected number at position 0 - got `:` in: ':'")
    expect_parse_error("-", "Expected number at position 0 - got `-` in: '-'")
    expect_parse_error(",", "Expected number at position 0 - got `,` in: ','")
    expect_parse_error("text", "Expected number at position 0 - got `t` in: 'text'")
    expect_parse_error("0,,", "Expected number at position 2 - got `,` in: '0,,'")


def test_invalid_range():
    expect_parse_error(
        "0-,",
        "Expected number for the end of the range at position 2 - got `,` in: '0-,'",
    )
    expect_parse_error(
        "0--",
        "Expected number for the end of the range at position 2 - got `-` in: '0--'",
    )
    expect_parse_error(
        "0-:",
        "Expected number for the end of the range at position 2 - got `:` in: '0-:'",
    )
    expect_parse_error(
        "0:",
        "Expected number, `,` or `-` at position 1 - got `:` in: '0:'",
    )
    expect_parse_error(
        "0-9:",
        "Expected number for the step size of the range at position 4 in: '0-9:'",
    )
    expect_parse_error(
        "0-9::",
        (
            "Expected number for the step size of the range at position 4 "
            "- got `:` in: '0-9::'"
        ),
    )
    expect_parse_error(
        "0-9:3:",
        "Expected number or `,` at position 5 - got `:` in: '0-9:3:'",
    )
    expect_parse_error(
        "0-9-3",
        "Expected number, `,` or `:` at position 3 - got `-` in: '0-9-3'",
    )
