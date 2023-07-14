from typing import List, Optional, Union


class RangeList:
    """
    A list of integer values that can either be specified as single values or ranges.
    The values are separated by commas (,) and ranges are given with hyphens (-).
    Each value can also be a range, meaning that it is the union of these ranges.
    Furthermore, the ranges support a step size by appending :N to them.

    Note: The ranges are *inclusive*, i.e. 1-3 == [1, 2, 3]

    Example:
        # Single value
        1 => [1]
        # Multiple values
        1,2,3,4 => [1, 2, 3, 4]
        # Range
        1-4, => [1, 2, 3, 4]
        # Range with steps of 2
        2-10:2 => [2, 4, 6, 8, 10]
        # Combination
        1,5-8,12-20:4 => [1, 5, 6, 7, 8, 12, 16, 20]
    """

    arg: str
    values: List[int]

    def __init__(self, arg: str):
        self.arg = arg
        self.values = RangeListParser(arg).parse()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(arg={repr(self.arg)}, values={self.values})"

    def __str__(self) -> str:
        return self.arg


class ParseError(Exception):
    def __init__(
        self,
        pos: int,
        expected: Union[str, List[str]],
        source: str,
        got: Optional[str] = None,
        reason: Optional[str] = None,
    ):
        self.pos = pos
        self.expected = expected
        self.source = source
        self.got = got
        self.reason = reason

    def expected_str(self) -> str:
        if isinstance(self.expected, str):
            return self.expected
        else:
            num_vals = len(self.expected)
            if num_vals == 0:
                return "{unknown}"
            elif num_vals == 1:
                return self.expected[0]
            else:
                return f'{", ".join(self.expected[:-1])} or {self.expected[-1]}'

    def __str__(self) -> str:
        msg = f"Expected {self.expected_str()} "
        if self.reason:
            msg += f"for {self.reason} "
        msg += f"at position {self.pos} "
        if self.got:
            msg += f"- got `{self.got}` "
        msg += f"in: {repr(self.source)}"
        return msg


class RangeListParser:
    """
    A parser for list of integer values and ranges (including steps)

    Grammar (EBNF-like):
        values = list | range | number
        list = number | range, { "," | " ", number | range } |
        range = number, "-", number, [ ":", number ]
        number = { digit }+
        digit = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" ;
    """

    def __init__(self, string: str):
        self.string = string
        self.len = len(string)
        self.pos = 0
        self.comma_allowed = False
        self.colon_allowed = False
        self.dash_allowed = False

    def reset(self):
        self.pos = 0
        self.comma_allowed = False
        self.colon_allowed = False
        self.dash_allowed = False

    def allowed(self) -> List[str]:
        values = ["number"]
        if self.comma_allowed:
            values.append("`,`")
        if self.colon_allowed:
            values.append("`:`")
        if self.dash_allowed:
            values.append("`-`")
        return values

    def parse_error(self, reason: Optional[str] = None) -> ParseError:
        return ParseError(
            pos=self.pos,
            expected=self.allowed(),
            source=self.string,
            got=self.current(),
            reason=reason,
        )

    def done(self) -> bool:
        return self.pos >= self.len

    def current(self) -> Optional[str]:
        return None if self.done() else self.string[self.pos]

    def peek(self, offset: int = 1) -> Optional[str]:
        return None if self.pos + offset >= self.len else self.string[self.pos + offset]

    def step(self, offset: int = 1) -> Optional[str]:
        self.pos += offset
        return self.current()

    def skip_whitespace(self) -> None:
        curr = self.current()
        while curr and curr.isspace():
            curr = self.step()

    def parse_number(self) -> int:
        acc = ""
        current = self.current()
        while current and current.isdigit():
            acc += current
            current = self.step()
        return int(acc)

    def parse_range_or_number(self) -> List[int]:
        start = self.parse_number()
        self.skip_whitespace()
        if self.current() != "-":
            # There is no dash, but it would be allowed in this context.
            # (To be included in the error message)
            self.dash_allowed = True
            self.colon_allowed = False
            # Single value
            return [start]
        # Skip over the "-"
        self.step()
        # Already consumed it, hence it is no longer allowed.
        self.dash_allowed = False
        self.skip_whitespace()
        current = self.current()
        if not (current and current.isdigit()):
            raise self.parse_error(reason="the end of the range")
        end = self.parse_number()
        step = 1
        self.skip_whitespace()
        self.colon_allowed = True
        if self.current() == ":":
            # Skip over the ":"
            self.step()
            # Already consumed it, hence it is no longer allowed.
            self.colon_allowed = False
            self.skip_whitespace()
            current = self.current()
            if not (current and current.isdigit()):
                raise self.parse_error(reason="the step size of the range")
            step = self.parse_number()
        return list(range(start, end + 1, step))

    def parse(self, from_beginning: bool = True) -> List[int]:
        if from_beginning:
            self.reset()
        values = []
        while True:
            self.skip_whitespace()
            if self.done():
                break
            current = self.current()
            if current and current.isdigit():
                parsed_values = self.parse_range_or_number()
                self.comma_allowed = True
                values.extend(parsed_values)
            elif self.comma_allowed and current and current == ",":
                # Skip over the ","
                self.step()
                # Only a number can follow a comma
                self.comma_allowed = False
                self.colon_allowed = False
                self.dash_allowed = False
            else:
                raise self.parse_error()
        # Keep only the unique values and sort them
        return sorted(set(values))
