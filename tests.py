import pytest


class Test_tokenize:
    @pytest.fixture
    def target(self):
        from pl0 import tokenize

        return tokenize

    def test_ident(self, target):
        result = list(target("x y zzz", False))
        assert result == [("ident", "x", 0), ("ident", "y", 2), ("ident", "zzz", 4)]

    def test_number(self, target):
        result = list(target("111 9 0", False))
        assert result == [("number", "111", 0), ("number", "9", 4), ("number", "0", 6)]

    def test_keywords(self, target):
        from pl0 import keywords

        for s in keywords:
            result = list(target(s, False))
            assert result == [(s.lower(), s, 0)]

    def test_symbols(self, target):
        from pl0 import symbols

        for s in symbols:
            result = list(target(s, False))
            assert result == [(s, s, 0)]

    def test_invalid(self, target):
        result = list(target("@", False))
        assert result == [("invalid", "@", 0)]

    def test_eof(self, target):
        result = list(target(""))
        assert result == [("eof", "", 0)]


class Test_linecol:
    @pytest.fixture
    def target(self):
        from pl0 import linecol

        return linecol

    def test(self, target):
        source = "\n".join(["aaaa", "bbbb", "cccc"])
        result = target(source, 10)
        assert result == (3, 1)


class Test_ParseContext:
    @pytest.fixture
    def target(self):
        from pl0 import ParseContext

        return ParseContext

    def test_accept(self, target):
        ctx = target("x * 1")
        assert ctx.accept("ident") == "ident"
        assert ctx.accept("*", "+", "-") == "*"
        assert ctx.accept("ident") is None
        assert ctx.accept("number") == "number"

    def test_expect(self, target):
        ctx = target("x * 1")
        assert ctx.expect("ident") == "ident"
        assert ctx.expect("*", "+", "-") == "*"
        assert ctx.expect("number") == "number"

    def test_expect_error(self, target):
        from pl0 import PL0Error

        ctx = target("@hello")
        with pytest.raises(PL0Error):
            ctx.expect("hello")


# class Test_parse:
#    @pytest.fixture
#    def target(self):
#        from pl0 import parse#
#
#        return parse#
#
#    def test_parse(self, target):
#        source = """
# CONST N = 10;
# VAR x, squ;#
#
# PROCEDURE square;
# BEGIN
#   squ:= x * x
# END;
#
# BEGIN
#   x := 1;
#   WHILE x <= N DO
#   BEGIN
#      CALL square;
#      ! squ;
#      x := x + 1
#   END
# END.
# """#
#
#        result = target(source)
#        assert result["const"] == [
#            {
#                "type": "const",
#                "ident": {"type": "ident", "name": "N"},
#                "number": {"type": "number", "value": 10},
#            }
#        ]
#        assert result["var"] == [
#            {"type": "ident", "name": "x"},
#            {"type": "ident", "name": "squ"},
#        ]
#        assert result["procedures"] == [
#            {
#                "type": "procedure",
#                "ident": {"type": "ident", "name": "square"},
#                "block": {
#                    "const": [],
#                    "var": [],
#                    "procedures": [],
#                    "statement": {
#                        "type": "begin",
#                        "statements": [
#                            {
#                                "type": "assignment",
#                                "ident": {"name": "squ", "type": "ident"},
#                                "expression": {
#                                    "type": "binary_expression",
#                                    "left": {"type": "ident", "name": "x"},
#                                    "operator": "*",
#                                    "right": {"type": "ident", "name": "x"},
#                                },
#                            }
#                        ],
#                    },
#                },
#            }
#        ]
#        assert result["statement"] == {
#            "type": "begin",
#            "statements": [
#                {
#                    "type": "assignment",
#                    "ident": {"name": "x", "type": "ident"},
#                    "expression": {"type": "number", "value": 1},
#                },
#                {
#                    "type": "while",
#                    "condition": {
#                        "type": "compare",
#                        "left": {"type": "ident", "name": "x"},
#                        "operator": "<=",
#                        "right": {"type": "ident", "name": "N"},
#                    },
#                    "statement": {
#                        "type": "begin",
#                        "statements": [
#                            {
#                                "type": "call",
#                                "ident": {"type": "ident", "name": "square"},
#                            },
#                            {
#                                "type": "write",
#                                "expression": {"type": "ident", "name": "squ"},
#                            },
#                            {
#                                "type": "assignment",
#                                "ident": {"type": "ident", "name": "x"},
#                                "expression": {
#                                    "type": "binary_expression",
#                                    "left": {"type": "ident", "name": "x"},
#                                    "operator": "+",
#                                    "right": {"type": "number", "value": 1},
#                                },
#                            },
#                        ],
#                    },
#                },
#            ],
#        }


class Test_parse_const:
    @pytest.fixture
    def target(self):
        from pl0 import parse_const

        return parse_const

    @pytest.fixture
    def context(self):
        from pl0 import ParseContext

        return ParseContext

    def test_const(self, target, context):
        source = "CONST x=1, y=2, z=3;"
        result = target(context(source))
        assert result == [
            {
                "type": "const",
                "ident": {"type": "ident", "name": "x", "pos": 6},
                "number": {"type": "number", "value": 1, "pos": 8},
                "pos": 6,
            },
            {
                "type": "const",
                "ident": {"type": "ident", "name": "y", "pos": 11},
                "number": {"type": "number", "value": 2, "pos": 13},
                "pos": 11,
            },
            {
                "type": "const",
                "ident": {"type": "ident", "name": "z", "pos": 16},
                "number": {"type": "number", "value": 3, "pos": 18},
                "pos": 16,
            },
        ]


class Test_parse_statement:
    @pytest.fixture
    def target(self):
        from pl0 import parse_statement

        return parse_statement

    @pytest.fixture
    def context(self):
        from pl0 import ParseContext

        return ParseContext

    def test_if(self, target, context):
        source = "IF ODD x THEN x := 1"
        result = target(context(source))
        assert result == {
            "type": "if",
            "condition": {
                "type": "odd",
                "expression": {"type": "ident", "name": "x", "pos": 7},
                "pos": 3,
            },
            "statement": {
                "type": "assignment",
                "ident": {"type": "ident", "name": "x", "pos": 14},
                "expression": {"type": "number", "value": 1, "pos": 19},
                "pos": 14,
            },
            "pos": 0,
        }

    def test_read(self, target, context):
        source = "?x"
        result = target(context(source))
        assert result == {
            "type": "read",
            "ident": {"type": "ident", "name": "x", "pos": 1},
            "pos": 0,
        }


class Test_parse_expression:
    @pytest.fixture
    def target(self):
        from pl0 import parse_expression

        return parse_expression

    @pytest.fixture
    def context(self):
        from pl0 import ParseContext

        return ParseContext

    def test_unary_expression(self, target, context):
        source = "-1"
        result = target(context(source))
        assert result == {
            "type": "unary_expression",
            "operator": "-",
            "argument": {"type": "number", "value": 1, "pos": 1},
            "pos": 0,
        }

    def test_factor_expression(self, target, context):
        source = "1 * (2 + 3)"
        result = target(context(source))
        assert result == {
            "type": "binary_expression",
            "left": {"type": "number", "value": 1, "pos": 0},
            "operator": "*",
            "right": {
                "type": "binary_expression",
                "left": {"type": "number", "value": 2, "pos": 5},
                "operator": "+",
                "right": {"type": "number", "value": 3, "pos": 9},
                "pos": 5,
            },
            "pos": 0,
        }
