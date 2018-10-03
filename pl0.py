import re
import io
import sys
from collections import namedtuple

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter("%(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


keywords = (
    "BEGIN",
    "CALL",
    "CONST",
    "DO",
    "END",
    "IF",
    "ODD",
    "PROCEDURE",
    "THEN",
    "VAR",
    "WHILE",
)
symbols = (
    ":=",
    "=",
    ",",
    ";",
    ".",
    "?",
    "!",
    "+",
    "-",
    "*",
    "/",
    "#",
    "<",
    "<=",
    ">",
    ">=",
    "(",
    ")",
)
patterns = [
    ("ignore", "[ \t\r\n]+"),
    ("ident", r"[_a-zA-Z][_a-zA-Z0-9]*"),
    ("number", r"0|[1-9]\d*"),
    ("symbol", "|".join(sorted((re.escape(sym) for sym in symbols), reverse=True))),
]

RE_TOKEN = re.compile("|".join(rf"(?P<{name}>{pat})" for name, pat in patterns))


def tokenize(source, return_eof=True):
    pos = 0
    while pos < len(source):
        m = RE_TOKEN.match(source, pos=pos)
        if m is None:
            yield "invalid", source[pos], pos
            pos += 1
        else:
            cap = [(k, v) for k, v in m.groupdict().items() if v is not None]
            assert len(cap) == 1
            tok, lit = cap[0]
            if tok == "ident" and lit in keywords:
                yield lit.lower(), lit, pos
            elif tok == "symbol":
                yield lit, lit, pos
            elif tok != "ignore":
                yield tok, lit, pos
            pos = m.end()
    if return_eof:
        yield "eof", "", pos


def linecol(source, pos):
    newlines = [-1] + [m.start() for m in re.finditer(r"\n", source[:pos])]
    return len(newlines), pos - newlines[-1]


class Position(namedtuple("Position", "off line col")):
    def __str__(self):
        return f"{self.line}:{self.col}"


def position(source, pos):
    return Position(pos, *linecol(source, pos))


class PL0Error(Exception):
    pass


class ParseContext:
    def __init__(self, source):
        self.source = source
        self.tokenizer = tokenize(source)
        self.tok = None
        self.lit = None
        self.pos = 0
        self.lastpos = 0
        self.accepted = None

        self.next()

    def next(self):
        try:
            self.tok, self.lit, self.pos = next(self.tokenizer)
        except StopIteration:
            pass

    def accept(self, *toks):
        if self.tok in toks:
            tok = self.tok
            self.lastpos = self.pos
            self.accepted = self.lit
            self.next()
            return tok
        return None

    def expect(self, *toks):
        tok = self.accept(*toks)
        if tok is not None:
            return tok

        pos = position(self.source, self.pos)
        msg = "".join(
            [
                f"{pos} expected ",
                ", ".join(repr(tok) for tok in toks),
                f", got {repr(self.tok)}",
            ]
        )
        raise PL0Error(msg)


def make_ident(ctx):
    return {"type": "ident", "name": ctx.accepted, "pos": ctx.lastpos}


def make_number(ctx):
    return {"type": "number", "value": int(ctx.accepted), "pos": ctx.lastpos}


def parse(source):
    """program = block "." ."""
    ctx = ParseContext(source)
    block = parse_block(ctx)
    ctx.expect(".")
    ctx.expect("eof")
    return block


def parse_block(ctx):
    """block = [ "const" ident "=" number {"," ident "=" number} ";"]
            [ "var" ident {"," ident} ";"]
            { "procedure" ident ";" block ";" } statement .
    """
    pos = ctx.pos
    const = parse_const(ctx)
    var = parse_var(ctx)
    procedures = parse_procedures(ctx)
    statement = parse_statement(ctx)
    return {
        "const": const,
        "var": var,
        "procedures": procedures,
        "statement": statement,
        "pos": pos,
    }


def parse_const(ctx):
    def parse_one():
        pos = ctx.pos
        ctx.expect("ident")
        ident = make_ident(ctx)
        ctx.expect("=")
        ctx.expect("number")
        number = make_number(ctx)
        return {"type": "const", "ident": ident, "number": number, "pos": pos}

    if not ctx.accept("const"):
        return []

    lst = [parse_one()]
    while True:
        tok = ctx.expect(",", ";")
        if tok == ";":
            break
        elif tok == ",":
            lst.append(parse_one())
    return lst


def parse_var(ctx):
    def parse_one():
        ctx.expect("ident")
        ident = make_ident(ctx)
        return {"type": "var", "ident": ident, "pos": ident["pos"]}

    if not ctx.accept("var"):
        return []

    lst = []
    lst.append(parse_one())
    while True:
        tok = ctx.expect(",", ";")
        if tok == ";":
            break
        elif tok == ",":
            lst.append(parse_one())
    return lst


def parse_procedures(ctx):
    procs = []
    while ctx.accept("procedure"):
        procs.append(parse_procedure(ctx))
    return procs


def parse_procedure(ctx):
    pos = ctx.lastpos
    ctx.expect("ident")
    ident = make_ident(ctx)
    ctx.expect(";")
    block = parse_block(ctx)
    ctx.expect(";")
    return {"type": "procedure", "ident": ident, "block": block, "pos": pos}


def parse_statement(ctx):
    """statement = [ ident ":=" expression | "call" ident
                | "?" ident | "!" expression
                | "begin" statement {";" statement } "end"
                | "if" condition "then" statement
                | "while" condition "do" statement ].
    """
    tok = ctx.expect("ident", "call", "?", "!", "begin", "if", "while")
    if tok == "ident":
        return parse_assignment(ctx)
    elif tok == "call":
        return parse_call(ctx)
    elif tok == "?":
        return parse_read(ctx)
    elif tok == "!":
        return parse_write(ctx)
    elif tok == "begin":
        return parse_begin(ctx)
    elif tok == "if":
        return parse_if(ctx)
    elif tok == "while":
        return parse_while(ctx)


def parse_read(ctx):
    pos = ctx.lastpos
    ctx.expect("ident")
    return {"type": "read", "ident": make_ident(ctx), "pos": pos}


def parse_write(ctx):
    pos = ctx.lastpos
    expr = parse_expression(ctx)
    return {"type": "write", "expression": expr, "pos": pos}


def parse_assignment(ctx):
    pos = ctx.lastpos
    ident = make_ident(ctx)
    ctx.expect(":=")
    return {
        "type": "assignment",
        "ident": ident,
        "expression": parse_expression(ctx),
        "pos": pos,
    }


def parse_call(ctx):
    pos = ctx.lastpos
    ctx.expect("ident")
    return {"type": "call", "ident": make_ident(ctx), "pos": pos}


def parse_begin(ctx):
    pos = ctx.lastpos
    stmts = [parse_statement(ctx)]
    while True:
        tok = ctx.expect("end", ";")
        if tok == "end":
            break
        elif tok == ";":
            stmt = parse_statement(ctx)
            stmts.append(stmt)
    return {"type": "begin", "statements": stmts, "pos": pos}


def parse_if(ctx):
    pos = ctx.lastpos
    cond = parse_condition(ctx)
    ctx.expect("then")
    stmt = parse_statement(ctx)
    return {"type": "if", "condition": cond, "statement": stmt, "pos": pos}


def parse_while(ctx):
    pos = ctx.lastpos
    cond = parse_condition(ctx)
    ctx.expect("do")
    stmt = parse_statement(ctx)
    return {"type": "while", "condition": cond, "statement": stmt, "pos": pos}


def parse_condition(ctx):
    """condition = "odd" expression |
                expression ("="|"#"|"<"|"<="|">"|">=") expression .
    """
    pos = ctx.pos
    if ctx.accept("odd"):
        expr = parse_expression(ctx)
        return {"type": "odd", "expression": expr, "pos": pos}
    left = parse_expression(ctx)
    op = ctx.expect("=", "#", "<", "<=", ">", ">=")
    right = parse_expression(ctx)
    return {"type": "compare", "operator": op, "left": left, "right": right, "pos": pos}


def parse_expression(ctx):
    """expression = [ "+"|"-"] term { ("+"|"-") term}."""
    pos = ctx.pos
    unary_op = ctx.accept("+", "-")
    term = parse_term(ctx)
    binary_op = ctx.accept("+", "-")
    if binary_op:
        term = {
            "type": "binary_expression",
            "operator": binary_op,
            "left": term,
            "right": parse_term(ctx),
            "pos": pos,
        }
    if unary_op:
        return {
            "type": "unary_expression",
            "operator": unary_op,
            "argument": term,
            "pos": pos,
        }
    return term


def parse_term(ctx):
    """term = factor {("*"|"/") factor}."""
    pos = ctx.pos
    factor = parse_factor(ctx)
    op = ctx.accept("*", "/")
    if op:
        return {
            "type": "binary_expression",
            "operator": op,
            "left": factor,
            "right": parse_factor(ctx),
            "pos": pos,
        }
    return factor


def parse_factor(ctx):
    """factor = ident | number | "(" expression ")"."""
    if ctx.accept("ident"):
        return make_ident(ctx)
    elif ctx.accept("number"):
        return make_number(ctx)
    else:
        ctx.expect("(")
        expr = parse_expression(ctx)
        ctx.expect(")")
        return expr


class Frame:
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent
        self.const_or_var = {}
        self.procs = {}
        self.vars = []

    def add_const_or_var(self, x):
        name = x["ident"]["name"]
        self.const_or_var[name] = x
        if x["type"] == "var":
            self.vars.append(x["ident"]["name"])

    def find_const_or_var(self, name, local=False):
        if name in self.const_or_var:
            node = self.const_or_var[name].copy()
            node["level"] = 0
            if node["type"] == "var":
                node["index"] = self.vars.index(name)
            return node
        if not local and self.parent:
            node = self.parent.find_const_or_var(name)
            if node:
                node["level"] += 1
                return node
        return None

    def add_procedure(self, proc):
        proc["frame"] = self
        self.procs[proc["ident"]["name"]] = proc

    def find_procedure(self, name, local=False):
        if name in self.procs:
            proc = self.procs[name].copy()
            proc["level"] = 0
            return proc
        if not local and self.parent:
            proc = self.parent.find_procedure(name)
            if proc:
                proc["level"] += 1
                return proc
        return None


PROGRAM_NAME = "_PL0MAIN"
WORD_SIZE = 8
STACK_ALIGNMENT = 16


def procedure_symbol_name(frame, proc_name):
    symbol_name = proc_name
    if frame.parent is not None:
        symbol_name = f"{frame.name[1:]}$${symbol_name}"
    return "_" + symbol_name


def stack_size(nvar):
    size = (nvar + 1) * WORD_SIZE
    return (((size - 1) // STACK_ALIGNMENT) + 1) * STACK_ALIGNMENT


def var_offset(index):
    return -(index + 1) * WORD_SIZE


class Compiler:
    def __init__(self, source, out, builtins):
        self.source = source
        self.out = out
        self.builtins = builtins
        self.program = parse(source)
        self.nlabel = 0

    def gen_label(self):
        label = f"L{self.nlabel}"
        self.nlabel += 1
        return label

    def compile(self):
        frame = Frame(PROGRAM_NAME, None)
        self.emit(".text")
        self.emit(f".globl {PROGRAM_NAME}")
        self.emit("")
        self.emit_block(frame, self.program)

    def error(self, node, msg):
        pos = position(self.source, node["pos"])
        raise PL0Error(f"{pos} {msg}")

    def emit(self, line):
        self.out.write(line + "\n")

    def emit_inst(self, op, *args):
        if args:
            self.emit(f"\t{op}\t" + ", ".join(args))
        else:
            self.emit(f"\t{op}")

    def emit_label(self, label):
        self.emit(label + ":")

    def add_const_or_var(self, frame, lst):
        for x in lst:
            name = x["ident"]["name"]
            if frame.find_const_or_var(name, local=True):
                self.error(x, f"duplicated identifier: {name}")
            frame.add_const_or_var(x)

    def emit_block(self, frame, block):
        self.add_const_or_var(frame, block["const"])
        self.add_const_or_var(frame, block["var"])

        for proc in block["procedures"]:
            self.emit_procedure(frame, proc)

        self.emit_label(frame.name)
        self.emit_inst("pushq", "%rbp")
        self.emit_inst("movq", "%rsp", "%rbp")
        ssize = stack_size(len(frame.vars))
        self.emit_inst("subq", f"${ssize}", "%rsp")
        self.emit_statement(frame, block["statement"])
        self.emit_inst("leaveq")
        self.emit_inst("retq")
        self.emit("")

    def emit_procedure(self, frame, proc):
        proc_name = proc["ident"]["name"]
        if frame.find_procedure(proc_name):
            self.error(proc, f"duplicated procedure: {proc_name}")
        symbol_name = procedure_symbol_name(frame, proc_name)
        proc["symbol_name"] = symbol_name
        self.emit_block(Frame(symbol_name, frame), proc["block"])
        frame.add_procedure(proc)

    def emit_statement(self, frame, stmt):
        typ = stmt["type"]
        if typ == "assignment":
            self.emit_assignment(frame, stmt)
        elif typ == "call":
            self.emit_call(frame, stmt)
        elif typ == "read":
            self.emit_read(frame, stmt)
        elif typ == "write":
            self.emit_write(frame, stmt)
        elif typ == "begin":
            self.emit_begin(frame, stmt)
        elif typ == "if":
            self.emit_if(frame, stmt)
        elif typ == "while":
            self.emit_while(frame, stmt)
        else:
            self.error(stmt, f"unknown statement: {stmt}")

    def emit_assignment(self, frame, stmt):
        name = stmt["ident"]["name"]
        ident = frame.find_const_or_var(name)
        if ident["type"] == "const":
            self.error(ident, f"cannot assign to {name}")
        elif ident["type"] == "var":
            self.emit_expression(frame, stmt["expression"])
            off = var_offset(ident["index"])
            level = ident["level"]
            if level > 0:
                self.emit_var_base_pointer(level)
                self.emit_inst("movq", "%rax", f"{off}(%rcx)")
            else:
                self.emit_inst("movq", "%rax", f"{off}(%rbp)")

    def emit_call(self, frame, stmt):
        proc_name = stmt["ident"]["name"]
        proc = frame.find_procedure(proc_name)
        if proc is None:
            self.error(stmt, f"undefined procedure: {proc_name}")
        self.emit_parent_base_pointer(frame, proc)
        self.emit_inst("callq", proc["symbol_name"])

    def emit_parent_base_pointer(self, frame, proc):
        off = -stack_size(len(frame.vars))
        level = proc["level"]
        if level > 0:
            self.emit_inst("movq", "%rbp", "%rcx")
            for i in range(level):
                self.emit_inst("movq", "16(%rcx)", "%rcx")
            self.emit_inst("movq", "%rcx", f"{off}(%rbp)")
        else:
            self.emit_inst("movq", "%rbp", f"{off}(%rbp)")

    def emit_var_base_pointer(self, level):
        self.emit_inst("movq", "%rbp", "%rcx")
        for i in range(level):
            self.emit_inst("movq", "16(%rcx)", "%rcx")

    def emit_read(self, frame, stmt):
        name = stmt["ident"]["name"]
        node = frame.find_const_or_var(name)
        if node is None:
            self.error(stmt["ident"], "undefined: {name}")
        if node["type"] == "const":
            self.error(stmt["ident"], f"cannot assign to {name}")
        self.emit_inst("callq", self.builtins["read"])
        level = node["level"]
        off = var_offset(node["index"])
        if level > 0:
            self.emit_var_base_pointer(level)
            self.emit_inst("movq", "%rax", f"{off}(%rcx)")
        else:
            self.emit_inst("movq", "%rax", f"{off}(%rbp)")

    def emit_write(self, frame, stmt):
        self.emit_expression(frame, stmt["expression"])
        self.emit_inst("movq", "%rax", "%rdi")
        self.emit_inst("callq", self.builtins["write"])

    def emit_begin(self, frame, stmt):
        for stmt in stmt["statements"]:
            self.emit_statement(frame, stmt)

    def emit_if(self, frame, stmt):
        label = self.gen_label()
        self.emit_condition(frame, stmt["condition"], label)
        self.emit_statement(frame, stmt["statement"])
        self.emit_label(label)

    def emit_while(self, frame, stmt):
        start_label = self.gen_label()
        end_label = self.gen_label()
        self.emit_label(start_label)
        self.emit_condition(frame, stmt["condition"], end_label)
        self.emit_statement(frame, stmt["statement"])
        self.emit_inst("jmp", start_label)
        self.emit_label(end_label)

    def emit_condition(self, frame, cond, label):
        typ = cond["type"]
        if typ == "odd":
            self.emit_odd(frame, cond, label)
        elif typ == "compare":
            self.emit_compare(frame, cond, label)
        else:
            self.error(cond, f"unknown condition: {cond}")

    def emit_odd(self, frame, cond, label):
        self.emit_expression(frame, cond["expression"])
        self.emit_inst("movq", "$2", "%rcx")
        self.emit_inst("xor", "%rdx", "%rdx")
        self.emit_inst("idivq", "%rcx")
        self.emit_inst("testq", "%rdx", "%rdx")
        self.emit_inst("jz", label)

    JMP_INSTS = {"=": "jne", "#": "je", "<": "jge", "<=": "jg", ">": "jle", ">=": "jl"}

    def emit_compare(self, frame, cond, label):
        self.emit_expression(frame, cond["left"])
        self.emit_inst("pushq", "%rax")
        self.emit_expression(frame, cond["right"])
        self.emit_inst("popq", "%rcx")
        op = cond["operator"]
        if op not in self.JMP_INSTS:
            self.error(cond, "unknown operator: {op}")
        self.emit_inst("cmpq", "%rax", "%rcx")
        self.emit_inst(self.JMP_INSTS[op], label)

    def emit_expression(self, frame, expr):
        typ = expr["type"]
        if typ == "ident":
            self.emit_ident(frame, expr)
        elif typ == "number":
            self.emit_number(frame, expr)
        elif typ == "binary_expression":
            self.emit_binary_expression(frame, expr)
        elif typ == "unary_expression":
            self.emit_unary_expression(frame, expr)
        else:
            self.error(expr, "unknown expression: {expr}")

    def emit_binary_expression(self, frame, expr):
        self.emit_expression(frame, expr["left"])
        self.emit_inst("pushq", "%rax")
        self.emit_expression(frame, expr["right"])
        self.emit_inst("popq", "%rcx")
        op = expr["operator"]
        if op == "+":
            inst = "addq"
        elif op == "-":
            inst = "subq"
        else:
            self.error(expr, "unknown operator: {op}")
        self.emit_inst(inst, "%rcx", "%rax")

    def emit_unary_expression(self, frame, expr):
        op = expr["operator"]
        if op == "+":
            self.emit_expression(expr)
        elif op == "-":
            self.emit_expression(expr)
            self.emit_inst("negq", "%rax")
        else:
            self.error(expr, "unknown operator: {op}")

    def emit_ident(self, frame, ident):
        name = ident["name"]
        node = frame.find_const_or_var(name)
        if node is None:
            self.error(ident, f"undefined identifier: {name}")
        elif node["type"] == "const":
            self.emit_number(frame, node["number"])
        elif node["type"] == "var":
            level = node["level"]
            off = var_offset(node["index"])
            if level > 0:
                self.emit_var_base_pointer(level)
                self.emit_inst("movq", f"{off}(%rcx)", "%rax")
            else:
                self.emit_inst("movq", f"{off}(%rbp)", "%rax")

    def emit_number(self, frame, num):
        val = num["value"]
        self.emit_inst("movq", f"${val}", "%rax")


def compile(source, out):
    builtins = {"read": "__pl0_read", "write": "__pl0_writeln"}
    compiler = Compiler(source, out, builtins)
    compiler.compile()


def main():
    if len(sys.argv) != 2:
        logger.error(f"Usage: {sys.argv[0]} FILENAME")
        sys.exit(1)

    with open(sys.argv[1]) as fp:
        source = fp.read()
    try:
        out = io.StringIO()
        compile(source, out)
        print(out.getvalue())
    except PL0Error as e:
        logger.error(e)


if __name__ == "__main__":
    main()
