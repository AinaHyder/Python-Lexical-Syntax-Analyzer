import re
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class Token:
    def __init__(self, token_type, value, line, column, description):
        self.type = token_type
        self.value = value
        self.line = line
        self.column = column
        self.description = description
    
    def to_dict(self):
        return {
            'type': self.type,
            'value': self.value,
            'line': self.line,
            'column': self.column,
            'description': self.description
        }

class ASTNode:
    def __init__(self, node_type, value=None, children=None):
        self.type = node_type
        self.value = value
        self.children = children or []
    
    def to_dict(self):
        return {
            'type': self.type,
            'value': self.value,
            'children': [child.to_dict() for child in self.children]
        }

class LexicalAnalyzer:
    def __init__(self):
        # Keywords from your original flex file
        self.keywords = {
            'def': 'DEF',
            'if': 'IF',
            'else': 'ELSE',
            'while': 'WHILE',
            'return': 'RETURN',
            'print': 'PRINT',
            'for': 'FOR',
            'in': 'IN',
            'and': 'AND',
            'or': 'OR',
            'not': 'NOT',
            'True': 'BOOLEAN',
            'False': 'BOOLEAN',
            'None': 'NONE'
        }
        
        # Operators from your original flex file
        self.operators = {
            '==': 'EQ',
            '!=': 'NEQ',
            '<=': 'LE',
            '>=': 'GE',
            '<': 'LT',
            '>': 'GT',
            '=': 'ASSIGN',
            '+': 'PLUS',
            '-': 'MINUS',
            '*': 'MUL',
            '/': 'DIV',
            '%': 'MOD',
            '//': 'FLOOR_DIV',
            '**': 'POWER'
        }
        
        # Delimiters
        self.delimiters = {
            '(': 'LPAREN',
            ')': 'RPAREN',
            '[': 'LBRACKET',
            ']': 'RBRACKET',
            '{': 'LBRACE',
            '}': 'RBRACE',
            ':': 'COLON',
            ',': 'COMMA',
            ';': 'SEMICOLON',
            '.': 'DOT'
        }
        
        self.tokens = []
        self.errors = []
        self.line_num = 1
        self.col_num = 1
    
    def analyze(self, code):
        self.tokens = []
        self.errors = []
        self.line_num = 1
        self.col_num = 1
        
        i = 0
        while i < len(code):
            # Skip whitespace except newlines
            if code[i] in ' \t':
                self.col_num += 1
                i += 1
                continue
            
            # Handle newlines
            elif code[i] == '\n':
                self.add_token('NEWLINE', '\\n', 'Line terminator')
                self.line_num += 1
                self.col_num = 1
                i += 1
                continue
            
            # Handle comments
            elif code[i] == '#':
                start = i
                while i < len(code) and code[i] != '\n':
                    i += 1
                comment = code[start:i]
                self.add_token('COMMENT', comment, 'Single-line comment')
                continue
            
            # Handle strings
            elif code[i] in '"\'':
                quote = code[i]
                start = i
                i += 1
                while i < len(code) and code[i] != quote:
                    if code[i] == '\\':
                        i += 2  # Skip escape sequence
                    else:
                        i += 1
                
                if i >= len(code):
                    self.errors.append(f"Unterminated string at line {self.line_num}, column {self.col_num}")
                    break
                
                i += 1  # Skip closing quote
                string_val = code[start:i]
                self.add_token('STRING', string_val, f'String literal ({quote}-delimited)')
                continue
            
            # Handle numbers
            elif code[i].isdigit():
                start = i
                while i < len(code) and (code[i].isdigit() or code[i] == '.'):
                    i += 1
                
                number = code[start:i]
                if '.' in number:
                    self.add_token('FLOAT', number, 'Floating-point literal')
                else:
                    self.add_token('INTEGER', number, 'Integer literal')
                continue
            
            # Handle identifiers and keywords
            elif code[i].isalpha() or code[i] == '_':
                start = i
                while i < len(code) and (code[i].isalnum() or code[i] == '_'):
                    i += 1
                
                identifier = code[start:i]
                if identifier in self.keywords:
                    self.add_token('KEYWORD', identifier, f'Reserved keyword: {identifier}')
                else:
                    self.add_token('IDENTIFIER', identifier, 'User-defined identifier')
                continue
            
            # Handle operators (check 2-char operators first)
            elif i + 1 < len(code) and code[i:i+2] in self.operators:
                op = code[i:i+2]
                self.add_token('OPERATOR', op, f'Operator: {op}')
                i += 2
                continue
            
            elif code[i] in self.operators:
                op = code[i]
                self.add_token('OPERATOR', op, f'Operator: {op}')
                i += 1
                continue
            
            # Handle delimiters
            elif code[i] in self.delimiters:
                delim = code[i]
                self.add_token('DELIMITER', delim, f'Delimiter: {self.get_delimiter_desc(delim)}')
                i += 1
                continue
            
            # Unknown character
            else:
                self.errors.append(f"Unknown character '{code[i]}' at line {self.line_num}, column {self.col_num}")
                i += 1
                continue
        
        return self.tokens, self.errors
    
    def add_token(self, token_type, value, description):
        token = Token(token_type, value, self.line_num, self.col_num, description)
        self.tokens.append(token)
        self.col_num += len(value) if value != '\\n' else 0
    
    def get_delimiter_desc(self, delim):
        descriptions = {
            '(': 'left parenthesis',
            ')': 'right parenthesis',
            '[': 'left bracket',
            ']': 'right bracket',
            '{': 'left brace',
            '}': 'right brace',
            ':': 'colon',
            ',': 'comma',
            ';': 'semicolon',
            '.': 'dot'
        }
        return descriptions.get(delim, delim)

class SyntaxAnalyzer:
    def __init__(self, tokens):
        self.tokens = [t for t in tokens if t.type not in ['NEWLINE', 'COMMENT']]  # Filter out newlines and comments for parsing
        self.current = 0
        self.errors = []
        self.variables = {}  # Symbol table for variables
        self.functions = {}  # Symbol table for functions
        self.output = []     # Predicted output
    
    def parse(self):
        try:
            ast = self.parse_program()
            self.execute_ast(ast)  # Execute to predict output
            return ast, self.errors, self.output
        except Exception as e:
            self.errors.append(f"Parse error: {str(e)}")
            return None, self.errors, self.output
    
    def current_token(self):
        if self.current < len(self.tokens):
            return self.tokens[self.current]
        return None
    
    def consume(self, expected_type=None):
        token = self.current_token()
        if token is None:
            raise Exception("Unexpected end of input")
        
        if expected_type and token.type != expected_type:
            raise Exception(f"Expected {expected_type}, got {token.type}")
        
        self.current += 1
        return token
    
    def peek(self, offset=0):
        pos = self.current + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return None
    
    def parse_program(self):
        statements = []
        while self.current_token():
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
        return ASTNode('PROGRAM', children=statements)
    
    def parse_statement(self):
        token = self.current_token()
        if not token:
            return None
        
        if token.value == 'def':
            return self.parse_function_def()
        elif token.value == 'if':
            return self.parse_if_statement()
        elif token.value == 'while':
            return self.parse_while_statement()
        elif token.value == 'return':
            return self.parse_return_statement()
        elif token.value == 'print':
            return self.parse_print_statement()
        elif token.type == 'IDENTIFIER':
            # Check if it's an assignment
            if self.peek(1) and self.peek(1).value == '=':
                return self.parse_assignment()
            else:
                # Expression statement
                expr = self.parse_expression()
                return ASTNode('EXPRESSION_STMT', children=[expr])
        else:
            # Expression statement
            expr = self.parse_expression()
            return ASTNode('EXPRESSION_STMT', children=[expr])
    
    def parse_function_def(self):
        self.consume('KEYWORD')  # 'def'
        name = self.consume('IDENTIFIER')
        self.consume('DELIMITER')  # '('
        
        params = []
        if self.current_token() and self.current_token().type == 'IDENTIFIER':
            params.append(self.consume('IDENTIFIER'))
            while self.current_token() and self.current_token().value == ',':
                self.consume('DELIMITER')  # ','
                params.append(self.consume('IDENTIFIER'))
        
        self.consume('DELIMITER')  # ')'
        self.consume('DELIMITER')  # ':'
        
        body = []
        while self.current_token() and self.current_token().value not in ['def', 'if', 'while']:
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
            else:
                break
        
        func_node = ASTNode('FUNCTION_DEF', name.value, [
            ASTNode('PARAMS', children=[ASTNode('PARAM', p.value) for p in params]),
            ASTNode('BODY', children=body)
        ])
        
        # Store function in symbol table
        self.functions[name.value] = {
            'params': [p.value for p in params],
            'body': body
        }
        
        return func_node
    
    def parse_if_statement(self):
        self.consume('KEYWORD')  # 'if'
        condition = self.parse_expression()
        self.consume('DELIMITER')  # ':'
        
        then_body = []
        while (self.current_token() and 
               self.current_token().value not in ['else', 'def', 'if', 'while']):
            stmt = self.parse_statement()
            if stmt:
                then_body.append(stmt)
            else:
                break
        
        else_body = []
        if self.current_token() and self.current_token().value == 'else':
            self.consume('KEYWORD')  # 'else'
            self.consume('DELIMITER')  # ':'
            while (self.current_token() and 
                   self.current_token().value not in ['def', 'if', 'while']):
                stmt = self.parse_statement()
                if stmt:
                    else_body.append(stmt)
                else:
                    break
        
        return ASTNode('IF_STMT', children=[
            condition,
            ASTNode('THEN_BODY', children=then_body),
            ASTNode('ELSE_BODY', children=else_body)
        ])
    
    def parse_while_statement(self):
        self.consume('KEYWORD')  # 'while'
        condition = self.parse_expression()
        self.consume('DELIMITER')  # ':'
        
        body = []
        while (self.current_token() and 
               self.current_token().value not in ['def', 'if', 'while']):
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
            else:
                break
        
        return ASTNode('WHILE_STMT', children=[
            condition,
            ASTNode('BODY', children=body)
        ])
    
    def parse_return_statement(self):
        self.consume('KEYWORD')  # 'return'
        expr = self.parse_expression()
        return ASTNode('RETURN_STMT', children=[expr])
    
    def parse_print_statement(self):
        self.consume('KEYWORD')  # 'print'
        expr = self.parse_expression()
        return ASTNode('PRINT_STMT', children=[expr])
    
    def parse_assignment(self):
        var = self.consume('IDENTIFIER')
        self.consume('OPERATOR')  # '='
        expr = self.parse_expression()
        return ASTNode('ASSIGNMENT', var.value, [expr])
    
    def parse_expression(self):
        return self.parse_comparison()
    
    def parse_comparison(self):
        left = self.parse_arithmetic()
        
        while (self.current_token() and 
               self.current_token().value in ['==', '!=', '<', '>', '<=', '>=']):
            op = self.consume('OPERATOR')
            right = self.parse_arithmetic()
            left = ASTNode('BINARY_OP', op.value, [left, right])
        
        return left
    
    def parse_arithmetic(self):
        left = self.parse_term()
        
        while (self.current_token() and 
               self.current_token().value in ['+', '-']):
            op = self.consume('OPERATOR')
            right = self.parse_term()
            left = ASTNode('BINARY_OP', op.value, [left, right])
        
        return left
    
    def parse_term(self):
        left = self.parse_factor()
        
        while (self.current_token() and 
               self.current_token().value in ['*', '/', '%']):
            op = self.consume('OPERATOR')
            right = self.parse_factor()
            left = ASTNode('BINARY_OP', op.value, [left, right])
        
        return left
    
    def parse_factor(self):
        token = self.current_token()
        
        if token.type == 'INTEGER':
            self.consume('INTEGER')
            return ASTNode('INTEGER', int(token.value))
        elif token.type == 'FLOAT':
            self.consume('FLOAT')
            return ASTNode('FLOAT', float(token.value))
        elif token.type == 'STRING':
            self.consume('STRING')
            return ASTNode('STRING', token.value[1:-1])  # Remove quotes
        elif token.type == 'IDENTIFIER':
            # Check if it's a function call
            if self.peek(1) and self.peek(1).value == '(':
                return self.parse_function_call()
            else:
                self.consume('IDENTIFIER')
                return ASTNode('IDENTIFIER', token.value)
        elif token.value == '(':
            self.consume('DELIMITER')  # '('
            expr = self.parse_expression()
            self.consume('DELIMITER')  # ')'
            return expr
        else:
            raise Exception(f"Unexpected token: {token.value}")
    
    def parse_function_call(self):
        name = self.consume('IDENTIFIER')
        self.consume('DELIMITER')  # '('
        
        args = []
        if self.current_token() and self.current_token().value != ')':
            args.append(self.parse_expression())
            while self.current_token() and self.current_token().value == ',':
                self.consume('DELIMITER')  # ','
                args.append(self.parse_expression())
        
        self.consume('DELIMITER')  # ')'
        return ASTNode('FUNCTION_CALL', name.value, args)
    
    def execute_ast(self, node):
        """Execute AST to predict program output"""
        if not node:
            return None
        
        if node.type == 'PROGRAM':
            for child in node.children:
                self.execute_ast(child)
        
        elif node.type == 'FUNCTION_DEF':
            # Function definitions don't produce output
            pass
        
        elif node.type == 'PRINT_STMT':
            value = self.evaluate_expression(node.children[0])
            self.output.append(f"Print: {value}")
        
        elif node.type == 'ASSIGNMENT':
            value = self.evaluate_expression(node.children[0])
            self.variables[node.value] = value
            self.output.append(f"Variable '{node.value}' assigned value: {value}")
        
        elif node.type == 'EXPRESSION_STMT':
            value = self.evaluate_expression(node.children[0])
            if value is not None:
                self.output.append(f"Expression result: {value}")
        
        elif node.type == 'IF_STMT':
            condition = self.evaluate_expression(node.children[0])
            if condition:
                self.output.append("If condition is TRUE - executing then block")
                for stmt in node.children[1].children:
                    self.execute_ast(stmt)
            else:
                self.output.append("If condition is FALSE - executing else block")
                for stmt in node.children[2].children:
                    self.execute_ast(stmt)
        
        elif node.type == 'WHILE_STMT':
            self.output.append("While loop detected (execution simulation limited)")
            # Simulate limited iterations to avoid infinite loops
            iterations = 0
            while iterations < 5:  # Limit iterations for demo
                condition = self.evaluate_expression(node.children[0])
                if not condition:
                    break
                self.output.append(f"While iteration {iterations + 1}")
                for stmt in node.children[1].children:
                    self.execute_ast(stmt)
                iterations += 1
        
        elif node.type == 'RETURN_STMT':
            value = self.evaluate_expression(node.children[0])
            self.output.append(f"Return: {value}")
    
    def evaluate_expression(self, node):
        """Evaluate expressions to predict values"""
        if not node:
            return None
        
        if node.type == 'INTEGER':
            return node.value
        elif node.type == 'FLOAT':
            return node.value
        elif node.type == 'STRING':
            return node.value
        elif node.type == 'IDENTIFIER':
            return self.variables.get(node.value, f"<undefined variable '{node.value}'>")
        elif node.type == 'BINARY_OP':
            left = self.evaluate_expression(node.children[0])
            right = self.evaluate_expression(node.children[1])
            
            if node.value == '+':
                return left + right
            elif node.value == '-':
                return left - right
            elif node.value == '*':
                return left * right
            elif node.value == '/':
                return left / right if right != 0 else "Division by zero"
            elif node.value == '%':
                return left % right if right != 0 else "Modulo by zero"
            elif node.value == '==':
                return left == right
            elif node.value == '!=':
                return left != right
            elif node.value == '<':
                return left < right
            elif node.value == '>':
                return left > right
            elif node.value == '<=':
                return left <= right
            elif node.value == '>=':
                return left >= right
        elif node.type == 'FUNCTION_CALL':
            if node.value == 'factorial' and len(node.children) == 1:
                # Special handling for factorial function
                n = self.evaluate_expression(node.children[0])
                if isinstance(n, int) and n >= 0:
                    result = 1
                    for i in range(1, n + 1):
                        result *= i
                    return result
                else:
                    return f"factorial({n})"
            else:
                return f"{node.value}(...)"
        
        return None

class CompilerAnalyzer:
    def __init__(self):
        self.lexer = LexicalAnalyzer()
    
    def analyze(self, code):
        # Lexical Analysis
        tokens, lex_errors = self.lexer.analyze(code)
        
        # Generate lexical summary
        lex_summary = {
            'total_tokens': len(tokens),
            'keywords': len([t for t in tokens if t.type == 'KEYWORD']),
            'identifiers': len([t for t in tokens if t.type == 'IDENTIFIER']),
            'operators': len([t for t in tokens if t.type == 'OPERATOR']),
            'literals': len([t for t in tokens if t.type in ['INTEGER', 'FLOAT', 'STRING']]),
            'delimiters': len([t for t in tokens if t.type == 'DELIMITER']),
            'comments': len([t for t in tokens if t.type == 'COMMENT'])
        }
        
        # Syntax Analysis
        parser = SyntaxAnalyzer(tokens)
        ast, parse_errors, predicted_output = parser.parse()
        
        # Generate parse tree representation
        parse_tree = None
        if ast:
            parse_tree = self.ast_to_tree_string(ast)
        
        return {
            'tokens': [token.to_dict() for token in tokens],
            'lexical_errors': lex_errors,
            'lexical_summary': lex_summary,
            'parse_tree': parse_tree,
            'ast': ast.to_dict() if ast else None,
            'parse_errors': parse_errors,
            'predicted_output': predicted_output,
            'symbol_table': {
                'variables': parser.variables,
                'functions': list(parser.functions.keys())
            }
        }
    
    def ast_to_tree_string(self, node, indent=0):
        """Convert AST to a readable tree string"""
        if not node:
            return ""
        
        result = "  " * indent + f"{node.type}"
        if node.value is not None:
            result += f": {node.value}"
        result += "\n"
        
        for child in node.children:
            result += self.ast_to_tree_string(child, indent + 1)
        
        return result

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_code():
    try:
        data = request.get_json()
        code = data.get('code', '')
        
        if not code:
            return jsonify({'error': 'No code provided'}), 400
        
        analyzer = CompilerAnalyzer()
        result = analyzer.analyze(code)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)