# Python-Lexical-Syntax-Analyzer
📌 Overview
This project is a Python-like lexical and syntax analyzer built using PLY (Python Lex-Yacc). It demonstrates core compiler design concepts by:

Tokenizing input code (keywords, identifiers, operators, etc.).

Parsing grammar rules (functions, conditionals, loops).

Generating detailed workflow reports.

Developed as part of the Theory of Programming Languages (TPL) course.

✨ Features
Lexical Analysis

Classifies tokens (e.g., INT, ID, PLUS).

Tracks line numbers and positions.

Syntax Analysis

Implements grammar rules (BNF) for Python-like constructs.

Handles expressions, function definitions, and control flow.

Detailed Output

Prints token streams with types.

Visualizes parsing steps (e.g., IF → CONDITION → ELSE).

🚀 Quick Start
Clone the repo:

bash
git clone https://github.com/your-username/python-lexer-parser.git
cd python-lexer-parser
Install dependencies (only PLY):

bash
pip install ply
Run the analyzer:

bash
python main.py
Enter code when prompted (e.g., print 5 + 3).

📚 Example Input/Output
Input:

python
def square(x):
    return x * x
print square(4)
Output:

text
[LEXER] Token: 'def' (Keyword)  
[LEXER] Token: 'square' (Identifier)  
[LEXER] Token: '(' (Punctuation)  
...  
[PARSER] Function defined: square with arg x  
[PARSER] Print: 16  
