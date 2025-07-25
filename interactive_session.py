#!/usr/bin/env python3
"""
Interactive Python Coding Session

A Python REPL with enhanced features including command history, 
syntax highlighting, error handling, and session management.
"""

import cmd
import code
import sys
import os
import traceback
import json
import datetime
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

try:
    import readline
    HAS_READLINE = True
except ImportError:
    HAS_READLINE = False


class InteractiveSession(cmd.Cmd):
    """Enhanced Python REPL with additional features."""
    
    intro = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🐍 Interactive Python Coding Session 🐍                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Welcome to an enhanced Python REPL with advanced features!                 ║
║                                                                              ║
║  Available Commands:                                                         ║
║    help          - Show this help message                                   ║
║    python <code> - Execute Python code                                      ║
║    multiline     - Enter multi-line code mode                               ║
║    history       - Show command history                                     ║
║    clear         - Clear the screen                                         ║
║    save <file>   - Save current session                                     ║
║    load <file>   - Load a saved session                                     ║
║    exercises     - Show available coding exercises                          ║
║    quit/exit     - Exit the session                                         ║
║                                                                              ║
║  You can also type Python code directly at the prompt!                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    
    prompt = "🐍 >>> "
    
    def __init__(self):
        super().__init__()
        self.locals = {}
        self.history = []
        self.session_data = {
            'commands': [],
            'outputs': [],
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Setup readline history if available
        if HAS_READLINE:
            self.setup_readline()
    
    def setup_readline(self):
        """Setup readline for command history and completion."""
        try:
            readline.set_completer_delims(' \t
            readline.parse_and_bind("tab: complete")
            
            # Load history file if it exists
            history_file = os.path.expanduser("~/.python_interactive_history")
            if os.path.exists(history_file):
                readline.read_history_file(history_file)
        except Exception as e:
            print(f"Warning: Could not setup readline: {e}")
    
    def save_history(self):
        """Save command history to file."""
        if HAS_READLINE:
            try:
                history_file = os.path.expanduser("~/.python_interactive_history")
                readline.write_history_file(history_file)
            except Exception as e:
                print(f"Warning: Could not save history: {e}")
    
    def default(self, line):
        """Handle Python code execution for lines that aren't commands."""
        if line.strip():
            self.execute_python_code(line)
    
    def execute_python_code(self, code_str):
        """Execute Python code with error handling and output capture."""
        try:
            # Capture stdout and stderr
            stdout_capture = StringIO()
            stderr_capture = StringIO()
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Try to compile as expression first (for immediate evaluation)
                try:
                    compiled_code = compile(code_str, '<interactive>', 'eval')
                    result = eval(compiled_code, globals(), self.locals)
                    if result is not None:
                        print(result)
                except SyntaxError:
                    # If not an expression, compile as statement
                    compiled_code = compile(code_str, '<interactive>', 'exec')
                    exec(compiled_code, globals(), self.locals)
            
            # Display captured output
            stdout_content = stdout_capture.getvalue()
            stderr_content = stderr_capture.getvalue()
            
            if stdout_content:
                print(stdout_content, end='')
            if stderr_content:
                print(f"🔴 Error: {stderr_content}", end='')
            
            # Store in history
            self.history.append({
                'code': code_str,
                'output': stdout_content,
                'error': stderr_content,
                'timestamp': datetime.datetime.now().isoformat()
            })
            
            # Store in session data
            self.session_data['commands'].append(code_str)
            self.session_data['outputs'].append(stdout_content + stderr_content)
            
        except Exception as e:
            error_msg = f"🔴 Error: {str(e)}
            print(error_msg)
            
            # Store error in history
            self.history.append({
                'code': code_str,
                'output': '',
                'error': error_msg,
                'timestamp': datetime.datetime.now().isoformat()
            })
    
    def do_python(self, line):
        """Execute Python code: python <code>"""
        if not line.strip():
            print("Usage: python <code>")
            return
        self.execute_python_code(line)
    
    def do_multiline(self, line):
        """Enter multi-line code mode."""
        print("📝 Multi-line mode (type 'END' on a new line to execute):")
        lines = []
        while True:
            try:
                line = input("... ")
                if line.strip() == 'END':
                    break
                lines.append(line)
            except (EOFError, KeyboardInterrupt):
                print("\n🚫 Multi-line input cancelled.")
                return
        
        if lines:
            code = '\n'.join(lines)
            print(f"🚀 Executing multi-line code:\n{code}\n")
            self.execute_python_code(code)
    
    def do_history(self, line):
        """Show command history."""
        if not self.history:
            print("📜 No command history available.")
            return
        
        print("📜 Command History:")
        print("=" * 50)
        for i, entry in enumerate(self.history[-10:], 1):  # Show last 10 entries
            print(f"{i}. [{entry['timestamp'][:19]}]")
            print(f"   Code: {entry['code']}")
            if entry['output']:
                print(f"   Output: {entry['output'].strip()}")
            if entry['error']:
                print(f"   Error: {entry['error'].strip()}")
            print("-" * 30)
    
    def do_clear(self, line):
        """Clear the screen."""
        os.system('clear' if os.name == 'posix' else 'cls')
        print(self.intro)
    
    def do_quit(self, line):
        """Exit the interactive session."""
        print("👋 Goodbye! Thanks for using the Interactive Python Session!")
        self.save_history()
        return True
    
    def do_exit(self, line):
        """Exit the interactive session."""
        return self.do_quit(line)
    
    def do_EOF(self, line):
        """Handle Ctrl+D to exit."""
        print()
        return self.do_quit(line)

