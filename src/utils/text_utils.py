import re
from typing import Tuple, Optional
from transformers import AutoTokenizer

# Default system prompt for MMLU-Pro style questions
MATH_SYSTEM_PROMPT = """The following are questions about mathematics. Think step by step and provide your answer in the format '\\boxed{}' with inside your final answer. The final answers should either be a number (in digits) or a latex expression."""


MMLU_SYSTEM_PROMPT = """The following are multiple choice questions (with answers) about science and reasoning. Think step by step and provide your answer in the format '\\boxed{LETTER}' where the letter is one of the options."""

prompt_pattern = re.compile(
    r"<\|im_start\|>system\n(.*?)"
    r"<\|im_end\|>\n<\|im_start\|>user\n(.*?)"
    r"<\|im_end\|>\n<\|im_start\|>assistant\n(.*)",
    re.DOTALL
)

def get_last_step_pos(text: str, tokenizer: AutoTokenizer) -> Tuple[int, str]:
    """
    Returns the token position after the last '\n\n' in the text.
    If not present, appends '\n\n' and returns the position after that.
    
    Args:
        text: Input text
        tokenizer: Tokenizer to use for tokenization
    
    Returns:
        Tuple of (token_position, processed_text)
    """
    if '\n\n' not in text:
        text = text + '\n\n'
    char_idx = text.rfind('\n\n') + 2
    prefix = text[:char_idx]
    prefix_token_ids = tokenizer(prefix, return_tensors="pt").input_ids[0].tolist()
    return len(prefix_token_ids) - 1, text


def count_tokens_after_marker(text: str, tokenizer: AutoTokenizer, marker: str = "<|im_start|>assistant") -> int:
    """
    Count tokens after a specific marker in the text.
    
    Args:
        text: Input text
        tokenizer: Tokenizer to use
        marker: Marker to search for
    
    Returns:
        Number of tokens after the marker
    """
    marker_idx = text.find(marker)
    if marker_idx == -1:
        return len(tokenizer(text, return_tensors="pt").input_ids[0])
    after_marker_text = text[marker_idx + len(marker):]
    return len(tokenizer(after_marker_text, return_tensors="pt").input_ids[0])


def ensure_think_ending(text: str) -> str:
    """
    Ensure the text ends with the think closing tag.
    
    Args:
        text: Input text
    
    Returns:
        Text with proper think ending
    """
    if not text.strip().endswith("</think>") and not text.strip().endswith("</think>\n"):
        return text.rstrip() + "\n</think>\n## Final Answer\n"
    return text


def format_prompt(question: str, system_prompt: str = MATH_SYSTEM_PROMPT, add_think: bool = False, cot="") -> str:
    """
    Format the question into a prompt for the model.

    Args:
        question: The question text to format.
        system_prompt: The system prompt to prepend to the question.

    Returns:
        A formatted prompt string.
    """

    user_question = question.strip()
    if add_think:
        user_question += " \\think"
        cot = "<think>\n" + cot.lstrip()

    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_question}<|im_end|>\n<|im_start|>assistant\n{cot}"


def normalize_latex_escapes(text: str) -> str:
    # turn JSON-escaped TeX commands back into literal backslashes
    # only for sequences TeX actually uses
    return (
        text
        .replace("\b", "\\b")       # backspace → \b
        .replace("\n", "\\n")       # newline → \n
        .replace("\t", "\\t")       # tab → \t
        .replace("\r", "\\r")
        .replace("\f", "\\f")
    )

def extract_final_answer(text: str) -> Optional[str]:
    """
    Normalize LaTeX escapes, then:

    1. Look for the last '\\boxed{' in the text.
       - If found, return everything between that '{' and the next '}'.
    2. Otherwise, look for the last 'ANSWER:' (case-insensitive).
       - If found, return everything after it to the end of the string, stripped.
    3. If neither is found, return None.
    """
    text = normalize_latex_escapes(text)

    # --- 1. Try last \boxed{...} or boxed{...} ---
    # We search for both patterns.
    # Note: re.findall or finditer could work, but we want the *last* occurrence.
    
    # Check for standard LaTeX format
    idx_latex = text.rfind(r'\boxed{')
    # Check for malformed format (missing backslash)
    idx_plain = text.rfind(r'boxed{')

    # Determine which one is actually the LAST one in the string
    last_pos = -1
    start_marker_len = 0

    if idx_latex != -1 and idx_latex >= idx_plain:
        last_pos = idx_latex
        start_marker_len = 7 # len(r'\boxed{')
    elif idx_plain != -1:
        last_pos = idx_plain
        start_marker_len = 6 # len(r'boxed{')

    if last_pos != -1:
        content_start_pos = last_pos + start_marker_len
        
        # Find the next '}' after content_start_pos
        closing_pos = text.find('}', content_start_pos)
        
        # Handle nested braces basics: simply counting braces is safer if models output \boxed{\text{A}}
        # But for MMLU/Math options usually it's simple. Let's stick to simple first
        # unless we want full brace matching logic.
        
        # Robust brace matching for nested brackets:
        open_braces = 1
        current_pos = content_start_pos
        while open_braces > 0 and current_pos < len(text):
            if text[current_pos] == '{':
                open_braces += 1
            elif text[current_pos] == '}':
                open_braces -= 1
            current_pos += 1
        
        if open_braces == 0:
            # We found the matching closing brace at current_pos - 1
            content = text[content_start_pos : current_pos - 1].strip()
            return content or None
        else:
            # Fallback: simple search if nesting logic failed (e.g. cut off)
             if closing_pos != -1:
                 content = text[content_start_pos:closing_pos].strip()
                 return content or None

    # --- 2. Fallback: last ANSWER: (case-insensitive) ---
    pattern = re.compile(r'ANSWER:\s*', re.IGNORECASE)
    last_match = None
    for m in pattern.finditer(text):
        last_match = m

    if last_match is not None:
        answer = text[last_match.end():].strip()
        return answer or None

    # --- 3. Nothing found ---
    return None




def normalize_answer(answer: str) -> str:
    """
    Normalizes a mathematical answer for robust comparison using safe string methods.
    This version does NOT evaluate mathematical expressions.
    """
    if not answer:
        return ""

    # 1. Initial text cleaning and equation handling
    normalized = str(answer).lower().strip()
    if '=' in normalized:
        normalized = normalized.split('=')[-1].strip()

    # 2. Remove LaTeX delimiters first
    normalized = re.sub(r'^(\$|\\\[|\\\(|\\\$)|(\$|\\\]|\\\)|\$$)$', '', normalized).strip()

    # 3. Handle LaTeX commands non-destructively
    # Remove sizing commands
    normalized = re.sub(r'\\left|\\right', '', normalized)
    # Keep the content of text-styling commands (FIXED)
    normalized = re.sub(r'\\(text|mathrm|mathbf|boldsymbol)\s*\{([^}]*)\}', r'\2', normalized)
    # Convert fractions, adding parentheses for safety
    normalized = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', normalized)
    # Convert LaTeX percent symbol
    normalized = re.sub(r'\\%', '%', normalized)

    # 4. Handle numerical conversions
    # Convert percentages to decimals
    if '%' in normalized:
        normalized = re.sub(r'(\d*\.?\d+)\s*%', lambda m: str(float(m.group(1)) / 100.0), normalized)
    # Remove thousands separators
    normalized = re.sub(r',(?=\d)', '', normalized)

    # 5. Standardize spacing (SAFER METHOD)
    # Replace multiple whitespace characters with a single space
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    # Remove spaces around common operators to standardize expressions like "2 + 1" to "2+1"
    normalized = re.sub(r'\s*([+\-*/=()^])\s*', r'\1', normalized)
    
    # 6. Final cleanup for numeric answers
    # Remove trailing zeros from decimals
    if '.' in normalized:
        normalized = normalized.rstrip('0').rstrip('.')
        if normalized == "": # Handles cases like "0.0" -> ""
            normalized = "0"
            
    return normalized


'''
def normalize_answer(answer: Optional[str]) -> str:
    """
    Normalize multiple-choice answers by returning the LAST letter A–J (case-insensitive),
    converted to uppercase. Returns an empty string when the input is None or
    no valid letter is found.
    """
    if answer is None:
        return ""

    normalized = str(answer).strip()
    for char in reversed(normalized):
        upper = char.upper()
        if 'A' <= upper <= 'J':
            return upper

    return ""

'''