#!/usr/bin/env python3
import re
import datetime
from typing import List, Tuple, Optional

# read text from file
def read_text(filepath: str) -> str:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return ""
    except Exception as e:
        return ""

# extract records from text
def extract_records(text: str) -> List[str]:
    # regex to find balanced curly brace blocks
    record_pattern = r'\{([^{}]*)\}'
    matches = re.findall(record_pattern, text)
    
    # clean records
    records = []
    for match in matches:
        cleaned = match.strip()
        if cleaned:  
            records.append(cleaned)
    
    return records

# clean punctuation and special characters from name while preserving spaces and accented characters
def clean_name(name: str) -> str:

    if not name:
        return name
    
    cleaned = re.sub(r'[*#@•★\(\)\.]', '', name)
    cleaned = re.sub(r'\s+', ' ', cleaned.strip())
    
    return cleaned

def classify_tokens(record: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:

    # Split the record by commas and clean whitespace
    tokens = [token.strip() for token in record.split(',')]
    
    # Remove empty tokens
    tokens = [token for token in tokens if token]
    
    if len(tokens) != 3:
        return None, None, None
    
    # Date pattern - comprehensive regex for various date formats
    date_pattern = r'''
        (?:
            # dd/mm/yyyy, dd/mm/yy, d/m/yy
            \b\d{1,2}/\d{1,2}/\d{2,4}\b
            |
            # yyyy/mm/dd, yyyy-mm-dd 
            \b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b
            |
            # dd-mm-yyyy, dd-mm-yy
            \b\d{1,2}-\d{1,2}-\d{2,4}\b
            |
            # dd.mm.yyyy
            \b\d{1,2}\.\d{1,2}\.\d{4}\b
            |
            # dd-Month-yyyy (15-Apr-1452)
            \b\d{1,2}-[A-Za-z]{3,9}-\d{4}\b
            |
            # dd Month yyyy (18 Jul 1918, 21 Aug 1973)
            \b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b
            |
            # Month dd, yyyy (January 15, 1929)
            \b[A-Za-z]{3,9}\s+\d{1,2},\s*\d{4}\b
            |
            # Month dd yyyy (December 18 2001)
            \b[A-Za-z]{3,9}\s+\d{1,2}\s+\d{4}\b
        )
    '''
    
    # Phone pattern - digit-heavy strings with separators
    phone_pattern = r'''
        (?:
            # Patterns with + prefix
            \+\d+[-\s\(\)]*\d+[-\s\(\)]*\d+[-\s\(\)]*\d*
            |
            # Patterns with parentheses like (021) 34567890
            \(\d+\)\s*\d+[-\s]*\d*
            |
            # Simple digit patterns like 124, 000-000-0000, 123-456-7890
            \b\d{3,}[-\s]*\d*[-\s]*\d*\b
            |
            # Complex patterns like 92-21-1111111
            \b\d+[-\s]\d+[-\s]\d+\b
        )
    '''
    
    # Name pattern - alphabetic with spaces, punctuation, special characters
    name_pattern = r'''
        ^[\w\s\.\*\#\@\•\★\'\-\(\)àáâãäåèéêëìíîïòóôõöùúûüýÿñçÀÁÂÃÄÅÈÉÊËÌÍÎÏÒÓÔÕÖÙÚÛÜÝŸÑÇ]+$
    '''
    
    name, phone, dob = None, None, None
    
    for token in tokens:
        # Check if token matches date pattern (preserve punctuation)
        if re.search(date_pattern, token, re.VERBOSE):
            dob = token
        # Check if token matches phone pattern (preserve punctuation)
        elif re.search(phone_pattern, token, re.VERBOSE):
            phone = token
        # Check if token matches name pattern (will clean punctuation later)
        elif re.match(name_pattern, token, re.VERBOSE):
            name = clean_name(token)  # Clean punctuation from name
        else:
            # If doesn't match any pattern, treat as name (fallback)
            name = clean_name(token)  # Clean punctuation from name
    
    return name, phone, dob

def normalize_dob(date_str: str) -> Optional[str]:
    if not date_str:
        return None
    
    date_str = date_str.strip()
    
    try:
        # Try different date parsing patterns
        patterns = [
            # dd/mm/yyyy or dd/mm/yy
            (r'(\d{1,2})/(\d{1,2})/(\d{2,4})', '%d/%m/%Y'),
            # yyyy/mm/dd  
            (r'(\d{4})/(\d{1,2})/(\d{1,2})', '%Y/%m/%d'),
            # dd-mm-yyyy or dd-mm-yy
            (r'(\d{1,2})-(\d{1,2})-(\d{2,4})', '%d-%m-%Y'),
            # yyyy-mm-dd
            (r'(\d{4})-(\d{1,2})-(\d{1,2})', '%Y-%m-%d'),
            # dd.mm.yyyy
            (r'(\d{1,2})\.(\d{1,2})\.(\d{4})', '%d.%m.%Y'),
            # dd-Month-yyyy (15-Apr-1452)
            (r'(\d{1,2})-([A-Za-z]+)-(\d{4})', '%d-%b-%Y'),
        ]
        
        # Try standard patterns first
        for pattern, format_str in patterns:
            match = re.match(pattern, date_str)
            if match:
                if pattern == patterns[0][0]:  # dd/mm/yyyy or dd/mm/yy
                    day, month, year = match.groups()
                    year = int(year)
                    # Apply two-digit year rule: 00-24 → 2000-2024, 25-99 → 1925-1999
                    if year < 100:
                        if year <= 24:
                            year += 2000  # 00-24 becomes 2000-2024
                        else:
                            year += 1900  # 25-99 becomes 1925-1999
                    return f"{year:04d}-{int(month):02d}-{int(day):02d}"
                elif pattern == patterns[2][0]:  # dd-mm-yyyy or dd-mm-yy  
                    day, month, year = match.groups()
                    year = int(year)
                    # Apply two-digit year rule: 00-24 → 2000-2024, 25-99 → 1925-1999
                    if year < 100:
                        if year <= 24:
                            year += 2000  # 00-24 becomes 2000-2024
                        else:
                            year += 1900  # 25-99 becomes 1925-1999
                    return f"{year:04d}-{int(month):02d}-{int(day):02d}"
                else:
                    # For other patterns, parse normally
                    parsed_date = datetime.datetime.strptime(date_str, format_str.replace('/%Y', f'/%Y'))
                    return parsed_date.strftime('%Y-%m-%d')
        
        # Try month name patterns
        month_patterns = [
            # Month dd, yyyy (January 15, 1929)
            r'([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})',
            # dd Month yyyy (18 Jul 1918)
            r'(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})',
            # Month dd yyyy (December 18 2001)
            r'([A-Za-z]+)\s+(\d{1,2})\s+(\d{4})'
        ]
        
        for i, pattern in enumerate(month_patterns):
            match = re.match(pattern, date_str)
            if match:
                if i == 0:  # Month dd, yyyy
                    month_name, day, year = match.groups()
                    try:
                        month_num = datetime.datetime.strptime(month_name, '%B').month
                    except ValueError:
                        month_num = datetime.datetime.strptime(month_name, '%b').month
                    return f"{int(year):04d}-{month_num:02d}-{int(day):02d}"
                elif i == 1:  # dd Month yyyy
                    day, month_name, year = match.groups()
                    try:
                        month_num = datetime.datetime.strptime(month_name, '%B').month
                    except ValueError:
                        month_num = datetime.datetime.strptime(month_name, '%b').month
                    return f"{int(year):04d}-{month_num:02d}-{int(day):02d}"
                elif i == 2:  # Month dd yyyy
                    month_name, day, year = match.groups()
                    try:
                        month_num = datetime.datetime.strptime(month_name, '%B').month
                    except ValueError:
                        month_num = datetime.datetime.strptime(month_name, '%b').month
                    return f"{int(year):04d}-{month_num:02d}-{int(day):02d}"
        
        return None
        
    except Exception as e:
        return None

def write_pairs(pairs: List[Tuple[str, str]], output_file: str) -> None:

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for name, dob in pairs:
                f.write(f"{name},{dob}\n")
    except Exception as e:
        pass

def main():
    
    # Read input file
    input_file = "Question2_input.txt"
    text = read_text(input_file)
    
    if not text:
        return
    
    # Extract records
    records = extract_records(text)
    
    # Process each record
    pairs = []
    processed_count = 0
    
    for i, record in enumerate(records, 1):        
        # Classify tokens
        name, phone, dob = classify_tokens(record)
        
        if not name or not dob:
            continue
        
        # Normalize date
        normalized_dob = normalize_dob(dob)
        
        if not normalized_dob:
            continue
        
        pairs.append((name, normalized_dob))
        processed_count += 1
    
    # Write output
    output_file = "Question2_output.txt"
    write_pairs(pairs, output_file)

if __name__ == "__main__":
    main()
