import html
import re
import subprocess
import tempfile
import unicodedata

from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Markdown -> HTML conversion (from OmniDocBench utils/table_utils.py)
# ---------------------------------------------------------------------------

def markdown_to_html(markdown_table):
    """Convert a single markdown table string to HTML."""
    rows = [row.strip() for row in markdown_table.strip().split('\n')]

    html_table = '<table>\n  <thead>\n    <tr>\n'

    header_cells = [cell.strip() for cell in rows[0].split('|')[1:-1]]
    for cell in header_cells:
        html_table += f'      <th>{cell}</th>\n'
    html_table += '    </tr>\n  </thead>\n  <tbody>\n'

    for row in rows[2:]:
        cells = [cell.strip() for cell in row.split('|')[1:-1]]
        html_table += '    <tr>\n'
        for cell in cells:
            html_table += f'      <td>{cell}</td>\n'
        html_table += '    </tr>\n'

    html_table += '  </tbody>\n</table>\n'
    return html_table



def find_md_table_mode(line):
    if re.search(r'-*?:', line) or re.search(r'---', line) or re.search(r':-*?', line):
        return True
    return False


def delete_table_and_body(input_list):
    res = []
    for line in input_list:
        if not re.search(r'</?t(able|head|body)>', line):
            res.append(line)
    return res


def merge_tables(input_str):
    input_str = re.sub(r'<!--[\s\S]*?-->', '', input_str)
    table_blocks = re.findall(r'<table>[\s\S]*?</table>', input_str)

    output_lines = []
    for block in table_blocks:
        block_lines = block.split('\n')
        for i, line in enumerate(block_lines):
            if '<th>' in line:
                block_lines[i] = line.replace('<th>', '<td>').replace('</th>', '</td>')
        final_tr = delete_table_and_body(block_lines)
        if len(final_tr) > 2:
            output_lines.extend(final_tr)

    merged_output = '<table>\n{}\n</table>'.format('\n'.join(output_lines))
    return "\n\n" + merged_output + "\n\n"


def replace_table_with_placeholder(input_string):
    lines = input_string.split('\n')
    output_lines = []

    in_table_block = False
    temp_block = ""
    last_line = ""

    for idx, line in enumerate(lines):
        if "<table>" in line:
            in_table_block = True
            temp_block += last_line
        elif in_table_block:
            if not find_md_table_mode(last_line) and "</thead>" not in last_line:
                temp_block += "\n" + last_line
            if "</table>" in last_line:
                if "<table>" not in line:
                    in_table_block = False
                    output_lines.append(merge_tables(temp_block))
                    temp_block = ""
        else:
            output_lines.append(last_line)

        last_line = line

    if last_line:
        if in_table_block or "</table>" in last_line:
            temp_block += "\n" + last_line
            output_lines.append(merge_tables(temp_block))
        else:
            output_lines.append(last_line)

    return '\n'.join(output_lines)


def convert_table(input_str):
    output_str = input_str.replace("<table>", '<table border="1" >')
    output_str = output_str.replace("<td>", '<td colspan="1" rowspan="1">')
    return output_str


def convert_markdown_to_html(markdown_content):
    """Convert markdown content containing tables to HTML."""
    markdown_content = markdown_content.replace('\r', '') + '\n'
    pattern = re.compile(r'\|\s*.*?\s*\|\n', re.DOTALL)

    matches = pattern.findall(markdown_content)

    for match in matches:
        html_table = markdown_to_html(match)
        markdown_content = markdown_content.replace(match, html_table, 1)

    res_html = convert_table(replace_table_with_placeholder(markdown_content))
    return res_html


# ---------------------------------------------------------------------------
# HTML normalization (from OmniDocBench utils/data_preprocess.py)
# ---------------------------------------------------------------------------

def normalized_html_table(text):
    def process_table_html(md_i):
        def _process_table_html(html_content):
            soup = BeautifulSoup(html_content, 'html.parser')
            th_tags = soup.find_all('th')
            for th in th_tags:
                th.name = 'td'
            thead_tags = soup.find_all('thead')
            for thead in thead_tags:
                thead.unwrap()
            math_tags = soup.find_all('math')
            for math_tag in math_tags:
                alttext = math_tag.get('alttext', '')
                alttext = f'${alttext}$'
                if alttext:
                    math_tag.replace_with(alttext)
            span_tags = soup.find_all('span')
            for span in span_tags:
                span.unwrap()
            return str(soup)

        table_res = ''
        if '<table' in md_i.replace(" ", "").replace("'", '"'):
            md_i = _process_table_html(md_i)
            table_res = html.unescape(md_i).replace('\n', '')
            table_res = unicodedata.normalize('NFKC', table_res).strip()
            pattern = r'<table\b[^>]*>(.*)</table>'
            tables = re.findall(pattern, table_res, re.DOTALL | re.IGNORECASE)
            table_res = ''.join(tables)
            table_res = re.sub('( style=".*?")', "", table_res)
            table_res = re.sub('( height=".*?")', "", table_res)
            table_res = re.sub('( width=".*?")', "", table_res)
            table_res = re.sub('( align=".*?")', "", table_res)
            table_res = re.sub('( class=".*?")', "", table_res)
            table_res = re.sub('</?tbody>', "", table_res)

            table_res = re.sub(r'\s+', " ", table_res)
            table_res = '<html><body><table border="1" >' + table_res + '</table></body></html>'

        return table_res

    def clean_table(input_str, flag=True):
        if flag:
            input_str = input_str.replace('<sup>', '').replace('</sup>', '')
            input_str = input_str.replace('<sub>', '').replace('</sub>', '')
            input_str = input_str.replace('<span>', '').replace('</span>', '')
            input_str = input_str.replace('<div>', '').replace('</div>', '')
            input_str = input_str.replace('<p>', '').replace('</p>', '')
            input_str = input_str.replace('<spandata-span-identity="">', '')
            input_str = re.sub('<colgroup>.*?</colgroup>', '', input_str)
        return input_str

    norm_text = process_table_html(text)
    norm_text = clean_table(norm_text)
    return norm_text


# ---------------------------------------------------------------------------
# LaTeX -> normalized HTML (from OmniDocBench utils/data_preprocess.py)
# ---------------------------------------------------------------------------

def normalized_latex_table(text):
    def latex_template(latex_code):
        template = r'''
        \documentclass[border=20pt]{article}
        \usepackage{subcaption}
        \usepackage{url}
        \usepackage{graphicx}
        \usepackage{caption}
        \usepackage{multirow}
        \usepackage{booktabs}
        \usepackage{color}
        \usepackage{colortbl}
        \usepackage{xcolor,soul,framed}
        \usepackage{fontspec}
        \usepackage{amsmath,amssymb,mathtools,bm,mathrsfs,textcomp}
        \setlength{\parindent}{0pt}''' + \
        r'''
        \begin{document}
        ''' + \
        latex_code + \
        r'''
        \end{document}'''

        return template

    def convert_latex_to_html(latex_content):
        with tempfile.TemporaryDirectory() as cache_dir:
            tex_path = f'{cache_dir}/table.tex'
            log_path = f'{cache_dir}/table.log'
            html_path = f'{cache_dir}/table.html'

            with open(tex_path, 'w', encoding='utf-8') as f:
                f.write(latex_template(latex_content))

            cmd = ['latexmlc', '--quiet', '--nocomments',
                   f'--log={log_path}', tex_path, f'--dest={html_path}']
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                with open(html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()

                pattern = r'<table\b[^>]*>(.*)</table>'
                tables = re.findall(pattern, html_content, re.DOTALL | re.IGNORECASE)
                tables = [f'<table>{table}</table>' for table in tables]
                html_content = '\n'.join(tables)

            except FileNotFoundError:
                raise RuntimeError(
                    "latexmlc not found. Install LaTeXML: "
                    "https://math.nist.gov/~BMiller/LaTeXML/get.html"
                )
            except Exception:
                html_content = ''

        return html_content

    html_text = convert_latex_to_html(text)
    normlized_tables = normalized_html_table(html_text)
    return normlized_tables


# ---------------------------------------------------------------------------
# Markdown -> normalized HTML (convenience wrapper)
# ---------------------------------------------------------------------------

def normalized_markdown_table(text):
    """Convert markdown table to normalized HTML matching OmniDocBench pipeline."""
    html_text = convert_markdown_to_html(text)
    return normalized_html_table(html_text)


def normalize_table(text):
    """Auto-detect table format (HTML, LaTeX, Markdown) and return normalized HTML."""
    stripped = text.strip()
    if "<table" in stripped.lower():
        return normalized_html_table(text)
    elif "\\begin{tabular}" in stripped or "\\begin{table}" in stripped:
        return normalized_latex_table(text)
    else:
        return normalized_markdown_table(text)
