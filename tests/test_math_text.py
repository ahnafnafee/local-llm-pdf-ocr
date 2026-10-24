"""Visible math must survive plain text and searchable-PDF export."""

import fitz

from pdf_ocr.core._layout import searchable_text
from pdf_ocr.core.pdf import PDFHandler
from pdf_ocr.core.text import TextHandler


def test_native_math_becomes_searchable_characters():
    assert searchable_text(r'$\text{Project 1-3}$') == 'Project 1-3'
    assert searchable_text(r'$A \rightarrow B$') == 'A → B'
    assert searchable_text(r'$\pi_{name}(R \bowtie S)$') == 'π_name(R ⋈ S)'
    assert searchable_text(r'$\boxed{A \rightarrow B}$') == 'A → B'
    assert searchable_text(r'$\left(R \cap S\right)$') == '(R ∩ S)'
    assert searchable_text(r'$\textcircled{6} \quad R$').split() == ['6', 'R']
    assert 'A → B' in searchable_text(r'$$\begin{aligned} A \rightarrow B \\ C \cap D \end{aligned}$$')


def test_literal_code_currency_and_unknown_math_are_preserved():
    for source in [r'Cost $12 and $20', r'`$\text{code}$`',
                   '```python\nx = "$value$"\n```', r'$\unknown{data}$',
                   '#include <stdio.h>', 'x < y']:
        assert searchable_text(source) == source


def test_plain_text_writer_decodes_math_and_table_cells(tmp_path):
    dest = tmp_path / 'text.txt'
    TextHandler().embed_structured_text('', str(dest), {0: [
        ([0, 0, 1, 1], r'$\text{Project 1-3}$'),
        ([0, 0, 1, 1], r'<table><tr><td>$A \rightarrow B$</td></tr></table>'),
    ]})
    assert dest.read_text(encoding='utf-8') == 'Project 1-3\nA → B\n'


def test_pdf_search_layer_preserves_greek_and_relational_operators():
    with fitz.open() as doc:
        page = doc.new_page()
        PDFHandler._draw_invisible_text(page, [.1, .1, .9, .2],
                                       r'$\pi_{name}(R \bowtie S) \rightarrow T$',
                                       page.rect.width, page.rect.height)
        text = page.get_text()
        assert 'π_name(R ⋈ S) → T' in text
        assert page.search_for('R ⋈ S')


def test_malformed_table_span_preserves_cell_text_without_guessing_a_grid():
    from pdf_ocr.core._layout import table_cell_segments

    source = '<table><tr><td colspan>Original words</td></tr></table>'
    assert table_cell_segments([0, 0, 1, 1], source) == []
    assert searchable_text(source) == 'Original words'


def test_full_page_text_fallback_preserves_unicode_math():
    with fitz.open() as doc:
        page = doc.new_page()
        PDFHandler._draw_invisible_text(page, [0, 0, 1, 1],
                                       'Relational algebra\n' + r'$\pi_{name}(R \bowtie S)$',
                                       page.rect.width, page.rect.height)
        assert 'π_name(R ⋈ S)' in page.get_text()
