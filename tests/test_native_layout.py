from pdf_ocr.core.native_layout import anchor_native_blocks


def test_native_prose_notation_uses_only_unique_source_reading():
    from pdf_ocr.core.native_layout import restore_literal_identifiers

    assert restore_literal_identifiers("- $a_p$ refers to the network", "- API refers to the network") == "- $a_p$ refers to the network"
    source = "- $i_p$ identifies this network interface\nNext line"
    assert restore_literal_identifiers(source, "- IP identifies this network interface") == "- IP identifies this network interface\nNext line"
    assert restore_literal_identifiers(source, "- IT identifies this network interface\n- IP identifies this network interface") == source
    assert restore_literal_identifiers(source, "IP identifies a different network interface") == source
    for math in ["$x_i$", "$a_2$ identifies this network interface", "$a_b + c$ is a valid equation"]:
        assert restore_literal_identifiers(math, "AB identifies this network interface") == math


def test_native_geometry_prevents_shifted_fields_and_preserves_spelling():
    boxes = [[.1, .3, .9, .34], [.1, .5, .9, .54], [.1, .7, .9, .74]]
    blocks = [([.1, .305, .9, .345], "Source typo"),
              ([.1, .705, .9, .745], "An answer")]
    assert anchor_native_blocks(blocks, boxes) == [(boxes[0], "Source typo"), (boxes[2], "An answer")]


def test_native_reading_order_and_unmatched_content_survive_detection():
    first = ([.5, .2, .9, .3], "Right column first")
    second = ([.1, .1, .3, .15], "Unmatched heading")
    assert anchor_native_blocks([first, second], []) == [first, second]


def test_ambiguous_detector_boxes_do_not_override_native_geometry():
    block = ([.1, .2, .9, .3], "A line")
    assert anchor_native_blocks([block], [[.1,.2,.9,.3], [.1,.201,.9,.301]]) == [block]


def test_native_multiline_block_anchors_each_line_once():
    boxes = [[.1,.2,.9,.25], [.1,.25,.9,.3]]
    assert anchor_native_blocks([([.1,.2,.9,.3], "One\nTwo")], boxes) == [
        (boxes[0], "One"), (boxes[1], "Two")]


def test_native_html_table_retains_structure():
    block = ([.1,.2,.9,.5], "<table>\n<tr><td>First</td></tr>\n</table>")
    assert anchor_native_blocks([block], []) == [block]
