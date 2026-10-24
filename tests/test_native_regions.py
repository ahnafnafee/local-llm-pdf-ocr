import pytest
from PIL import Image, ImageDraw

from pdf_ocr.core.grounded import GroundedBlock
from pdf_ocr.core.native_regions import (
    RegionBlocks, anchor_local_blocks, plan_regions, recover_regions, repair_targets,
)


def source_with_boxes(boxes):
    image = Image.new('RGB', (800, 1000), 'white')
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(tuple(v * image.size[i % 2] for i, v in enumerate(box)), fill='black')
    return image


def test_region_planner_preserves_columns_without_cutting_lines():
    boxes = [[x, .03 + i * .06, x + .2, .05 + i * .06]
             for x in [.05, .4, .75] for i in range(14)]
    regions = plan_regions(boxes, limit=8)
    assert sorted(map(tuple, sum(regions, []))) == sorted(map(tuple, boxes))
    assert all(len(region) <= 8 for region in regions)
    assert all(len({box[0] for box in region}) == 1 for region in regions)


def test_visual_regions_recover_literal_labels_only():
    boxes = [[.1, .1, .3, .13], [.5, .1, .8, .13], [.1, .7, .8, .73]]
    source = source_with_boxes(boxes)
    blocks = [GroundedBlock(boxes[-1], 'Existing paragraph', 0)]
    dense, targets = repair_targets(blocks, boxes, source, [[0, 0, 1, .5]])
    assert not dense
    assert targets == boxes[:2]
    assert repair_targets(blocks, boxes, Image.new('RGB', source.size, 'white'), [])[1] == []


def test_local_anchor_corrects_width_without_swapping_neighbor_lines():
    boxes = [[.1, .2, .8, .23], [.1, .25, .9, .28]]
    blocks = [GroundedBlock([.1, .201, .3, .231], 'First', 0),
              GroundedBlock([.1, .252, .3, .282], 'Second', 0)]
    assert [b.bbox for b in anchor_local_blocks(blocks, boxes)] == boxes


def test_isolated_label_uses_its_source_box_despite_crop_coordinate_drift():
    box = [.526,.404,.551,.413]
    block = GroundedBlock([.533,.400,.544,.405], 'Short label', 0)
    result = anchor_local_blocks([block], [box])
    assert result[0].text == block.text
    assert result[0].bbox == box


def test_display_math_is_not_split_into_delimiters_and_body():
    text = '$$\n\\boxed{X \\mid Y}\n$$'
    block = GroundedBlock([.1, .1, .5, .2], text, 0)
    result = anchor_local_blocks([block], [])
    assert result == [block]


def test_overview_deduplication_preserves_short_labels_and_distant_repetitions():
    from pdf_ocr.core.native_regions import superseded_overview

    text = 'A long printed question about relational databases'
    local = GroundedBlock([.6, .1, .9, .15], text, 0)
    assert superseded_overview(GroundedBlock([.92,.1,.99,.15], text, 0), [local])
    assert not superseded_overview(GroundedBlock([.6,.8,.9,.85], text, 0), [local])
    assert not superseded_overview(GroundedBlock([.92,.1,.99,.15], 'Name', 0),
                                  [GroundedBlock([.6,.1,.9,.15], 'Name', 0)])


def test_table_columns_anchor_to_their_detected_cells():
    boxes = [[.1,.2,.25,.24], [.4,.2,.6,.24]]
    block = GroundedBlock([.1,.2,.6,.24], '<table><tr><td>First</td><td>Second</td></tr></table>', 0)
    result = anchor_local_blocks([block], boxes)
    assert [(b.bbox,b.text) for b in result] == [(boxes[0],'First'), (boxes[1],'Second')]


async def test_image_only_region_subdivides_and_preserves_existing_text():
    boxes = [[.1, .1, .3, .13], [.5, .1, .8, .13], [.1, .7, .8, .73]]
    source = source_with_boxes(boxes)
    existing = GroundedBlock(boxes[-1], 'Existing paragraph', 0)
    calls = []

    async def recognize(bounds, budget):
        calls.append((bounds, budget))
        contained = [b for b in boxes[:2] if bounds[0] <= b[0] and bounds[2] >= b[2]]
        if len(contained) > 1:
            return []  # Native image description was deliberately excluded.
        return [GroundedBlock(contained[0], 'Column label', 0)]

    result = await recover_regions(source, [existing], boxes, [[0, 0, 1, .5]], recognize)
    assert [b.text for b in result] == ['Column label', 'Column label', 'Existing paragraph']
    assert len(calls) == 3
    assert all(budget == 512 for _, budget in calls)


async def test_region_truncation_is_bounded_and_not_silently_accepted():
    from pdf_ocr.inference import IncompleteCompletionError

    box = [.1, .1, .3, .13]
    async def recognize(*args):
        raise IncompleteCompletionError('truncated')
    with pytest.raises(IncompleteCompletionError):
        await recover_regions(source_with_boxes([box]), [], [box], [[0, 0, 1, 1]], recognize)


async def test_dense_recovery_preserves_overview_when_a_region_has_no_text():
    boxes = [[.05 + x*.12, .05 + y*.12, .11 + x*.12, .07 + y*.12]
             for y in range(6) for x in range(8)]
    existing = GroundedBlock(boxes[-1], 'Retained words', 0)

    async def recognize(bounds, budget):
        if bounds[0] < .1:
            return [GroundedBlock(boxes[0], 'Recovered words', 0)]
        return []

    result = await recover_regions(source_with_boxes(boxes), [existing], boxes, [], recognize)
    assert {'Recovered words', 'Retained words'} <= {block.text for block in result}


async def test_refined_cell_replaces_its_unanchored_duplicate():
    boxes = [[.1,.4,.15,.412], [.3,.4,.4,.412], [.55,.4,.58,.412],
             [.65,.4,.7,.412], [.8,.4,.85,.412]]
    words = ['First', 'Second', 'Third', 'Fourth', 'Fifth']
    calls = 0

    async def recognize(bounds, budget):
        nonlocal calls
        calls += 1
        if calls == 1:
            blocks = [GroundedBlock(box, word, 0) for box, word in zip(boxes, words)]
            blocks[2].bbox = [.43,.395,.54,.419]
            return RegionBlocks(blocks, visual_regions=[[0,0,1,1]])
        return [GroundedBlock(boxes[2], 'Third', 0)]

    result = await recover_regions(source_with_boxes(boxes), [], boxes, [[0,0,1,1]], recognize)
    third = [block for block in result if block.text == 'Third']
    assert len(third) == 1
    assert third[0].bbox == boxes[2]


def test_refined_merge_keeps_distinct_repeated_labels():
    from pdf_ocr.core.native_regions import merge_refined

    boxes = [[.1,.4,.2,.42], [.8,.4,.9,.42]]
    first, second = [GroundedBlock(box, 'Name', 0) for box in boxes]
    assert merge_refined([first], [second], boxes) == [first, second]
    displaced = GroundedBlock([.1,.3,.2,.32], 'Name', 0)
    assert merge_refined([displaced], [second], boxes) == [displaced, second]


async def test_detailed_source_is_lazy_and_initial_pixels_are_preserved():
    import base64
    import io
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from pdf_ocr.core.grounded import PromptedGroundedOCR

    image = source_with_boxes([[.1,.1,.4,.14]])
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    encoded = base64.b64encode(buffer.getvalue()).decode()
    complete = AsyncMock(return_value='![text](100,100,400,140) Source words')
    loader = Mock(side_effect=AssertionError('A complete page must not render a second source'))
    backend = PromptedGroundedOCR(model='lightonai/LightOnOCR-3-4B',
                                  client=SimpleNamespace(complete=complete))
    result = await backend.ocr_image(encoded, detected_boxes=[[.1,.1,.4,.14]], source_loader=loader)
    assert not result.page_errors
    loader.assert_not_called()
    assert complete.await_count == 1
    assert complete.call_args.args[1] == encoded


async def test_omitted_figure_labels_load_detailed_pixels_once():
    import base64
    import io
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from pdf_ocr.core.grounded import PromptedGroundedOCR

    box = [.1,.1,.4,.14]
    image = source_with_boxes([box])
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    complete = AsyncMock(side_effect=[
        '![image](50,50,800,500) A generated diagram description',
        '![text](40,140,960,860) Source label',
    ])
    loader = Mock(return_value=image.resize((1600,2000)))
    backend = PromptedGroundedOCR(model='lightonai/LightOnOCR-3-4B',
                                  client=SimpleNamespace(complete=complete))
    result = await backend.ocr_image(base64.b64encode(buffer.getvalue()).decode(),
                                     detected_boxes=[box], source_loader=loader)
    assert not result.page_errors
    loader.assert_called_once_with()
    assert complete.await_count == 2
    assert [b.text for b in result.blocks] == ['Source label']
    assert result.blocks[0].bbox == box
