"""Unit tests for convfinqa.graph.FinancialGraph with mocked Graphiti client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.convfinqa.dataset.models import ConvFinQARecord

# pre/post text must exceed the 20-char filter in graph.py to produce episode chunks
OTHER_RECORD = ConvFinQARecord.model_validate({
    "id": "Single_OTHER/2010/page_1.pdf",
    "doc": {
        "pre_text": "Some pre text describing the data.",
        "post_text": "Some post text discussing the figures.",
        "table": {"2010": {"revenue": 200.0}},
    },
    "dialogue": {
        "conv_questions": ["q"], "conv_answers": ["a"],
        "turn_program": ["200"], "executed_answers": [200.0], "qa_split": [False],
    },
    "features": {
        "num_dialogue_turns": 1, "has_type2_question": False,
        "has_duplicate_columns": False, "has_non_numeric_values": False,
    },
})

SAMPLE_RECORD = ConvFinQARecord.model_validate({
    "id": "Single_TEST/2009/page_1.pdf",
    "doc": {
        "pre_text": "Context text describing the financial document for 2009.",
        "post_text": "Post context with additional narrative about the report.",
        "table": {"2009": {"net income": 100.0}, "2008": {"net income": 90.0}},
    },
    "dialogue": {
        "conv_questions": ["what was net income?"],
        "conv_answers": ["100"],
        "turn_program": ["100"],
        "executed_answers": [100.0],
        "qa_split": [False],
    },
    "features": {
        "num_dialogue_turns": 1,
        "has_type2_question": False,
        "has_duplicate_columns": False,
        "has_non_numeric_values": False,
    },
})


@pytest.fixture
def mock_graphiti() -> MagicMock:
    mock = MagicMock()
    mock.build_indices_and_constraints = AsyncMock()
    mock.add_episode = AsyncMock()
    mock.add_episode_bulk = AsyncMock()
    mock.add_triplet = AsyncMock()
    # search() returns an object with .edges and .episodes (current API)
    search_result = MagicMock()
    search_result.edges = []
    search_result.episodes = []
    mock.search_with_config = AsyncMock(return_value=search_result)
    mock.driver = MagicMock()
    mock.driver.execute_query = AsyncMock(return_value=([{"cnt": 0}], None, None))
    return mock


@pytest.fixture
def financial_graph(mock_graphiti: MagicMock):
    with patch("src.lib.graph.graph.Neo4jDriver"), \
         patch("src.lib.graph.graph.OpenAIClient"), \
         patch("src.lib.graph.graph.OpenAIEmbedder"), \
         patch("src.lib.graph.graph._FinancialGraphiti", return_value=mock_graphiti):
        from src.lib.graph import FinancialGraph
        graph = FinancialGraph()
        graph._graphiti = mock_graphiti
        return graph


@pytest.mark.asyncio
async def test_index_record_creates_episodes(financial_graph, mock_graphiti) -> None:
    await financial_graph.index_record(SAMPLE_RECORD)
    # Hybrid: triplets for table + episodes for text
    assert mock_graphiti.add_triplet.call_count > 0
    assert mock_graphiti.add_episode_bulk.call_count == 1
    episodes = mock_graphiti.add_episode_bulk.call_args[0][0]
    assert len(episodes) == 2  # pre_text + post_text
    call_kwargs = mock_graphiti.add_episode_bulk.call_args[1]
    assert call_kwargs["group_id"] == "Single_TEST-2009-page_1-pdf"


@pytest.mark.asyncio
async def test_table_edge_uuid_is_deterministic(financial_graph, mock_graphiti) -> None:
    """Each table cell maps to a stable edge UUID, so re-indexing the same cell
    upserts instead of inserting a duplicate edge (one (row, column) == one edge)."""
    import hashlib
    import re
    import uuid as _uuid

    await financial_graph.index_record(SAMPLE_RECORD)
    edges = [call.args[1] for call in mock_graphiti.add_triplet.call_args_list]
    gid = "Single_TEST-2009-page_1-pdf"

    def expected(metric: str, column: str) -> str:
        h = hashlib.md5(f"{gid}::{metric}|{column}|HAS_VALUE_IN_COLUMN".encode()).hexdigest()
        return str(_uuid.UUID(h))

    assert len(edges) == 2  # two cells -> two edges
    assert len({e.uuid for e in edges}) == 2  # distinct per cell, no collision
    for e in edges:
        column, metric = re.search(r"In column \[(.+?)\], the row \[(.+?)\]", e.fact).groups()
        assert e.uuid == expected(metric, column)  # stable upsert key, not random


@pytest.mark.asyncio
async def test_index_record_skips_when_group_already_in_graph(financial_graph, mock_graphiti) -> None:
    """A fresh process (empty in-memory set) must still skip a group that already
    has nodes in the graph, so re-indexing never duplicates."""
    mock_graphiti.driver.execute_query = AsyncMock(return_value=([{"cnt": 5}], None, None))
    mock_graphiti.add_triplet = AsyncMock()
    mock_graphiti.add_episode_bulk = AsyncMock()

    await financial_graph.index_record(SAMPLE_RECORD)

    assert mock_graphiti.add_triplet.call_count == 0  # nothing re-inserted
    assert mock_graphiti.add_episode_bulk.call_count == 0


@pytest.mark.asyncio
async def test_index_record_idempotent(financial_graph, mock_graphiti) -> None:
    await financial_graph.index_record(SAMPLE_RECORD)
    await financial_graph.index_record(SAMPLE_RECORD)
    assert mock_graphiti.add_episode_bulk.call_count == 1


@pytest.mark.asyncio
async def test_index_bulk(financial_graph, mock_graphiti, tmp_path) -> None:
    # Use a fresh checkpoint file so the production checkpoint doesn't skip the test record
    ckpt = tmp_path / "ckpt.json"
    await financial_graph.index_bulk([SAMPLE_RECORD], checkpoint_path=str(ckpt))
    assert mock_graphiti.add_episode_bulk.call_count == 1


@pytest.mark.asyncio
async def test_financial_graphiti_search_with_config_delegates_to_search(mock_graphiti) -> None:
    """_FinancialGraphiti.search_with_config() must call the underlying _search."""
    from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF

    from src.lib.graph.graph import _FinancialGraphiti

    instance = _FinancialGraphiti.__new__(_FinancialGraphiti)
    instance._search = AsyncMock(return_value=MagicMock(edges=[], episodes=[]))
    config = COMBINED_HYBRID_SEARCH_RRF.model_copy(update={"limit": 10})
    await instance.search_with_config(query="net income", group_ids=["grp"], config=config)
    instance._search.assert_awaited_once_with(query="net income", group_ids=["grp"], config=config)


@pytest.mark.asyncio
async def test_search_returns_edge_facts(financial_graph, mock_graphiti) -> None:
    edge = MagicMock()
    edge.fact = "net income in 2009 was 100.0"
    search_result = MagicMock()
    search_result.edges = [edge]
    search_result.episodes = []
    search_result.edge_reranker_scores = [0.5]
    search_result.episode_reranker_scores = []
    mock_graphiti.search_with_config = AsyncMock(return_value=search_result)

    results = await financial_graph.search("Single_TEST/2009/page_1.pdf", "net income 2009")
    assert results == ["[relevance 0.500] net income in 2009 was 100.0"]
    mock_graphiti.search_with_config.assert_called_once()


@pytest.mark.asyncio
async def test_search_returns_empty_list(financial_graph, mock_graphiti) -> None:
    search_result = MagicMock()
    search_result.edges = []
    search_result.episodes = []
    search_result.edge_reranker_scores = []
    search_result.episode_reranker_scores = []
    mock_graphiti.search_with_config = AsyncMock(return_value=search_result)
    assert await financial_graph.search("Single_TEST/2009/page_1.pdf", "unknown") == []


# build_indices error handling -----------------------------------------------

@pytest.mark.asyncio
async def test_build_indices_swallows_equivalent_schema_error(
    financial_graph, mock_graphiti
) -> None:
    """EquivalentSchemaRuleAlreadyExists is benign on idempotent re-runs and must not raise."""
    mock_graphiti.build_indices_and_constraints = AsyncMock(
        side_effect=Exception("EquivalentSchemaRuleAlreadyExists: index already exists")
    )
    await financial_graph.build_indices()


@pytest.mark.asyncio
async def test_build_indices_reraises_unrelated_errors(
    financial_graph, mock_graphiti
) -> None:
    """Non-schema errors must still propagate."""
    mock_graphiti.build_indices_and_constraints = AsyncMock(
        side_effect=Exception("Connection refused")
    )
    with pytest.raises(Exception, match="Connection refused"):
        await financial_graph.build_indices()


# is_indexed caching ---------------------------------------------------------

@pytest.mark.asyncio
async def test_is_indexed_caches_after_db_hit(financial_graph, mock_graphiti) -> None:
    """First call queries Neo4j; subsequent calls hit the in-memory cache."""
    mock_graphiti.driver.execute_query = AsyncMock(
        return_value=([{"cnt": 5}], None, None)
    )
    assert await financial_graph.is_indexed("some-gid") is True
    assert await financial_graph.is_indexed("some-gid") is True
    assert mock_graphiti.driver.execute_query.call_count == 1


@pytest.mark.asyncio
async def test_is_indexed_returns_false_when_no_rows(
    financial_graph, mock_graphiti
) -> None:
    mock_graphiti.driver.execute_query = AsyncMock(
        return_value=([{"cnt": 0}], None, None)
    )
    assert await financial_graph.is_indexed("missing-gid") is False


# index_record edge cases ----------------------------------------------------

@pytest.mark.asyncio
async def test_index_record_skips_empty_pre_post_text(
    financial_graph, mock_graphiti
) -> None:
    """Whitespace-only pre/post text should not produce episodes."""
    record = ConvFinQARecord.model_validate({
        "id": "Single_EMPTY/2009/page_1.pdf",
        "doc": {
            "pre_text": "   ",
            "post_text": "",
            "table": {"2009": {"net income": 100.0}},
        },
        "dialogue": {
            "conv_questions": ["q"], "conv_answers": ["a"],
            "turn_program": ["100"], "executed_answers": [100.0], "qa_split": [False],
        },
        "features": {
            "num_dialogue_turns": 1, "has_type2_question": False,
            "has_duplicate_columns": False, "has_non_numeric_values": False,
        },
    })
    await financial_graph.index_record(record)
    assert mock_graphiti.add_episode_bulk.call_count == 0
    assert mock_graphiti.add_triplet.call_count > 0


@pytest.mark.asyncio
async def test_index_record_drops_short_chunks(
    financial_graph, mock_graphiti
) -> None:
    """Chunks shorter than MIN_CHUNK_CHARS should be filtered out."""
    short_chunk = MagicMock()
    short_chunk.text = "tiny"  # < MIN_CHUNK_CHARS
    long_chunk = MagicMock()
    long_chunk.text = "This chunk has more than twenty characters of content."

    with patch("src.lib.graph.index._CHUNKER", return_value=[short_chunk, long_chunk]):
        await financial_graph.index_record(SAMPLE_RECORD)

    episodes = mock_graphiti.add_episode_bulk.call_args[0][0]
    # pre_text contributes 1 valid chunk + post_text contributes 1 valid chunk
    assert len(episodes) == 2
    assert all("twenty characters" in ep.content for ep in episodes)


# _index_table_triplets edge cases -------------------------------------------

@pytest.mark.asyncio
async def test_index_table_triplets_skips_empty_keys(
    financial_graph, mock_graphiti
) -> None:
    """Empty column or metric keys are skipped (defensive against dirty data)."""
    record = ConvFinQARecord.model_validate({
        "id": "Single_DIRTY/2009/page_1.pdf",
        "doc": {
            "pre_text": "Some pre text describing the data.",
            "post_text": "Some post text discussing the figures.",
            "table": {
                "": {"net income": 100.0},          # empty column key
                "2009": {"": 50.0, "revenue": 200.0},  # empty metric key
            },
        },
        "dialogue": {
            "conv_questions": ["q"], "conv_answers": ["a"],
            "turn_program": ["100"], "executed_answers": [100.0], "qa_split": [False],
        },
        "features": {
            "num_dialogue_turns": 1, "has_type2_question": False,
            "has_duplicate_columns": False, "has_non_numeric_values": False,
        },
    })
    await financial_graph.index_record(record)
    # Only the "2009"/"revenue" cell is valid all others have empty keys
    assert mock_graphiti.add_triplet.call_count == 1


@pytest.mark.asyncio
async def test_index_table_triplets_unparseable_year_uses_reference_time(
    financial_graph, mock_graphiti
) -> None:
    """Columns that don't parse to a year fall back to reference_time."""
    record = ConvFinQARecord.model_validate({
        "id": "Single_NOYEAR/2009/page_1.pdf",
        "doc": {
            "pre_text": "Some pre text describing the data.",
            "post_text": "Some post text discussing the figures.",
            "table": {"total": {"net income": 100.0}},  # "total" has no year
        },
        "dialogue": {
            "conv_questions": ["q"], "conv_answers": ["a"],
            "turn_program": ["100"], "executed_answers": [100.0], "qa_split": [False],
        },
        "features": {
            "num_dialogue_turns": 1, "has_type2_question": False,
            "has_duplicate_columns": False, "has_non_numeric_values": False,
        },
    })
    await financial_graph.index_record(record)
    # Triplet was still created year just couldn't be parsed (uses ref time instead)
    assert mock_graphiti.add_triplet.call_count == 1


# index_bulk failure paths ---------------------------------------------------

@pytest.mark.asyncio
async def test_index_bulk_skips_completed_from_checkpoint(
    financial_graph, mock_graphiti, tmp_path
) -> None:
    """Records already in the checkpoint should not be re-indexed."""
    ckpt = tmp_path / "ckpt.json"
    ckpt.write_text('{"completed": ["Single_TEST/2009/page_1.pdf"]}')
    await financial_graph.index_bulk([SAMPLE_RECORD], checkpoint_path=str(ckpt))
    assert mock_graphiti.add_episode_bulk.call_count == 0


@pytest.mark.asyncio
async def test_index_bulk_recovers_from_corrupted_checkpoint(
    financial_graph, mock_graphiti, tmp_path
) -> None:
    """A truncated/corrupted checkpoint file is ignored; indexing starts fresh."""
    ckpt = tmp_path / "ckpt.json"
    ckpt.write_text('{"completed": ["Single_TEST/2009/page_1.pdf"')  # truncated JSON
    # Should not raise; the corrupted file is treated as an empty checkpoint.
    await financial_graph.index_bulk([SAMPLE_RECORD], checkpoint_path=str(ckpt))
    # Record was re-indexed (checkpoint was not trusted).
    assert mock_graphiti.add_episode_bulk.call_count == 1


@pytest.mark.asyncio
async def test_index_bulk_stops_on_quota_exhausted(
    financial_graph, mock_graphiti, tmp_path
) -> None:
    """OpenAI quota exhaustion halts the loop without raising."""
    ckpt = tmp_path / "ckpt.json"
    mock_graphiti.add_triplet = AsyncMock(side_effect=Exception("insufficient_quota"))
    await financial_graph.index_bulk([SAMPLE_RECORD, OTHER_RECORD], checkpoint_path=str(ckpt))
    # Nothing should be in the checkpoint (both records failed at the quota wall)
    import json as _json
    saved = _json.loads(ckpt.read_text()) if ckpt.exists() else {"completed": []}
    assert SAMPLE_RECORD.id not in saved.get("completed", [])


@pytest.mark.asyncio
async def test_index_bulk_quota_short_circuits_later_records(
    financial_graph, mock_graphiti, tmp_path
) -> None:
    """Once one record hits insufficient_quota, later records are skipped without retrying."""
    ckpt = tmp_path / "ckpt.json"
    call_counts = {"index": 0}

    async def fail_first_record(record):
        call_counts["index"] += 1
        if record.id == SAMPLE_RECORD.id:
            raise Exception("insufficient_quota")
        # Other records should never get here because the quota flag was set

    with patch.object(financial_graph._indexer, "index_record", side_effect=fail_first_record):
        await financial_graph.index_bulk(
            [SAMPLE_RECORD, OTHER_RECORD], checkpoint_path=str(ckpt), concurrency=1
        )
    # Only the first record was attempted the quota flag short-circuited the rest
    assert call_counts["index"] == 1


@pytest.mark.asyncio
async def test_index_bulk_logs_and_gives_up_after_5_generic_failures(
    financial_graph, mock_graphiti, tmp_path
) -> None:
    """A non-quota, non-rate-limit error retries up to 5 times then gives up."""
    ckpt = tmp_path / "ckpt.json"
    attempts = {"n": 0}

    async def always_fail(record):
        attempts["n"] += 1
        raise Exception("transient neo4j blip")

    with patch.object(financial_graph._indexer, "index_record", side_effect=always_fail):
        await financial_graph.index_bulk([SAMPLE_RECORD], checkpoint_path=str(ckpt))
    # All 5 attempts were made (covers both the "Retry N" warning + the final "Gave up" error branches)
    assert attempts["n"] == 5
    import json as _json
    saved = _json.loads(ckpt.read_text()) if ckpt.exists() else {"completed": []}
    assert SAMPLE_RECORD.id not in saved.get("completed", [])


@pytest.mark.asyncio
async def test_index_bulk_aborts_mid_retry_when_quota_flag_flips(
    financial_graph, mock_graphiti, tmp_path
) -> None:
    """A record stuck mid-retry must abort when a concurrent record exhausts the quota.

    Race choreography:
      - Record A's first attempt raises rate_limit, then awaits asyncio.sleep
        (patched to wait on an Event).
      - Record B raises insufficient_quota the except handler sets
        quota_exhausted=True, then calls logger.error.
      - logger.error is patched to release A by setting the Event.
      - When A's fake_sleep returns, the next attempt checks the flag
        (now True) and short-circuits at the `if quota_exhausted: return`
        guard inside the attempt loop.
    """
    import asyncio as _asyncio
    ckpt = tmp_path / "ckpt.json"

    record_a = SAMPLE_RECORD
    record_b = OTHER_RECORD

    quota_signal = _asyncio.Event()
    a_attempts = {"n": 0}

    async def dispatch_index(record):
        if record.id == record_a.id:
            a_attempts["n"] += 1
            if a_attempts["n"] == 1:
                raise Exception("rate_limit_exceeded")
            # Second attempt should never be reached.
            raise AssertionError("record A should not run a second attempt")
        # Record B: raise quota immediately (no awaits) so the except handler
        # sets quota_exhausted=True before A's sleep is released.
        raise Exception("insufficient_quota")

    async def fake_sleep(_secs):
        # Block A until B's exception handler has set the quota flag.
        await quota_signal.wait()

    from src.lib.graph import index as index_mod
    real_error = index_mod.logger.error

    def signal_on_quota_log(msg, *args, **kwargs):
        # graph_mod.logger.error is called by the except handler AFTER
        # `quota_exhausted = True`, so flipping the event here guarantees
        # A wakes up to a True flag.
        if "quota" in msg.lower():
            quota_signal.set()
        return real_error(msg, *args, **kwargs)

    with patch.object(financial_graph._indexer, "index_record", side_effect=dispatch_index), \
         patch("src.lib.graph.index.asyncio.sleep", new=fake_sleep), \
         patch.object(index_mod.logger, "error", side_effect=signal_on_quota_log):
        await financial_graph.index_bulk(
            [record_a, record_b], checkpoint_path=str(ckpt), concurrency=2
        )

    # Only A's first attempt ran the quota guard handled the rest.
    assert a_attempts["n"] == 1


@pytest.mark.asyncio
async def test_index_bulk_third_record_skips_at_inner_semaphore_guard(
    financial_graph, mock_graphiti, tmp_path
) -> None:
    """With concurrency=2 and 3 records: A and B fill the semaphore, C waits.
    A raises quota (sets flag), exits semaphore, C acquires it and must hit
    the inner quota guard (line 318: `if quota_exhausted: return`)."""
    ckpt = tmp_path / "ckpt.json"

    def _rec(rid):
        return ConvFinQARecord.model_validate({
            "id": rid,
            "doc": {"pre_text": "x", "post_text": "x", "table": {"2010": {"v": 1.0}}},
            "dialogue": {"conv_questions": ["q"], "conv_answers": ["a"],
                         "turn_program": ["1"], "executed_answers": [1.0], "qa_split": [False]},
            "features": {"num_dialogue_turns": 1, "has_type2_question": False,
                         "has_duplicate_columns": False, "has_non_numeric_values": False},
        })

    records = [_rec(f"Single_R{i}/2010/page_1.pdf") for i in range(3)]

    async def fail_quota(record):
        raise Exception("insufficient_quota")

    with patch.object(financial_graph._indexer, "index_record", side_effect=fail_quota):
        await financial_graph.index_bulk(records, checkpoint_path=str(ckpt), concurrency=2)

    import json as _json
    saved = _json.loads(ckpt.read_text()) if ckpt.exists() else {"completed": []}
    assert saved.get("completed", []) == []


@pytest.mark.asyncio
async def test_index_bulk_retries_on_rate_limit(
    financial_graph, mock_graphiti, tmp_path
) -> None:
    """Rate-limit errors trigger backoff retry instead of skipping the record."""
    ckpt = tmp_path / "ckpt.json"
    calls = {"n": 0}

    async def flaky_triplet(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise Exception("rate_limit_exceeded")
        # succeed on the 3rd attempt onwards

    mock_graphiti.add_triplet = AsyncMock(side_effect=flaky_triplet)
    # Patch asyncio.sleep so the test doesn't actually wait 60s
    with patch("src.lib.graph.index.asyncio.sleep", new=AsyncMock()):
        await financial_graph.index_bulk([SAMPLE_RECORD], checkpoint_path=str(ckpt))
    assert calls["n"] >= 3


# search episode content -----------------------------------------------------

@pytest.mark.asyncio
async def test_search_includes_episode_content_and_dedupes(
    financial_graph, mock_graphiti
) -> None:
    """Episodes' content joins the facts list; duplicate content is filtered."""
    edge = MagicMock()
    edge.fact = "fact A"
    ep1 = MagicMock()
    ep1.content = "narrative chunk one"
    ep2 = MagicMock()
    ep2.content = "narrative chunk one"  # duplicate, should be deduped
    ep3 = MagicMock()
    ep3.content = "narrative chunk two"
    ep_empty = MagicMock()
    ep_empty.content = ""  # empty content skipped

    search_result = MagicMock()
    search_result.edges = [edge]
    search_result.episodes = [ep1, ep2, ep3, ep_empty]
    search_result.edge_reranker_scores = [0.9]
    search_result.episode_reranker_scores = [0.8, 0.8, 0.7, 0.0]
    mock_graphiti.search_with_config = AsyncMock(return_value=search_result)

    # Sorted best-first by reranker score; scores prefixed; duplicates filtered.
    results = await financial_graph.search("Single_TEST/2009/page_1.pdf", "any query")
    assert results == [
        "[relevance 0.900] fact A",
        "[relevance 0.800] narrative chunk one",
        "[relevance 0.700] narrative chunk two",
    ]


@pytest.mark.asyncio
async def test_search_sorts_best_first_without_trimming(
    financial_graph, mock_graphiti
) -> None:
    """A high-scoring narrative chunk ranks ahead of lower-scoring table cells,
    and nothing is trimmed (Variant A: surface the score, let the LLM decide)."""
    # 20 table edges scored 0.01..0.20, one narrative chunk scored 0.99.
    edges = [MagicMock(fact=f"cell {i}") for i in range(20)]
    edge_scores = [round(0.01 * (i + 1), 2) for i in range(20)]
    ep = MagicMock()
    ep.content = "the narrative answer"

    search_result = MagicMock()
    search_result.edges = edges
    search_result.episodes = [ep]
    search_result.edge_reranker_scores = edge_scores
    search_result.episode_reranker_scores = [0.99]
    mock_graphiti.search_with_config = AsyncMock(return_value=search_result)

    results = await financial_graph.search("Single_TEST/2009/page_1.pdf", "q")
    assert len(results) == 21  # nothing dropped
    assert results[0] == "[relevance 0.990] the narrative answer"  # highest score first
    # the next line is the highest-scoring table cell (0.20), confirming the sort
    assert results[1] == "[relevance 0.200] cell 19"


@pytest.mark.asyncio
async def test_search_handles_missing_scores(financial_graph, mock_graphiti) -> None:
    """If the backend returns no reranker scores, facts still come back unprefixed."""
    edge = MagicMock(fact="fact A")
    search_result = MagicMock()
    search_result.edges = [edge]
    search_result.episodes = []
    search_result.edge_reranker_scores = []
    search_result.episode_reranker_scores = []
    mock_graphiti.search_with_config = AsyncMock(return_value=search_result)

    results = await financial_graph.search("Single_TEST/2009/page_1.pdf", "q")
    assert results == ["fact A"]  # no [relevance ...] prefix when score is absent


# search_narrative ------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_narrative_tags_pre_and_post_sources(
    financial_graph, mock_graphiti
) -> None:
    """search_narrative parses the episode name to assign 'pre'/'post'/'unknown' source."""
    ep_pre = MagicMock(content="pre text content", name="rec::pre_text::0")
    ep_pre.name = "rec::pre_text::0"
    ep_post = MagicMock(content="post text content", name="rec::post_text::1")
    ep_post.name = "rec::post_text::1"
    ep_unknown = MagicMock(content="weirdly-named content")
    ep_unknown.name = "some_other_name"
    ep_empty = MagicMock(content="   ")
    ep_empty.name = "rec::pre_text::2"

    search_result = MagicMock()
    search_result.episodes = [ep_pre, ep_post, ep_unknown, ep_empty]
    mock_graphiti.search_with_config = AsyncMock(return_value=search_result)

    results = await financial_graph.search_narrative("rec", "any query")
    assert results == [
        {"source": "pre", "snippet": "pre text content"},
        {"source": "post", "snippet": "post text content"},
        {"source": "unknown", "snippet": "weirdly-named content"},
    ]


@pytest.mark.asyncio
async def test_search_narrative_dedupes_repeated_chunks(
    financial_graph, mock_graphiti
) -> None:
    """Repeated content (same string) is returned only once."""
    ep1 = MagicMock(content="dup")
    ep1.name = "rec::pre_text::0"
    ep2 = MagicMock(content="dup")
    ep2.name = "rec::pre_text::1"
    search_result = MagicMock()
    search_result.episodes = [ep1, ep2]
    mock_graphiti.search_with_config = AsyncMock(return_value=search_result)

    results = await financial_graph.search_narrative("rec", "q")
    assert len(results) == 1


@pytest.mark.asyncio
async def test_search_narrative_strips_amp_and_slash(
    financial_graph, mock_graphiti
) -> None:
    """`&` and `/` must be stripped to avoid BM25 tokenizer breaks."""
    search_result = MagicMock()
    search_result.episodes = []
    mock_graphiti.search_with_config = AsyncMock(return_value=search_result)
    await financial_graph.search_narrative("rec", "rd&e and segment a/b")
    call_kwargs = mock_graphiti.search_with_config.call_args.kwargs
    assert "&" not in call_kwargs["query"]
    assert "/" not in call_kwargs["query"]


# search_table ----------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_table_parses_cell_facts(financial_graph, mock_graphiti) -> None:
    """search_table parses 'In column [X], the row [Y] has value V' edges back into cells."""
    edge1 = MagicMock(fact="In column [2009], the row [net income] has value 100")
    edge2 = MagicMock(fact="In column [2008], the row [net income] has value 90.5")
    edge3 = MagicMock(fact="In column [2009], the row [net income] has value 100")  # dup
    edge_bad = MagicMock(fact="not a cell fact")
    edge_none = MagicMock(fact=None)
    edge_str_val = MagicMock(fact="In column [2009], the row [region] has value APAC")

    search_result = MagicMock()
    search_result.edges = [edge1, edge2, edge3, edge_bad, edge_none, edge_str_val]
    mock_graphiti.search_with_config = AsyncMock(return_value=search_result)

    cells = await financial_graph.search_table("rec", "net income 2009")
    assert {"column": "2009", "row": "net income", "value": 100.0} in cells
    assert {"column": "2008", "row": "net income", "value": 90.5} in cells
    # Non-numeric value passes through as string
    assert {"column": "2009", "row": "region", "value": "APAC"} in cells
    # Duplicate cell (same column+row) is dropped
    assert len([c for c in cells if c["column"] == "2009" and c["row"] == "net income"]) == 1
    # Malformed facts are skipped (no exception)
    assert len(cells) == 3


@pytest.mark.asyncio
async def test_search_table_strips_amp_and_slash(financial_graph, mock_graphiti) -> None:
    """`&` and `/` must be stripped from the search_table query too."""
    search_result = MagicMock()
    search_result.edges = []
    mock_graphiti.search_with_config = AsyncMock(return_value=search_result)
    await financial_graph.search_table("rec", "r&d 2008/2009")
    call_kwargs = mock_graphiti.search_with_config.call_args.kwargs
    assert "&" not in call_kwargs["query"]
    assert "/" not in call_kwargs["query"]


# close ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_close_calls_graphiti_close(financial_graph, mock_graphiti) -> None:
    mock_graphiti.close = AsyncMock()
    await financial_graph.close()
    mock_graphiti.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_is_reachable_returns_true_when_neo4j_responds(financial_graph, mock_graphiti) -> None:
    """is_reachable() returns True when the driver responds without error."""
    mock_graphiti.driver.execute_query = AsyncMock(return_value=([{"1": 1}], None, None))
    assert await financial_graph.is_reachable() is True


@pytest.mark.asyncio
async def test_is_reachable_returns_false_on_connection_error(financial_graph, mock_graphiti) -> None:
    """is_reachable() returns False when the driver raises (Neo4j unreachable)."""
    mock_graphiti.driver.execute_query = AsyncMock(side_effect=Exception("connection refused"))
    assert await financial_graph.is_reachable() is False
