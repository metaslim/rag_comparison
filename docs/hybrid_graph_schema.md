# Hybrid Graph Schema: A Worked Example

A concrete walkthrough of how the hybrid indexing strategy (direct triplets for table cells + LLM-extracted episodes for narrative) materialises in Neo4j for one real, indexed ConvFinQA record. Every node, edge, and count below is taken from the live graph, so it shows exactly how `search_table` and `search_narrative` hit different parts of the same `group_id`.

**Record**: `Single_MRO/2007/page_134.pdf-1` (Marathon Oil, 2007, stock-option fair-value assumptions). One of the 100 pre-indexed dev records; scores **5/5** in both Graph-RAG and Agentic RAG.

**Neo4j `group_id`**: `Single_MRO-2007-page_134-pdf` (the `-N` record suffix is stripped, so the four `page_134` records share one graph).

---

## The table

| Assumption | 2007 | 2006 | 2005 |
|---|---|---|---|
| **weighted average exercise price per share** | **60.94** | **37.84** | **25.14** |
| expected annual dividends per share | 0.96 | 0.80 | 0.66 |
| expected life in years | 5.0 | 5.1 | 5.5 |
| expected volatility | -27.0 | -28.0 | -28.0 |
| risk-free interest rate | -4.1 | -5.0 | -3.8 |
| weighted average grant date fair value of stock option awards granted | 17.24 | 10.19 | 6.15 |

(Verbatim from the dataset; the negative volatility/rate cells are how the source stores them.)

---

## Two indexing paths, one `group_id`

| Source | Path | Edge in Neo4j | Edges |
|---|---|---|---|
| Table cells | Direct triplet, **no LLM** | `(:Entity)-[:RELATES_TO {name:"HAS_VALUE_IN_COLUMN"}]->(:Entity)` | 18 |
| Narrative | Chunked, **GPT** extracts | `(:Entity)-[:RELATES_TO {name:<relation>}]->(:Entity)` + `:Episodic` | 14 |

Both write `:RELATES_TO` edges, told apart by the `name` property.

```
group_id: Single_MRO-2007-page_134-pdf
  |
  |-- TABLE      18 triplets   (:row)-[HAS_VALUE_IN_COLUMN]->(:year)   no LLM, exact value
  |-- NARRATIVE  14 relations  (:entity)-[RELATES_TO {name}]->(:entity)   GPT-extracted
  |-- EPISODIC   13 chunks      the raw ~400-char narrative slices GPT read
  |
  `-- shared nodes: (:Entity {name:"2007"}), (:Entity {name:"2006"})
       the year columns are the only structural link between the two paths
```

---

## The graph

### Table side: 18 triplets (no LLM)

6 row-entities x 3 year-entities (`2007`, `2006`, `2005`), one edge per cell. The `fact` carries the exact value:

```
(:Entity {name:"weighted average exercise price per share"})
   -[:RELATES_TO {name:"HAS_VALUE_IN_COLUMN",
                  fact:"In column [2007], the row [weighted average
                        exercise price per share] has value 60.94"}]->
(:Entity {name:"2007"})
```

All 18 edges land on the three year nodes:

```
  ┌───────────────────────────────────────────────────────────────────────┐
  │ weighted average exercise price per share                             │
  │ expected annual dividends per share                                   │
  │ expected life in years                                                │
  │ expected volatility                                                   │
  │ risk-free interest rate                                               │
  │ weighted average grant date fair value of stock option awards granted │
  └───────────────────────────────────┬───────────────────────────────────┘
                                      │ HAS_VALUE_IN_COLUMN: every one of the 6 rows → each year column (18 edges, no LLM)
                         ┌────────────┼────────────┐
                         ▼            ▼            ▼
                     ┌───────┐    ┌───────┐    ┌───────┐
                     │  2007 │    │  2006 │    │  2005 │◄── 2007 and 2006 also carry narrative
                     └───────┘    └───────┘    └───────┘    edges (see "the bridge" below).
```

### Narrative side: 14 relations (GPT-extracted)

Qualitative content (plans, vesting, dividend equivalents) that is not in the table. The extraction is **noisy**: GPT fragments the same real-world entity into several nodes (`sars` vs `stock appreciation rights` vs `stock appreciation rights 2013`; `2013 marathon` vs `Marathon Oil Corporation`). That noise is why the table side is indexed as deterministic triplets, not via the LLM.

The bulk of the narrative forms one cluster around the option plans (`Marathon` links the two plan hubs, being `2007 plan`'s target and `2003 plan`'s source):

```
(Marathon Oil Corporation) ─GRANTS───────────────────────────┐
(2013 marathon) ─MAINTAINS───────────────────────────────────┤
(stock options 2013 marathon grants) ─GRANTS_OPTIONS_UNDER───┤
                                                             ▼
                                                       ┌───────────┐
                                                       │ 2007 plan │
                                                       └───────────┘
                                                             │ NO_STOCK_BASED_PERFORMANCE_AWARDS_GRANTED_UNDER
                                                             ▼
                                                       ┌──────────┐
                                                       │ Marathon │
                                                       └──────────┘

(Marathon Oil Corporation) ─GRANTED_AWARDS_UNDER────────────────────┐
(Marathon Oil Corporation) ─GRANTS_TIME_BASED_RESTRICTED_STOCK_TO───┤
(Marathon) ─GRANTED_STOCK_BASED_PERFORMANCE_AWARDS_UNDER────────────┤
(sars) ─GRANTED_UNDER───────────────────────────────────────────────┤
                                                                    ▼
                                                              ┌───────────┐
                                                              │ 2003 plan │
                                                              └───────────┘
```

Four more narrative edges are standalone pairs, unconnected to the plans or the table:

```
(stock appreciation rights) ─REPRESENTS─► (stock options)
(marathon common stock) ─HAS_DIVIDEND_EQUIVALENTS─► (employee stock-based compensation expense)
(stock option awards) ─REALIZED─► (tax benefits)
(stock option awards) ─EXCEEDED─► (stock-based compensation expense)
```

### The bridge: the only link between the two sides

Querying the graph for nodes that carry **both** a table edge and a narrative edge returns exactly two: `2007` and `2006`. The table writes them (`HAS_VALUE_IN_COLUMN`), and two narrative edges start from them:

```
                                      ┌──────┐                                        ┌──────┐
6 table rows ─HAS_VALUE_IN_COLUMN────►│ 2007 │─VESTS_RATELY_OVER─────────────────────►│ 2006 │──GRANTED_BEFORE──► (stock appreciation rights 2013)
                                      └──────┘                                        └──────┘
2007 and 2006 are the only two nodes shared by both sides: table edges arrive (HAS_VALUE_IN_COLUMN),
narrative edges leave (VESTS_RATELY_OVER, GRANTED_BEFORE). Every other node belongs to one side only.
```

The exact bridge here (two shared year nodes) is specific to this record: which nodes are shared depends on what GPT names when it extracts the narrative, so another record might share more nodes, or none at all. The general point holds regardless of that: the value of indexing both paths is **coverage, not graph traversal**. A number resolves through `search_table`, a plan or schedule through `search_narrative`, and both live under one `group_id` so they are retrieved together for a query, whether or not their nodes happen to touch. (Where the table holds a ratio and the dollar amount lives only in the narrative, you need both; see CB/2010 in REPORT.md.)

---

## Conversation (Q&A)

| Turn | Question | Gold | Source |
|---|---|---|---|
| Q1 | what was the weighted average exercise price per share in 2007? | 60.94 | table triplet |
| Q2 | and what was it in 2005? | 25.14 | table triplet |
| Q3 | what was, then, the change over the years? | 35.8 | computed (60.94 - 25.14) |
| Q4 | what was the weighted average exercise price per share in 2005? | 25.14 | table triplet / history |
| Q5 | and how much does that change represent in relation to this 2005 price? | 1.42403 | computed (35.8 / 25.14) |

**Agent score: 5/5** (code and judge, both Graph-RAG and Agentic RAG).
