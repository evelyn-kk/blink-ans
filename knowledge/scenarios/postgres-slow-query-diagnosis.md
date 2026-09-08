---
title: Diagnosing a slow SQL query on a production PostgreSQL server
---

## pg_stat_statements ranks statement shapes, not individual slow executions
source: postgresql https://www.postgresql.org/docs/17/pgstatstatements.html#PGSTATSTATEMENTS-PG-STAT-STATEMENTS

The `pg_stat_statements` view holds one row per distinct combination of database ID,
user ID, query ID and top-level flag — not one row per execution. Plannable statements
(`SELECT`, `INSERT`, `UPDATE`, `DELETE`, `MERGE`) and utility commands "are combined
into a single `pg_stat_statements` entry whenever they have identical query structures
according to an internal hash calculation. Typically, two queries will be considered
the same for this purpose if they are semantically equivalent except for the values of
literal constants appearing in the query." Ignored constants are displayed as parameter
symbols such as `$1`, and the rest of the text shown "is that of the first query that
had the particular queryid" — so the `query` column is a representative sample of a
shape, not the text of whichever call was slow. That shapes how the view should be
read during triage: `calls`, `total_exec_time` and `mean_exec_time` are per-shape
aggregates, so sorting by `total_exec_time` finds the statement shape that consumed the
most server time in total, which is not necessarily the shape with the worst per-call
latency — that ordering comes from `mean_exec_time` (or `max_exec_time`). Getting any
of these numbers at all requires loading the module through
`shared_preload_libraries`, which the documentation notes needs a server restart,
plus query identifier calculation enabled.

## auto_explain captures the plan of the actual slow execution, but log_analyze times every statement, not only the logged ones
source: postgresql https://www.postgresql.org/docs/17/auto-explain.html#AUTO-EXPLAIN-CONFIGURATION-PARAMETERS

`auto_explain` logs execution plans of slow statements without anyone having to
reproduce them by hand, which is what makes it the tool for a query that is only slow
in production. Its default behavior "is to do nothing, so you must set at least
`auto_explain.log_min_duration` if you want any results": that value is the minimum
execution time in milliseconds that causes a plan to be logged, `0` logs every plan,
and `-1` (the default) disables logging. The trap is the option that makes the log
useful. `auto_explain.log_analyze` produces `EXPLAIN ANALYZE` output rather than plain
`EXPLAIN`, but "when this parameter is on, per-plan-node timing occurs for all
statements executed, whether or not they run long enough to actually get logged. This
can have an extremely negative impact on performance." The cost is therefore paid by
the fast statements too, not only by the slow ones that end up in the log. The
documentation names two levers against that: turning off `auto_explain.log_timing`,
which "ameliorates the performance cost, at the price of obtaining less information"
(actual row counts without exact per-node times), and `auto_explain.sample_rate`, which
explains only a fraction of the statements in each session.

## In EXPLAIN ANALYZE output, compare estimated against actual rows — cost and actual time are not the same unit
source: postgresql https://www.postgresql.org/docs/17/using-explain.html#USING-EXPLAIN-ANALYZE

`EXPLAIN ANALYZE` actually executes the query and then displays "the true row counts
and true run time accumulated within each plan node, along with the same estimates
that a plain `EXPLAIN` shows." Comparing the two columns is the point of the exercise,
but only one comparison is meaningful: "the actual time values are in milliseconds of
real time, whereas the cost estimates are expressed in arbitrary units; so they are
unlikely to match up," while "the thing that's usually most important to look for is
whether the estimated row counts are reasonably close to reality." Two further reading
rules matter for a node inside a nested loop: when a subplan is executed more than
once, `loops` reports the number of executions and the actual time and rows shown are
averages per execution, so the total time spent in that node is the per-loop figure
multiplied by `loops`. And a `Rows Removed by Filter` line reports how many scanned
rows a filter condition rejected, which is what distinguishes "this node read little"
from "this node read a lot and threw most of it away." Adding the `BUFFERS` option
reports shared block hits and reads per node, whose "numbers help to identify which
parts of the query are the most I/O-intensive."

## An estimate that does not match the actual count is not automatically an estimation error
source: postgresql https://www.postgresql.org/docs/17/using-explain.html#USING-EXPLAIN-CAVEATS

Before treating a mismatch as a planner problem worth fixing, the documentation lists
cases where "the actual and estimated values won't match up well, but nothing is really
wrong." A node stopped short by a `LIMIT` shows its estimated cost and row count "as
though it were run to completion," while the actual row count reflects only the rows
requested — "this is not an estimation error, only a discrepancy in the way the
estimates and true values are displayed." Merge joins produce their own artifacts: a
rescanned inner child has its repeated emissions counted "as if they were real
additional rows," so its reported actual row count can be significantly larger than the
relation. `BitmapAnd` and `BitmapOr` nodes "always report their actual row counts as
zero, due to implementation limitations." The measured times carry caveats of their own:
no rows are delivered to the client, so network transmission costs are excluded (and
I/O conversion costs too unless `SERIALIZE` is specified), and "the measurement overhead
added by `EXPLAIN ANALYZE` can be significant, especially on machines with slow
`gettimeofday()` operating-system calls." Finally, results "should not be extrapolated
to situations much different from the one you are actually testing" — the planner's
cost estimates are not linear, so a plan measured on a small table says little about
the same query on a large one.

## When the bad estimate comes from correlated columns, re-running ANALYZE alone will not fix it
source: postgresql https://www.postgresql.org/docs/17/planner-stats.html#PLANNER-STATS-EXTENDED

A row estimate that is off by orders of magnitude on a multi-condition `WHERE` clause
has a specific documented cause: "it is common to see slow queries running bad
execution plans because multiple columns used in the query clauses are correlated. The
planner normally assumes that multiple conditions are independent of each other, an
assumption that does not hold when column values are correlated." The reason a fresh
`ANALYZE` does not repair this is structural rather than a matter of staleness —
regular statistics, "because of their per-individual-column nature, cannot capture any
knowledge about cross-column correlation." The remedy the documentation describes is an
extended statistics object created with `CREATE STATISTICS` over the interesting set of
columns; multivariate statistics are not computed automatically because the number of
possible column combinations is too large. Note the two-step nature of it: creating the
object "merely creates a catalog entry expressing interest in the statistics. Actual
data collection is performed by `ANALYZE` (either a manual command, or background
auto-analyze)." So `CREATE STATISTICS` on its own changes no plan until an `ANALYZE`
has run, and the collected values can then be examined in `pg_statistic_ext_data`.

## A covering index only avoids heap access while the visibility map says those pages are all-visible
source: postgresql https://www.postgresql.org/docs/17/indexes-index-only-scans.html#INDEXES-INDEX-ONLY-SCANS

Adding an index that contains every column a query touches is supposed to turn a plan
into an index-only scan and drop the random heap I/O. Two requirements are described as
fundamental: the index type must support index-only scans (B-tree always does; GiST and
SP-GiST only for some operator classes; GIN cannot, because each entry typically holds
only part of the original value), and the query must reference only columns stored in
the index — `INCLUDE` columns count, and exist precisely so payload columns can be
carried without joining the search key. But meeting both is not sufficient, because
every table scan "must verify that each retrieved row be visible to the query's MVCC
snapshot," and "visibility information is not stored in index entries, only in heap
entries." What rescues the scan is the visibility map: a per-heap-page bit saying all
rows on that page are old enough to be visible to everyone. An index-only scan checks
that bit for the candidate row's page, and "if it's not set, the heap entry must be
visited to find out whether it's visible, so no performance advantage is gained over a
standard index scan." Hence the documented condition on the whole technique: "while an
index-only scan is possible given the two fundamental requirements, it will be a win
only if a significant fraction of the table's heap pages have their all-visible map
bits set" — which is why the same covering index that transforms a read-mostly table
can do nothing for a heavily updated one.
