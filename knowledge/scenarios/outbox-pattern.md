---
title: Outbox pattern for reliable event publishing
---

## Why a database write and a message publish cannot be one operation
source: debezium https://debezium.io/documentation/reference/3.6/transformations/outbox-event-router.html

A service that updates its own database and then separately publishes an event about
that change is really performing two independent writes against two different
systems: the local database transaction, and the message broker. There is no single
commit that covers both. If the process crashes after the database commit but before
the broker accepts the message, the change happened but no one downstream ever hears
about it. If it crashes the other way around — publish first, database write second —
downstream consumers can react to a change that the source service itself never
finished persisting. This is the "dual write" problem, and no amount of retrying the
second half of the pair removes the underlying inconsistency window; it only makes
the window smaller or bigger.

The outbox pattern avoids this by turning the second write into something the first
write's transaction already covers. Instead of calling the broker directly, the
service inserts a row describing the event into an outbox table in the same database,
as part of the same transaction as the business change. A separate process reads that
table and forwards the events to the broker. Because "record the business change" and
"record the intent to publish" now happen atomically in one transaction, the
inconsistency window described above disappears — you can lose the forwarder for a
while, but you cannot lose the event once the transaction has committed.

## How the Outbox Event Router turns a table row into a Kafka message
source: debezium https://debezium.io/documentation/reference/3.6/transformations/outbox-event-router.html#basic-outbox-table

Debezium's Outbox Event Router is the "separate process" from the paragraph above,
implemented as a single message transform (SMT) on top of a normal Debezium CDC
connector. The connector already streams every row change from the outbox table via
the database's own change data capture mechanism (for example logical decoding on
PostgreSQL), so there is nothing bespoke to poll — the outbox table is read exactly
the same way any other captured table would be.

By default the transform expects a fixed outbox table shape: an `id` column that
becomes the Kafka message key, an `aggregatetype` column used to route the event to
a topic, and a `payload` column holding the event body. The router reads a captured
`INSERT` on that table and turns it into a Kafka record — key, destination topic and
value are all derived from the row's columns, not invented by the connector. Column
names and the routing rule are configurable, but the shape is deliberately narrow: an
outbox table is meant to behave like a queue that only ever receives inserts, not a
general-purpose events table that also gets updated or read by other application
logic.

## The outbox pattern covers the write side, not the read side
source: kafka https://kafka.apache.org/43/design/design/#message-delivery-semantics

Everything above only fixes the boundary between the database and the broker: once a
service commits its business change, the corresponding event is guaranteed to
eventually reach the broker too. It says nothing about what happens once a consumer
picks that event up on the other end — that half of the problem is still governed by
whatever delivery guarantee Kafka itself provides to consumers by default, which is
not "exactly once" (Kafka's design documentation on message delivery semantics spells
out the narrower conditions under which exactly-once actually holds).

So the outbox pattern and consumer-side idempotency solve two different halves of the
same end-to-end reliability problem, and neither substitutes for the other. Adopting
the outbox pattern does not remove the need to design consumers that can safely see
the same event more than once — for example by keying on the outbox row's `id` — it
only guarantees that a committed change will produce that event at all, not that the
event will be seen exactly one time downstream.
