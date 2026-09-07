---
title: Preventing inventory oversell under concurrent Redis stock decrements
---

## Why checking stock and decrementing it as two separate Redis commands can oversell
source: spring-data-redis https://docs.spring.io/spring-data/redis/reference/4.1/redis/scripting.html

A naive stock-decrement implementation reads the current count with `GET`, checks in
application code whether it is still positive, and then issues a separate `DECR` only
if that check passes. Those are two independent round trips to Redis, and nothing on
the server links them together. Spring Data Redis's own documentation describes
exactly this shape of problem under the name "check-and-set": a scenario that
"requires that running a set of commands atomically, and the behavior of one command
is influenced by the result of another." When two concurrent checkout requests each
run their `GET` before either has run its `DECR`, both can observe the same last unit
of stock, both conclude the sale is allowed, and both `DECR` — the count goes negative
and more units are sold than exist. The fix has to make the read-and-conditional-write
a single operation the server executes without interleaving, not a smarter retry loop
wrapped around the same two commands.

## Wrapping the check in a Spring-managed Redis transaction does not add the missing atomicity
source: spring-data-redis https://docs.spring.io/spring-data/redis/reference/4.1/redis/transactions.html#tx.spring

Enabling `RedisTemplate.setEnableTransactionSupport(true)` and running the
read-then-decrement inside a `@Transactional` method looks like it should close the
gap, but Spring Data Redis's transaction documentation spells out what commands issued
inside that transaction actually do: write commands are only queued and applied on
commit, and a read run during the transaction returns null immediately, because
"values set within a transaction are not visible" until the transaction commits. There
is no point during the transaction where application code can read the real current
stock value and branch on it — the `GET` is queued the same way the `DECR` is, so the
code has nothing to condition on. The transaction only batches commands that were
already decided on before it opened; it never hands the application a real value to
decide with.

## A Lua script runs as a single atomic command, so no other client's request can land between the read and the write
source: spring-data-redis https://docs.spring.io/spring-data/redis/reference/4.1/redis/scripting.html

Spring Data Redis ships the `checkandset.lua` example for exactly this class of
problem: a script that reads a key with `redis.call('GET', ...)`, compares it against
an argument, and conditionally writes a result — all inside one script that the Redis
server runs to completion before it processes any other client's command. The same
shape covers inventory: replace the equality check with a threshold check (current
stock at least the requested quantity) and the unconditional `SET` with a `DECRBY`,
and the script becomes an atomic reserve-or-refuse operation. Because Redis executes
the whole script as a single unit, there is no window between the read and the write
for a second checkout request to see the pre-decrement count — the property both the
naive two-command version and the transaction wrapper above were missing.

## RedisScript caches the script's SHA1, so the atomicity doesn't cost an extra round trip on every call
source: spring-data-redis https://docs.spring.io/spring-data/redis/reference/4.1/redis/scripting.html

Running a script through `RedisTemplate`/`ReactiveRedisTemplate` does not mean sending
the full script text on every invocation. The default `ScriptExecutor` retrieves the
script's SHA1 and first attempts `evalsha`, only falling back to sending the full
script body with `eval` if the Redis instance does not already have that script
cached. Spring Data Redis's own guidance is to configure a single `DefaultRedisScript`
instance in the application context so the SHA1 is computed once rather than
recalculated on every run. In practice this means the atomic check-and-decrement
script costs about the same as a single command over the wire once the script is
warm — the atomicity is not purchased by paying for an extra round trip on the hot
path.

## The atomic counters in spring-data-redis's support classes only wrap a single command — they do not add the conditional check inventory needs
source: spring-data-redis https://docs.spring.io/spring-data/redis/reference/4.1/redis/support-classes.html

`org.springframework.data.redis.support` offers JDK-style atomic counters that "make
it easy to wrap Redis key incrementation" — a convenience wrapper around a single
Redis command, atomic in exactly the sense that any single Redis command is atomic.
That is a different guarantee from what inventory needs: such a counter will happily
decrement a key below zero, because the command it wraps has no notion of a floor and
nothing to compare against before writing. Reaching for one of these counter classes
on a stock key does not reproduce the check-and-set behavior from the sections above;
it only makes the unconditional decrement atomic, which was never the part of the
problem that caused overselling in the first place. The conditional threshold check
still has to come from a script, not from swapping `DECR` for a nicer-looking Java
wrapper around the same command.
