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
inside that transaction actually do, and the two kinds of commands are not treated the
same way: "Spring Data Redis distinguishes between read-only and write commands in an
ongoing transaction. Read-only commands, such as `KEYS`, are piped to a fresh
(non-thread-bound) `RedisConnection` to allow reads. Write commands are queued by
`RedisTemplate` and applied upon commit." So a `GET` on the stock key does not sit
blocked behind the pending transaction — it runs on that separate connection and
returns the real, live count. (The documentation's own null example is a different
case: it reads back a key that the same still-open transaction had just queued a `SET`
for, so there is nothing committed yet for that specific key to read.) The `DECR`,
being a write, is still only queued and applied later at `EXEC`. That is exactly the
gap: the transaction gives the application a real value to read, but nothing ties that
read to the later commit — a second checkout's `GET` can land, see the same live
count, and queue its own `DECR` before the first transaction commits, so both still go
through. Wrapping the pair in `@Transactional` batches the commands; it does not add
the missing condition that would make the second transaction's `EXEC` fail once the
first one has already spent the stock.

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

## The plain increment/decrement operation the support classes wrap has no notion of a stock floor
source: spring-data-redis https://docs.spring.io/spring-data/redis/reference/4.1/redis/support-classes.html

`org.springframework.data.redis.support` offers JDK-style atomic counters that "make
it easy to wrap Redis key incrementation." The documentation describes what these
classes wrap — a Redis key-increment command — but does not describe them adding any
inventory-specific rule on top of it, such as a minimum-value check. An unconditional
decrement, called directly the way the documentation describes, inherits exactly the
atomicity of that one underlying command and nothing more: the command itself has no
notion of a floor, so nothing stops the key from going negative if it runs when there
is not enough stock left. That means reaching for one of these counters on a stock key
and calling its plain decrement operation does not, by itself, reproduce the
check-and-set behavior from the sections above — the documentation for these classes
does not describe them doing that. The missing threshold check has to be added
deliberately, whether by scripting the whole read-and-decide server-side as in the
sections above, or by some other mechanism outside what this documentation covers.
