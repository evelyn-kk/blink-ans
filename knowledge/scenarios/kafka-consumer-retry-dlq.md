---
title: Kafka consumer retry and dead-letter topics in Spring Kafka
---

## Why one bad record can block an entire partition, not just itself
source: spring-kafka https://docs.spring.io/spring-kafka/reference/4.1/kafka/annotation-error-handling.html#default-eh

Kafka only guarantees ordering within a partition, and a listener container processes
the records it polls in that order. When a `@KafkaListener` method throws while
handling a record, Spring Kafka's `DefaultErrorHandler` does not simply skip ahead to
the next record: it seeks the consumer back so that the failed record — and everything
after it in that partition — will be redelivered on the next poll, retrying according
to a configured `BackOff` (for example a `FixedBackOff` with a one second delay and a
fixed number of attempts). Nothing later in that partition advances until the failed
record is either processed successfully or explicitly recovered. This is why an
unbounded backoff is dangerous: configuring `FixedBackOff.UNLIMITED_ATTEMPTS` means the
handler keeps retrying that one record forever, which in practice means the partition
stops making progress at all. By default, once the configured attempts are exhausted,
the handler recovers the record by logging it at `ERROR` level and moving on —
recovery, not infinite retry, is what actually lets consumption continue past a record
that will never succeed on its own.

## Retrying is not always the right response to an exception
source: spring-kafka https://docs.spring.io/spring-kafka/reference/4.1/kafka/annotation-error-handling.html#default-eh

The `DefaultErrorHandler` only ever sees exceptions that inherit from
`RuntimeException`; anything inheriting from `Error` bypasses the error handler
entirely, terminates the consumer and closes the Kafka connection — no retry, no
recovery, no dead-letter publish. The documentation calls out the consequence
explicitly: an application can report itself as healthy while the consumer thread that
was supposed to be processing that topic has already died, silently, because the
failure happened to surface as an `Error` rather than a `RuntimeException`. Separately,
several `RuntimeException` subtypes are classified as fatal by default —
`DeserializationException`, `MessageConversionException`, `ConversionException`,
`MethodArgumentResolutionException`, `NoSuchMethodException` and `ClassCastException` —
because retrying the exact same delivery is very unlikely to change the outcome; for
these the recoverer runs on the very first failure instead of after the configured
backoff attempts. The fatal list is not fixed: `DefaultErrorHandler.addNotRetryableExceptions()`
lets an application add its own exception types to it, for the same reason — some
failures are a property of the message itself, not a transient condition that a retry
could fix.

## Recovering a record usually means publishing it to a dead-letter topic
source: spring-kafka https://docs.spring.io/spring-kafka/reference/4.1/kafka/annotation-error-handling.html#dead-letters

The framework's built-in recoverer for this is `DeadLetterPublishingRecoverer`,
configured alongside a `DefaultErrorHandler` and invoked once retries — if any — are
exhausted. Its default behavior is to publish the failed record to a topic named
`<originalTopic>-dlt` (the original topic name suffixed with `-dlt`), targeting the
same partition number as the original record; because of that, the dead-letter topic
must be provisioned with at least as many partitions as the original topic, or the
default resolver has no matching partition to publish into. The record that lands on
the dead-letter topic is not a bare copy of the original: the recoverer adds headers
carrying the original topic, partition, offset and timestamp, plus the exception's
fully-qualified class name, message and stack trace, so whoever consumes the
dead-letter topic later has enough information to triage the failure without
correlating it back against application logs by hand.

## Non-blocking retries move the wait off the partition's consumer thread
source: spring-kafka https://docs.spring.io/spring-kafka/reference/4.1/retrytopic/retry-config.html#using-the-retryabletopic-annotation

Everything above still retries in place: the same consumer thread keeps polling the
same partition and blocking through each `BackOff` delay before it can move on, which
is exactly the mechanism that lets one record hold up everything behind it. Annotating
a `@KafkaListener` method with `@RetryableTopic` takes a different approach: Spring for
Apache Kafka bootstraps a set of dedicated retry topics and a dead-letter topic
automatically, and a failed record is republished to the next retry topic in the
chain — with its own consumer group — instead of being redelivered on the original
partition. The original topic's consumer is therefore free to keep making progress on
later records while a failing one works through its retry topics on a separate
consumer entirely. A method on the same class can be annotated `@DltHandler` to process
whatever ends up on the dead-letter topic; if none is provided, a default consumer is
created that only logs what it receives.

## Not every listener can use non-blocking retry topics
source: spring-kafka https://docs.spring.io/spring-kafka/reference/4.1/kafka/annotation-error-handling.html#batch-listener-error-handling-dlt

`@RetryableTopic` is explicitly not supported for batch listeners — a `@KafkaListener`
method that receives a list of records per invocation rather than one record at a time.
A batch listener that needs dead-letter behavior has to fall back to the same
`DefaultErrorHandler` plus `DeadLetterPublishingRecoverer` combination described above
for the record-at-a-time case, which means it also inherits that combination's
tradeoff: retries for the batch happen in place, on the same consumer thread and
partition, rather than being offloaded to separate retry topics.
