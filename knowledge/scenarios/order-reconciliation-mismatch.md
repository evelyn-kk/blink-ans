---
title: An order service whose books do not balance at reconciliation time
---

## In the default configuration a checked exception commits the transaction instead of rolling it back
source: spring-framework https://docs.spring.io/spring-framework/reference/7.0/data-access/transaction/declarative/rolling-back.html

The way to ask Spring for a rollback is to let an exception escape: "The recommended way
to indicate to the Spring Framework's transaction infrastructure that a transaction's work
is to be rolled back is to throw an `Exception` from code that is currently executing in
the context of a transaction. The Spring Framework's transaction infrastructure code
catches any unhandled `Exception` as it bubbles up the call stack and makes a determination
whether to mark the transaction for rollback." The determination is the part that decides
whether an order row and its ledger row stay consistent, and its default is narrower than
most people assume: "In its default configuration, the Spring Framework's transaction
infrastructure code marks a transaction for rollback only in the case of runtime, unchecked
exceptions. That is, when the thrown exception is an instance or subclass of
`RuntimeException`. (`Error` instances also, by default, result in a rollback)." The
converse is stated just as plainly — "Checked exceptions that are thrown from a
transactional method do not result in a rollback in the default configuration" — so a
business exception declared as a checked `Exception`, thrown after the order row was
already written, leaves that half-finished write committed. Two escape hatches exist. The
declarative one is rollback rules: `rollbackFor`/`noRollbackFor` (types) and
`rollbackForClassName`/`noRollbackForClassName` (patterns), where "the strongest matching
rule wins". Patterns match on substring, which is a documented foot-gun: a rule configured
for `"com.example.CustomException"` "will match against an exception named
`com.example.CustomExceptionV2`" or a nested `com.example.CustomException$AnotherException`,
and `"Exception"` "will match nearly anything and will probably hide other rules". The
programmatic one is
`TransactionAspectSupport.currentTransactionStatus().setRollbackOnly()`, which the
documentation permits but discourages: "You are strongly encouraged to use the declarative
approach to rollback, if at all possible."

## A `@Transactional` method called from inside the same object never gets the transaction it declares — it runs in whatever the caller already had
source: spring-framework https://docs.spring.io/spring-framework/reference/7.0/data-access/transaction/declarative/annotations.html

The annotation is only metadata: "the mere presence of the `@Transactional` annotation is
not enough to activate the transactional behavior", and it is `@EnableTransactionManagement`
(or `<tx:annotation-driven/>`) that switches the behavior on at runtime. Once on, the
default mechanism is a proxy, and a proxy can only intercept calls that pass through it:
"For both kinds of proxies, only external method calls coming in through the proxy are
intercepted." The consequence is spelled out for the exact shape that a service class
grows into over time: "In proxy mode (which is the default), only external method calls
coming in through the proxy are intercepted. This means that self-invocation (in effect, a
method within the target object calling another method of the target object) does not lead
to an actual transaction at runtime even if the invoked method is marked with
`@Transactional`." So `placeOrder()` calling `this.writeLedgerEntry()` runs the ledger
write with whatever transaction the caller already had — the annotation on the inner method
is inert. Read that sentence precisely, because the two cases it covers fail in different
ways and the difference decides what you look for during an incident. If `placeOrder()` is
itself transactional, the ledger write silently *joins that outer transaction*: the write
is still transactional, but everything `writeLedgerEntry()` declared for itself is ignored
— a `REQUIRES_NEW` does not start a second transaction, and a different isolation level or
timeout is not applied. If the caller is not transactional either, then this call does not
start one either — and that is where the quoted rule stops. It says something about **this
call**, not about everything that happens afterwards, so three things it does *not* settle
are worth naming, because each of them has been read into it at some point:

- *Whether a transaction exists further down.* The rule limits interception to calls that
  enter through a proxy — so the moment `writeLedgerEntry()` calls a **different** bean,
  that call does go through that bean's proxy, and a transactional method there (a
  repository method, say) establishes its own boundary. Programmatic transaction
  management is available at that point too. Neither is excluded by anything quoted here.
- *What happens to an unmanaged write.* If nothing downstream establishes a boundary
  either, the outcome is decided by the data access API and its configuration, not by this
  page: such a write may go through on its own or be rejected outright, and this section
  deliberately cites no evidence for either.
- *Whether anything will roll it back.* That follows from the previous two, not from the
  annotation.

What the rule does establish is narrow and still worth acting on: the boundary
`writeLedgerEntry()` declares for itself is not created at this call, so any reasoning
that starts with 'it is annotated, therefore this method runs in the transaction it asked
for' is unsupported — and so is its mirror image, 'the annotation did not fire, therefore
nothing here is transactional'. Both cases look identical in the
source code, which is why 'the annotation is there, so the method is transactional' is the
wrong question to ask — the right one is which method the call entered the object
through. The same section adds a second timing trap: "the proxy
must be fully initialized to provide the expected behavior, so you should not rely on this
feature in your initialization code -- for example, in a `@PostConstruct` method." If self-invocation genuinely must be transactional, the documented alternative is
AspectJ mode — "In this case, there is no proxy in the first place. Instead, the target
class is woven (that is, its byte code is modified)". One further silent failure is worth
knowing because it produces the same symptom without any self-call: Spring recommends
annotating methods of concrete classes rather than interfaces, because in AspectJ mode
"Java annotations are not inherited from interfaces … As a consequence, your transaction
annotations may be silently ignored: Your code might appear to 'work' until you test a
rollback scenario."

## Under the default propagation an inner rollback drags the outer one down — and reports it as UnexpectedRollbackException
source: spring-framework https://docs.spring.io/spring-framework/reference/7.0/data-access/transaction/declarative/tx-propagation.html#tx-propagation-required

`PROPAGATION_REQUIRED` "enforces a physical transaction, either locally for the current
scope if no transaction exists yet or participating in an existing 'outer' transaction
defined for a larger scope". The distinction that explains most surprises is logical versus
physical scope: "When the propagation setting is `PROPAGATION_REQUIRED`, a logical
transaction scope is created for each method upon which the setting is applied. Each such
logical transaction scope can determine rollback-only status individually … In the case of
standard `PROPAGATION_REQUIRED` behavior, all these scopes are mapped to the same physical
transaction. So a rollback-only marker set in the inner transaction scope does affect the
outer transaction's chance to actually commit." Catching the inner exception in the outer
method therefore does not save the outer work — the marker is already on the shared
physical transaction. What the caller sees instead is an exception at commit time: "the
rollback (silently triggered by the inner transaction scope) is unexpected. A corresponding
`UnexpectedRollbackException` is thrown at that point. This is expected behavior so that
the caller of a transaction can never be misled to assume that a commit was performed when
it really was not." Read that way, `UnexpectedRollbackException` is not the bug; it is the
report that something inside already decided to roll back. A second detail of the same
propagation mode quietly weakens settings that a reconciliation job thinks it declared: "By
default, a participating transaction joins the characteristics of the outer scope, silently
ignoring the local isolation level, timeout value, or read-only flag (if any)." So
`@Transactional(readOnly = true, isolation = REPEATABLE_READ)` on a method that happens to
be invoked inside an existing transaction gets neither. Setting
`validateExistingTransaction` to `true` on the transaction manager turns those mismatches
into rejections instead of silence — it "also rejects read-only mismatches (that is, an
inner read-write transaction that tries to participate in a read-only outer scope)".

## REQUIRES_NEW really is a second transaction, and it borrows a second connection
source: spring-framework https://docs.spring.io/spring-framework/reference/7.0/data-access/transaction/declarative/tx-propagation.html#tx-propagation-requires_new

When a failed reconciliation line has to be recorded even though the main transaction rolls
back, the propagation mode that delivers it is `PROPAGATION_REQUIRES_NEW`, which "in
contrast to `PROPAGATION_REQUIRED`, always uses an independent physical transaction for
each affected transaction scope, never participating in an existing transaction for an
outer scope. In such an arrangement, the underlying resource transactions are different
and, hence, can commit or roll back independently, with an outer transaction not affected
by an inner transaction's rollback status and with an inner transaction's locks released
immediately after its completion." Independence extends to the settings that
`PROPAGATION_REQUIRED` swallows: "Such an independent inner transaction can also declare
its own isolation level, timeout, and read-only settings and not inherit an outer
transaction's characteristics." The price is a resource, and the documentation states it as
an operational constraint rather than a note: "The resources attached to the outer
transaction will remain bound there while the inner transaction acquires its own resources
such as a new database connection. This may lead to exhaustion of the connection pool and
potentially to a deadlock if several threads have an active outer transaction and wait to
acquire a new connection for their inner transaction, with the pool not being able to hand
out any such inner connection anymore. Do not use `PROPAGATION_REQUIRES_NEW` unless your
connection pool is appropriately sized, exceeding the number of concurrent threads by at
least 1." A batch job that runs one `REQUIRES_NEW` audit insert per order line on a pool
sized to the thread count is therefore not merely slow — it is a documented deadlock shape.

## NESTED is a savepoint inside one transaction, and it only works on JDBC
source: spring-framework https://docs.spring.io/spring-framework/reference/7.0/data-access/transaction/declarative/tx-propagation.html#tx-propagation-nested

The other way to let one line fail without losing the batch is `PROPAGATION_NESTED`, and it
is a different mechanism from `REQUIRES_NEW` despite solving a similar-sounding problem:
"`PROPAGATION_NESTED` uses a single physical transaction with multiple savepoints that it
can roll back to. Such partial rollbacks let an inner transaction scope trigger a rollback
for its scope, with the outer transaction being able to continue the physical transaction
despite some operations having been rolled back." Because there is one physical transaction,
there is one connection and one commit — the inner work is not durable on its own, and it
disappears if the outer transaction later rolls back. That is the opposite of
`REQUIRES_NEW`, where the inner commit survives an outer rollback, and it is what decides
which of the two an audit trail actually needs: a record that must outlive the failure needs
`REQUIRES_NEW`, a step that merely needs to be retryable within the same unit of work needs
`NESTED`. The applicability is also narrower: "This setting is typically mapped onto JDBC
savepoints, so it works only with JDBC resource transactions."

## Publishing the order event after commit is what `@TransactionalEventListener` is for — and it does nothing without a transaction
source: spring-framework https://docs.spring.io/spring-framework/reference/7.0/data-access/transaction/event.html

Sending the "order created" message from inside the transaction means consumers can see an
event whose row later rolls back. Spring binds listeners to transaction phases for exactly
this case: "the listener of an event can be bound to a phase of the transaction. The
typical example is to handle the event when the transaction has completed successfully."
The mechanism is annotation-level — "You can register a regular event listener by using the
`@EventListener` annotation. If you need to bind it to the transaction, use
`@TransactionalEventListener`. When you do so, the listener is bound to the commit phase of
the transaction by default." The documented phases are `BEFORE_COMMIT`, `AFTER_COMMIT`
(default), `AFTER_ROLLBACK` and `AFTER_COMPLETION`, the last of which "aggregates the
transaction completion (be it a commit or a rollback)". One behavior surprises people who
reuse the same listener from a non-transactional path, such as a test or an admin endpoint:
"If no transaction is running, the listener is not invoked at all, since we cannot honor
the required semantics. You can, however, override that behavior by setting the
`fallbackExecution` attribute of the annotation to `true`." Note what this does and does not
buy: `AFTER_COMMIT` guarantees the database work committed before the message is sent, but
the send itself is outside the transaction, so a broker failure at that moment still loses
the message — the durable-handoff problem that the Outbox card covers.

## Read Committed gives each statement its own snapshot, so two counts in one transaction can disagree
source: postgresql https://www.postgresql.org/docs/17/transaction-iso.html#XACT-READ-COMMITTED

"Read Committed is the default isolation level in PostgreSQL. When a transaction uses this
isolation level, a `SELECT` query (without a `FOR UPDATE/SHARE` clause) sees only data
committed before the query began; it never sees either uncommitted data or changes
committed by concurrent transactions during the query's execution. In effect, a `SELECT`
query sees a snapshot of the database as of the instant the query begins to run." The
sentence that matters for a reconciliation job which sums orders in one query and payments
in the next is the one about repeating a read: "two successive `SELECT` commands can see
different data, even though they are within a single transaction, if other transactions
commit changes after the first `SELECT` starts and before the second `SELECT` starts." A
mismatch of exactly the orders that committed between the two queries is therefore not a
bug in the arithmetic — it is the isolation level working as documented. Writes behave the
same way at statement granularity and add a re-check: `UPDATE`, `DELETE`, `SELECT FOR
UPDATE` and `SELECT FOR SHARE` "will only find target rows that were committed as of the
command start time", and if a concurrent transaction has already updated a found row, the
would-be updater waits for it, then "the search condition of the command (the `WHERE`
clause) is re-evaluated to see if the updated version of the row still matches the search
condition". If it no longer matches, the row is skipped — an `UPDATE … WHERE status =
'PENDING'` can legitimately report fewer rows than the count taken a moment earlier.

## Repeatable Read gives the whole transaction one snapshot, at the cost of serialization failures
source: postgresql https://www.postgresql.org/docs/17/transaction-iso.html#XACT-REPEATABLE-READ

The fix for a reconciliation run that must compare several tables as of one instant is a
transaction-level snapshot: "The Repeatable Read isolation level only sees data committed
before the transaction began; it never sees either uncommitted data or changes committed by
concurrent transactions during the transaction's execution." Explicitly contrasted with the
default: "a query in a repeatable read transaction sees a snapshot as of the start of the
first non-transaction-control statement in the transaction, not as of the start of the
current statement within the transaction. Thus, successive `SELECT` commands within a
single transaction see the same data." PostgreSQL's implementation is stronger than the SQL
standard requires at this level, preventing "all of the phenomena described … except for
serialization anomalies". The cost lands on any transaction at this level that also writes:
"Applications using this level must be prepared to retry transactions due to serialization
failures." Concretely, when a concurrent transaction has already updated or deleted a target
row and commits, "the repeatable read transaction will be rolled back with the message
`ERROR: could not serialize access due to concurrent update`". A read-only reconciliation
query rarely hits this; a job that both reads a consistent snapshot and posts adjustments
from it will.

## Serialization failures are the application's job to retry, and the retry must cover the whole transaction
source: postgresql https://www.postgresql.org/docs/17/mvcc-serialization-failure-handling.html#MVCC-SERIALIZATION-FAILURE-HANDLING

"Both Repeatable Read and Serializable isolation levels can produce errors that are designed
to prevent serialization anomalies. As previously stated, applications using these levels
must be prepared to retry transactions that fail due to serialization errors. Such an
error's message text will vary according to the precise circumstances, but it will always
have the SQLSTATE code `40001` (`serialization_failure`)." That code, not the message text,
is the thing to branch on. The documentation lists what else is worth retrying and how
confidently: deadlock failures with SQLSTATE `40P01` (`deadlock_detected`) — "It may also be
advisable to retry deadlock failures" — and, with more care, unique-key failures (`23505`,
`unique_violation`) and exclusion constraint failures (`23P01`, `exclusion_violation`),
because those "might represent persistent error conditions rather than transient failures";
the blanket advice is that "it's recommendable to just retry `serialization_failure` errors
unconditionally". The scope of a retry is the part that a Spring service usually gets wrong
by wrapping only the failing statement: "It is important to retry the complete transaction,
including all logic that decides which SQL to issue and/or which values to use. Therefore,
PostgreSQL does not offer an automatic retry facility, since it cannot do so with any
guarantee of correctness." Retrying is also not a bounded promise: "Transaction retry does
not guarantee that the retried transaction will complete; multiple retries may be needed. In
cases with very high contention, it is possible that completion of a transaction may take
many attempts."
