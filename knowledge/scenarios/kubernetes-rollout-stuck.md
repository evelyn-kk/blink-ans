---
title: A Kubernetes rolling update that stalls in production
---

## maxUnavailable is a capacity floor, and its default 25% is what paces every rolling update
source: kubernetes https://v1-36.docs.kubernetes.io/docs/concepts/workloads/controllers/deployment/#max-unavailable

`.spec.strategy.rollingUpdate.maxUnavailable` "is an optional field that specifies the
maximum number of Pods that can be unavailable during the update process." The value is
"an absolute number (for example, 5) or a percentage of desired Pods (for example, 10%)",
and when a percentage is given "the absolute number is calculated from percentage by
rounding down" — so on a small Deployment a percentage can round to a smaller number of
Pods than the operator expected. The default value is 25%, which is what applies to every
Deployment that never set the field. The documentation's own example makes the semantics
concrete: with the value set to 30%, "the old ReplicaSet can be scaled down to 70% of
desired Pods immediately when the rolling update starts. Once new Pods are ready, old
ReplicaSet can be scaled down further, followed by scaling up the new ReplicaSet, ensuring
that the total number of Pods available at all times during the update is at least 70% of
the desired Pods." Two consequences follow for a rollout that appears frozen. First, the
guarantee is expressed in *available* Pods, so it is the availability of the new Pods —
not the fact that they were created — that lets the controller take the next step. Second,
the pair of knobs cannot both be zero: "the value cannot be 0 if
`.spec.strategy.rollingUpdate.maxSurge` is 0", because a rollout that may neither remove
an old Pod nor add an extra one has no way to make progress at all.

## A rollout that never finishes is reported as a status condition; Kubernetes does not roll it back for you
source: kubernetes https://v1-36.docs.kubernetes.io/docs/concepts/workloads/controllers/deployment/#failed-deployment

"Your Deployment may get stuck trying to deploy its newest ReplicaSet without ever
completing." The documented causes are worth reading as a triage list, because they span
four different subsystems: "insufficient quota", "readiness probe failures", "image pull
errors", "insufficient permissions", "limit ranges" and "application runtime
misconfiguration". Detection is opt-in through a deadline: `.spec.progressDeadlineSeconds`
"denotes the number of seconds the Deployment controller waits before indicating (in the
Deployment status) that the Deployment progress has stalled." Once it is exceeded, the
controller adds a condition with `type: Progressing`, `status: "False"` and
`reason: ProgressDeadlineExceeded`; the same condition can also "fail early" with a reason
such as `ReplicaSetCreateError`. What the controller does *not* do is the part that
surprises people during an incident: "Kubernetes takes no action on a stalled Deployment
other than to report a status condition with `reason: ProgressDeadlineExceeded`. Higher
level orchestrators can take advantage of it and act accordingly, for example, rollback
the Deployment to its previous version." So an automatic rollback is a property of your
CD tooling, never of the Deployment controller. Two details matter when scripting this:
`kubectl rollout status` "returns a non-zero exit code if the Deployment has exceeded the
progression deadline", which is the machine-readable gate a pipeline should use, and a
paused rollout is exempt — "if you pause a Deployment rollout, Kubernetes does not check
progress against your specified deadline." A quota failure additionally shows up as its
own `ReplicaFailure` condition with `reason: FailedCreate`, and the documented example
output pairs it with `Available: True` / `MinimumReplicasAvailable` — a stalled rollout and
a Deployment that still has minimum availability are not mutually exclusive states.

## A container with no probe is treated as Success, which is why a rollout can "complete" before the app is usable
source: kubernetes https://v1-36.docs.kubernetes.io/docs/concepts/workloads/pods/probes/#probe-results

The kubelet reduces every probe execution to one of three results, and the action taken
depends on which probe produced it. On `Failure`, "for liveness and startup probes, the
kubelet kills the container, and the container is subjected to its restart policy. For
readiness probes, the kubelet marks the container as not ready, and the Pod stops
receiving traffic from matching Services." That asymmetry is the first thing to establish
when a Pod is restarting versus merely sitting at `0/1 READY`: restarts point at liveness
or startup, a Pod that stays not-ready without restarting points at readiness. `Unknown`
means "the diagnostic failed (no action should be taken, and the kubelet will make further
checks)" — it is not a failure and does not count toward a threshold. The rule that most
often explains a rollout which reported success while users saw errors is the default for
a probe that was never configured: "if a container does not provide a particular probe,
the kubelet always considers the result as Success." A container with no readiness probe
is therefore ready the moment it is running, so nothing in the rollout is waiting on the
new process to finish loading configuration or warming caches. One boundary case cuts the
other way and is easy to misread as a failing check: "for readiness probes specifically,
the result is considered `Failure` before the initial delay."

## For a slow-starting application, a startup probe is the documented alternative to stretching the liveness probe
source: kubernetes https://v1-36.docs.kubernetes.io/docs/concepts/workloads/pods/probes/#when-should-you-use-a-startup-probe

"Startup probes are useful for Pods that have containers that take a long time to come
into service. Rather than set a long liveness interval, you can configure a separate
configuration for probing the container as it starts up, allowing a time longer than the
liveness interval would allow." The documentation gives an explicit threshold for when the
liveness probe alone is no longer safe: "if your container usually starts in more than
`initialDelaySeconds + failureThreshold × periodSeconds`, you should specify a startup
probe that checks the same endpoint as the liveness probe." Note that this budget is a sum
of the liveness probe's own settings, so the question to ask about a JVM service that
takes ninety seconds to serve traffic is not "is the app slow" but "does ninety seconds
fit inside that expression" — if it does not, the liveness probe will kill the container
mid-startup and the Deployment will never converge. The prescribed shape of the fix is
also specific: "the default for `periodSeconds` is 10s. You should then set its
`failureThreshold` high enough to allow the container to start, without changing the
default values of the liveness probe." In other words the startup probe absorbs the long
wait and the liveness probe keeps its tight settings for steady state, which is what
preserves its purpose — the same paragraph notes this "helps to protect against
deadlocks", the failure mode liveness probes exist to catch.

## CrashLoopBackOff names the backoff, not the cause — the cause is in the logs and events
source: kubernetes https://v1-36.docs.kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#container-restarts

Kubernetes handles a failing container as a sequence, and only the third step is the one
that shows up in `kubectl get pods`: an "initial crash" triggers an immediate restart
according to the Pod's `restartPolicy`; "repeated crashes" make Kubernetes apply "an
exponential backoff delay for subsequent restarts … this prevents rapid, repeated restart
attempts from overloading the system"; the `CrashLoopBackOff` state "indicates that the
backoff delay mechanism is currently in effect for a given container that is in a crash
loop, failing and restarting repeatedly"; and finally "if a container runs successfully
for a certain duration (e.g., 10 minutes), Kubernetes resets the backoff delay, treating
any new crash as the first one." Read literally, the status names the throttle, not the
fault — which is why it is a symptom to explain rather than an error to fix. The documented
causes cover more than application bugs: "application errors that cause the container to
exit", "configuration errors, such as incorrect environment variables or missing
configuration files", "resource constraints, where the container might not have enough
memory or CPU to start properly", "health checks failing if the application doesn't start
serving within the expected time", and "container liveness probes or startup probes
returning a `Failure` result". The investigation order given is `kubectl logs` first
("often the most direct way to diagnose the issue causing the crashes"), then
`kubectl describe pod` for events, then configuration, then resource limits.

## A container killed for exceeding its memory limit says OOMKilled in Last State, not in the application log
source: kubernetes https://v1-36.docs.kubernetes.io/docs/concepts/configuration/manage-resources-containers/#my-container-is-terminated

When a container disappears without a stack trace, the documented check is to "call
`kubectl describe pod` on the Pod of interest" and read the container's state block rather
than the log. The example output shows the shape to look for: a container whose `State` is
`Running` but whose `Last State` is `Terminated` with `Reason: OOMKilled`, `Exit Code: 137`
and a `Restart Count` above zero. The interpretation is given directly: "the
`Restart Count: 5` indicates that the `simmemleak` container in the Pod was terminated and
restarted five times (so far). The `OOMKilled` reason shows that the container tried to use
more memory than its limit." Two things follow for a service that only dies under
production load. The kill is a comparison against the container's own configured
`limits.memory` — in the example, a `memory: 50Mi` limit on that one container — so a node
that still had free memory is not evidence against an OOM kill. And the
remedy is a fork in the road rather than a reflex: "your next step might be to check the
application code for a memory leak. If you find that the application is behaving how you
expect, consider setting a higher memory limit (and possibly request) for that container."
Raising the limit without answering the first question converts a fast crash into a slow
one.

## Rolling back restores the Pod template only, and a revision exists only because the template changed
source: kubernetes https://v1-36.docs.kubernetes.io/docs/concepts/workloads/controllers/deployment/#rolling-back-a-deployment

`kubectl rollout undo deployment/<name>` reverts to the previous revision, and
`--to-revision=<n>` selects a specific one. What that operation covers is narrower than
"undo the last change": "a Deployment's revision is created when a Deployment's rollout is
triggered. This means that the new revision is created if and only if the Deployment's Pod
template (`.spec.template`) is changed, for example if you update the labels or container
images of the template. Other updates, such as scaling the Deployment, do not create a
Deployment revision, so that you can facilitate simultaneous manual- or auto-scaling. This
means that when you roll back to an earlier revision, only the Deployment's Pod template
part is rolled back." So a rollback does not restore the replica count, and an incident
caused by a scale change is not addressed by `rollout undo` at all. The documentation's
worked example is the classic bad image: after `kubectl set image` with a typo, the rollout
gets stuck and `kubectl get pods` shows one new Pod in `ImagePullBackOff` while the three
old Pods keep running — and, importantly, "the Deployment controller stops the bad rollout
automatically, and stops scaling up the new ReplicaSet. This depends on the rollingUpdate
parameters (`maxUnavailable` specifically) that you have specified. Kubernetes by default
sets the value to 25%." That containment is why a broken image usually degrades a release
rather than taking the service down.

## revisionHistoryLimit: 0 silently removes the ability to roll back at all
source: kubernetes https://v1-36.docs.kubernetes.io/docs/concepts/workloads/controllers/deployment/#clean-up-policy

Rollback history is not stored separately — it lives in the old ReplicaSets, and
`.spec.revisionHistoryLimit` "specifies how many old ReplicaSets for this Deployment you
want to retain. The rest will be garbage-collected in the background. By default, it is
10." Setting it to zero is the trap, because it reads like a tidiness preference and is not
rejected: "explicitly setting this field to 0, will result in cleaning up all the history
of your Deployment thus that Deployment will not be able to roll back." Nothing warns at
apply time; the cost is discovered during the incident, when `rollout undo` has no revision
to return to. Two timing details change how the cleanup behaves in exactly the situations
where it matters. "The cleanup only starts **after** a Deployment reaches a complete
state", and even with the limit at zero "any rollout nonetheless triggers creation of a new
ReplicaSet before Kubernetes removes the old one." And the count is a ceiling only for
healthy Deployments: "even with a non-zero revision history limit, you can have more
ReplicaSets than the limit you configure. For example, if pods are crash looping, and there
are multiple rolling updates events triggered over time, you might end up with more
ReplicaSets than the `.spec.revisionHistoryLimit` because the Deployment never reaches a
complete state." A pile of old ReplicaSets in `kubectl get rs` is therefore a signal that
rollouts have been failing, not that garbage collection is broken.
