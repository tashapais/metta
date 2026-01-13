# ThunderdomeSingleEpisodeRunner

Our goal is to evaluate cogames policies that are contributed by external researchers (which involves granting a limited
amount of remote code execution), while maintaining or raising our security bar.

## Current approach

As of 2026-01-12, we use Linux containers to run evaluations for the tournament, with all of the involved policies plus
the game engine running in a single Python process.

### k8s pod per episode

The unit of evaluation work is an episode. Each evaluation is a separate Kubernetes pod.

### SingleEpisodeRunner in a k8s container

The pod today is a SingleEpisode job, which creates a PureSingleEpisode job as a local JSON file. The "impure" parent
process then passes that job description to a "pure" Python subprocess.

That subprocess downloads the zip file for each policy involved in the evaluation, runs the simulation of that single
episode, and writes out the results.json and the replay.json.z files. We call it "pure" because it does not generate
other side effects, especially those visible from outside of the pod/machine.

The "impure" parent process then is responsible for the side-effects: it uploads the replay to S3 and posts the results
to Observatory.

## The path to VM-level isolation

We'll still use k8s jobs, and their config will be similar to what it is today.

The jobs will run on a dedicated set of machines that are able to run small ephemeral VMs. These will be "metal" EC2
instances, such as `c7i.metal-24xl`. (This depends on the CPU/RAM ratio we need, the desired hardware generation, and
the desired blast radius for single-machine failure.) These machines will have hypervisor software installed, such as
the pairing of kata-containers and containerd.

The new SingleEpisode jobs run with the usual container image, to include our Python code and virtualenv. But the
container will provide no meaningful isolation; it will mount the host's root filesystem (such as at `/host`), and will
allow trivial escape via the `nsenter` command.

### Preparing the job directory

Our new Python code will download the necessary policy zip files to the host's filesystem (such as to
`/host/tmp/job-xyz/input/policy/dinkyv15.zip`).

It will generate and write a PureSingleEpisodeConfig to the host's filesystem (such as to
`/host/tmp/job-xyz/input/episode.json`). The config will use relative paths for the inputs and outputs.

It will create a directory for the outputs on the host's filesystem (such as `/host/tmp/job-xyz/output/`).

It will generate a pod config for a single VM-isolated container. It will write it to a path on the host's filesystem
(such as `/host/tmp/job-xyz/pod.yaml`). The pod config will include instructions to mount the input directory as
read-only, and the output directory as read-write.

### Running the job

We'll then use `nsenter` to talk to the host machine's `containerd`, asking it to run the pod we've specified and to use
VM-level isolation.

### Upload and cleanup

Once we hear from the machine-wide VM manager that the job is complete, we process the result and replay as usual and
clean up the tmpdir.

On timeout, we kill it and clean up and report failure.

Do we leak resources? Maybe! We could drain and cycle these machines somewhat often, especially if episodes only run for
a few minutes.

## What Thunderdome gets us

That gives us VM-level isolation, which raises our security bar.

### Benefits of the Thunderdome approach

Quick shipping of the "don't get owned" goal; there's a much higher bar for policies to escape a VM vs to escape a
container.

The name is neat :)

Zero network access for policies.

### Shortcomings of the Thunderdome approach

It does not prevent policies from taking over the episode and reporting that they've earned a million hearts.

It does not help us identify which policy is responsible for an OOM.

No parallelism; we step policies one at a time.

Using a single Python process for the game runner and all policies means class names may conflict.

## What's after Thunderdome

### Split into multiple Python processes within one VM

We'll still be coupled to Python, and policy authors will need to make do with whatever version of the Python
dependencies we provide.

But they won't need to worry about class name collisions.

We'll need to build (finish building) a wire protocol for the game engine to communicate with the policies.

### Split into multiple Python processes, each in a separate VM.

Still coupled to Python. Still need the wire protocol.

But we get clear attribution for memory usage, including the right policy getting OOM-killed if it uses too much.

### Policies take the form of container images, which each run in separate VMs.

Requires the most building, but this gives policy authors a lot of flexibility. They can choose their own version of any
Python dependency (and perhaps even the Python interpreter). They can write in other languages (after we/they write a
small SDK for the Policy Protocol).

The contract between the game engine and the participating policies is in the form of JSON/HTTP requests for which we
write clear documentation, with us also providing a Python SDK as a reference implementation.
