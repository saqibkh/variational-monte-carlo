# xpu additions, fixes, and migrations

Tracks what xpu needs so VMC can drop its own backend code, plus xpu defects and slow paths found while doing that. Names below are proposals unless marked as existing.

Baseline: xpu `10efcac` (2026-09-25). VMC tracks xpu `main`, and CI records which revision each run used. Evidence levels:

- **Measured**: timed on the Ryzen 5 8500G (AVX-512, GCC 15, 6 OpenMP threads).
- **Verified**: confirmed by compiling, running, or reading generated GPU machine code (SASS).
- **Code**: read from source only, not yet measured. GPU items are mostly here because the RTX 3070 was unavailable.

## 1. Bugs

Fix these first. Item 1 silently breaks consumers.

| # | Problem | Location | Evidence | Fix |
| --- | --- | --- | --- | --- |
| 1 | `assume_aligned` calls host-only `std::assume_aligned` from device code. Without `--expt-relaxed-constexpr`, nvcc warns (#20013/#20015) and compiles every `soa_view` access into a trap. VMC crashed with error 719 on every CUDA run. | `memory.hpp:97` | Verified: a one-line kernel writing through `soa_view` has no store instruction without the flag, only `BPT.TRAP`. With the flag the store is present. xpu's own CUDA build emits 66 of these warnings. | Use a device-safe path (`cuda::std::assume_aligned` or `__builtin_assume_aligned`). Add a CUDA test that indexes an `soa_view` inside a kernel; no current test does. |
| 2 | `parallel_reduce_sum` without scratch works on CPU and aborts on CUDA ("insufficient reduction scratch storage"). Code that passes CPU tests dies on GPU. | `launch.hpp` `parallel_reduce_sum_impl` | Code | Adopt the workspace design in section 3.4. |
| 3 | CPU and CUDA random streams differ (`mt19937_64` vs Philox), so no cross-backend parity is possible. CPU skip-ahead is linear in the offset. | `random.hpp` | Measured: `seed(offset=1e6)` takes 159 us vs 4 us at offset 0. Generator state is 2504 B on CPU. | Use counter-based Philox on CPU too: identical streams, O(1) skip-ahead, 64 B state. |
| 4 | `-Warray-bounds` fires for every 1-D reduce: the grain-splitting loop indexes `counts[++split]` even though it can't run for `dims == 1`. | `launch.hpp:443` | Verified: 3 warnings in the VMC build | Guard the split loop with `if constexpr (dims > 1)`. |
| 5 | `free` ignores `cudaFree`'s status, which can swallow earlier asynchronous errors. | `memory.hpp` `free` | Code | Check the status. |

## 2. Performance audit

Every component was measured against a hand-written baseline doing the same work. The CPU numbers come from wall-clock timings (best of 3-5 runs) plus `perf` profiles. The GPU numbers come from the compiled sm_86 machine code (SASS: instruction count, registers, software 64-bit divisions, trap paths), because the GPU was unavailable for runtime profiling. Each item's fix is the change expected to give the best performance.

### 2.1 Compiler flags xpu should pass to consumers (CPU)

| # | Problem | Evidence | Fix |
| --- | --- | --- | --- |
| 1 | GCC's default `-fmath-errno` makes every `xpu::sqrt`, `exp`, `log` and `pow` able to set `errno`, which blocks vectorization and inlining of the loops that call them. | Measured: `xpu::rsqrt` loop **2.86 ns → 0.65 ns (4.4x)** with `-fno-math-errno`. Results are unchanged; only `errno` stops being set. | Add `-fno-math-errno` to the xpu target's exported compile flags. Nobody reads `errno` from math functions in numerical code. |
| 2 | Math calls in loops never use the vectorized math library (libmvec). glibc only declares the vector versions under `__FAST_MATH__`. | Measured: a `sincos` loop takes **9.8 ns/element scalar vs 1.77 ns with libmvec (5.5x)**. | Offer an opt-in vector math path: either declare the libmvec/SLEEF variants (`#pragma omp declare simd`) behind an xpu option, or compile only the math-heavy translation unit with `-ffast-math`. Keep it opt-in, since it changes the last bits of results. |
| 3 | On CPU, `xpu::sincos` is a separate `sin` plus `cos`. | Measured: **10.4 vs 9.4 ns** for glibc `sincos` (10%). | Call `::sincos`/`::sincosf` on CPU too. |

### 2.2 SoA views

| # | Problem | Evidence | Fix |
| --- | --- | --- | --- |
| 1 | `soa_view::operator[]` recomputes the padded stride with overflow-checked arithmetic on every access. The compiler can't hoist it out of loops, because the check can call `abort`. | Measured (CPU update loop `c += a*b`): **3.6x slower at n=485, 1.76x at n=1e5**. Copying the three pointers out of the view before the loop matches raw pointers exactly (0.045 us), which isolates the cause. Read-only loops are unaffected. | Compute the stride once when the view is constructed and store it in the view. Accessors do no checks (validate once in the constructor), and array pointers are returned `restrict`-qualified. |
| 2 | Same issue on GPU, plus every accessor carries a `printf` + trap path. | Verified (SASS): writing through a view takes **72 instructions and 24 registers** vs **32 and 14** with raw pointers, and includes a trap path. | Same fix. Fewer registers also means more resident threads per SM. |
| 3 | `soa_batch_view::view(b)` inside a kernel runs `checked_mul`/`checked_bytes` in the `soa_view` constructor. | Verified (SASS): **3 trap paths and 4 absolute calls** in a trivial kernel (136 instructions). | Unchecked device-side constructors. Validate on the host where the batch is created. |

### 2.3 `parallel_for`

| # | Problem | Evidence | Fix |
| --- | --- | --- | --- |
| 1 | On CPU, every call forks and joins an OpenMP team, with no serial fallback. That costs about 2 us per call regardless of size. | Measured: **2.0 us** at n=64-4096 vs **0.012-0.22 us** serial SIMD (**60-170x slower** at n ≤ 485). Parallel only wins above about **32k** elements. `perf`: **68% of the time is in libgomp** fork/join/barrier. | Run a serial SIMD loop when the total work is below a threshold (calibrated per machine, or about 16-32k simple elements). Also let callers pass a grain size or cost hint. |
| 2 | Called from inside an existing OpenMP region (a resident walker loop), it still enters a nested region that runs serially. | Measured: **1.36 us per call** at n=485, vs 0.028 us for the loop itself. | Check `omp_in_parallel()` and run the loop inline. |
| 3 | On CUDA, multi-dimensional ranges decode each index with runtime 64-bit division and remainder (`itr_index`). | Verified (SASS): 2-D kernel **560 instructions with 7 software-division calls** vs **56** for 1-D. | Launch a real 2-D/3-D grid, or precompute fast divisors (multiply-shift). Treat a dense range as 1-D when the functor only needs the linear index. |
| 4 | No thread-count setting or nesting policy. | Code | Accept a thread budget as part of the launch policy (section 3.1). |

Checked and fine: 3-D CPU `parallel_for` matches a hand-written `omp parallel for` + `omp simd` loop (11.7 ms both, W=48 × N=485 × K=248). Large 1-D ranges also match (103 vs 107 us at n=2M).

### 2.4 `parallel_reduce_sum`

| # | Problem | Evidence | Fix |
| --- | --- | --- | --- |
| 1 | On CPU, small ranges still enter an OpenMP region (`if(boxes > 1)` is false but the region is still created). | Measured: **0.43-0.48 us** at n=64-485 vs **0.007-0.048 us** serial SIMD (**10-60x**). `perf`: about 34% of the time is in libgomp. | Same serial fast path as 2.3 item 1. |
| 2 | The fixed grain of 8192 under-parallelizes mid-size ranges (32k elements makes only 4 boxes for 6 threads). | Measured: n=32k is **3.03 us vs 1.95 us** for a hand-written OpenMP reduction (1.55x). At n=262k, 9.5 vs 8.6 us. At n=2M xpu is *faster* (66.7 vs 81.9 us). | Size the grain from the thread count, for example `max(min_grain, n / (4 × threads))`. |
| 3 | On CUDA, multi-dimensional reductions decode indices with 64-bit division inside the cub transform iterator. | Verified (SASS): 3-D reduce kernels are **3,864 instructions with 87 software-division calls** vs **688-984 with none** for 1-D. | Same as 2.3 item 3. For dense ranges, give the contribution the linear index. |
| 4 | On CUDA, the caller manages scratch memory, and it's re-queried every call. | Code | The workspace design in section 3.4. |

### 2.5 Memory and buffers

| # | Problem | Evidence | Fix |
| --- | --- | --- | --- |
| 1 | `buffer` always zero-fills, which is one extra full write pass. On CUDA that's also an extra kernel launch per allocation. | Measured (CPU): 8 MB allocate-and-write goes from **63 us to 310 us** when the memory is already mapped (matches an explicit `memset` + write, 327 us). For a fresh 400 MB allocation the page faults dominate and the difference is within noise. An earlier "+39%" figure was an ordering artifact. | Add an uninitialized constructor or tag, and use it where the buffer is overwritten right away. |
| 2 | Small CPU allocations use aligned `operator new`. | Measured: **21 ns vs 5.8 ns** for `malloc`/`free` (1 KB). | Low priority. Only matters for per-call temporaries; avoid those rather than pool them. |
| 3 | Every CUDA buffer is a synchronous `cudaMalloc` + fill kernel + `cudaFree`. VMC's per-operation Jastrow path allocates a 1-element buffer on every call. | Code. Needs GPU timing. | Stream-ordered or pooled allocation (`cudaMallocAsync`), plus the uninitialized option. |

### 2.6 Math and atomics

| # | Problem | Evidence | Fix |
| --- | --- | --- | --- |
| 1 | CPU `atomic_add` is a compare-and-swap loop (`atomic_ref<double>`). | Measured: **4.72 vs 0.63 ns/op (7.5x)** for a single owner. | Document it as concurrency-only. Provide team and segmented reductions (sections 3.1, 3.4) as the fast path. |
| 2 | `rnorm3d`/`rsqrt` on CPU: after the errno fix (2.1 item 1) they match hand-written code (0.66 vs 0.68 ns). | Measured | No change needed beyond 2.1. |

### 2.7 Random numbers

| # | Problem | Evidence | Fix |
| --- | --- | --- | --- |
| 1 | CPU generators are `mt19937_64`: expensive to seed, skip-ahead is linear in the offset, and each is large. | Measured: `seed` **4.0 us**; `seed(offset=1e6)` **157 us**; `seed_n` **1.2 us/generator**. State is **2,504 B each (25 MB for 10k generators)**. | Philox4x32-10 on CPU: counter-based, O(1) skip-ahead, 64 B of state, and the same stream as CUDA (bug 1.3). |
| 2 | CUDA `uniform_index` uses 64-bit rejection plus 64-bit modulo. | Verified (SASS): **552 instructions, 48 registers, 2 software-division calls** vs **144 instructions** for 32-bit `curand() % n`. | When the bound fits in 32 bits, use a 32-bit draw with Lemire's multiply-shift method: no division, and still unbiased. |
| 3 | CPU `uniform<double>` (1.40 ns) and `uniform(a, b)` (1.73 ns) vs 1.00 ns for a raw `mt19937_64` draw. | Measured | Fine. Philox would change these numbers, so re-measure then. |

### 2.8 Linear algebra

| # | Problem | Evidence | Fix |
| --- | --- | --- | --- |
| 1 | CPU wrappers call LAPACKE in row-major mode. On every call LAPACKE allocates a temporary, transposes the matrix in and out, and scans the whole input for NaNs. | `perf`: **13.4% in `LAPACKE_dge_trans` + 6.2% in `LAPACKE_dge_nancheck`** of all LU time. Measured with OpenBLAS: `getrs` **2.1x (N=485) and 3.7x (N=1000)** slower than calling column-major directly; `getrf` +8-16%; invert (plus VMC's transpose) +11-21%. With reference LAPACK the overhead is smaller (+2-15%), because the O(N^3) work dominates. | Call LAPACK in column-major on the stored layout. A row-major matrix is its column-major transpose, so solve with `trans = 'T'` and expose the inverse in whichever layout the consumer wants (VMC wants the transpose). Skip the NaN scan (`LAPACKE_set_nancheck(0)`, or call `dgetrf_` directly). This removes up to 3 N^2 passes and an allocation per call. |
| 2 | Which LAPACK gets used is left to the system. Reference LAPACK is what gets picked when OpenBLAS isn't installed. | Measured: `getrf` N=485 **17.2 ms (reference) vs 2.0 ms (OpenBLAS)**, **8.6x**. | Prefer an optimized BLAS in CMake (`BLA_VENDOR`, with a clear message on fallback), and document it. |
| 3 | CUDA: one cuSOLVER handle per `lu_factorization`, and a blocking device-to-host status copy in every `factorize`/`solve`/`invert`. | Code. GPU init at W=48, N=485 spent about 230 ms in per-walker cuSOLVER (nsys, 2026-09-26). | Status that stays on the device, a shared or stream-bound handle, then batched LU (section 3.6). |
| 4 | CUDA `transpose_square` launches N^2 threads where half exit immediately, and one side of each swap is a strided (uncoalesced) access. It also inherits 2.3 item 3's division cost. | Verified (SASS): **608 instructions, 7 software-division calls**. Access pattern from code. | Tiled shared-memory transpose (32×32 tiles), or avoid the transpose entirely by using `CUBLAS_OP_T` / 2.8 item 1's layout choice. On CPU it's 56 us at N=485 (not a bottleneck). |

### 2.9 Still needs GPU runtime measurement

These need a working GPU to put numbers on (the Windows reboot is still pending): default-stream serialization, the cost of per-buffer `cudaMalloc` + fill, the per-call synchronization in linear algebra, `parallel_for` launch overhead for small ranges, and the SASS-based items in 2.2-2.8. Use nsys for timelines. ncu needs GPU counter access enabled in the Windows NVIDIA driver settings.

## 3. Missing features VMC needs

Grouped by the VMC code they would replace. Today VMC has 31 `#if defined(XPU_CUDA)` blocks and 13 `__CUDA_ARCH__` uses.

### 3.1 Team (block) execution

The resident walker kernels need one team per walker, with team-local index, width, barrier, scratch, and reductions. `parallel_for` can't do this: it's host-dispatched, one call per index, with no shared memory or barriers.

- VMC prototypes: [execution.hpp](src/utilities/execution.hpp) (`thread`, `stride`, `sync`), `store_partial_sum` in [energy_tracking_stencils.hpp](src/energy_tracking/energy_tracking_stencils.hpp).
- Proposed: a team launch policy (league = walkers, team size = block width, typed scratch), with `team_for`, `team_reduce_sum` and `team_barrier`. On CPU, a team is one thread: width 1, no-op barrier, reduce returns the local value. On GPU, use warp shuffles, not shared-memory atomics or a tree of barriers.
- Requirements: uniform participation, inactive lanes contribute zero, no dynamic allocation, and it must preserve the one-team-per-walker Markov chain. Test on real hardware at several team sizes.
- This is the largest single removal of VMC backend forks.

### 3.2 Streams and synchronization

Nothing in xpu takes a stream, and there's no `synchronize()`. CUDA calls return before they finish while CPU calls don't, and nothing documents that. VMC calls `cudaDeviceSynchronize` and `cudaMemcpy` directly ([wavefunction_kernels.hpp](src/wavefunction/wavefunction_kernels.hpp), [slater_plane_wave_kernels.hpp](src/slater_plane_wave/slater_plane_wave_kernels.hpp), [jastrow_pade_kernels.hpp](src/jastrow_pade/jastrow_pade_kernels.hpp)).

- Proposed: a stream/queue object accepted by every operation (`parallel_for`, reduce, `fill_n`, `copy_n`, linear algebra), plus `fence`/`synchronize`. On CPU, the queue is immediate.
- Do this before the API grows further: it changes every signature.

### 3.3 Memory model

- Host mirrors and explicit host/device copy helpers. The README says transfers aren't wrapped, which is stale: `copy_n`/`memcpy` exist but are synchronous, use the default stream, and are undocumented.
- Pinned host allocation for staging.
- Async copies on a queue.
- Uninitialized buffers (section 2.5 item 1).
- Pooled/stream-ordered allocation (section 2.5 item 3).
- Non-aborting allocation, so VMC can do graceful capacity checks (PLAN GPU-006). Today, running out of memory aborts.

### 3.4 Reductions

- **Workspace without a raw-byte contract.** Keep reductions as functions; the workspace is an object the caller owns and prepares once.

  ```cpp
  xpu::reduction_workspace<fp_t> workspace;

  xpu::prepare_reduce_sum(workspace, range, contribution);
  xpu::parallel_reduce_sum(range, result, contribution, workspace);
  ```

  - Preparation sizes and allocates backend scratch once, using the actual contribution type. It must not read input elements or keep input pointers.
  - Execution reuses the storage without allocating. Growing past the prepared size needs explicit re-preparation, and misuse must be reported clearly.
  - xpu owns byte counts, alignment and capacity checks. The CPU workspace can be empty. The raw pointer/byte API can stay underneath for advanced callers.
  - Specify stream/device compatibility, and forbid overlapping reuse of one workspace unless explicitly supported.
  - Fixes bug 2.
- **Segmented reduce:** one result per walker without a host dispatch per segment. It would replace VMC's custom partial-sum init kernels in [energy_tracking_kernels.hpp](src/energy_tracking/energy_tracking_kernels.hpp).
- **General transform-reduce:** min, max and custom operations, not just sum. A scan would also help.
- **Small-range serial fast path and an adaptive grain size** (sections 2.3-2.4).

### 3.5 Lambdas

`parallel_for` already accepts any callable. On CUDA, only device-compiled ones work. Verified: a plain lambda fails to compile, `[=] __host__ __device__` needs `--extended-lambda`, and with the flag it compiles. Functions passed by name can't work on CUDA.

- xpu should add `--extended-lambda` to the flags consumers inherit, and provide a portable lambda macro (`XPU_LAMBDA`: `[=] __host__ __device__` on CUDA, `[=]` on CPU; same pattern as Kokkos).
- Document nvcc's rules: capture by value; capture `*this`, not `this`; the enclosing function can't have a deduced return type or private/protected access.
- VMC would then replace the functor structs from the `parallel_for` port with lambdas at the call site.

### 3.6 Linear algebra

1. Status output that stays on the device, with defined meanings, pivot lifetime and ordering, plus the existing synchronous convenience API. VMC keeps per-walker status and log determinants on the device, selects failed walkers there, and retrieves one aggregate status. That removes per-walker synchronization without batched LU.
2. Batched factorization and inversion with one status per matrix. Support padded leading dimensions, batch strides or pointer arrays, backend pivot/output storage, stream/handle ownership, and reusable workspace (no allocation per call). Define how failed matrices are excluded from inversion and retried.
3. Compare sequential cuSOLVER, bounded concurrent calls, and vendor batched routines on real hardware. Don't assume batched is fastest at N=485.
4. Column-major CPU path without NaN checks, and a preferred optimized BLAS (section 2.8).

VMC keeps walker retry selection, deterministic RNG streams, and its 100-attempt limit.

### 3.7 Random numbers

- Philox on CPU (bug 3).
- Batched seeding exists (`xpu::random::seed_n`) and VMC uses it. It needs a device-resident array of stream IDs; an overload that takes a first ID and derives the rest would avoid VMC's temporary buffer and host-to-device copy.

### 3.8 Launch and occupancy queries

- Make the occupancy query public: blocks per SM × SM count for a given kernel and block size (`detail::wave_blocks` today). VMC needs it to pick a walker count that fills the GPU.
- Launch-size validation (grid limits, narrowing), currently VMC's `validate_initialization_size` in [energy_tracking_kernels.hpp](src/energy_tracking/energy_tracking_kernels.hpp).
- Checked allocation arithmetic is now in xpu (`detail::checked_*`) but internal. Expose it publicly so VMC's `EnergyTracker::initialize_reduction_storage` in [energy_tracking.cu](src/energy_tracking/energy_tracking.cu) can use it.

### 3.9 Error handling

Everything aborts, and almost everything is `noexcept`. Add non-aborting variants, or an error callback, for allocation, linear algebra and launch failures.

### 3.10 Optional

- `norm_squared3d(x, y, z)`: a convenience for `x*x + y*y + z*z`, not a demonstrated speedup.
- On CPU, `norm3d` doesn't guard against overflow the way CUDA's intrinsic does. Document the difference or match it.

## 4. Build, packaging, CI

| # | Problem | Evidence | Fix |
| --- | --- | --- | --- |
| 1 | An installed xpu can't be found with `find_package`: only `xpuTargets.cmake` is installed. | Verified: "Could not find xpuConfig.cmake" | Generate `xpuConfig.cmake` + a version file, and add `find_dependency` for CUDAToolkit, OpenMP and LAPACK. |
| 2 | No versions or tags. API changes land on `main` and break consumers (VMC broke on the macro renames and the `soa_batch_view` change). | Verified: `git tag` is empty | Semver tags + a changelog. |
| 3 | `XPU_ARCH_NATIVE` defaults to ON and passes `-march=native` to every consumer, so binaries are tied to the build machine. | Code | Default it to OFF. |
| 4 | Required nvcc flags aren't passed to consumers. | Verified (bug 1, section 3.5) | Export `--expt-relaxed-constexpr` (until bug 1 is fixed) and `--extended-lambda`. |
| 5 | PRs never get GPU testing: the CUDA job needs a self-hosted runner and a repository variable, and never runs on PRs. | Code (`.github/workflows/test.yml`) | Add a compile-only CUDA job on PRs with nvcc warnings as errors. It would have caught bug 1 without a GPU. |
| 6 | The README is stale: it says transfers aren't wrapped, and doesn't cover async semantics, reduce scratch, random-number behavior, or the `invert` layout. | Code | Add an API reference and an async/sync semantics section. |

## 5. Already in xpu

| Feature | xpu status | VMC status |
| --- | --- | --- |
| `soa_batch_view` | Exists; at `10efcac` the storage array count is part of the type | Consumed; migrated (section 6) |
| `parallel_reduce_sum` | Exists; CPU path optimized in `65131b2` | Consumed for log-det, determinant ratio, kinetic energy (per-operation path) |
| `parallel_for` | Exists; CPU nested-SIMD path in `7f2e49e` | Consumed for position and Slater init (2026-09-26 port) |
| `random::seed_n` | Exists | Consumed |
| `xpu::numbers`, `xpu::numeric_limits` | Exist (`numeric.hpp`) | Not yet: VMC still uses `xstd::` directly in [energy_tracking_stencils.hpp](src/energy_tracking/energy_tracking_stencils.hpp) and [simulation_stencils.hpp](src/simulation/simulation_stencils.hpp) |
| `atomic_add` | Exists | Don't use in CPU hot loops (section 2.6); prefer the ownership-based `+=` or team reductions |
| `buffer_view` | Exists | Not consumed; candidate for the raw-pointer fields in VMC's views |
| `norm3d`, `rnorm3d`, `rsqrt`, `sincos` | Exist | Partly consumed; the pair loops could use `rnorm3d` |

## 6. VMC migration to `10efcac` (done 2026-09-27)

Verified the way CI runs it: all 4 CPU test configurations pass (19/19 each), the CUDA compile-only job builds with 0 warnings, the fp64 validation suite passes (7/7), and energies and acceptance are bit-for-bit identical to the pre-migration build. CUDA tests have not been run on a GPU yet (the GPU needs a Windows reboot).

1. Renamed `CUDA_CALLABLE`, `DEVICE_ONLY` and `RESTRICT` to `XPU_CUDA_CALLABLE`, `XPU_DEVICE_ONLY` and `XPU_RESTRICT` (18 files in `src/`). This rename is what caused the cascading "scast", "_fp" and "idx not declared" errors.
2. `soa_batch_view` takes 3 constructor arguments, not 4: updated the 8 default initializers.
3. `Particles::BatchView` views 3 and 4 arrays of a 7-array batch, so its member types now carry the storage count (`Particles::PosBatchView`, `Particles::DerivativesBatchView`). Removed the now-unused `vec3_batch_view_t`.
4. Kept `--expt-relaxed-constexpr` in VMC's CMake until bug 1 is fixed in xpu.
5. Build tracking: `GIT_TAG main`. `scripts/test.sh` and `scripts/validate.sh` pass `FETCHCONTENT_UPDATES_DISCONNECTED=OFF` explicitly, because the old conditional ON setting stays cached in existing build directories and froze xpu. `validate.sh` no longer reuses the test build's xpu checkout. CI keys its dependency cache on xpu `main`'s current commit and prints the revision it built against.

## 7. Priority

1. Bug 1, the compile-only CUDA CI job with warnings as errors, and a device-side SoA test.
2. Cheap, measured, large wins: `-fno-math-errno` exported (2.1 item 1, 4.4x); `soa_view` stride stored once with unchecked accessors (2.2, up to 3.6x); serial fast path for small `parallel_for`/`reduce` plus `omp_in_parallel` inlining (2.3-2.4, 10-170x on small ranges); column-major LAPACK without NaN checks (2.8 item 1, up to 3.7x on solves).
3. Packaging and releases (section 4, items 1-4).
4. Streams and synchronization, plus the memory model (3.2, 3.3). This is the foundation, and it changes every signature.
5. Team execution with team reductions (3.1).
6. Multi-dimensional GPU index decoding without division (2.3 item 3, 2.4 item 3); adaptive reduce grain (2.4 item 2); 32-bit `uniform_index` (2.7 item 2).
7. Lambdas, the reduction workspace, segmented and general reduce, the public occupancy query, non-aborting errors.
8. Philox on CPU (bug 1.3, 2.7 item 1).
9. Linear algebra: device-resident status, then batched; tiled GPU transpose (2.8 items 3-4).
10. Opt-in vector math (2.1 item 2), uninitialized and pooled buffers (2.5).

## Ownership and verification

- VMC keeps the physics: Ewald, Jastrow and orbital formulas, walker state, Metropolis orchestration, and scientific validation. Generic primitives go in xpu.
- xpu tests its own primitives. VMC tests observable behavior and numerical results, and doesn't duplicate xpu's tests.
- Migrations must preserve RNG streams, precision support and numerical results. The check is bit-for-bit energy comparisons at fixed seeds on both backends.
- Compiling CUDA is not the same as verifying it on a GPU. Team reductions, synchronization, RNG seeding and LU need tests on real hardware, compute-sanitizer where it applies, and profiling for any performance claim.
