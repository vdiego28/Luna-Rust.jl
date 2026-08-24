using TestItems

@testitem "QueueExec topology, exact-once execution, and cleanup" tags=[:io] begin
    import Test: @test, @testset, @test_throws
    using Amalthea
    using Distributed
    import SHA
    Core.eval(Main, :(using Amalthea))

    @test Scans.QueueExec().threads_per_worker == 1
    @test Scans.QueueExec(2, "queue.h5").threads_per_worker == 1
    @test Scans.QueueExec(2, "queue.h5"; threads_per_worker=3).threads_per_worker == 3
    @test_throws ArgumentError Scans.QueueExec(threads_per_worker=0)

    before = sort(workers())
    mktempdir() do dir
        function run_small(name, subdir; fail_at=0)
            mkpath(subdir)
            scan = Scan(name, Scans.QueueExec(nproc=2, threads_per_worker=1); value=1:8)
            Main.eval(quote
                Amalthea.Scans.runscan($scan) do scanidx, value
                    write(joinpath($subdir, string(scanidx)),
                          "$(scanidx):$(value):$(Threads.nthreads())")
                    scanidx == $fail_at && error("intentional QueueExec test failure")
                    # Exercise two resident handles concurrently in distinct workers.
                    linop = zeros(ComplexF64, 16)
                    field = fill(ComplexF64(1, -0.25), 16)
                    stepper = Amalthea.RK45.RustNativeStepper(linop, field, 0.0, 1e-3)
                    Amalthea.RK45.step!(stepper)
                end
            end)
        end

        name_a = "queue_concurrent_a_$(getpid())"
        name_b = "queue_concurrent_b_$(getpid())"
        task_a = @async run_small(name_a, joinpath(dir, "a"))
        task_b = @async run_small(name_b, joinpath(dir, "b"))
        fetch(task_a)
        fetch(task_b)
        for sub in ("a", "b")
            files = sort(parse.(Int, readdir(joinpath(dir, sub))))
            @test files == collect(1:8)
            values = read.(joinpath.(joinpath(dir, sub), string.(1:8)), String)
            @test all(s -> endswith(s, ":1"), values)
        end

        fail_name = "queue_failure_$(getpid())"
        run_small(fail_name, joinpath(dir, "failure"); fail_at=3)
        @test sort(parse.(Int, readdir(joinpath(dir, "failure")))) == collect(1:8)

        for name in (name_a, name_b, fail_name)
            digest = bytes2hex(SHA.sha256(name))[1:16]
            @test !isfile(joinpath(Utils.cachedir(), "qfile_$digest.h5"))
        end
    end
    @test sort(workers()) == before
end
