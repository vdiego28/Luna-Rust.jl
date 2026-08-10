"""
    _target_triple(kernel=Sys.KERNEL, arch=Sys.ARCH)

Return the exact release-binary target for a supported OS/architecture pair,
or `nothing` so the installer compiles from source. Never return a binary for a
different architecture merely because the operating system matches.
"""
function _target_triple(kernel::Symbol=Sys.KERNEL, arch::Symbol=Sys.ARCH)
    kernel === :Linux && arch === :x86_64 && return "x86_64-unknown-linux-gnu"
    kernel === :Linux && arch === :aarch64 && return "aarch64-unknown-linux-gnu"
    kernel === :Darwin && arch === :aarch64 && return "aarch64-apple-darwin"
    kernel === :NT && arch === :x86_64 && return "x86_64-pc-windows-msvc"
    nothing
end

"""Return whether a CPU-only release binary satisfies the requested build policy."""
function _cpu_prebuilt_allowed(cuda_build::AbstractString, require_cuda_tests::AbstractString)
    require_cuda_tests == "1" && return false
    lowercase(strip(cuda_build)) == "off"
end
