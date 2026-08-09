use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const CUDA_RAMAN_MAX_OSCILLATORS: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CudaBuildMode {
    Off,
    Auto,
    Required,
}

fn main() {
    println!("cargo:rerun-if-changed=src/kernels.cu");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=AMALTHEA_CUDA_BUILD");
    println!("cargo:rerun-if-env-changed=AMALTHEA_REQUIRE_CUDA_TESTS");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let dest_path = out_dir.join("kernels.ptx");
    let require_cuda_tests =
        env::var("AMALTHEA_REQUIRE_CUDA_TESTS").is_ok_and(|value| value == "1");
    let cuda_mode = cuda_build_mode(
        env::var("AMALTHEA_CUDA_BUILD").ok().as_deref(),
        require_cuda_tests,
    )
    .unwrap_or_else(|message| panic!("{message}"));

    write_cuda_raman_limits(&out_dir).expect("failed to write CUDA Raman capacity contract");

    let nvcc = (cuda_mode != CudaBuildMode::Off).then(find_nvcc).flatten();
    if let Err(message) = configure_cuda(cuda_mode, nvcc.as_deref(), &dest_path) {
        panic!("{message}");
    }
}

fn cuda_build_mode(value: Option<&str>, require_cuda_tests: bool) -> Result<CudaBuildMode, String> {
    if require_cuda_tests {
        return Ok(CudaBuildMode::Required);
    }
    match value.unwrap_or("auto").trim().to_ascii_lowercase().as_str() {
        "off" => Ok(CudaBuildMode::Off),
        "auto" => Ok(CudaBuildMode::Auto),
        "required" => Ok(CudaBuildMode::Required),
        value => Err(format!(
            "invalid AMALTHEA_CUDA_BUILD={value:?}; expected off, auto, or required"
        )),
    }
}

fn write_cuda_raman_limits(out_dir: &Path) -> std::io::Result<()> {
    fs::write(
        out_dir.join("cuda_raman_limits.h"),
        format!("#define AMALTHEA_CUDA_RAMAN_MAX_OSCILLATORS {CUDA_RAMAN_MAX_OSCILLATORS}\n"),
    )?;
    fs::write(
        out_dir.join("cuda_raman_limits.rs"),
        format!("pub const CUDA_RAMAN_MAX_OSCILLATORS: usize = {CUDA_RAMAN_MAX_OSCILLATORS};\n"),
    )
}

fn find_nvcc() -> Option<PathBuf> {
    if let Some(nvcc) = env::var_os("NVCC") {
        return Some(PathBuf::from(nvcc));
    }
    for root_var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Some(root) = env::var_os(root_var) {
            let candidate = PathBuf::from(root).join("bin").join(if cfg!(windows) {
                "nvcc.exe"
            } else {
                "nvcc"
            });
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    let conventional = PathBuf::from("/usr/local/cuda/bin/nvcc");
    if conventional.is_file() {
        return Some(conventional);
    }
    Command::new("nvcc")
        .arg("--version")
        .status()
        .ok()
        .filter(|status| status.success())
        .map(|_| PathBuf::from("nvcc"))
}

fn configure_cuda(
    mode: CudaBuildMode,
    nvcc: Option<&Path>,
    dest_path: &Path,
) -> Result<(), String> {
    match mode {
        CudaBuildMode::Off => write_dummy_ptx(dest_path)
            .map_err(|error| format!("failed to write CPU-only PTX marker: {error}")),
        CudaBuildMode::Auto => compile_or_fallback(nvcc, dest_path, false),
        CudaBuildMode::Required => compile_or_fallback(nvcc, dest_path, true),
    }
}

fn compile_or_fallback(
    nvcc: Option<&Path>,
    dest_path: &Path,
    require_cuda: bool,
) -> Result<(), String> {
    let failure = match nvcc {
        Some(nvcc) => {
            let status = match std::fs::remove_file(dest_path) {
                Ok(()) => run_nvcc(nvcc, dest_path),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    run_nvcc(nvcc, dest_path)
                }
                Err(error) => return Err(format!("failed to clear prior PTX output: {error}")),
            };
            match status {
                Ok(status) if status.success() && is_real_ptx(dest_path) => return Ok(()),
                Ok(status) if status.success() => {
                    "nvcc reported success but did not produce real PTX".to_owned()
                }
                Ok(status) => format!("nvcc compilation failed with status {status}"),
                Err(error) => format!("failed to invoke nvcc: {error}"),
            }
        }
        None => "nvcc compiler not found".to_owned(),
    };

    if require_cuda {
        return Err(format!(
            "CUDA support is required but nvcc did not produce real PTX: {failure}. \
             Set NVCC or CUDA_HOME/CUDA_PATH to a working CUDA toolkit, or use \
             AMALTHEA_CUDA_BUILD=off for a CPU-only build"
        ));
    }

    println!("cargo:warning={failure}; creating dummy PTX for CPU-only build");
    write_dummy_ptx(dest_path).map_err(|error| format!("failed to write dummy PTX: {error}"))
}

fn run_nvcc(nvcc: &Path, dest_path: &Path) -> std::io::Result<std::process::ExitStatus> {
    Command::new(nvcc)
        .args(["--ptx", "-I"])
        .arg(dest_path.parent().unwrap_or_else(|| Path::new(".")))
        .args(["src/kernels.cu", "-o"])
        .arg(dest_path)
        .status()
}

fn is_real_ptx(path: &Path) -> bool {
    std::fs::read_to_string(path).is_ok_and(|ptx| {
        ptx.contains(".version")
            && ptx.contains(".target")
            && ptx.contains(".address_size")
            && ptx.contains(".visible .entry")
    })
}

fn write_dummy_ptx(dest_path: &Path) -> std::io::Result<()> {
    // Keep include_str! valid for CPU-only development; strict CUDA mode never reaches this path.
    std::fs::write(dest_path, "// DUMMY PTX\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_ptx_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "amalthea-build-rs-{name}-{}-{}.ptx",
            std::process::id(),
            line!()
        ))
    }

    #[test]
    fn cpu_only_build_writes_dummy_ptx_without_nvcc() {
        let dest_path = test_ptx_path("cpu-fallback");
        let _ = std::fs::remove_file(&dest_path);

        compile_or_fallback(None, &dest_path, false).unwrap();
        assert_eq!(
            std::fs::read_to_string(&dest_path).unwrap(),
            "// DUMMY PTX\n"
        );
        assert!(!is_real_ptx(&dest_path));

        std::fs::remove_file(dest_path).unwrap();
    }

    #[test]
    fn explicit_off_never_invokes_configured_nvcc() {
        let dest_path = test_ptx_path("explicit-off");
        let _ = std::fs::remove_file(&dest_path);

        configure_cuda(
            CudaBuildMode::Off,
            Some(Path::new("/definitely/not/a/real/nvcc")),
            &dest_path,
        )
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(&dest_path).unwrap(),
            "// DUMMY PTX\n"
        );

        std::fs::remove_file(dest_path).unwrap();
    }

    #[test]
    fn cuda_build_mode_has_explicit_policy_and_strict_test_precedence() {
        assert_eq!(cuda_build_mode(None, false).unwrap(), CudaBuildMode::Auto);
        assert_eq!(
            cuda_build_mode(Some("off"), false).unwrap(),
            CudaBuildMode::Off
        );
        assert_eq!(
            cuda_build_mode(Some("AUTO"), false).unwrap(),
            CudaBuildMode::Auto
        );
        assert_eq!(
            cuda_build_mode(Some(" required "), false).unwrap(),
            CudaBuildMode::Required
        );
        assert_eq!(
            cuda_build_mode(Some("off"), true).unwrap(),
            CudaBuildMode::Required
        );
        assert!(
            cuda_build_mode(Some("maybe"), false)
                .unwrap_err()
                .contains("expected off, auto, or required")
        );
    }

    #[test]
    fn strict_cuda_build_rejects_missing_nvcc() {
        let dest_path = test_ptx_path("strict-missing-nvcc");
        let _ = std::fs::remove_file(&dest_path);

        let error = compile_or_fallback(None, &dest_path, true).unwrap_err();
        assert!(error.contains("CUDA support is required"));
        assert!(!dest_path.exists());
    }

    #[test]
    fn real_ptx_requires_nvcc_output_markers() {
        let dest_path = test_ptx_path("ptx-markers");
        std::fs::write(
            &dest_path,
            ".version 8.0\n.target sm_80\n.address_size 64\n.visible .entry kernel() {}\n",
        )
        .unwrap();
        assert!(is_real_ptx(&dest_path));

        std::fs::write(&dest_path, "// DUMMY PTX\n").unwrap();
        assert!(!is_real_ptx(&dest_path));

        std::fs::remove_file(dest_path).unwrap();
    }
}
