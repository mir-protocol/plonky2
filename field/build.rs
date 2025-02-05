use rustc_version::{Channel, VersionMeta};

fn main() {
    let version_meta = VersionMeta::for_command(std::process::Command::new("rustc")).unwrap();

    if version_meta.channel == Channel::Nightly {
        println!("cargo:rustc-cfg=nightly");
    }

    // Declare `nightly` as a valid `cfg` condition for rustc to prevent warnings
    println!("cargo::rustc-check-cfg=cfg(nightly)")
}
