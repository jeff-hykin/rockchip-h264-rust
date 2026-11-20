{
  description = "rockchip-h264: pure-Rust MJPEG -> H.264 on the Rockchip RK3588 hardware";

  inputs = {
    nix-filter.url = "github:numtide/nix-filter";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    crate2nix.url = "github:nix-community/crate2nix";
    crate2nix.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nix-filter, nixpkgs, flake-utils, crate2nix }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin" ] (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        name = "rockchip-h264";
        src = nix-filter.lib { root = ./.; exclude = [ "target" "result" "tools/captures" ]; };
        generated = crate2nix.tools.${system}.generatedCargoNix { inherit name src; };

        # builds for `targetPkgs` (native, or pkgsCross.aarch64-multiplatform for the board)
        crateFor = targetPkgs: (import generated {
          pkgs = targetPkgs;
          buildRustCrateForPkgs = cratePkgs: crate:
            cratePkgs.buildRustCrate (crate // pkgs.lib.optionalAttrs (crate.crateName == name) {
              # the test harness lives in examples/; build it as this crate's binary
              crateBin = [{ name = "mjpeg2h264"; path = "examples/mjpeg2h264.rs"; }];
            });
        }).rootCrate.build;

        native = crateFor pkgs;
      in {
        packages = {
          # the library (its `lib` output) and the mjpeg2h264 harness (`bin`)
          default = native;
        } // pkgs.lib.optionalAttrs (system != "aarch64-linux") {
          # the RK3588 board's build, cross-compiled
          aarch64-linux = crateFor pkgs.pkgsCross.aarch64-multiplatform;
        };
        checks.tests = native.override { runTests = true; };

        devShells.default = pkgs.mkShell { packages = [ pkgs.cargo pkgs.rustc pkgs.clippy pkgs.rustfmt pkgs.python3 ]; };
      });
}
