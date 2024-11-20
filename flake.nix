{
  description = "OpenCV project using Nix Flakes";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    pkgs = import nixpkgs {
        # config.cudaVersion = "12";
        system = "x86_64-linux"; # Adjust this for your system if needed
      config.allowUnfree = true;
        # config.cudaSupport = true;
    };
    opencvGtk-py = pkgs.python311Packages.opencv4.override (old: {enableGtk3 = true;});
  in {
    devShells.x86_64-linux.default = pkgs.mkShell {

        #export CUDA_PATH=${pkgs.cudatoolkit}
        #export LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH:${pkgs.linuxPackages.nvidia_x11}/lib
        # export CUDA_PATH=${pkgs.cudatoolkit}
        # export LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH:${pkgs.linuxPackages.nvidia_x11}/lib
      shellHook = ''
        export LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH
        export QT_QPA_PLATFORM_PLUGIN_PATH="${pkgs.libsForQt5.qt5.qtbase.bin}/lib/qt-${pkgs.libsForQt5.qt5.qtbase.version}/plugins"; 
        source ./.venv/bin/activate
      '';
      buildInputs = with pkgs; [
          # cudaPackages.cudatoolkit 
          # cudaPackages.cudnn
        (pkgs.writeShellScriptBin "python" ''
                export LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH
          exec ${pkgs.python3}/bin/python "$@"
        '')
        python311
        python311Packages.pip
        python311Packages.pyqt5
        # python311Packages.torch-bin
        # python311Packages.torchvision-bin
        # python311Packages.transformers
        # python3Packages.virtualenv
        # python311Packages.opencv4
        python311Packages.numpy
        # python311Packages.numba
        # python311Packages.tqdm
        # python311Packages.scikitlearn
        python311Packages.matplotlib
        opencvGtk-py
          # python312Packages.conda
        ffmpeg
                 # nvtop
      ];
    };
  };
}
