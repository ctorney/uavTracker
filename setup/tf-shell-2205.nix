with import <nixos> {};
let python39 =
    let
    packageOverrides = self:
    super: {
      opencv4 = super.opencv4.override {
        enableGtk2 = true;
        gtk2 = pkgs.gtk2;
        # enableGtk3 = true
        # gtk3 = pkgs.gtk3;
        enableVtk = true;
        vtk = pkgs.vtk;
        enableFfmpeg = true;
        ffmpeg = pkgs.ffmpeg-full; #it is without _3 for 21.05
        # enableGtk3 = pkgs.gtk3;
        # doCheck = true;
        };
    };
    in
      pkgs.python39.override {inherit packageOverrides; self = python39;
    };
in

mkShell {
  #allowUnfree = true; #for some reason that needs to be added in > $ vim ~/.config/nixpkgs/config.nix
  name = "tensorflow-cuda-shell";
  buildInputs = with python3.pkgs; [
  cudatoolkit
  (python39.withPackages(ps: with ps; [
    opencv4
    numpy
    scikitlearn
    pyaml
    pandas
    tensorflowWithCuda
    tensorflow-tensorboard
    Keras
    tqdm
    filterpy
    seaborn
    folium
    pyproj
    ]))
  ];
  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.cudnn_8_3_2}/lib:${pkgs.cudaPackages.cudatoolkit.lib}/lib:$LD_LIBRARY_PATH
    alias pip="PIP_PREFIX='$(pwd)/_build/pip_packages' TMPDIR='$HOME' \pip"
    export PYTHONPATH="$(pwd)/_build/pip_packages/lib/python3.9/site-packages:$PYTHONPATH"
    export PATH="$(pwd)/_build/pip_packages/bin:$PATH"
    unset SOURCE_DATE_EPOCH
  '';
}
