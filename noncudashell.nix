
# add old nixos channel: sudo nix-channel --add https://nixos.org/channels/nixos-20.09 nixos-old
#sudo nix-channel --update
#using old because new version of hp5 library is shit.
with (import <nixos-old> {});

let python37 =
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
        ffmpeg_3 = pkgs.ffmpeg-full; #it is without _3 for 21.05
        # enableGtk3 = pkgs.gtk3;
        # doCheck = true;
        };
    };
    in
      pkgs.python37.override {inherit packageOverrides; self = python37;
    };
in
mkShell {
  buildInputs = [
  (python37.withPackages(ps: with ps; [
    opencv4
    numpy
    scikitlearn
    pyaml
    pandas
    tensorflow
    Keras
    filterpy
    #only the above are for uavtracker
    notebook
    jupyterlab
    matplotlib
    scipy
    scikitimage
    flask
    flask-appbuilder
    email_validator
    bokeh
    seaborn
    shapely
    folium
    statsmodels
    yapf
    xlrd
    pyproj #map projections
    python-dotenv
    psycopg2
  ]))
  ];
}
  