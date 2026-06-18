{pkgs}: {
  deps = [
    pkgs.xorg.libICE
    pkgs.xorg.libSM
    pkgs.xorg.libXext
    pkgs.xorg.libX11
    pkgs.xorg.libxcb
  ];
}
