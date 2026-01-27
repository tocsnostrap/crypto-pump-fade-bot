{ pkgs }: {
  deps = [
    pkgs.bashInteractive
    pkgs.nodejs_20
    pkgs.python311
    pkgs.python311Packages.numpy
    pkgs.python311Packages.pandas
  ];
}