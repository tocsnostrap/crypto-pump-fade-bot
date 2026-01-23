{ pkgs }: {
  deps = [
    pkgs.nodejs_22
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.python311Packages.numpy
    pkgs.python311Packages.pandas
    pkgs.python311Packages.ccxt
    pkgs.pkg-config
    pkgs.ta-lib
  ];
}
