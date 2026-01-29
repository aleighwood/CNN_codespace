package mobilenet_v1_pkg;
  parameter int DATA_W = 8;
  parameter int ACC_W = 32;

  parameter int IN_H = 224;
  parameter int IN_W = 224;
  parameter int IN_C = 3;

  parameter int TILE_H = 16;
  parameter int TILE_W = 16;

  typedef logic signed [DATA_W-1:0] q8_t;
  typedef logic signed [ACC_W-1:0] q32_t;
endpackage
