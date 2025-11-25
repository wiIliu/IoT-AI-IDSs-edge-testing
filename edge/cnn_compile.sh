#!/bin/bash
vai_c_xir -x quantize_result/MultiAttnCNN_int.xmodel -a arch.json -o multiAttn_out_dpu -n cap_multiAttnCNN_first
