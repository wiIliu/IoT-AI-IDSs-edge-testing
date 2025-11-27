#!/bin/bash
vai_c_xir -x quantize_result/MultiCNN_int.xmodel -a arch.json -o multiAttn_out_dpu -n cap_multiCNN_first
