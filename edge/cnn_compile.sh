#!/bin/bash
vai_c_xir -x quantize_result/MultiClassAttackCNN_int.xmodel -a arch.json -o c_out_dpu -n cap_cnnMulti_first
