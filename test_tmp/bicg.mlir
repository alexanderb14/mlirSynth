func.func @fn_0_raised(%arg2: tensor<5x3xf64>, %arg3: tensor<3xf64>, %arg4: tensor<5xf64>, %arg5: tensor<3xf64>, %arg6: tensor<5xf64>) -> (tensor<3xf64>, tensor<5xf64>) attributes {changed_sizes = "2100:5,1900:3", irsynth.raised} {
    %0 = "mhlo.dot" (%arg6, %arg2) : (tensor<5xf64>, tensor<5x3xf64>) -> tensor<3xf64>
    %1 = "mhlo.dot" (%arg2, %arg5) : (tensor<5x3xf64>, tensor<3xf64>) -> tensor<5xf64>
    return %0, %1 : tensor<3xf64>, tensor<5xf64>
}
