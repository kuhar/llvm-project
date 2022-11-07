// RUN: mlir-opt --arith-emulate-wide-int="widest-int-supported=32" %s | FileCheck %s

// CHECK-LABEL: func.func @shrsi_scalar
func.func @shrsi_scalar(%a : i64, %b : i64) -> i64 {
    %c = arith.shrsi %a, %b : i64
    return %c : i64
}

// CHECK-LABEL: func.func @shrsi_vector
// CHECK-SAME:    ({{%.+}}: vector<3x2xi32>, {{%.+}}: vector<3x2xi32>) -> vector<3x2xi32>
// CHECK:         {{%.+}} = arith.shli {{%.+}}, {{%.+}} : vector<3x1xi32>
// CHECK:         {{%.+}} = arith.shrui {{%.+}}, {{%.+}} : vector<3x1xi32>
// CHECK:         {{%.+}} = arith.shli {{%.+}}, {{%.+}} : vector<3x1xi32>
// CHECK:         {{%.+}} = arith.shli {{%.+}}, {{%.+}} : vector<3x1xi32>
// CHECK:         {{%.+}} = arith.shrui {{%.+}}, {{%.+}} : vector<3x1xi32>
// CHECK:         {{%.+}} = arith.shrui {{%.+}}, {{%.+}} : vector<3x1xi32>
// CHECK:         {{%.+}} = arith.shli {{%.+}}, {{%.+}} : vector<3x1xi32>
// CHECK:         {{%.+}} = arith.shrui {{%.+}}, {{%.+}} : vector<3x1xi32>
// CHECK:       return {{%.+}} : vector<3x2xi32>
func.func @shrsi_vector(%a : vector<3xi64>, %b : vector<3xi64>) -> vector<3xi64> {
    %m = arith.shrsi %a, %b : vector<3xi64>
    return %m : vector<3xi64>
}