func.func nested @op_wide(i16, i16) -> i16
func.func nested @op_emulated(vector<2xi8>, vector<2xi8>) -> vector<2xi8>

func.func @emulate_op(%a : i16, %b : i16, %emulate : i16) -> i16 {
  %bool_cond = arith.trunci %emulate: i16 to i1
  %res = scf.if %bool_cond -> (i16) {
    %aa = llvm.bitcast %a : i16 to vector<2xi8>
    %bb = llvm.bitcast %b : i16 to vector<2xi8>
    %r = func.call @op_emulated(%aa, %bb) : (vector<2xi8>, vector<2xi8>) -> vector<2xi8>
    %rr = llvm.bitcast %r : vector<2xi8> to i16
    scf.yield %rr : i16
  } else {
    %rr = func.call @op_wide(%a, %b) : (i16, i16) -> i16
    scf.yield %rr : i16
  }

  return %res : i16
}
