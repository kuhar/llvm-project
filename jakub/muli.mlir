func.func @op_wide(%a : i16, %b : i16) -> i16 {
  %res = arith.muli %a, %b : i16
  return %res : i16
}
