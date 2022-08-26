func.func @op_wide(%a : i16, %b : i16) -> i16 {
  %x = arith.trunci %a : i16 to i8
  %y = arith.extui %x : i8 to i16
  return %y : i16
}
