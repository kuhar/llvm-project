add_mlir_library(MLIRDialectUtils
  CombiningOpsUtils.cpp
  IndexingUtils.cpp
  ReshapeOpsUtils.cpp
  StructuredOpsUtils.cpp
  StaticValueUtils.cpp

  LINK_LIBS PUBLIC
  MLIRArithUtils
  MLIRIR
)
