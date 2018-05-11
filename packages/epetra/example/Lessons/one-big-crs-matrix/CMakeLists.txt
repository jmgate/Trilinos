
TRIBITS_ADD_EXECUTABLE_AND_TEST(
  one-big-crs-matrix
  SOURCES one-big-crs-matrix.cpp
  ARGS 
  COMM serial mpi
  STANDARD_PASS_OUTPUT
)
