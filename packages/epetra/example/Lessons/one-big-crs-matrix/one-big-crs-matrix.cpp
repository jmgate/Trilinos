#include <mpi.h>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
#include <vector>
#include <Epetra_config.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Import.h>
#include <Epetra_Map.h>
#include <Epetra_MpiComm.h>
#include <Epetra_Vector.h>
#include <Epetra_Version.h>

///////////////////////////////////////////////////////////////////////////////
//
//  The goal here is take a blocked matrix
//
//        ⎡ 2 -1          │   ⎤
//        ⎢-1  2 -1       │   ⎥
//        ⎢   -1  2 -1    │1  ⎥
//    A = ⎢      -1  2 -1 │  1⎥
//        ⎢         -1  2 │   ⎥
//        ⎢───────────────┼───⎥
//        ⎢    1          │1  ⎥
//        ⎣          1    │  1⎦
//
//  and turn it into one big Epetra_CrsMatrix.  The 00 block is already an
//  Epetra_CrsMatrix.  The remaining three blocks are Epetra_MultiVectors.  The
//  Epetra_MultiVector corresponding to the 10 block is actually the transpose
//  of the 10 block shown above, that is
//
//          ⎡   ⎤
//          ⎢1  ⎥
//    A10 = ⎢   ⎥.
//          ⎢  1⎥
//          ⎣   ⎦
//
///////////////////////////////////////////////////////////////////////////////
int
main(
  int   argc,
  char* argv[])
{
  using std::cout;
  using std::endl;
  using std::size_t;
  using std::stringstream;
  using std::vector;
  const int delay(1);

  MPI_Init(&argc, &argv);
  Epetra_MpiComm comm(MPI_COMM_WORLD);

  const int myRank(comm.MyPID()), numProcs(comm.NumProc());

  if (myRank == 0)
    cout << Epetra_Version() << endl << endl
         << "Total number of processes: " << numProcs << endl;

  /////////////////////////////////////////////////////////////////////////////
  //
  //  The first step is to create the four blocks of the 2x2 blocked matrix A.
  //
  /////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////
  //
  //  Create the 00 Block
  //
  /////////////////////////////////////////////////////////////////////////////

  // Construct a Map that puts approximately the same number of
  // equations on each processor.
  const int n(5), m(2), indexBase(0);
  Epetra_Map nMap(n, indexBase, comm);

  // Get the list of global indices that this process owns.
  const int numMyElements(nMap.NumMyElements());
  int* myGlobalElements(nMap.MyGlobalElements());

  // Create a Epetra sparse matrix whose rows have distribution given
  // by the Map.  The max number of entries per row is 3.
  Epetra_CrsMatrix A00(Copy, nMap, 3);

  // Local error code for use below.
  int lclerr(0);

  // Fill the sparse matrix, one row at a time.  InsertGlobalValues
  // adds entries to the sparse matrix, using global column indices.
  // It changes both the graph structure and the values.
  double tempVals[3];
  int tempGblInds[3];
  for (int i(0); i < numMyElements; ++i)
  {
    if (myGlobalElements[i] == 0)
    {
      // A00(0, 0:1) = [2, -1]
      tempVals[0] = 2;
      tempVals[1] = -1;
      tempGblInds[0] = myGlobalElements[i];
      tempGblInds[1] = myGlobalElements[i] + 1;
      if (lclerr == 0)
        lclerr = A00.InsertGlobalValues(myGlobalElements[i], 2, tempVals,
          tempGblInds);
      if (lclerr != 0)
        break;
    }
    else if (myGlobalElements[i] == n - 1)
    {
      // A00(N-1, N-2:N-1) = [-1, 2]
      tempVals[0] = -1;
      tempVals[1] = 2;
      tempGblInds[0] = myGlobalElements[i] - 1;
      tempGblInds[1] = myGlobalElements[i];
      if (lclerr == 0)
        lclerr = A00.InsertGlobalValues(myGlobalElements[i], 2, tempVals,
          tempGblInds);
      if (lclerr != 0)
        break;
    }
    else // if we're looking at any of the rows in the middle
    {
      // A00(i, i-1:i+1) = [-1, 2, -1]
      tempVals[0] = -1;
      tempVals[1] = 2;
      tempVals[2] = -1;
      tempGblInds[0] = myGlobalElements[i] - 1;
      tempGblInds[1] = myGlobalElements[i];
      tempGblInds[2] = myGlobalElements[i] + 1;
      if (lclerr == 0)
        lclerr = A00.InsertGlobalValues(myGlobalElements[i], 3, tempVals,
          tempGblInds);
      if (lclerr != 0)
        break;
    } // end if this is the first, last, or other row
  } // end loop over the rows owned by this process

  // If any process failed to insert at least one entry, throw.
  int gblerr(0);
  comm.MaxAll(&lclerr, &gblerr, 1);
  if (gblerr != 0)
    throw std::runtime_error("Some process failed to insert an entry.");

  // Tell the sparse matrix that we are done adding entries to it.
  gblerr = A00.FillComplete();
  if (gblerr != 0)
  {
    std::ostringstream os;
    os << "A00.FillComplete() failed with error code " << gblerr << ".";
    throw std::runtime_error(os.str());
  }

//  if (myRank == 0)
//    cout << endl << "-----[ A00 ]---------------------------------------------"
//         << endl << endl;
//  for (int i(0); i < numProcs; ++i)
//  {
//    if (myRank == i)
//      A00.Print(cout);
//  }
//  sleep(delay);

  /////////////////////////////////////////////////////////////////////////////
  //
  //  Create the 01 Block
  //
  /////////////////////////////////////////////////////////////////////////////
  
  Epetra_MultiVector A01(nMap, m);
  A01.ReplaceGlobalValue(2, 0, 1);
  A01.ReplaceGlobalValue(3, 1, 1);
//  if (myRank == 0)
//    cout << endl << "-----[ A01 ]---------------------------------------------"
//         << endl << endl;
//  for (int i(0); i < numProcs; ++i)
//  {
//    if (myRank == i)
//      A01.Print(cout);
//  }
//  sleep(delay);

  /////////////////////////////////////////////////////////////////////////////
  //
  //  Create the 10 Block
  //
  /////////////////////////////////////////////////////////////////////////////
 
  // Note that this is actually the transpose of the 10 block.
  Epetra_MultiVector A10(nMap, m);
  A10.ReplaceGlobalValue(1, 0, 1);
  A10.ReplaceGlobalValue(3, 1, 1);
//  if (myRank == 0)
//    cout << endl << "-----[ A10 ]---------------------------------------------"
//         << endl << endl;
//  for (int i(0); i < numProcs; ++i)
//  {
//    if (myRank == i)
//      A10.Print(cout);
//  }
//  sleep(delay);

  /////////////////////////////////////////////////////////////////////////////
  //
  //  Create the 11 Block
  //
  /////////////////////////////////////////////////////////////////////////////

  Epetra_Map mMap(m, indexBase, comm);
  Epetra_MultiVector A11(mMap, m);
  A11.ReplaceGlobalValue(0, 0, 1);
  A11.ReplaceGlobalValue(1, 1, 1);
//  if (myRank == 0)
//    cout << endl << "-----[ A11 ]---------------------------------------------"
//         << endl << endl;
//  for (int i(0); i < numProcs; ++i)
//  {
//    if (myRank == i)
//      A11.Print(cout);
//  }
//  sleep(delay);

  /////////////////////////////////////////////////////////////////////////////
  //
  //  Now we can start combining them into a bigger Epetra_CrsMatrix.  Here we
  //  must assume that we know nothing of the structure of the blocks or their
  //  contents in order to have a method that should work for the general case.
  //
  /////////////////////////////////////////////////////////////////////////////

  // First get the dimensions of the blocked system.
  const int newN(A00.NumGlobalRows()), newM(A01.NumVectors());

  // Then get the row and column maps from the 00 block.
  Epetra_Map rowMap(A00.RowMap());
//  Epetra_Map colMap(A00.ColMap());
//  if (myRank == 0)
//    cout << endl << "-----[ rowMap ]------------------------------------------"
//         << endl << endl;
//  for (int i(0); i < numProcs; ++i)
//  {
//    if (myRank == i)
//      rowMap.Print(cout);
//  }
//  sleep(delay);
//  if (myRank == 0)
//    cout << endl << "-----[ colMap ]------------------------------------------"
//         << endl << endl;
//  for (int i(0); i < numProcs; ++i)
//  {
//    if (myRank == i)
//      colMap.Print(cout);
//  }
//  sleep(delay);

  // Get a vector of the global indices each processor owns in the rowMap.
  const int numMyRowMapElements(rowMap.NumMyElements());
  const int* rowMapGlobalElementsPtr(rowMap.MyGlobalElements());
  vector<int> myGlobalRowMapElements(rowMapGlobalElementsPtr,
    rowMapGlobalElementsPtr + numMyRowMapElements);
//  stringstream ss;
//  ss << "p" << myRank << ":  myGlobalRowMapElements = {";
//  for (size_t i(0); i < myGlobalRowMapElements.size(); ++i)
//  {
//    ss << myGlobalRowMapElements[i];
//    if (i < myGlobalRowMapElements.size() - 1)
//      ss << ", ";
//  }
//  ss << "} (size = " << myGlobalRowMapElements.size() << ")" << endl;
//  if (myRank == 0)
//    cout << endl << "-----[ rowMap ]------------------------------------------"
//         << endl << endl;
//  for (int i(0); i < numProcs; ++i)
//  {
//    if (myRank == i)
//      cout << ss.str();
//  }
//  sleep(delay);

  /*
  // Get a vector of the global indices each processor owns in the colMap.
  const int numMyColMapElements(colMap.NumMyElements());
  const int* colMapGlobalElementsPtr(colMap.MyGlobalElements());
  vector<int> myGlobalColMapElements(colMapGlobalElementsPtr,
    colMapGlobalElementsPtr + numMyColMapElements);
  ss = stringstream("");
  ss << "p" << myRank << ":  myGlobalColMapElements = {";
  for (size_t i(0); i < myGlobalColMapElements.size(); ++i)
  {
    ss << myGlobalColMapElements[i];
    if (i < myGlobalColMapElements.size() - 1)
      ss << ", ";
  }
  ss << "} (size = " << myGlobalColMapElements.size() << ")" << endl;
  if (myRank == 0)
    cout << endl << "-----[ colMap ]------------------------------------------"
         << endl << endl;
  for (int i(0); i < numProcs; ++i)
  {
    if (myRank == i)
      cout << ss.str();
  }
  sleep(delay);
  */

  // Append the extra rows and columns to the list on processor 0.
  if (myRank == 0)
  {
    for (int i(0); i < newM; ++i)
    {
      myGlobalRowMapElements.push_back(newN + i);
//      myGlobalColMapElements.push_back(newN + i);
    }
  }
//  ss = stringstream("");
//  ss << "p" << myRank << ":  myGlobalRowMapElements = {";
//  for (size_t i(0); i < myGlobalRowMapElements.size(); ++i)
//  {
//    ss << myGlobalRowMapElements[i];
//    if (i < myGlobalRowMapElements.size() - 1)
//      ss << ", ";
//  }
//  ss << "} (size = " << myGlobalRowMapElements.size() << ")" << endl;
//  if (myRank == 0)
//    cout << endl << "-----[ rowMap ]------------------------------------------"
//         << endl << endl;
//  for (int i(0); i < numProcs; ++i)
//  {
//    if (myRank == i)
//      cout << ss.str();
//  }
//  sleep(delay);
//  ss = stringstream("");
//  ss << "p" << myRank << ":  myGlobalColMapElements = {";
//  for (size_t i(0); i < myGlobalColMapElements.size(); ++i)
//  {
//    ss << myGlobalColMapElements[i];
//    if (i < myGlobalColMapElements.size() - 1)
//      ss << ", ";
//  }
//  ss << "} (size = " << myGlobalColMapElements.size() << ")" << endl;
//  if (myRank == 0)
//    cout << endl << "-----[ colMap ]------------------------------------------"
//         << endl << endl;
//  for (int i(0); i < numProcs; ++i)
//  {
//    if (myRank == i)
//      cout << ss.str();
//  }
//  sleep(delay);

  // Create the new row and column maps.
  Epetra_Map newRowMap(-1, myGlobalRowMapElements.size(),
    myGlobalRowMapElements.data(), indexBase, comm);
//  Epetra_Map newColMap(-1, myGlobalColMapElements.size(),
//    myGlobalColMapElements.data(), indexBase, comm);
//  if (myRank == 0)
//    cout << endl << "-----[ newRowMap ]---------------------------------------"
//         << endl << endl;
//  for (int i(0); i < numProcs; ++i)
//  {
//    if (myRank == i)
//      newRowMap.Print(cout);
//  }
//  sleep(delay);
//  if (myRank == 0)
//    cout << endl << "-----[ newColMap ]---------------------------------------"
//         << endl << endl;
//  for (int i(0); i < numProcs; ++i)
//  {
//    if (myRank == i)
//      newColMap.Print(cout);
//  }
//  sleep(delay);

  /*
  // Need to figure out the number of entries per row from A00.
  Epetra_CrsGraph graph(A00.Graph());
  const int numLocalRows(rowMap.NumMyElements());
  vector<int> numEntriesPerRow(numLocalRows);
  for (int i(0); i < numLocalRows; ++i)
    numEntriesPerRow[i] = graph.NumMyIndices(i);
  ss = stringstream("");
  ss << "p" << myRank << ":  numEntriesPerRow = {";
  for (size_t i(0); i < numEntriesPerRow.size(); ++i)
  {
    ss << numEntriesPerRow[i];
    if (i < numEntriesPerRow.size() - 1)
      ss << ", ";
  }
  ss << "} (size = " << numEntriesPerRow.size() << ")" << endl;
  if (myRank == 0)
    cout << endl << "-----[ numEntriesPerRow ]--------------------------------"
         << endl << endl;
  for (int i(0); i < numProcs; ++i)
  {
    if (myRank == i)
      cout << ss.str();
  }
  sleep(delay);
  */

  /*
  // Add the number of entries per row from A01.
  for (int i(0); i < A01.NumVectors(); ++i)
    for (int j(0); j < numLocalRows; ++j)
      if (A01[i][j] != 0)
        ++numEntriesPerRow[j];
  ss = stringstream("");
  ss << "p" << myRank << ":  numEntriesPerRow = {";
  for (size_t i(0); i < numEntriesPerRow.size(); ++i)
  {
    ss << numEntriesPerRow[i];
    if (i < numEntriesPerRow.size() - 1)
      ss << ", ";
  }
  ss << "} (size = " << numEntriesPerRow.size() << ")" << endl;
  if (myRank == 0)
    cout << endl << "-----[ numEntriesPerRow ]--------------------------------"
         << endl << endl;
  for (int i(0); i < numProcs; ++i)
  {
    if (myRank == i)
      cout << ss.str();
  }
  sleep(delay);
  */

  // Add the number of entries per row from A10.
  // Loop over the vectors in the multivector; count the number in each one.
  // Add one for the 11 block.

  /*
  // Add the number of entries per row from A11.
  Epetra_BlockMap mapA11(A11.Map());
  const int numMyA11Rows(mapA11.NumMyElements());
  cout << "numMyA11Rows = " << numMyA11Rows << endl;
  for (int i(0); i < A11.NumVectors(); ++i)
  {
    cout << "i = " << i << endl;
    numEntriesPerRow.push_back(0);
    for (int j(0); j < numMyA11Rows; ++j)
    {
      cout << "j = " << j << endl;
      cout << "A11[i][j] = " << A11[i][j] << endl;
      cout << "numLocalRows + j = " << numLocalRows + j << endl;
      if (A11[i][j] != 0)
        ++numEntriesPerRow[numLocalRows + j];
    }
  }
  ss = stringstream("");
  ss << "p" << myRank << ":  numEntriesPerRow = {";
  for (size_t i(0); i < numEntriesPerRow.size(); ++i)
  {
    ss << numEntriesPerRow[i];
    if (i < numEntriesPerRow.size() - 1)
      ss << ", ";
  }
  ss << "} (size = " << numEntriesPerRow.size() << ")" << endl;
  if (myRank == 0)
    cout << endl << "-----[ numEntriesPerRow ]--------------------------------"
         << endl << endl;
  for (int i(0); i < numProcs; ++i)
  {
    if (myRank == i)
      cout << ss.str();
  }
  sleep(delay);
  */

  Epetra_Import A01Importer(A00.RowMap(), A01.Map());
  Epetra_MultiVector ghostedA01(A00.RowMap(), newM);
  ghostedA01.Import(A01, A01Importer, Add);
//  if (myRank == 0)
//    cout << endl << "-----[ ghostedA01 ]--------------------------------------"
//         << endl << endl;
//  ghostedA01.Print(cout);
//  sleep(delay);
  // doing this such that the entries in A01 in rows that p1 ownes in the final matrix now exist on p1
  //
  Epetra_Map p0ownesA10(-1, (myRank == 0) ? A10.GlobalLength() : 0, 0, comm);
  Epetra_MultiVector ghostedA10(p0ownesA10, A10.NumVectors());
  Epetra_Import A10Importer(p0ownesA10, A10.Map());
  ghostedA10.Import(A10, A10Importer, Add);
//  if (myRank == 0)
//    cout << endl << "-----[ ghostedA10 ]--------------------------------------"
//         << endl << endl;
//  ghostedA10.Print(cout);
//  sleep(delay);

  Epetra_Map p0ownesA11(-1, (myRank == 0) ? A11.GlobalLength() : 0, 0, comm);
  Epetra_MultiVector ghostedA11(p0ownesA11, A11.NumVectors());
  Epetra_Import A11Importer(p0ownesA11, A11.Map());
  ghostedA11.Import(A11, A11Importer, Add);
//  if (myRank == 0)
//    cout << endl << "-----[ ghostedA11 ]--------------------------------------"
//         << endl << endl;
//  ghostedA11.Print(cout);
//  sleep(delay);

//  Epetra_CrsMatrix A(Copy, newRowMap, newColMap, 0); // change this to use numEntriesPerRow, true later.
  Epetra_CrsMatrix A(Copy, newRowMap, 0); // change this to use numEntriesPerRow, true later.
//  if (myRank == 0)
//    cout << endl << "-----[ A ]-----------------------------------------------"
//         << endl << endl;
//  A.Print(cout);
//  sleep(delay);

  vector<double> values(A00.MaxNumEntries());
  vector<int> colGIDs(A00.MaxNumEntries());
  for (int i(0); i < A00.NumMyRows(); ++i)
  {
    int numEntries, rowGID(A00.RowMap().GID(i));
    A00.ExtractGlobalRowCopy(rowGID, A00.MaxNumEntries(), numEntries,
      values.data(), colGIDs.data());
    A.InsertGlobalValues(rowGID, numEntries, values.data(), colGIDs.data());
  }
//  if (myRank == 0)
//    cout << endl << "-----[ after A00 ]---------------------------------------"
//         << endl << endl;
//  A.Print(cout);
//  sleep(delay);

  for (int i(0); i < ghostedA01.MyLength(); ++i)
  {
    int rowGID(ghostedA01.Map().GID(i));
    for (int j(0); j < ghostedA01.NumVectors(); ++j)
    {
      if (ghostedA01[j][i] != 0)
      {
        int colGID(newN + j);
        A.InsertGlobalValues(rowGID, 1, &ghostedA01[j][i], &colGID);
//        cout << "Insert " << rowGID << "; p" << myRank << "; colGID " << colGID << endl;
      }
    }
  }
//  if (myRank == 0)
//    cout << endl << "-----[ after A01 ]---------------------------------------"
//         << endl << endl;
//  A.Print(cout);
//  sleep(delay);

  for (int i(0); i < ghostedA10.MyLength(); ++i)
  {
    int colGID(ghostedA10.Map().GID(i));
    for (int j(0); j < ghostedA10.NumVectors(); ++j)
    {
      int rowGID(newN + j);
      if (ghostedA10[j][i] != 0)
      {
        A.InsertGlobalValues(rowGID, 1, &ghostedA10[j][i], &colGID);
//        cout << "Insert " << rowGID << "; p" << myRank << "; colGID " << colGID << endl;
      }
    }
  }
//  if (myRank == 0)
//    cout << endl << "-----[ after A10 ]---------------------------------------"
//         << endl << endl;
//  A.Print(cout);
//  sleep(delay);


  for (int i(0); i < ghostedA11.MyLength(); ++i)
  {
    // Left out second loop because our use case is diagonal.  Need it in general though.
    int rowGID(newN + i);
    A.InsertGlobalValues(rowGID, 1, &ghostedA11[i][i], &rowGID);
//    cout << "Insert " << rowGID << "; p" << myRank << endl;
  }
  A.FillComplete();
  A.Print(cout);
//  sleep(delay);

  /////////////////////////////////////////////////////////////////////////////
  //
  //  Finish Up
  //
  /////////////////////////////////////////////////////////////////////////////

  MPI_Finalize();
  return 0;
} // end of main()
