#include <mpi.h>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
#include <vector>
#include <Epetra_config.h>
#include <Epetra_CrsMatrix.h>
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

  // Get the list of global indices that this process owns.  In this
  // example, this is unnecessary, because we know that we created a
  // contiguous Map (see above).  (Thus, we really only need the min
  // and max global index on this process.)  However, in general, we
  // don't know what global indices the Map owns, so if we plan to add
  // entries into the sparse matrix using global indices, we have to
  // get the list of global indices this process owns.
  const int numMyElements(nMap.NumMyElements());
  int* myGlobalElements(NULL);
  myGlobalElements = nMap.MyGlobalElements();

  // In general, tests like this really should synchronize across all
  // processes.  However, the likely cause for this case is a
  // misconfiguration of Epetra, so we expect it to happen on all
  // processes, if it happens at all.
  if (numMyElements > 0 && myGlobalElements == NULL)
    throw std::logic_error("Failed to get the list of global indices");

  if (myRank == 0)
    cout << endl << "Creating the sparse matrix" << endl;

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

  if (myRank == 0)
    cout << endl << "-----[ A00 ]---------------------------------------------"
         << endl << endl;
  for (int i(0); i < numProcs; ++i)
  {
    if (myRank == i)
      A00.Print(cout);
  }
  sleep(delay);

  /////////////////////////////////////////////////////////////////////////////
  //
  //  Create the 01 Block
  //
  /////////////////////////////////////////////////////////////////////////////
  
  Epetra_MultiVector A01(nMap, m);
  A01.ReplaceGlobalValue(2, 0, 1);
  A01.ReplaceGlobalValue(3, 1, 1);
  if (myRank == 0)
    cout << endl << "-----[ A01 ]---------------------------------------------"
         << endl << endl;
  for (int i(0); i < numProcs; ++i)
  {
    if (myRank == i)
      A01.Print(cout);
  }
  sleep(delay);

  /////////////////////////////////////////////////////////////////////////////
  //
  //  Create the 10 Block
  //
  /////////////////////////////////////////////////////////////////////////////
 
  // Note that this is actually the transpose of the 10 block.
  Epetra_MultiVector A10(nMap, m);
  A10.ReplaceGlobalValue(1, 0, 1);
  A10.ReplaceGlobalValue(3, 1, 1);
  if (myRank == 0)
    cout << endl << "-----[ A10 ]---------------------------------------------"
         << endl << endl;
  for (int i(0); i < numProcs; ++i)
  {
    if (myRank == i)
      A10.Print(cout);
  }
  sleep(delay);

  /////////////////////////////////////////////////////////////////////////////
  //
  //  Create the 11 Block
  //
  /////////////////////////////////////////////////////////////////////////////

  Epetra_Map mMap(m, indexBase, comm);
  Epetra_MultiVector A11(mMap, m);
  A11.ReplaceGlobalValue(0, 0, 1);
  A11.ReplaceGlobalValue(1, 1, 1);
  if (myRank == 0)
    cout << endl << "-----[ A11 ]---------------------------------------------"
         << endl << endl;
  for (int i(0); i < numProcs; ++i)
  {
    if (myRank == i)
      A11.Print(cout);
  }
  sleep(delay);

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
  Epetra_Map rowMap(A00.RowMap()), colMap(A00.ColMap());
  if (myRank == 0)
    cout << endl << "-----[ rowMap ]------------------------------------------"
         << endl << endl;
  for (int i(0); i < numProcs; ++i)
  {
    if (myRank == i)
      rowMap.Print(cout);
  }
  sleep(delay);
  if (myRank == 0)
    cout << endl << "-----[ colMap ]------------------------------------------"
         << endl << endl;
  for (int i(0); i < numProcs; ++i)
  {
    if (myRank == i)
      colMap.Print(cout);
  }
  sleep(delay);

  // Get a vector of the global indices each processor owns in the rowMap.
  const int numMyRowMapElements(rowMap.NumMyElements());
  const int* rowMapGlobalElementsPtr(rowMap.MyGlobalElements());
  vector<int> myGlobalRowMapElements(rowMapGlobalElementsPtr,
    rowMapGlobalElementsPtr + numMyRowMapElements);
  stringstream ss;
  ss << "p" << myRank << ":  myGlobalRowMapElements = {";
  for (size_t i(0); i < myGlobalRowMapElements.size(); ++i)
  {
    ss << myGlobalRowMapElements[i];
    if (i < myGlobalRowMapElements.size() - 1)
      ss << ", ";
  }
  ss << "} (size = " << myGlobalRowMapElements.size() << ")" << endl;
  if (myRank == 0)
    cout << endl << "-----[ rowMap ]------------------------------------------"
         << endl << endl;
  for (int i(0); i < numProcs; ++i)
  {
    if (myRank == i)
      cout << ss.str();
  }
  sleep(delay);

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

  // Append the extra rows and columns to the list on processor 0.
  if (myRank == 0)
  {
    for (int i(0); i < newM; ++i)
    {
      myGlobalRowMapElements.push_back(newN + i);
      myGlobalColMapElements.push_back(newN + i);
    }
  }
  ss = stringstream("");
  ss << "p" << myRank << ":  myGlobalRowMapElements = {";
  for (size_t i(0); i < myGlobalRowMapElements.size(); ++i)
  {
    ss << myGlobalRowMapElements[i];
    if (i < myGlobalRowMapElements.size() - 1)
      ss << ", ";
  }
  ss << "} (size = " << myGlobalRowMapElements.size() << ")" << endl;
  if (myRank == 0)
    cout << endl << "-----[ rowMap ]------------------------------------------"
         << endl << endl;
  for (int i(0); i < numProcs; ++i)
  {
    if (myRank == i)
      cout << ss.str();
  }
  sleep(delay);
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

  // Create the new row and column maps.
  Epetra_Map newRowMap(-1, myGlobalRowMapElements.size(),
    myGlobalRowMapElements.data(), indexBase, comm);
  Epetra_Map newColMap(-1, myGlobalColMapElements.size(),
    myGlobalColMapElements.data(), indexBase, comm);
  if (myRank == 0)
    cout << endl << "-----[ newRowMap ]---------------------------------------"
         << endl << endl;
  for (int i(0); i < numProcs; ++i)
  {
    if (myRank == i)
      newRowMap.Print(cout);
  }
  sleep(delay);
  if (myRank == 0)
    cout << endl << "-----[ newColMap ]---------------------------------------"
         << endl << endl;
  for (int i(0); i < numProcs; ++i)
  {
    if (myRank == i)
      newColMap.Print(cout);
  }
  sleep(delay);

  // Need to figure out const int* numEntriesPerRow.
  // Loop over the local rows of A00
  // ExtractGlobalRowCopy(int GlobalRow, int Length, int& NumEntries, double* Values, int* Indices)

  // Epetra_CrsMatrix A(Copy, newRowMap, newColMap, const int* numEntriesPerRow, true);

  /////////////////////////////////////////////////////////////////////////////
  //
  //  Finish Up
  //
  /////////////////////////////////////////////////////////////////////////////

  MPI_Finalize();
  return 0;
} // end of main()
