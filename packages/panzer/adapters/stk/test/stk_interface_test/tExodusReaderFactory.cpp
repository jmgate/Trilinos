#include <Teuchos_ConfigDefs.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include "Teuchos_DefaultComm.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Panzer_STK_Version.hpp"
#include "Panzer_STK_config.hpp"
#include "Panzer_STK_Interface.hpp"
#include "Panzer_STK_ExodusReaderFactory.hpp"

#include "Shards_BasicTopologies.hpp"

#ifdef HAVE_MPI
   #include "Epetra_MpiComm.h"
#else
   #include "Epetra_SerialComm.h"
#endif

#ifdef HAVE_IOSS

namespace panzer_stk {

TEUCHOS_UNIT_TEST(tExodusReaderFactory, basic_test)
{
   int numprocs = stk::parallel_machine_size(MPI_COMM_WORLD);
   int rank = stk::parallel_machine_rank(MPI_COMM_WORLD);
   out << "Running numprocs = " << numprocs << " rank = " << rank << std::endl;

   // read from file and build mesh
   STK_ExodusReaderFactory fact("meshes/basic.gen");
   Teuchos::RCP<STK_Interface> mesh = fact.buildMesh(MPI_COMM_WORLD);

   TEST_ASSERT(mesh!=Teuchos::null);
   TEST_ASSERT(mesh->getDimension()==2);
   TEST_ASSERT(mesh->isWritable());
   // TEST_ASSERT(not mesh->isModifiable());

   std::vector<std::string> eBlocks;
   mesh->getElementBlockNames(eBlocks);
   TEST_EQUALITY((int) eBlocks.size(),2);
   out << "E-Blocks: ";
   for(std::size_t i=0;i<eBlocks.size();i++)  
      out << "\"" << eBlocks[i] << "\" ";
   out << std::endl;

   std::vector<std::string> sidesets;
   mesh->getSidesetNames(sidesets);
   TEST_EQUALITY((int) sidesets.size(),7);
   out << "Sides: ";
   for(std::size_t i=0;i<sidesets.size();i++)  
      out << "\"" << sidesets[i] << "\" ";
   out << std::endl;
}

}

#endif
