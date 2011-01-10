/*
 *  Falcon DataMatrix - Service
 */

#define FALCON_EXPORT_SERVICE

#include "dmtx_srv.h"
#include "dmtx_mod.h"

#include <falcon/vm.h>

namespace Falcon
{

DataMatrixService::DataMatrixService()
    :
    Falcon::Service( DMTX_SERVICENAME )
{
}

DataMatrixService::~DataMatrixService()
{
}

Falcon::Dmtx::DataMatrix*
DataMatrixService::createCodec()
{
    VMachine* vm = VMachine::getCurrent();
    Item* wki = vm->findWKI( "DataMatrix" );
    return new Falcon::Dmtx::DataMatrix( wki->asClass() );
}

} // !namespace Falcon
