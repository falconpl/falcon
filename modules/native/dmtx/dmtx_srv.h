/*
 *  Falcon DataMatrix - Service
 */

#ifndef DMTX_SRV_H
#define DMTX_SRV_H

#include <falcon/service.h>

#define DMTX_SERVICENAME     "dmtx"

namespace Falcon
{

namespace Dmtx
{
class DataMatrix;
} // !namespace Dmtx

class FALCON_SERVICE DataMatrixService
    :
    public Falcon::Service
{
public:

    DataMatrixService();
    virtual ~DataMatrixService();

    virtual Falcon::Dmtx::DataMatrix* createCodec();

};

} // !namespace Falcon

#endif // !DMTX_SRV_H
