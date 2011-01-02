/*
 *  Falcon MongoDB - Extension
 */

#ifndef DMTX_EXT_H
#define DMTX_EXT_H

#include <falcon/setup.h>

#include <falcon/error_base.h>
#include <falcon/error.h>

#ifndef FALCON_DATAMATRIX_ERROR_BASE
#define FALCON_DATAMATRIX_ERROR_BASE   16100
#endif

// #define MONGODB_ERR_NOMEM           (FALCON_MONGODB_ERROR_BASE + 0)
// #define MONGODB_ERR_CREATE_CONN     (FALCON_MONGODB_ERROR_BASE + 1)
// #define MONGODB_ERR_CONNECT         (FALCON_MONGODB_ERROR_BASE + 2)
// #define MONGODB_ERR_CREATE_BSON     (FALCON_MONGODB_ERROR_BASE + 3)

namespace Falcon
{

class VMachine;

namespace Ext
{

class DataMatrixError
    :
    public ::Falcon::Error
{
public:

    DataMatrixError()
        :
        Error( "DataMatrixError" )
    {}

    DataMatrixError( const ErrorParam &params )
        :
        Error( "DataMatrixError", params )
    {}

};

FALCON_FUNC DataMatrixError_init( VMachine* vm );

FALCON_FUNC DataMatrix_init( VMachine* vm );
FALCON_FUNC DataMatrix_encode( VMachine* vm );
FALCON_FUNC DataMatrix_decode( VMachine* vm );

} // !namespace Ext
} // !namespace Falcon

#endif // !DMTX_EXT_H
