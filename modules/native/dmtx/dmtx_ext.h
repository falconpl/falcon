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

//FALCON_FUNC DataMatrix_init( VMachine* vm );
FALCON_FUNC DataMatrix_encode( VMachine* vm );
FALCON_FUNC DataMatrix_decode( VMachine* vm );
FALCON_FUNC DataMatrix_resetOptions( VMachine* vm );

} // !namespace Ext
} // !namespace Falcon

#endif // !DMTX_EXT_H
