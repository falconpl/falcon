/*
 *  Falcon DataMatrix - Extension
 */

#include "dmtx_ext.h"
#include "dmtx_mod.h"
//#include "dmtx_srv.h"
//#include "dmtx_st.h"

#include <falcon/engine.h>
#include <falcon/vm.h>

#include <stdio.h>//debug..

//extern Falcon::DataMatrixService theDataMatrixService;

/*#
    @beginmodule dmtx
 */

namespace Falcon
{
namespace Ext
{

/*#
    @class DataMatrixError
    @brief Error generated DataMatrix operations.
    @optparam code The error code
    @optparam desc The description for the error code
    @optparam extra Extra information specifying the error conditions.
    @from Error( code, desc, extra )
*/
FALCON_FUNC DataMatrixError_init( VMachine* vm )
{
    CoreObject *einst = vm->self().asObject();

    if ( einst->getUserData() == 0 )
        einst->setUserData( new DataMatrixError );

    ::Falcon::core::Error_init( vm );
}


/*#
    @class DataMatrix
    @brief DataMatrix codec
 */
FALCON_FUNC DataMatrix_init( VMachine* vm )
{
    Dmtx::DataMatrix* self = static_cast<Dmtx::DataMatrix*>( vm->self().asObjectSafe() );
    vm->retval( self );
}


/*#
    @method encode DataMatrix
    @param data A string or membuf
    @param context A context object
    @return true on success
 */
FALCON_FUNC DataMatrix_encode( VMachine* vm )
{
    Item* i_data = vm->param( 0 );
    Item* i_ctxt = vm->param( 1 );

    if ( !i_data || !( i_data->isString() || i_data->isMemBuf() )
        || !i_ctxt || !i_ctxt->isObject() )
    {
        throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                .extra( "S|M,O" ) );
    }

    Dmtx::DataMatrix* self = static_cast<Dmtx::DataMatrix*>( vm->self().asObjectSafe() );
    vm->retval( self->encode( *i_data, *i_ctxt ) );
}

} /* !namespace Ext */
} /* !namespace Falcon */
