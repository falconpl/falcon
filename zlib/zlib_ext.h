
/**
 * \file
 *
 * ZLib module main file - extension definitions
 */

#ifndef FLC_ZLIB_EXT_H
#define FLC_ZLIB_EXT_H

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/error.h>

namespace Falcon {
namespace Ext {

FALCON_FUNC ZLib_getVersion( ::Falcon::VMachine *vm );
FALCON_FUNC ZLib_compress( ::Falcon::VMachine *vm );
FALCON_FUNC ZLib_uncompress( ::Falcon::VMachine *vm );

class ZlibError: public ::Falcon::Error
{
public:
   ZlibError():
      Error( "ZlibError" )
   {}

   ZlibError( const ErrorParam &params  ):
      Error( "ZlibError", params )
      {}
};

FALCON_FUNC  ZlibError_init ( ::Falcon::VMachine *vm );

}
}

#endif /* FLC_ZLIB_EXT_H */
