
/**
 * \file
 *
 * ZLib module main file - extension definitions
 */

#ifndef FLC_ZLIB_EXT_H
#define FLC_ZLIB_EXT_H

#include <falcon/setup.h>
#include <falcon/module.h>

namespace Falcon {
namespace Ext {

FALCON_FUNC ZLib_init( ::Falcon::VMachine *vm );
FALCON_FUNC ZLib_compress( ::Falcon::VMachine *vm );
FALCON_FUNC ZLib_uncompress( ::Falcon::VMachine *vm );

}
}

#endif /* FLC_ZLIB_EXT_H */
