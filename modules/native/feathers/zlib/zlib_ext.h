
/**
 * \file
 *
 * ZLib module main file - extension definitions
 */

#ifndef FLC_ZLIB_EXT_H
#define FLC_ZLIB_EXT_H

#include <falcon/setup.h>
#include <falcon/error.h>
#include <falcon/error_base.h>

#include <falcon/function.h>

#ifndef FALCON_ZLIB_ERROR_BASE
   #define FALCON_ZLIB_ERROR_BASE        1190
#endif


namespace Falcon {
namespace Ext {



//====================================================
// Functions.
//


FALCON_DECLARE_FUNCTION( getVersion, "" );
FALCON_DECLARE_FUNCTION( compress, "buffer:M|S" );
FALCON_DECLARE_FUNCTION( uncompress, "buffer:M|S" );
FALCON_DECLARE_FUNCTION( compressText, "text:S" );
FALCON_DECLARE_FUNCTION( uncompressText, "buffer:M|S" );


class ClassZLibError: public ClassError
      {
      private:
         static ClassZLibError* m_instance;
      public:
         inline ClassZLibError(): ClassError( "ZLibError" ) {} 
         inline virtual ~ClassZLibError(){} 
         virtual void* createInstance() const;
         static ClassZLibError* singleton();
      };

class ZLibError: public ::Falcon::Error
{
public:
   ZLibError():
      Error( ClassZLibError::singleton() )
   {}

   ZLibError( const ErrorParam &params  ):
      Error( ClassZLibError::singleton(), params )
      {}
};

}
}

#endif /* FLC_ZLIB_EXT_H */
