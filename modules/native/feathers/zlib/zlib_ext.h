
/**
 * \file
 *
 * ZLib module main file - extension definitions
 */

#ifndef FLC_ZLIB_EXT_H
#define FLC_ZLIB_EXT_H

#include <falcon/setup.h>
#include <falcon/error.h>
#include <falcon/errorclasses.h>
#include <falcon/error_base.h>

#include <falcon/classes/classuser.h>
#include <falcon/property.h>
#include <falcon/method.h>

#ifndef FALCON_ZLIB_ERROR_BASE
   #define FALCON_ZLIB_ERROR_BASE        1190
#endif


namespace Falcon {
namespace Ext {

class ClassZLib: public ClassUser
{
public:

   ClassZLib();
   virtual ~ClassZLib();

   virtual void serialize( DataWriter* stream, void* self ) const;
   virtual void* deserialize( DataReader* stream ) const;

   //=============================================================
   //
   virtual void* createInstance( ) const;

private:

   //====================================================
   // Properties.
   //


   //====================================================
   // Methods.
   //


   FALCON_DECLARE_METHOD( getVersion, "" );
   FALCON_DECLARE_METHOD( compress, "buffer:M|S" );
   FALCON_DECLARE_METHOD( uncompress, "buffer:M|S" );
   FALCON_DECLARE_METHOD( compressText, "text:S" );
   FALCON_DECLARE_METHOD( uncompressText, "buffer:M|S" );

};

class ClassZLibError: public ClassError
      {
      private:
         static ClassZLibError* m_instance;
      public:
         inline ClassZLibError(): ClassError( "ZLibError" ) {} 
         inline virtual ~ClassZLibError(){} 
         virtual void* createInstance() const;
         static inline ClassZLibError* singleton();
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
