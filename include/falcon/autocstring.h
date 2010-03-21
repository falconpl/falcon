/*
   FALCON - The Falcon Programming Language.
   FILE: autocstring.h

   SUtility to convert falcon items and strings into C Strings.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab ago 4 2007

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Utility to convert falcon items and strings into C Strings.
   Header file.
*/

#ifndef flc_autocstring_H
#define flc_autocstring_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/string.h>

namespace Falcon {

class Item;
class VMachine;


/** Automatically converts and allocate temporary memory for C strings.

   Falcon has a complete API model in which every representation, naming,
   and in general string operation is performed through Falcon::String.

   Embedding applications may use that class inside their own code to access
   the advanced features that the Falcon::String class provides.

   However, when it is necessary to embed Falcon in previously existing
   applications, or when it is necessary to interface Falcon with plain C code,
   it is necessary to turn items and strings into something C functions can
   understand.

   Falcon::String has a toCString() member that can fill a C string with
   an UTF-8 representation of the internal string data (as UTF-8 is manageable
   by any C function and is thus the standard for internationalized C libraries
   maintaining char * as their primary string type). Using that interface is a
   bit tricky, as it requires to provide enough buffer space; C applications
   may find themselves in checking Falcon::String size and creating temporary
   buffers all the time.

   This class provides a very simple mean for applications interfacing to C to
   store small strings in a temporary stack based buffer, while longer strings
   are placed in a heap allocated space. The casts and the member c_str() allow
   to access the buffer or the heap space transparently, and on destruction
   if a buffer was allocated, it is automatically deleted.

   The stack space provided is 128 bytes, which is enough for the vast majority
   of strings an ordinary program has to manage. If the converted string is longer
   a wide enough char buffer is allocated with memAlloc() and automatically deleted
   at scope termination.

   The class provides also a constructor that allows to automatically convert any
   falcon item to string (with eventually a falcon format to be applied)
   by providing a VM that will be eventually called to execute "toString()" methods,
   in case the provided item is an object.

   Usage is:
   \code
      Module *module = .... // a mean to create the module.

      AutoCString modName( module->name() );

      // printf cannot cast to (char*), we'll have to do...
      printf( "The module name is %s\n", modName.c_str() );

      // but strlen and strcpy does, so there's no need for that.
      char *retval = (char *) malloc( strlen( modName ) );
      strcpy( retval, modName );
   \endcode

   If you want to convert item to plain old C strings on the fly:
   \code
      Item number = 3.24;

      // also pass the optional format (right aling, 10 size fixed 4 decimals)
      AutoCString numrep( vm, number, "r10.4" );
      if ( ! numrep.isValid() )
      {
         ... conversion didn't work, i.e. because of errors in the toString() call.
      }

      // printf cannot cast to (char*), we'll have to do...
      printf( "The item is %s\n", numrep.c_str() );
   \endcode

   It is possible also to convert an item without using a VM; in that case convertion
   of items is performed using the default to string representation.
   \code
      Item number = 3.24;

      AutoCString numrep( number );

      // printf cannot cast to (char*), we'll have to do...
      printf( "The item is %s\n", numrep.c_str() );
   \endcode

*/

class FALCON_DYN_CLASS AutoCString
{
   typedef enum {
      AutoCString_BUF_SPACE = 128
   } e_consts;

   char *m_pData;
   uint32 m_len;
   char m_buffer[ AutoCString_BUF_SPACE ];
   void init_vm_and_format( VMachine *vm, const Item &itm, const String &fmt );

public:
   AutoCString();

   AutoCString( const Falcon::String &str );
   AutoCString( const Falcon::Item &itm );

   AutoCString( Falcon::VMachine *vm, const Falcon::Item &itm ):
      m_pData( 0 )
   {
      init_vm_and_format( vm, itm, "" );
   }

   AutoCString( Falcon::VMachine *vm, const Falcon::Item &itm, const Falcon::String &fmt ):
       m_pData( 0 )
   {
      init_vm_and_format( vm, itm, fmt );
   }

   ~AutoCString();

   void set( const Falcon::String &str );
   void set( const Falcon::Item &itm );
   void set( Falcon::VMachine *vm, const Falcon::Item &itm );
   void set( Falcon::VMachine *vm, const Falcon::Item &itm, const Falcon::String &fmt );

   const char *c_str() const { return m_pData+3; }
   const char *bom_str();
   operator const char *() const { return m_pData+3; }
   bool isValid() const { return m_pData[3] != (char) 255; }

   /** Size of the returned buffer.
      This returns the number of bytes in the returned buffer, not the number
      of charcaters actually contained in the string.

      It's the distance between c_str() begin and the terminating 0.
   */
   uint32 length() const { return m_len; }
};

}

#endif

/* end of autocstring.h */
