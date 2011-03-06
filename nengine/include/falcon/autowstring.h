/*
   FALCON - The Falcon Programming Language.
   FILE: autowstring.h

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

#ifndef flc_autowstring_H
#define flc_autowstring_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/string.h>

namespace Falcon {

class Item;
class VMachine;


/** Automatically converts and allocate temporary memory for C wide character strings.
   Works exactly as Falcon::AutoCString, but this is meant to translate Falcon::String
   into the wchar_t representation of the string.
   \see Falcon::AutoCString
*/

class FALCON_DYN_CLASS AutoWString
{
   typedef enum {
      AutoWString_BUF_SPACE = 128
   } e_consts;

   wchar_t *m_pData;
   wchar_t m_buffer[AutoWString_BUF_SPACE];
   uint32 m_len;

   void init_vm_and_format( VMachine *vm, const Item &itm, const String &fmt );


public:

   AutoWString();

   AutoWString( const Falcon::String &str );
   AutoWString( const Falcon::Item &itm );

   AutoWString( Falcon::VMachine *vm, const Falcon::Item &itm ):
      m_pData( 0 )
   {
      init_vm_and_format( vm, itm, "" );
   }

   AutoWString( Falcon::VMachine *vm, const Falcon::Item &itm, const Falcon::String &fmt ):
       m_pData( 0 )
   {
      init_vm_and_format( vm, itm, fmt );
   }

   ~AutoWString();

   void set( const Falcon::String &str );
   void set( const Falcon::Item &itm );
   void set( Falcon::VMachine *vm, const Falcon::Item &itm );
   void set( Falcon::VMachine *vm, const Falcon::Item &itm, const Falcon::String &fmt );

   const wchar_t *w_str() const { return m_pData; }
   operator const wchar_t *() const { return m_pData; }
   bool isValid() const { return m_pData[0] != (wchar_t) 0xFFFF; }

   /** Size of the returned buffer.
      This returns the number of wide characters that have been transcoded.
   */
   uint32 length() const { return m_len; }
};

}

#endif

/* end of autowstring.h */
