/*
   FALCON - The Falcon Programming Language.
   FILE: autoucsstring.h

   Utility to convert falcon items and strings into UCS-2 Strings.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 12 Sep 2010 12:53:18 +0200

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Utility to convert falcon items and strings into UCS Strings.
   Header file.
*/

#ifndef FALCON_AUTOUCSSTRING_H
#define FALCON_AUTOUCSSTRING_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/string.h>

namespace Falcon {

class Item;
class VMachine;

/** Automatically converts and allocate temporary memory for UCS-2 strings.

   Falcon has a complete API model in which every representation, naming,
   and in general string operation is performed through Falcon::String.

   However, it is often necessary to pass Falcon String to outer world.

   This class converts automatically falcon strings to UCS-2 strings, that is,
   strings having character long exactly 2 bytes, with machine-dependent
   byte ordering (pure number) without any encoding in place. A 2-bytes long
   position corresponds 1:1 to a UNICODE character in the range 0-0xFFFF.

   UNICODE Characters above that range are translated into a square (U25A1),
   but this default can be changed in the constructor.

   @note This behavior is similar to AutoWString, but AutoWString uses the
   platform-dependent wchar_t type. Some libraries that may be bound into
   Falcon modules may use UCS-2 (16-bit) character encoding as an hard-coded
   choice, hence the need to avoid relying on wchar_t to deal with those.
   libraries.

   @see AutoCString
   @see AutoWString
*/

class FALCON_DYN_CLASS AutoUCSString
{
   typedef enum {
      AutoUCSString_BUF_SPACE = 128,
      DefaultUnknownChar = 0x25A1
   } e_consts;

   uint16 *m_pData;
   uint32 m_len;
   uint16 m_buffer[ AutoUCSString_BUF_SPACE ];

public:
   AutoUCSString();
   AutoUCSString( const Falcon::String &str, uint16 defChar = DefaultUnknownChar );
   ~AutoUCSString();

   void set( const Falcon::String &str, uint16 defChar = DefaultUnknownChar );
   const uint16 *ucs_str() const { return m_pData; }
   uint32 length() const { return m_len; }
};

}

#endif

/* end of autowstring.h */
