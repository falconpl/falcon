/*
   FALCON - The Falcon Programming Language.
   FILE: stringwithid.h
   $Id: stringwithid.h,v 1.1.1.1 2006/10/08 15:05:34 gian Exp $

   An extenson of the Falcon::String that saves also the module string ID.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: gio lug 21 2005
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

/** \file
   An extenson of the Falcon::String that saves also the module string ID.
*/

#ifndef flc_stringwithid_H
#define flc_stringwithid_H

#include <falcon/string.h>

namespace Falcon 
{

/** Pair of strings and their ID.
   Useful during compilation to create objects that need to know the string ID in
   the module rather than the physical string.
*/
class StringWithID: public String
{
   int32 m_id;
public:
   /** allows on-the-fly core string creation from static data */
   StringWithID( const char *data ):
      String( data )
   {}

   StringWithID( const char *data, uint32 len ):
      String( data, len )
   {}
   
   StringWithID():
      String()
   {}

   StringWithID( byte *buffer, uint32 size, uint32 allocated ):
      String( buffer, size, allocated )
   {}

   explicit StringWithID( uint32 prealloc ):
      String( prealloc )
   {}
   
   StringWithId( const StringWithID &other ):
      String( other )
   {
      m_id = other.m_id;
   }
   
   int32 id() const { return m_id; }
   void id( int32 val ) { m_id = val; }
};

}

#endif

/* end of stringwithid.h */
