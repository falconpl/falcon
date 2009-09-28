/*
   FALCON - The Falcon Programming Language.
   FILE: json_mod.h

   JSON transport format interface - inner logic (serviceable)
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 27 Sep 2009 18:28:44 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Compiler module main file - extension definitions.
*/

#ifndef flc_json_mod_H
#define flc_json_mod_H

#include <falcon/setup.h>
#include <falcon/item.h>
#include <falcon/stream.h>
#include <falcon/error_base.h>


namespace Falcon {

class JSON: public BaseAlloc
{
public:
   JSON( bool bEncUni = false, bool bPretty=false, bool bReadale = false );
   ~JSON();

   bool encode( const Item& source, Stream* tgt );
   bool decode( Item& target, Stream* src ) const;

private:
   void encode_string( const String& source, Stream* tgt ) const;
   CoreArray* decodeArray( Stream* src ) const;
   CoreDict* decodeDict( Stream* src ) const;

   bool m_bEncUnicode;
   bool m_bPretty;
   bool m_bReadable;
   int m_level;
};

}

#endif

/* end of json_mod.h */
