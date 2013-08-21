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

class TextWriter;
class TextReader;

class JSON
{
public:
   JSON( Stream* stream, bool bPretty=false, bool bReadale = false );
   ~JSON();

   bool encode( const Item& source, String& error );
   bool decode( Item& target, String& error );

   void put( uint32 chr );
   void putLine( uint32 chr );
private:

   void encode_string( const String& source ) const;
   bool decodeArray( Item& target, String& error );
   bool decodeDict( Item& target, String& error );
   bool decodeKey( String& tgt );
   bool getChar( uint32& chr );
   void ungetChar( uint32 chr );
   void setError( const String& error, String& target ) const;

   bool m_bPretty;
   bool m_bReadable;
   int m_level;

   Stream* m_stream;
   TextReader* m_tr;
   TextWriter* m_tw;

   int32 m_charPos;
   int32 m_oldCharPos;
   int32 m_linePos;

};

}

#endif

/* end of json_mod.h */
