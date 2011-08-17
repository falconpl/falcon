/*
   FALCON - The Falcon Programming Language.
   FILE: scriptdata.cpp

   Simple helper to help keeping a the unit test a bit cleaner.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun feb 13 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Simple helper to help keeping a the unit test a bit cleaner.
*/

#include <falcon/string.h>
#include <falcon/stringstream.h>
#include "scriptdata.h"

namespace Falcon {

ScriptData::ScriptData( const String &filename, int id ):
   m_id( id ),
   m_filename( filename )
{
	m_filename.bufferize();
}


void ScriptData::IdToIdCode( int id, String &code )
{
   // code is just CODE * 100 + letter...
   int64 numcode = id /100;
   char letter = (id % 100);
   code = "";

   if ( letter != 0 )
   {
      letter += 'a'-1;
      code.writeNumber( numcode );
      code.append( letter );
   }
   else
      code.writeNumber( numcode );
}

int ScriptData::IdCodeToId( const String &code )
{

   if ( code.length() == 0 )
      return 0;

   uint32 lettercode = code.getCharAt( code.length() - 1 );
   String copy;

   if ( lettercode >= (uint32)'a' && lettercode <= (uint32) 'z' )
   {
      lettercode = lettercode - 'a' + 1;
      copy = code.subString( 0, code.length() - 1 );
   }
   else {
      copy = code;
      lettercode = 0;
   }

   int64 ret;
   if ( copy.parseInt( ret ) )
      return (int)(ret * 100 + lettercode);
   return 0;
}

void ScriptData::setProperty( const String &name, const String &value )
{
   m_properties[ name ] = value;
}

bool ScriptData::getProperty( const String &name, String &value ) const
{
   t_stringMap::const_iterator iter = m_properties.find( name );
   if( iter != m_properties.end() )
   {
      value = iter->second;
      return true;
   }

   value = "";
   return false;
}

}


/* end of scriptdata.cpp */
