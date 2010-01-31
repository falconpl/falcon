/*
   FALCON - The Falcon Programming Language.
   FILE: utils.cpp

   Utilities for Falcon packager
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 30 Jan 2010 12:42:48 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "utils.h"
#include <falcon/fstream.h>

namespace Falcon
{

Stream* stdOut;
Stream* stdErr;

static void message_1( const String &msg )
{
   stdOut->writeString( "falpack: " );
   stdOut->writeString( msg );
   stdOut->writeString( "\n" );
   stdOut->flush();
}

// for non-verbose operations
static void message_2( const String &msg )
{}

void (*message)( const String &msg );

void error( const String &msg )
{
   stdErr->writeString( "falpack: ERROR - " );
   stdErr->writeString( msg );
   stdErr->writeString( "\n" );
   stdErr->flush();
}


void warning( const String &msg )
{
   stdOut->writeString( "falpack: WARN - " );
   stdOut->writeString( msg );
   stdOut->writeString( "\n" );
   stdOut->flush();
}


void setVerbose( bool mode )
{
   if ( mode )
      message = message_1;
   else
      message = message_2;
}

void splitPaths( const String& path, std::vector<String>& tgt )
{
   uint32 pos = 0, pos1;
   while( (pos1 = path.find( ";", pos )) != String::npos )
   {
      String sRes = path.subString( pos, pos1 );
      sRes.trim();
      tgt.push_back( sRes );
      pos = pos1+1;
   }

   String sRes = path.subString( pos );
   sRes.trim();
   tgt.push_back( sRes );
}

bool copyFile( const String& source, const String& dest )
{
   message( String("Copying ").A( source ).A(" => ").A( dest ) );

   // NOTE: streams are closed by the destructor.
   FileStream instream, outstream;

   instream.open( source, ::Falcon::BaseFileStream::e_omReadOnly );
   if ( ! instream.good() )
   {
      return false;
   }

   outstream.create( dest, (Falcon::BaseFileStream::t_attributes) 0644 );
   if ( ! outstream.good() )
   {
      return false;
   }

   byte buffer[8192];
   int count = 0;
   while( ( count = instream.read( buffer, 8192) ) > 0 )
   {
      if ( outstream.write( buffer, count ) < 0 )
      {
         return false;
      }
   }

   return true;
}


}

/* end of utils.cpp */
