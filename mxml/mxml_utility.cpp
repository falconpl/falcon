/*
   Mini XML lib PLUS for C++

   Utility routines

   Author: Giancarlo Niccolai <gian@niccolai.ws>

*/

#include <falcon/string.h>
#include <falcon/stream.h>

namespace MXML {

/** Escapes a string.
   \todo have this function to work...
*/
Falcon::String escape( const Falcon::String &unescaped )
{
   return "";
}


Falcon::uint32 parseEntity( const Falcon::String &entity )
{
   Falcon::uint32 chr = 0;

   if ( entity == "amp" ) chr = '&';
   else if ( entity == "lt" ) chr = '<';
   else if ( entity == "gt" ) chr = '>';
   else if ( entity == "quot" ) chr = '"';
   else if ( entity == "apos" ) chr = '\'';

   return chr;
}


Falcon::Stream & writeEscape( Falcon::Stream &stream, const Falcon::String &src )
{

   for( Falcon::uint32 i = 0; i < src.length(); i++ ) {
      switch ( src[i] ) {
         case '"': stream.write( "&quot;", 6 ); break;
         case '\'': stream.write( "&apos;", 6 ); break;
         case '&': stream.write( "&amp;", 5 ); break;
         case '<': stream.write( "&lt;", 4 ); break;
         case '>': stream.write( "&gt;", 4 ); break;
         default: stream.put( src[i] );
      }
      if ( stream.bad() ) break;
   }

   return stream;
}

}

/* end of utility.cpp */
