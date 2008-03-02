/*
   Mini XML lib PLUS for C++

   Utility routines

   Author: Giancarlo Niccolai <gian@niccolai.ws>

   $Id: mxml_utility.cpp,v 1.3 2004/10/14 13:16:37 jonnymind Exp $
*/

#include <string>
#include <iostream>

namespace MXML {

/** Escapes a string.
   \todo have this function to work...
*/
std::string escape( const std::string unescaped )
{
   return "";
}


char parseEntity( const std::string entity )
{
   char chr = 0;

   if ( entity == "amp" ) chr = '&';
   else if ( entity == "lt" ) chr = '<';
   else if ( entity == "gt" ) chr = '>';
   else if ( entity == "quot" ) chr = '"';
   else if ( entity == "apos" ) chr = '\'';

   return chr;
}


std::ostream & writeEscape( std::ostream &stream, const std::string &src )
{

   for( int i = 0; i < src.length(); i++ ) {
      switch ( src[i] ) {
         case '"': stream << "&quot;"; break;
         case '\'': stream << "&apos;"; break;
         case '&': stream << "&amp;"; break;
         case '<': stream << "&lt;"; break;
         case '>': stream << "&gt;"; break;
         default: stream << src[i];
      }
      if ( stream.bad() ) break;
   }

   return stream;
}

}

/* end of utility.cpp */
