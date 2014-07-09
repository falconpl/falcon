/*
   Mini XML lib PLUS for C++

   Attribute class

   Author: Giancarlo Niccolai <gc@falconpl.org>
*/

#include "mxml.h"
#include "mxml_attribute.h"
#include "mxml_error.h"
#include "mxml_utility.h"

#include <falcon/fassert.h>

#include <ctype.h>

namespace MXML {

Attribute::Attribute( Falcon::TextReader &in, int style, int l, int p ):
   Element( l, p )
{
   Falcon::uint32 chr, quotechr=0;
   int iStatus = 0;
   Falcon::String entity;
   markBegin(); // default start

   m_value = "";
   m_name = "";

   while ( iStatus < 6 && ! in.eof() )
   {
      chr = nextChar(in);
      //std::cout << "LINE: " << line() << "  POS: " << character() << std::endl;
      switch ( iStatus ) {
         // begin
         case 0:
            // no attributes found - should not happen as I have been called by
            // node only if an attribute is to be read.
            fassert( chr != '>' && chr !='/');
            switch ( chr ) {
               // We repeat line terminator here for portability
               case MXML_SOFT_LINE_TERMINATOR: break;
               case ' ': case '\t':
                  throw new MalformedError( Error::errInvalidAtt, this );

               default:
                  if ( isalpha( chr ) ) {
                     m_name = chr;
                     iStatus = 1;
                     markBegin();
                  }
                  else {
                     throw new MalformedError( Error::errInvalidAtt, this );
                  }
                  break;
            }
         break;

         // scanning for a name
         case 1:
            if ( isalnum( chr ) || chr == '_' || chr == '-' || chr == ':' ) {
               m_name += chr;
            }
            else if( chr == MXML_LINE_TERMINATOR ) {
               iStatus = 2; // waiting for a '='
            }
            // We repeat line terminator here for portability
            else if ( chr == ' ' || chr == '\t' || chr == '\n' || chr == '\r' ) {
               iStatus = 2;
            }
            else if ( chr == '=' ) {
               iStatus = 3;
            }
            else {
               throw MalformedError( Error::errMalformedAtt, this );
            }
         break;

         // waiting for '='
         case 2:
            if ( chr == '=' ) {
               iStatus = 3;
            }
            // We repeat line terminator here for portability
            else if ( chr == ' ' || chr == '\t' || chr == '\n' || chr == '\r' ) {
            }
            else {
               throw MalformedError( Error::errMalformedAtt, this );
            }
         break;

         // waiting for ' or "
         case 3:
            if ( chr == '\'' || chr == '"' ) {
               iStatus = 4;
               quotechr = chr;
            }
            // We repeat line terminator here for portability
            else if ( chr == ' ' || chr == '\t' || chr == '\n' || chr == '\r' ) {
            }
            else {
               throw MalformedError( Error::errMalformedAtt, this );
            }
         break;

         // scanning the attribute content ( until next quotechr )
         case 4:
            if ( chr == quotechr ) {
               iStatus = 6;
            }
            else if ( chr == '&' && !( style & MXML_STYLE_NOESCAPE) ) {
               iStatus = 5;
               entity = "";
            }
            else if( chr == MXML_LINE_TERMINATOR ) {
               m_value += chr;
            }
            else {
               m_value += chr;
            }
         break;

         case 5:
            if ( chr == quotechr ) {
               iStatus = 6;
            }
            else if ( chr == ';' ) {
               if ( entity == "" ) {
                  throw MalformedError( Error::errWrongEntity, this );
               }

               iStatus = 4;

               // we see if we have a predef entity (also known as escape)
               if ( entity == "amp" ) chr = '&';
               else if ( entity == "lt" ) chr = '<';
               else if ( entity == "gt" ) chr = '>';
               else if ( entity == "quot" ) chr = '"';
               else if ( entity == "apos" ) chr = '\'';
               else {
                  // for now we take save the unexisting entity
                  chr = ';';
                  m_value += "&" + entity;
               }

               m_value += chr;
            }
            else if ( !isalnum( chr ) && chr != '-' && chr != '_' && chr != '#' ) {
               //error
               throw MalformedError( Error::errUnclosedEntity, this );
            }
            else {
               entity += chr;
            }
         break;
      }
      if ( iStatus < 6 )
         chr = nextChar(in);
   }

   if ( iStatus < 6 ) {
      throw MalformedError( Error::errMalformedAtt, this );
   }
}

void Attribute::write( Falcon::TextWriter &out, const int style ) const
{
   out.write( m_name );
   out.write( "=\"" );

   if ( style & MXML_STYLE_NOESCAPE )
      out.write( m_value );
   else
      MXML::writeEscape( out, m_value );

   out.putChar( '\"' );
}

}

/* end of mxml_attribute.cpp */
