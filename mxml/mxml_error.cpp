/*
   Mini XML lib PLUS for C++

   Error class - implementation

   Author: Giancarlo Niccolai <gian@niccolai.ws>

   $Id: mxml_error.cpp,v 1.2 2004/04/10 23:50:29 jonnymind Exp $
*/

#include <mxml_error.h>

namespace MXML {

const Falcon::String Error::description()
{
   switch( m_code )
   {
      case errNone: return "No error";
      case errIo: return "Input/output error";
      case errNomem: return "Not enough memory";
      case errOutChar: return "Character outside tags";
      case errInvalidNode: return "Invalid character as tag name";
      case errInvalidAtt: return "Invalid character as attribute name";
      case errMalformedAtt: return "Malformed attribute definition";
      case errInvalidChar: return "Invalid character";
      case errUnclosed: return "Unbalanced tag opening";
      case errUnclosedEntity: return "Unbalanced entity opening";
      case errWrongEntity: return "Escape/entity '&;' found";
      case errChildNotFound: return "Unexisting child request";
      case errAttrNotFound: return "Attribute name cannot be found";
      case errHyerarcy: return "Node is not in a hierarcy - no parent";
      case errCommentInvalid: return "Invalid comment ( -- sequence is not followed by '>')";

   }
   return "Undefined error code";
}

void Error::toString( Falcon::String &stream )
{
   switch( this->type() ) {
      case malformedError: stream = "MXML::MalformedError"; break;
      case ioError: stream = "MXML::IOError"; break;
      case notFoundError: stream = "MXML::NotFoundError"; break;
      default: stream = "MXML::Unknown error";
   }
   stream += " (";
   stream.writeNumber( (Falcon::int64) this->m_code );
   stream += "):";

   stream += this->description();

   if ( this->type() != notFoundError ) {
      stream += " in line ";
      stream.writeNumber( (Falcon::int64) this->m_generator->beginLine() );
      stream += ":";
      stream.writeNumber( (Falcon::int64) this->m_generator->beginChar() );
      if( this->m_generator->line() )
      {
         stream += " ( realized in line ";
         stream.writeNumber( (Falcon::int64) this->m_generator->line()); 
         stream += ":";
         stream.writeNumber( (Falcon::int64) this->m_generator->character() ); 
         stream += ")";
      }
   }

   stream.append( '\n' );
}

}

