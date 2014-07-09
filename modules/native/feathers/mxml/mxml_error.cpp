/*
   Mini XML lib PLUS for C++

   Error class - implementation

   Author: Giancarlo Niccolai <gian@niccolai.ws>
*/

#include "mxml_error.h"

namespace MXML {

Error::Error( const codes code, const Element *generator ):
   m_code( code )
{
   m_beginLine = generator->beginLine();   
   m_beginChar = generator->beginChar();
   m_line = generator->line();
   m_char = generator->character();
}

Error::~Error()
{
}

const Falcon::String Error::description() const
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
      case errHierarchy: return "Node is not in a hierarcy - no parent";
      case errCommentInvalid: return "Invalid comment ( -- sequence is not followed by '>')";
      case errMultipleXmlDecl: return "Multiple XML top node delcarations";
   }
   return "Undefined error code";
}

int Error::numericCode() const
{
   return ((int) m_code );
}

void Error::toString( Falcon::String &stream ) const
{
   switch( this->type() ) {
      case malformedError: stream = "MXML::MalformedError"; break;
      case ioError: stream = "MXML::IOError"; break;
      case notFoundError: stream = "MXML::NotFoundError"; break;
      default: stream = "MXML::Unknown error"; break;
   }
   stream += " (";
   stream.writeNumber( (Falcon::int64) this->m_code );
   stream += "):";

   stream += this->description();

   if ( this->type() != notFoundError ) {
      describeLine( stream );
   }
   stream.append( '\n' );
}

void Error::describeLine( Falcon::String &stream ) const
{
   if ( m_beginLine )
   {
      stream += "at ";
      stream.writeNumber( (Falcon::int64) m_beginLine );
      stream += ":";
      stream.writeNumber( (Falcon::int64) m_beginChar );
   }

   if ( m_line )
   {
      stream += " (from  ";
      stream.writeNumber( (Falcon::int64) m_line );
      stream += ":";
      stream.writeNumber( (Falcon::int64) m_char );
      stream += ")";
   }
}

}

