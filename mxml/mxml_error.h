/*
   Mini XML lib PLUS for C++

   Error class

   Author: Giancarlo Niccolai <gian@niccolai.ws>

   $Id: mxml_error.h,v 1.1.1.1 2003/08/13 00:13:29 jonnymind Exp $
*/

#ifndef MXML_ERROR_H
#define MXML_ERROR_H

#include <mxml_element.h>
#include <falcon/string.h>
#include <falcon/userdata.h>

#define FALCON_MXML_ERROR_BASE   1100

namespace MXML {

typedef enum {
   malformedError=1,
   ioError,
   notFoundError
} errorType;

class Error: public Falcon::UserData
{
private:
   int m_code;
   int m_beginLine;
   int m_beginChar;
   int m_line;
   int m_char;

public:

/** Error codes
   This define what kind of error happened while decoding the XML document
*/
enum codes
{
   errNone = 0,
   errIo,
   errNomem,
   errOutChar,
   errInvalidNode,
   errInvalidAtt,
   errMalformedAtt,
   errInvalidChar,
   errUnclosed,
   errUnclosedEntity,
   errWrongEntity,
   errChildNotFound,
   errAttrNotFound,
   errHyerarcy,
   errCommentInvalid,
   errMultipleXmlDecl
};

protected:
   Error( const codes code, const Element *generator );

public:
   virtual ~Error();
   virtual const errorType type() const = 0;
   int numericCode() const;
   const Falcon::String description() const;
   void toString( Falcon::String &target ) const;
   void describeLine( Falcon::String &target ) const;

   const Falcon::String describeLine() const { Falcon::String s; describeLine(s); return s; }
};


class MalformedError: public Error
{
public:
   MalformedError( const codes code, const Element *generator ):
      Error( code, generator ) {};
   virtual const errorType type() const  { return malformedError; }
};

class IOError: public Error
{
public:
   IOError( const codes code, const Element *generator  ): 
      Error( code, generator ) {};
   virtual const errorType type() const { return ioError; }
};

class NotFoundError: public Error
{
public:
   NotFoundError( const codes code, const Element *generator ): 
      Error( code, generator ) {};
   virtual const errorType type() const { return notFoundError; }
};

}

#endif
