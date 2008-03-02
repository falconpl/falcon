/*
   Mini XML lib PLUS for C++

   Error class

   Author: Giancarlo Niccolai <gian@niccolai.ws>

   $Id: mxml_error.h,v 1.1.1.1 2003/08/13 00:13:29 jonnymind Exp $
*/

#ifndef MXML_ERROR_H
#define MXML_ERROR_H

#include <mxml_element.h>
#include <string>

namespace MXML {

typedef enum {
   malformedError=1,
   ioError,
   notFoundError
} errorType;

class Error
{
private:
   int m_code;
   const Element *m_generator;

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
   errCommentInvalid
};

protected:
   Error( const codes code, const Element *generator )
   {
      m_code = code;
      m_generator = generator;
   }

public:
   virtual const errorType type() const = 0;
   const std::string description();

   /** Serializes the error.
      Use this operator to display what kind of error has happened
   */
   friend std::ostream& operator<<( std::ostream &stream, Error &err );
};


class MalformedError: public Error
{
public:
   MalformedError( const codes code, const Element *generator  ):
      Error( code, generator ) {};
   virtual const errorType type() const  { return malformedError; }
};

class IOError: public Error
{
public:
   IOError( const codes code, const Element *generator  ): Error( code, generator ) {};
   virtual const errorType type() const { return ioError; }
};

class NotFoundError: public Error
{
public:
   NotFoundError( const codes code, const Element *generator  ): Error( code, generator ) {};
   virtual const errorType type() const { return notFoundError; }
};

}

#endif
