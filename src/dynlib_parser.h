/*
   The Falcon Programming Language
   FILE: dynlib_mod.cpp

   Direct dynamic library interface for Falcon
   Interface extension functions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 13 Oct 2009 23:17:09 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: The Falcon Committee

   See the LICENSE file distributed with this package for licensing details.
*/

#ifndef DYNLIB_PARSER_H_
#define DYNLIB_PARSER_H_

#include <falcon/setup.h>
#include <falcon/string.h>

namespace Falcon
{

class Parameter;
class Tokenizer;

class ParamList
{
public:
   ParamList();
   ParamList( const ParamList& other );
   ~ParamList();

   void add(Parameter* p);
   Parameter* first() const { return m_head; }
   bool empty() const { return m_size == 0; }
   int size() const { return m_size; }

   String toString() const;

private:
   Parameter* m_head;
   Parameter* m_tail;
   int m_size;
};


class Parameter
{
public:
   typedef enum
   {
      e_void,
      e_char,
      e_unsigned_char,
      e_signed_char,
      e_short,
      e_unsigned_short,
      e_int,
      e_unsigned_int,
      e_long,
      e_unsigned_long,
      e_long_long,
      e_unsigned_long_long,
      e_float,
      e_double,
      e_long_double,
      e_struct,
      e_union,
      e_enum,
      e_varpar
   } e_integral_type;

   Parameter( e_integral_type ct, const String& name, int pointers = 0, int subs = 0, bool isFunc = false );

   Parameter( e_integral_type ct, const String& name, const String& tag, int pointers = 0, int subs = 0, bool isFunc = false );

   Parameter( const Parameter& other );

   ~Parameter();

   /** Type of this parameter. */
   e_integral_type m_type;

   /** Name of the parameters.

      We don't accept unnamed parameters for practical reasons.
   */
   String m_name;

   /** Tag for tagged types. */
   String m_tag;

   /** Number of pointer indirections */
   int m_pointers;

   /** Array subscript count (-1 for [] ) */
   int m_subscript;

   /** True if this is a function pointer (returning this type)
    * m_pointers will be zero unless this is a pointer to a function pointer.
    * */
   bool m_isFuncPtr;

   /** If this is a function pointer, it may have one or more function parameters. */
   ParamList m_funcParams;

   Parameter* m_next;

   /** returns the number o indirections (pointers/array subscripts) */
   int indirections() const { return m_pointers + m_subscript != 0 ? 1:0; }

   String toString() const;

   static String typeToString( e_integral_type type );
};


class FunctionDef2 //: public FalconData
{
   String m_definition;
   String m_name;
   Parameter* m_return;
   ParamList m_params;

public:

   FunctionDef2():
      m_return(0)
   {}

   FunctionDef2( const String& definition ):
      m_return(0)
   {
      parse( definition );
   }

   FunctionDef2( const FunctionDef2& other );
   virtual ~FunctionDef2();

   /** Parses a string definition.
    * Throws ParseError* on error.
    */
   bool parse( const String& definition );

   const String& name() const { return m_name; }
   const String& definition() const { return m_definition; }

   String toString() const;

private:
   Parameter* parseNextParam( Tokenizer& tok, bool isFuncName = false);
   void parseFuncParams( ParamList& params, Tokenizer& tok );

};


}

#endif /* DYNLIB_PARSER_H_ */
