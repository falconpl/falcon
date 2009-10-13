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

namespace Falcon
{

class Parameter;
class ParamList;

class CType
{
public:
   typedef enum
   {
      e_void,
      e_char,
      e_unsigned_char,
      e_signed_char,
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

   CType( e_integral_type id, const char *repr, int pointers = 0, int subs = 0, bool isFunc = false ):
      m_id(id),
      m_normal_repr(repr),
      m_tag(0),
      m_pointers(pointers),
      m_subscript(subs),
      m_isFuncPtr(isFunc),
      m_funcParams(0)
      {}

   CType( e_integral_type id, const char *repr, const char* tag, int pointers = 0, int subs = 0, bool isFunc = false ):
      m_id(id),
      m_normal_repr(repr),
      m_tag(tag),
      m_pointers(pointers),
      m_subscript(subs),
      m_isFuncPtr(isFunc),
      m_funcParams(0)
      {
         fassert( m_id == e_struct || m_id == e_union || m_id == e_enum );
      }

   CType( const CType& other );

   e_integral_type m_id;

   /** Normalized declaration of this type.
    * Tagged types have a terminal "*" that is substituted with exactly a word.
    * */
   const char* m_normal_repr;

   /** Tag for tagged types. */
   const char* m_tag;

   /** Number of pointer indirections */
   int m_pointers;

   /** Array subscript count (-1 for [] ) */
   int m_subscript;

   /** returns the number o indirections (pointers/array subscripts) */
   int indirections() const { return m_pointers + m_subscript != 0 ? 1:0; }

   /** True if this is a function pointer (returning this type)
    * m_pointers will be zero unless this is a pointer to a function pointer.
    * */
   bool m_isFuncPtr;

   /** If this is a function pointer, it may have one or more function parameters. */
   ParamList m_funcParams;
};



class Parameter
{
public:
   Parameter( const CType& t, const char* name );

   CType m_type;
   const char* m_name;
   Parameter* m_next;
};

class ParamList
{
public:
   ParamList();
   ParamList( const ParamList& other );
   ~ParamList();

   void add(Parameter* p);
   void first() const { return m_head; }
   bool empty() const { return m_size == 0; }
   int size() const { return m_size; }

private:
   Parameter* m_head;
   Parameter* m_tail;
   int m_size;
};

}

#endif /* DYNLIB_PARSER_H_ */
