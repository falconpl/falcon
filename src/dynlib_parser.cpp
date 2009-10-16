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

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/fassert.h>
#include <falcon/tokenizer.h>

#include "dynlib_parser.h"

namespace Falcon
{

BaseCType::BaseCType( e_integral_type id, const String& repr, int size ):
   m_id(id),
   m_size( size ),
   m_normal_repr( repr )
{
}

BaseCType::BaseCType( e_integral_type id, const char *repr, int size ):
   m_id(id),
   m_size( size ),
   m_normal_repr( repr )
{
}

BaseCType ctypes[] =
{
   BaseCType( BaseCType::e_void, "void", 0 ),
   BaseCType( BaseCType::e_char, "char", sizeof(char) ),
   BaseCType( BaseCType::e_unsigned_char, "unsigned char", sizeof(char) ),
   BaseCType( BaseCType::e_signed_char, "signed char", sizeof(char) ),
   // ... then the modern version
   BaseCType( BaseCType::e_short, "short", sizeof(short) ),
   BaseCType( BaseCType::e_unsigned_short, "unsigned short", sizeof(short) ),
   BaseCType( BaseCType::e_int, "int", sizeof(int) ),
   BaseCType( BaseCType::e_unsigned_int, "unsigned int", sizeof(int) ),
   BaseCType( BaseCType::e_long, "long", sizeof(long) ),
   BaseCType( BaseCType::e_unsigned_long, "unsigned long", sizeof(long) ),
   BaseCType( BaseCType::e_long_long, "long long", sizeof(int64) ),
   BaseCType( BaseCType::e_unsigned_long_long, "unsigned long long", sizeof(int64) ),
   BaseCType( BaseCType::e_float, "float", sizeof(float) ),
   BaseCType( BaseCType::e_double, "double", sizeof(double) ),
   BaseCType( BaseCType::e_long_double, "long double", sizeof(long double) ),

   // tagged types
   BaseCType( BaseCType::e_struct, "struct *", sizeof(void*) ),
   BaseCType( BaseCType::e_union, "union *", sizeof(void*) ),
   BaseCType( BaseCType::e_enum, "enum *", sizeof(void*) ),

   // some aliases
   BaseCType( BaseCType::e_int, "BOOL", sizeof(int) ),
   BaseCType( BaseCType::e_long_long, "__int64", sizeof(int64) ),
   BaseCType( BaseCType::e_unsigned_long_long, "unsigned __int64", sizeof(int64) ),
   BaseCType( BaseCType::e_unsigned_short, "WORD", 2 ),
   BaseCType( BaseCType::e_unsigned_int, "DWORD", 4 ),
   BaseCType( BaseCType::e_unsigned_long, "HANDLE", sizeof(long) ),

   // Finally, varadic params
   BaseCType( BaseCType::e_varpar, "...", sizeof(void*) )

};


CType::CType( BaseCType* ct, int pointers, int subs, bool isFunc ):
   m_ctype(ct),
   m_pointers(pointers),
   m_subscript(subs),
   m_isFuncPtr(isFunc)
{}


CType::CType( BaseCType* ct, const String &tag, int pointers, int subs, bool isFunc ):
   m_ctype(ct),
   m_tag(tag),
   m_pointers(pointers),
   m_subscript(subs),
   m_isFuncPtr(isFunc)
{
   fassert( ct->m_id == BaseCType::e_struct || ct->m_id == BaseCType::e_union || ct->m_id == BaseCType::e_enum );
}


CType::CType( const CType& other ):
   m_ctype( other.m_ctype ),
   m_tag(other.m_tag),
   m_pointers(other.m_pointers),
   m_subscript(other.m_subscript),
   m_isFuncPtr(other.m_isFuncPtr),
   m_funcParams(other.m_funcParams)
{
}

CType::~CType()
{}

//=======================================================
//

Parameter::Parameter( const CType& t, const char* name ):
   m_type(t),
   m_name(name),
   m_next(0)
{
}

//=======================================================
//

ParamList::ParamList():
   m_head(0),
   m_tail(0),
   m_size(0)
   {}

ParamList::ParamList( const ParamList& other ):
   m_head(0),
   m_tail(0),
   m_size(0)
{
   Parameter* p = other.m_head;
   while (p != 0 )
   {
      add( new Parameter(*p) );
      p = p->m_next;
   }
}

ParamList::~ParamList()
{
   Parameter* p = m_head;
   while ( p != 0 )
   {
      Parameter* old = p;
      p = p->m_next;
      delete old;
   }
}

void ParamList::add(Parameter* p)
{
   if ( m_head == 0 )
   {
      m_head = m_tail = p;
   }
   else
   {
      m_tail->m_next = p;
      m_tail = p;
   }
   m_size++;
   p->m_next = 0; // just to be on the bright side.
}

//===================================================
// Forming parameter
//===================================================

class FormingParam: public BaseAlloc
{
public:
   
   class state
   {
   public:
      bool m_bTypeComplete;
      bool m_bGettingTag;
      bool m_bOpenSub;
      int m_nPointers;
      int m_nSubscripts;
      String m_tag;
      String m_name;

      FormingParam* nextState;
      Parameter* complete;
      
      state():
         m_bTypeComplete(false),
         m_bGettingTag(false),
         m_bOpenSub(false),
         m_nPointers(0),
         m_nSubscripts(0),
         nextState( this ),
         complete( 0 )
      {}
   };

   FormingParam( BaseCType::e_integral_type t ):
      m_type( t )
   {}
   
   virtual Parameter* parseNext( const String &next, state& st ) = 0;

protected:
   BaseCType::e_integral_type m_type;
   bool parseCommon( const String &next, state& st );
};

class cFP_start: public FormingParam
{
public:
   cFP_start( ):
      FormingParam( BaseCType::e_void )
      {}

   virtual Parameter* parseNext( const String &next, state& st );
} FP_start;

class cFP_unsigned: public FormingParam
{
public:
   cFP_unsigned( ):
      FormingParam( BaseCType::e_unsigned_int )
      {}

   virtual Parameter* parseNext( const String &next, state& st );
} FP_unsigned;

class cFP_signed: public FormingParam
{
public:
   cFP_signed():
      FormingParam( BaseCType::e_void )
      {}

   virtual Parameter* parseNext( const String &next, state& st );
} FP_signed;

class cFP_char: public FormingParam
{
public:
   cFP_char():
      FormingParam( BaseCType::e_char )
      {}

   virtual Parameter* parseNext( const String &next, state& st );
} FP_char;

class cFP_unsigned_char: public FormingParam
{
public:
   cFP_unsigned_char():
      FormingParam( BaseCType::e_unsigned_char )
      {}

   virtual Parameter* parseNext( const String &next, state& st );
} FP_unsigned_char;

class cFP_signed_char: public FormingParam
{
public:
   cFP_signed_char():
      FormingParam( BaseCType::e_signed_char )
      {}

   virtual Parameter* parseNext( const String &next, state& st );
} FP_signed_char;

class cFP_short: public FormingParam
{
public:
   cFP_short():
      FormingParam( BaseCType::e_short )
      {}

   virtual Parameter* parseNext( const String &next, state& st );
} FP_short;

class cFP_unsigned_short: public FormingParam
{
public:
   cFP_unsigned_short():
      FormingParam( BaseCType::e_unsigned_short )
      {}

   virtual Parameter* parseNext( const String &next, state& st );
} FP_unsigned_short;

class cFP_int: public FormingParam
{
public:
   cFP_int():
      FormingParam( BaseCType::e_unsigned_short )
      {}

   virtual Parameter* parseNext( const String &next, state& st );
} FP_int;

class cFP_unsigned_int: public FormingParam
{
public:
   cFP_unsigned_int():
      FormingParam( BaseCType::e_unsigned_int )
      {}

   virtual Parameter* parseNext( const String &next, state& st );
} FP_unsigned_int;

class cFP_long: public FormingParam
{
public:
   cFP_long():
      FormingParam( BaseCType::e_long )
      {}

   virtual Parameter* parseNext( const String &next, state& st );
} FP_long;

class cFP_unsigned_long: public FormingParam
{
public:
   cFP_unsigned_long():
      FormingParam( BaseCType::e_unsigned_long )
      {}

   virtual Parameter* parseNext( const String &next, state& st );
} FP_unsigned_long;

class cFP_long_long: public FormingParam
{
public:
   cFP_long_long():
      FormingParam( BaseCType::e_long_long )
      {}

   virtual Parameter* parseNext( const String &next, state& st );
} FP_long_long;

class cFP_unsigned_long_long: public FormingParam
{
public:
   cFP_unsigned_long_long():
      FormingParam( BaseCType::e_unsigned_long_long )
      {}

   virtual Parameter* parseNext( const String &next, state& st );
} FP_unsigned_long_long;

class cFP_float: public FormingParam
{
public:
   cFP_float():
      FormingParam( BaseCType::e_float )
      {}

   virtual Parameter* parseNext( const String &next, state& st );
} FP_float;

class cFP_double: public FormingParam
{
public:
   cFP_double():
      FormingParam( BaseCType::e_double )
      {}

   virtual Parameter* parseNext( const String &next, state& st );
} FP_double;

class cFP_long_double: public FormingParam
{
public:
   cFP_long_double():
      FormingParam( BaseCType::e_long_double )
      {}

   virtual Parameter* parseNext( const String &next, state& st );
} FP_long_double;


class cFP_void: public FormingParam
{
public:
   cFP_void( ):
      FormingParam( BaseCType::e_void )
      {}

   virtual Parameter* parseNext( const String &next, state& st );
} FP_void;


class cFP_struct: public FormingParam
{
public:
   cFP_struct( ):
      FormingParam( BaseCType::e_struct )
      {}

   virtual Parameter* parseNext( const String &next, state& st );
} FP_struct;

class cFP_union: public FormingParam
{
public:
   cFP_union( ):
      FormingParam( BaseCType::e_union )
      {}

   virtual Parameter* parseNext( const String &next, state& st );
} FP_union;

class cFP_enum: public FormingParam
{
public:
   cFP_enum( ):
      FormingParam( BaseCType::e_enum )
      {}

   virtual Parameter* parseNext( const String &next, state& st );
} FP_enum;


bool FormingParam::parseCommon( const String &next, FormingParam::state& st )
{
   return true;
}

Parameter* cFP_start::parseNext( const String &next, FormingParam::state& st )
{
   if ( next == "signed" )
     st.nextState = &FP_signed;

   return 0;
}

Parameter* cFP_unsigned::parseNext( const String &next, FormingParam::state& st )
{
   return 0;
}

Parameter* cFP_signed::parseNext( const String &next, FormingParam::state& st )
{
   return 0;
}

Parameter* cFP_char::parseNext( const String &next, FormingParam::state& st )
{
   return 0;
}

Parameter* cFP_unsigned_char::parseNext( const String &next, FormingParam::state& st )
{
   return 0;
}

Parameter* cFP_signed_char::parseNext( const String &next, FormingParam::state& st )
{
   return 0;
}


Parameter* cFP_short::parseNext( const String &next, FormingParam::state& st )
{
   return 0;
}

Parameter* cFP_unsigned_short::parseNext( const String &next, FormingParam::state& st )
{
   return 0;
}

Parameter* cFP_int::parseNext( const String &next, FormingParam::state& st )
{
   return 0;
}

Parameter* cFP_unsigned_int::parseNext( const String &next, FormingParam::state& st )
{
   return 0;
}

Parameter* cFP_long::parseNext( const String &next, FormingParam::state& st )
{
   return 0;
}

Parameter* cFP_unsigned_long::parseNext( const String &next, FormingParam::state& st )
{
   return 0;
}

Parameter* cFP_long_long::parseNext( const String &next, FormingParam::state& st )
{
   return 0;
}

Parameter* cFP_unsigned_long_long::parseNext( const String &next, FormingParam::state& st )
{
   return 0;
}

Parameter* cFP_float::parseNext( const String &next, FormingParam::state& st )
{
   return 0;
}

Parameter* cFP_double::parseNext( const String &next, FormingParam::state& st )
{
   return 0;
}

Parameter* cFP_long_double::parseNext( const String &next, FormingParam::state& st )
{
   return 0;
}

Parameter* cFP_void::parseNext( const String &next, FormingParam::state& st )
{
   return 0;
}

Parameter* cFP_struct::parseNext( const String &next, FormingParam::state& st )
{
   return 0;
}

Parameter* cFP_union::parseNext( const String &next, FormingParam::state& st )
{
   return 0;
}

Parameter* cFP_enum::parseNext( const String &next, FormingParam::state& st )
{
   return 0;
}



//===================================================
// Function Definition
//===================================================

FunctionDef2::FunctionDef2( const FunctionDef2& other ):
   m_definition( other.m_definition ),
   m_name( other.m_name ),
   m_params( other.m_params )
{
   if ( other.m_return != 0 )
      m_return = new Parameter( *other.m_return );
   else
      m_return = 0;
}

FunctionDef2::~FunctionDef2()
{
   delete m_return;
}

bool FunctionDef2::parse( const String& definition )
{
   return false;
}

}
