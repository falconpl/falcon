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
#include <falcon/error.h>

#include "dynlib_parser.h"

namespace Falcon
{


Parameter::Parameter( Parameter::e_integral_type ct, const String& name, int pointers, int subs, bool isFunc ):
   m_type(ct),
   m_name( name ),
   m_pointers(pointers),
   m_subscript(subs),
   m_isFuncPtr(isFunc)
{}


Parameter::Parameter( Parameter::e_integral_type ct, const String& name, const String &tag, int pointers, int subs, bool isFunc ):
   m_type(ct),
   m_name( name ),
   m_tag(tag),
   m_pointers(pointers),
   m_subscript(subs),
   m_isFuncPtr(isFunc)
{
}


Parameter::Parameter( const Parameter& other ):
   m_type( other.m_type ),
   m_name( other.m_name ),
   m_tag(other.m_tag),
   m_pointers(other.m_pointers),
   m_subscript(other.m_subscript),
   m_isFuncPtr(other.m_isFuncPtr),
   m_funcParams(other.m_funcParams)
{
}

Parameter::~Parameter()
{}


String Parameter::toString() const
{
   String ret = typeToString( m_type );

   if( m_type == e_struct || m_type == e_union || m_type == e_enum )
   {
      ret += " ";
      ret += m_tag;
   }

   for( int i = 0; i < m_pointers; ++i )
      ret += "*";

   if( m_isFuncPtr )
   {
      ret += "(";
      ret += m_name;
      ret += "*)(";
      ret += m_funcParams.toString();
      ret += ")";
   }
   else
   {
      ret += " ";
      ret += m_name;
   }

   if( m_subscript == -1 )
   {
      ret += "[]";
   }
   else if ( m_subscript > 0 )
   {
      ret += "[";
      ret.N( m_subscript );
      ret += "]";
   }

   return ret;
}


String Parameter::typeToString( Parameter::e_integral_type type )
{
   switch( type )
   {
   case e_void: return "void"; break;
   case e_char: return "char"; break;
   case e_unsigned_char: return "unsigned char"; break;
   case e_signed_char: return "signed char"; break;
   case e_short: return "short"; break;
   case e_unsigned_short: return "unsigned short"; break;
   case e_int: return "int"; break;
   case e_unsigned_int: return "unsigned int"; break;
   case e_long: return "long"; break;
   case e_unsigned_long: return "unsigned long"; break;
   case e_long_long: return "long long"; break;
   case e_unsigned_long_long: return "unsigned long long"; break;
   case e_float: return "float"; break;
   case e_double: return "double"; break;
   case e_long_double: return "long double"; break;
   case e_struct: return "struct"; break;
   case e_union: return "union"; break;
   case e_enum: return "enum"; break;
   case e_varpar: return "..."; break;
   }
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


String ParamList::toString() const
{
   String ret;

   Parameter* child = first();
   while( child != 0 )
   {
      ret += child->toString();
      child = child->m_next;
      if ( child != 0 )
         ret += ", ";
   }

   return ret;
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
      int m_nPointers;
      int m_nSubscripts;
      Parameter::e_integral_type m_type;
      String m_tag;
      String m_name;
      FormingParam* nextState;
      Parameter* m_forming;
      
      state():
         m_nPointers(0),
         m_nSubscripts(0),
         m_type( Parameter::e_void ),
         nextState( 0 ),
         m_forming(0)
      {}

      ~state()
      {
         // this ensures we don't have leaks at throws.
         delete m_forming;
      }
   };

   FormingParam()
   {}
   
   virtual Parameter* parseNext( const String &next, state& st ) = 0;

};

class cFP_start: public FormingParam
{
public:
   virtual Parameter* parseNext( const String &next, state& st );
} FP_start;

class cFP_unsigned: public FormingParam
{
public:
   virtual Parameter* parseNext( const String &next, state& st );
} FP_unsigned;

class cFP_signed: public FormingParam
{
public:
   virtual Parameter* parseNext( const String &next, state& st );
} FP_signed;

class cFP_char: public FormingParam
{
public:
   virtual Parameter* parseNext( const String &next, state& st );
} FP_char;

class cFP_unsigned_char: public FormingParam
{
public:
   virtual Parameter* parseNext( const String &next, state& st );
} FP_unsigned_char;

class cFP_signed_char: public FormingParam
{
public:
   virtual Parameter* parseNext( const String &next, state& st );
} FP_signed_char;

class cFP_short: public FormingParam
{
public:
   virtual Parameter* parseNext( const String &next, state& st );
} FP_short;

class cFP_unsigned_short: public FormingParam
{
public:
   virtual Parameter* parseNext( const String &next, state& st );
} FP_unsigned_short;

class cFP_int: public FormingParam
{
public:
   virtual Parameter* parseNext( const String &next, state& st );
} FP_int;

class cFP_unsigned_int: public FormingParam
{
public:
   virtual Parameter* parseNext( const String &next, state& st );
} FP_unsigned_int;

class cFP_long: public FormingParam
{
public:
   virtual Parameter* parseNext( const String &next, state& st );
} FP_long;

class cFP_unsigned_long: public FormingParam
{
public:
   virtual Parameter* parseNext( const String &next, state& st );
} FP_unsigned_long;

class cFP_long_long: public FormingParam
{
public:
   virtual Parameter* parseNext( const String &next, state& st );
} FP_long_long;

class cFP_unsigned_long_long: public FormingParam
{
public:
   virtual Parameter* parseNext( const String &next, state& st );
} FP_unsigned_long_long;

class cFP_float: public FormingParam
{
public:
   virtual Parameter* parseNext( const String &next, state& st );
} FP_float;

class cFP_double: public FormingParam
{
public:
   virtual Parameter* parseNext( const String &next, state& st );
} FP_double;

class cFP_long_double: public FormingParam
{
public:
   virtual Parameter* parseNext( const String &next, state& st );
} FP_long_double;


class cFP_void: public FormingParam
{
public:
   virtual Parameter* parseNext( const String &next, state& st );
} FP_void;


class cFP_struct: public FormingParam
{
public:
   virtual Parameter* parseNext( const String &next, state& st );
} FP_struct;

class cFP_union: public FormingParam
{
public:
   virtual Parameter* parseNext( const String &next, state& st );
} FP_union;

class cFP_enum: public FormingParam
{
public:
   virtual Parameter* parseNext( const String &next, state& st );
} FP_enum;

class cFP_tagdef: public FormingParam
{
public:
   virtual Parameter* parseNext( const String &next, state& st );
} FP_tagdef;

class cFP_varpar: public FormingParam
{
public:
   virtual Parameter* parseNext( const String &next, state& st );
} FP_varpar;


class cFP_paramname: public FormingParam
{
public:
   virtual Parameter* parseNext( const String &next, state& st );
} FP_paramname;

class cFP_postname: public FormingParam
{
public:
   virtual Parameter* parseNext( const String &next, state& st );
} FP_postname;

class cFP_subscript: public FormingParam
{
public:
   virtual Parameter* parseNext( const String &next, state& st );
} FP_subscript;

class cFP_postsubscript: public FormingParam
{
public:
   virtual Parameter* parseNext( const String &next, state& st );
} FP_postsubscript;

class cFP_paramcomplete: public FormingParam
{
public:
   virtual Parameter* parseNext( const String &next, state& st );
} FP_paramcomplete;

class cFP_funcdecl: public FormingParam
{
public:
   virtual Parameter* parseNext( const String &next, state& st );
} FP_funcdecl;

class cFP_funcdecl_1: public FormingParam
{
public:
   virtual Parameter* parseNext( const String &next, state& st );
} FP_funcdecl_1;

class cFP_funcdecl_2: public FormingParam
{
public:
   virtual Parameter* parseNext( const String &next, state& st );
} FP_funcdecl_2;

inline void tpe( int l )
{
   throw new ParseError( ErrorParam( e_syntax, l ) );
}

inline bool isKeyword( const String& next )
{
   return
      next == "void" ||
      next == "signed" ||
      next == "unsigned" ||
      next == "char" ||
      next == "short" ||
      next == "int" ||
      next == "long" ||
      next == "float" ||
      next == "double" ||
      next == "struct" ||
      next == "union" ||
      next == "enum" ||

      next == "WORD" ||
      next == "DWORD" ||
      next == "HANDLE" ||
      next == "__int64" ||

      next == "...";
}

inline bool isPuntaction( const String& next )
{
   return
      next == "." ||
      next == "," ||
      next == "(" ||
      next == ")" ||
      next == "*";
}

inline bool isKwordOrPunc( const String& next )
{
   return isKeyword( next ) || isPuntaction( next );
}


inline void parseTerminal( const String& next, FormingParam::state& st )
{
   if ( next == "*" )
   {
      st.m_nPointers = 1;
      st.nextState = &FP_paramname;
   }
   else if ( next == "(" )
   {
      // it's a function declaration.
      st.nextState = &FP_funcdecl;
   }
   else if ( isKwordOrPunc(next) )
      tpe( __LINE__ );
   else
   {
      st.m_name = next;
      st.nextState = &FP_postname;
   }
}


Parameter* cFP_start::parseNext( const String &next, FormingParam::state& st )
{
   if ( next == "signed" )
   {
      st.nextState = &FP_signed;
   }
   else if ( next == "void" )
   {
      st.nextState = &FP_void;
   }
   else if ( next == "unsigned" )
   {
      st.nextState = &FP_unsigned;
   }
   else if ( next == "char" )
   {
      st.nextState = &FP_char;
   }
   else if ( next == "short" )
   {
      st.nextState = &FP_short;
   }
   else if ( next == "int" )
   {
      st.nextState = &FP_int;
   }
   else if ( next == "long" )
   {
      st.nextState = &FP_long;
   }
   else if ( next == "float" )
   {
      st.nextState = &FP_float;
   }
   else if ( next == "double" )
   {
      st.nextState = &FP_double;
   }
   else if ( next == "struct" )
   {
      st.nextState = &FP_struct;
   }
   else if ( next == "union" )
   {
      st.nextState = &FP_union;
   }
   else if ( next == "enum" )
   {
      st.nextState = &FP_enum;
   }
   else if ( next == "..." )
   {
      st.nextState = &FP_varpar;
   }
   else if ( next == "WORD" )
   {
      st.nextState = &FP_unsigned_short;
   }
   else if ( next == "DWORD" )
   {
      st.nextState = &FP_unsigned_int;
   }
   else if ( next == "HANDLE" )
   {
      st.nextState = &FP_unsigned_long;
   }
   else if ( next == "__int64" )
   {
      st.nextState = &FP_long_long;
   }
   else
      tpe( __LINE__ );

   return 0;
}

Parameter* cFP_unsigned::parseNext( const String &next, FormingParam::state& st )
{
   // tentative default type
   st.m_type = Parameter::e_unsigned_int;

   if ( next == "char" )
      st.nextState = &FP_unsigned_char;
   else if ( next == "int" )
      st.nextState = &FP_unsigned_int;
   else if ( next == "short" )
      st.nextState = &FP_unsigned_short;
   else if ( next == "long" )
      st.nextState = &FP_unsigned_long;
   else
   {
      parseTerminal( next, st );
   }

   return 0;
}

Parameter* cFP_signed::parseNext( const String &next, FormingParam::state& st )
{
   // tentative default type
   st.m_type = Parameter::e_int;

   if ( next == "char" )
      st.nextState = &FP_signed_char;
   else if ( next == "int" )
      st.nextState = &FP_int;
   else if ( next == "short" )
      st.nextState = &FP_short;
   else if ( next == "long" )
      st.nextState = &FP_long;
   else
   {
       parseTerminal( next, st );
   }

   return 0;
}

Parameter* cFP_char::parseNext( const String &next, FormingParam::state& st )
{
   // tentative default type
   st.m_type = Parameter::e_char;
   parseTerminal( next, st );

   return 0;
}

Parameter* cFP_unsigned_char::parseNext( const String &next, FormingParam::state& st )
{
   // tentative default type
   st.m_type = Parameter::e_unsigned_char;
   parseTerminal( next, st );

   return 0;
}

Parameter* cFP_signed_char::parseNext( const String &next, FormingParam::state& st )
{
   // tentative default type
   st.m_type = Parameter::e_signed_char;
   parseTerminal( next, st );

   return 0;
}


Parameter* cFP_short::parseNext( const String &next, FormingParam::state& st )
{
   // tentative default type
   st.m_type = Parameter::e_short;

   if( next == "int" )
   {
      // stay in this state.
      return 0;
   }

   parseTerminal( next, st );

   return 0;
}

Parameter* cFP_unsigned_short::parseNext( const String &next, FormingParam::state& st )
{
   // tentative default type
   st.m_type = Parameter::e_unsigned_short;

   if( next == "int" )
   {
      // stay in this state.
      return 0;
   }

   parseTerminal( next, st );

   return 0;
}

Parameter* cFP_int::parseNext( const String &next, FormingParam::state& st )
{
   st.m_type = Parameter::e_int;
   parseTerminal( next, st );
   return 0;
}

Parameter* cFP_unsigned_int::parseNext( const String &next, FormingParam::state& st )
{
   st.m_type = Parameter::e_unsigned_int;
   parseTerminal( next, st );

   return 0;
}

Parameter* cFP_long::parseNext( const String &next, FormingParam::state& st )
{
   st.m_type = Parameter::e_long;
   if( next == "long" )
   {
      st.nextState = &FP_long_long;
   }
   else if (next == "double" )
   {
      st.nextState = &FP_long_double;
   }
   else if (next == "int" )
   {
      // stay here
      return 0;
   }
   else
   {
      parseTerminal( next, st );
   }

   return 0;
}

Parameter* cFP_unsigned_long::parseNext( const String &next, FormingParam::state& st )
{
   st.m_type = Parameter::e_unsigned_long;
   if( next == "long" )
   {
      st.nextState = &FP_unsigned_long_long;
   }
   else if (next == "int" )
   {
      // stay here
      return 0;
   }
   else
   {
      parseTerminal( next, st );
   }

   return 0;
}

Parameter* cFP_long_long::parseNext( const String &next, FormingParam::state& st )
{
   st.m_type = Parameter::e_long_long;
   if (next == "int" )
   {
      // stay here
      return 0;
   }
   else
   {
      parseTerminal( next, st );
   }

   return 0;
}

Parameter* cFP_unsigned_long_long::parseNext( const String &next, FormingParam::state& st )
{
   st.m_type = Parameter::e_unsigned_long_long;
   if (next == "int" )
   {
      // stay here
      return 0;
   }
   else
   {
      parseTerminal( next, st );
   }

   return 0;
}

Parameter* cFP_float::parseNext( const String &next, FormingParam::state& st )
{
   st.m_type = Parameter::e_float;
   parseTerminal( next, st );

   return 0;
}

Parameter* cFP_double::parseNext( const String &next, FormingParam::state& st )
{
   st.m_type = Parameter::e_double;
   parseTerminal( next, st );
   return 0;
}

Parameter* cFP_long_double::parseNext( const String &next, FormingParam::state& st )
{
   st.m_type = Parameter::e_long_double;
   parseTerminal( next, st );
   return 0;
}

Parameter* cFP_void::parseNext( const String &next, FormingParam::state& st )
{
   st.m_type = Parameter::e_void;
   // only in void, accept ")" that can be used to declare no parameter.
   if( next == ")" )
      return 0;

   parseTerminal( next, st );
   return 0;
}

Parameter* cFP_struct::parseNext( const String &next, FormingParam::state& st )
{
   st.m_type = Parameter::e_struct;
   st.nextState = &FP_tagdef;
   return 0;
}

Parameter* cFP_union::parseNext( const String &next, FormingParam::state& st )
{
   st.m_type = Parameter::e_union;
   st.nextState = &FP_tagdef;

   return 0;
}

Parameter* cFP_enum::parseNext( const String &next, FormingParam::state& st )
{
   st.m_type = Parameter::e_enum;
   st.nextState = &FP_tagdef;

   return 0;
}

Parameter* cFP_tagdef::parseNext( const String &next, FormingParam::state& st )
{
   if( isKwordOrPunc( next ) )
      tpe( __LINE__ );

   st.m_tag = next;
   st.nextState = &FP_paramname;

   return 0;
}

Parameter* cFP_paramname::parseNext( const String &next, state& st )
{
   if( next == "*" )
   {
      st.m_nPointers++;
   }
   else if ( isKwordOrPunc(next) )
   {
      tpe( __LINE__ );
   }
   else
   {
      st.m_name = next;
      st.nextState = &FP_postname;
   }

   return 0;
}

Parameter* cFP_postname::parseNext( const String &next, state& st )
{
   if( next == "," || next == ")" )
   {
      if( st.m_forming )
      {
         Parameter* p = st.m_forming;
         st.m_forming = 0;
         return p;
      }

      // we're done...
      return new Parameter( st.m_type, st.m_name, st.m_tag, st.m_nPointers );
   }
   else if ( next == "[" )
   {
      st.nextState = &FP_subscript;
   }
   else
      tpe( __LINE__ );

   return 0;
}

Parameter* cFP_subscript::parseNext( const String &next, state& st )
{
   int64 subCount;

   if( next == "]" )
   {
      st.m_nSubscripts = -1;
      st.nextState = &FP_paramcomplete;
   }
   else if ( next.parseInt( subCount ) )
   {
      st.m_nSubscripts = (int) subCount;
      st.nextState = &FP_postsubscript;
   }
   else
      tpe( __LINE__ );

   return 0;
}

Parameter* cFP_postsubscript::parseNext( const String &next, state& st )
{
   int64 subCount;

   if( next == "]" )
   {
      st.nextState = &FP_paramcomplete;
   }
   else
      tpe( __LINE__ );

   return 0;
}


Parameter* cFP_paramcomplete::parseNext( const String &next, state& st )
{
   if( next == ")" || next == "," )
   {
      if( st.m_forming )
      {
         Parameter* p = st.m_forming;
         st.m_forming = 0;
         p->m_subscript = st.m_nSubscripts;
         return p;
      }

      // we're done...
      return new Parameter( st.m_type, st.m_name, st.m_tag, st.m_nPointers, st.m_nSubscripts );
   }
   else
      tpe( __LINE__ );

   return 0; // just to make the compiler happy
}

Parameter* cFP_varpar::parseNext( const String &next, state& st )
{
   if( next == ")" || next == "," )
   {
      // we're done... and can't be forming
      return new Parameter( st.m_type, st.m_name, st.m_tag );
   }
   else
      tpe( __LINE__ );

   return 0; // just to make the compiler happy
}



Parameter* cFP_funcdecl::parseNext( const String &next, state& st )
{
   // TODO: Should we accept also void (funcname)(...) ?

   if( next != "*" )
      tpe( __LINE__ );
   else
   {
      st.nextState = &FP_funcdecl_1;
   }

   return 0;
}


Parameter* cFP_funcdecl_1::parseNext( const String &next, state& st )
{
   if( isKwordOrPunc(next) )
      tpe( __LINE__ );
   else
   {
      st.m_name = next;
      st.nextState = &FP_funcdecl_2;
   }
   return 0;
}

Parameter* cFP_funcdecl_2::parseNext( const String &next, state& st )
{
   if( next != ")" )
      tpe( __LINE__ );
   else
   {
      // seeing the "forming parameter" non-empty, the upper parser will
      // start a sub-parameter list loop.
      st.m_forming = new Parameter( st.m_type, st.m_name, st.m_tag, st.m_nPointers, 0, true );
      // prepare as we should be called after the sub-loop, that is, as post name.
      st.nextState = &FP_postname;
   }

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
   Tokenizer tok(TokenizerParams().wsIsToken().returnSep(), "();[],*");
   tok.parse( definition );
   if ( ! tok.hasCurrent() )
   {
      return false;
   }

   // parse the outer return type; as "(" is found, we get a formingParam in state,
   // and we know we can start the parameter loop
   m_return = parseNextParam( tok, true );

   if ( tok.getToken() != "(" )
   {
      throw new ParseError( ErrorParam( e_syntax, __LINE__ ) );
   }

   // if we didn't throw, it means we have a return value.
   // the return will have our name.
   m_name = m_return->m_name;

   // now we can process the comma separated parameters.
   tok.next();
   if( tok.getToken() == ")" )
   {
      // we're done.
      return true;
   }

   // parse the main paramter list
   parseFuncParams( m_params, tok );

   return true;
}

Parameter* FunctionDef2::parseNextParam( Tokenizer& tok, bool isFuncName )
{
   FormingParam::state state;
   Parameter *ret = 0;

   state.nextState = &FP_start;

   while( tok.hasCurrent() )
   {
      String next = tok.getToken();
      ret = state.nextState->parseNext( next, state );

      if ( state.m_forming != 0 )
      {
         // ok, we should parse the parameter list of this parameter.
         if( tok.next() )
         {
            // this is to skip the initial "(".
            if ( tok.getToken() != "(" || ! tok.next() )
               tpe( __LINE__ );

            parseFuncParams( state.m_forming->m_funcParams, tok );
         }
      }
      else if ( ret != 0 )
      {
         return ret;
      }

      tok.next();
      if( isFuncName && tok.getToken() == "(" )
      {
         return new Parameter( state.m_type, state.m_name, state.m_tag, state.m_nPointers );
      }
   }

   return ret;
}


void FunctionDef2::parseFuncParams( ParamList& params, Tokenizer& tok )
{
   while( tok.hasCurrent() && tok.getToken() != ")" )
   {
      Parameter* p = parseNextParam(tok);
      if( tok.hasCurrent() )
      {
         if (tok.getToken() == ","  )
         {
            params.add(p);
            tok.next();
            continue;
         }

         if( tok.getToken() == ")" )
         {
            // we're done
            params.add(p);
            return;
         }
      }

      // we must be filtered before reaching here.
      throw new ParseError( ErrorParam( e_syntax, __LINE__ ) );
   }
}


String FunctionDef2::toString() const
{
   if ( m_return == 0 )
   {
      return m_name;
   }

   String ret = Parameter::typeToString( m_return->m_type );
   for ( int i = 0; i < m_return->m_pointers; ++i )
      ret += "*";

   ret += " " + m_name + "(" + m_params.toString() + ")";
   return ret;
}

}
