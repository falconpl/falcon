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
#include <falcon/vm.h>

#include "dynlib_mod.h"
#include "dynlib_ext.h"
#include "dynlib_st.h"

namespace Falcon
{
//===================================================
// Forming parameter
// Inner private class that's not visible outside.
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
      bool m_bConst;
      bool m_bFunc;
      FormingParam* nextState;
      Parameter* m_forming;

      state():
         m_nPointers(0),
         m_nSubscripts(0),
         m_type( Parameter::e_void ),
         m_bConst(false),
         m_bFunc(false),
         nextState( 0 ),
         m_forming(0)
      {}

      ~state()
      {
         // this ensures we don't have leaks at throws.
         delete m_forming;
      }

      Parameter* makeParameter()
      {
         return new Parameter( m_type, m_bConst, m_name, m_tag, m_nPointers, m_nSubscripts, m_bFunc );
      }
   };

   FormingParam()
   {}

   virtual Parameter* parseNext( const String &next, state& st ) = 0;

};

//===================================================
// Inline utilities for this module.
//===================================================

inline void tpe( int l )
{
   throw new ParseError(
         ErrorParam( FALCON_DYNLIB_ERROR_BASE+5, l )
         .desc( *VMachine::getCurrent()->currentModule()->getString(
               dyl_invalid_syn ) )
         );
}

inline bool isKeyword( const String& next )
{
   return
      next == "const" ||
      next == "void" ||
      next == "signed" ||
      next == "unsigned" ||
      next == "char" ||
      next == "wchar_t" ||
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

//===================================================
// Forming parameter declarations
//===================================================

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

class cFP_wchar_t: public FormingParam
{
public:
   virtual Parameter* parseNext( const String &next, state& st );
} FP_wchar_t;

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

//==================================================
// Forming parameter implementations
//==================================================


Parameter* cFP_start::parseNext( const String &next, FormingParam::state& st )
{
   if ( next == "const" )
   {
      if( st.m_bConst )
         tpe( __LINE__ );

      st.m_bConst = true;

   }
   else if ( next == "signed" )
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
   else if ( next == "wchar_t" )
   {
      st.nextState = &FP_wchar_t;
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
      st.m_type = Parameter::e_struct;
      st.nextState = &FP_tagdef;
   }
   else if ( next == "union" )
   {
      st.m_type = Parameter::e_union;
      st.nextState = &FP_tagdef;
   }
   else if ( next == "enum" )
   {
      st.m_type = Parameter::e_enum;
      st.nextState = &FP_tagdef;

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

Parameter* cFP_wchar_t::parseNext( const String &next, FormingParam::state& st )
{
   // tentative default type
   st.m_type = Parameter::e_wchar_t;
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
      return st.makeParameter();
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
      return st.makeParameter();
   }
   else
      tpe( __LINE__ );

   return 0; // just to make the compiler happy
}

Parameter* cFP_varpar::parseNext( const String &next, state& st )
{
   if( next == ")" )
   {
      // the only thing that can't be const
      if( st.m_bConst )
      {
         tpe( __LINE__ );
      }

      // we're done... and can't be forming
      return new Parameter( Parameter::e_varpar, false, "" );
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
      st.m_bFunc = true;
      // seeing the "forming parameter" non-empty, the upper parser will
      // start a sub-parameter list loop.
      st.m_forming = st.makeParameter();

      // prepare as we should be called after the sub-loop, that is, as post name.
      st.nextState = &FP_postname;
   }

   return 0;
}

//=================================================
// Part of the FunctionDef using the private parser
//

Parameter* FunctionDef::parseNextParam( Tokenizer& tok, bool isFuncName )
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
         return state.makeParameter();
      }
   }

   return ret;
}

}
