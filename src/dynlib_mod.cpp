/*
   The Falcon Programming Language
   FILE: dynlib_mod.cpp

   Direct dynamic library interface for Falcon
   Interface extension functions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 28 Oct 2008 22:23:29 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: The Falcon Committee

   See the LICENSE file distributed with this package for licensing details.
*/

/** \file
   Direct dynamic library interface for Falcon
   Internal logic functions - implementation.
*/

#include <falcon/fassert.h>
#include <falcon/tokenizer.h>
#include <falcon/error.h>
#include <falcon/membuf.h>
#include <falcon/vm.h>

#include <string.h>

#include "dynlib_mod.h"
#include "dynlib_ext.h"
#include "dynlib_st.h"

namespace Falcon {



Parameter::Parameter( Parameter::e_integral_type ct, bool bConst, const String& name, int pointers, int subs, bool isFunc ):
   m_type(ct),
   m_name( name ),
   m_bConst( bConst ),
   m_pointers(pointers),
   m_subscript(subs),
   m_isFuncPtr(isFunc)
{}


Parameter::Parameter( Parameter::e_integral_type ct, bool bConst, const String& name, const String &tag, int pointers, int subs, bool isFunc ):
   m_type(ct),
   m_name( name ),
   m_tag(tag),
   m_bConst( bConst ),
   m_pointers(pointers),
   m_subscript(subs),
   m_isFuncPtr(isFunc)
{
}


Parameter::Parameter( const Parameter& other ):
   m_type( other.m_type ),
   m_name( other.m_name ),
   m_tag(other.m_tag),
   m_bConst( other.m_bConst),
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
   String ret;
   if( m_bConst )
      ret = "const ";

   ret += typeToString( m_type );

   if( m_type == e_varpar )
      return ret;

   if( m_tag.size() != 0 )
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
   String ret;

   switch( type )
   {
   case e_void: ret += "void"; break;
   case e_char: ret += "char"; break;
   case e_wchar_t: ret += "wchar_t"; break;
   case e_unsigned_char: ret += "unsigned char"; break;
   case e_signed_char: ret += "signed char"; break;
   case e_short: ret += "short"; break;
   case e_unsigned_short: ret += "unsigned short"; break;
   case e_int: ret += "int"; break;
   case e_unsigned_int: ret += "unsigned int"; break;
   case e_long: ret += "long"; break;
   case e_unsigned_long: ret += "unsigned long"; break;
   case e_long_long: ret += "long long"; break;
   case e_unsigned_long_long: ret += "unsigned long long"; break;
   case e_float: ret += "float"; break;
   case e_double: ret += "double"; break;
   case e_long_double: ret += "long double"; break;
   case e_struct: ret += "struct"; break;
   case e_union: ret += "union"; break;
   case e_enum: ret += "enum"; break;
   case e_varpar: ret += "..."; break;
   }

   return ret;
}

//=======================================================
//

ParamList::ParamList():
   m_head(0),
   m_tail(0),
   m_size(0),
   m_bVaradic(false)
   {}

ParamList::ParamList( const ParamList& other ):
   m_head(0),
   m_tail(0),
   m_size(0),
   m_bVaradic(false)
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

   if ( p->m_type == Parameter::e_varpar )
   {
      m_bVaradic = true;
   }
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
// Function Definition
//===================================================

FunctionDef::FunctionDef( const String& definition ):
      m_return(0),
      m_fAddress(0),
      m_fDeletor(0)
{
   try {
      parse( definition );
   }
   catch(...)
   {
      delete m_return;
      throw;
   }
}


FunctionDef::FunctionDef( const String& definition, const String& deletor ):
   m_return(0),
   m_fAddress(0),
   m_fDeletor(0)
{
   try {
      parse( definition );
      if( m_return == 0 || m_return->indirections() == 0 ||
         (m_return->m_type != Parameter::e_union && m_return->m_type != Parameter::e_struct)
         )
      {
         throw new ParseError( ErrorParam( FALCON_DYNLIB_ERROR_BASE+7, __LINE__ )
               .desc( *VMachine::getCurrent()->currentModule()->getString(
                  dyl_undeletable ) ) );
      }
      parseDeletor( deletor );
   }
   catch(...)
   {
      delete m_return;
      throw;
   }
}


FunctionDef::FunctionDef( const FunctionDef& other ):
   m_definition( other.m_definition ),
   m_name( other.m_name ),
   m_params( other.m_params ),
   m_fAddress( other.m_fAddress ),
   m_fDeletor( other.m_fDeletor )
{
   if ( other.m_return != 0 )
      m_return = new Parameter( *other.m_return );
   else
      m_return = 0;
}

FunctionDef::~FunctionDef()
{
   delete m_return;
}


bool FunctionDef::parse( const String& definition )
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
      throw new ParseError( ErrorParam( FALCON_DYNLIB_ERROR_BASE+5, __LINE__ )
            .desc( *VMachine::getCurrent()->currentModule()->getString(
               dyl_invalid_syn ) ) );
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

bool FunctionDef::parseDeletor( const String& definition )
{
   Tokenizer tok(TokenizerParams().wsIsToken().returnSep(), "();[],*");
   tok.parse( definition );
   if ( ! tok.hasCurrent() )
   {
      return false;
   }

   // parse the outer return type; as "(" is found, we get a formingParam in state,
   // and we know we can start the parameter loop
   Parameter* delret = parseNextParam( tok, true );
   if( delret->m_type != Parameter::e_void )
   {
      delete delret;
      throw new ParseError( ErrorParam( FALCON_DYNLIB_ERROR_BASE+6, __LINE__)
            .desc( *VMachine::getCurrent()->currentModule()->getString(
                  dyl_invalid_del ) ) );
   }

   delete delret;
   if ( tok.getToken() != "(" )
   {
      throw new ParseError( ErrorParam( FALCON_DYNLIB_ERROR_BASE+5, __LINE__ )
                  .desc( *VMachine::getCurrent()->currentModule()->getString(
                     dyl_invalid_syn ) ) );
   }

   // if we didn't throw, it means we have a return value.
   // the return will have our name.
   m_deletorName = m_return->m_name;

   // now we can process the comma separated parameters.
   tok.next();
   if( tok.getToken() == ")" )
   {
      throw new ParseError( ErrorParam( FALCON_DYNLIB_ERROR_BASE+6, __LINE__)
         .desc( *VMachine::getCurrent()->currentModule()->getString(
               dyl_invalid_del ) ) );
   }

   // parse the main paramter list
   ParamList pars;
   parseFuncParams( pars, tok );
   if ( pars.size() != 1 ||
         pars.first()->m_type != m_return->m_type ||
         pars.first()->indirections() == 0 ||
         pars.first()->indirections() != m_return->indirections()
      )
   {
      throw new ParseError( ErrorParam( FALCON_DYNLIB_ERROR_BASE+6, __LINE__)
               .desc( *VMachine::getCurrent()->currentModule()->getString(
                     dyl_invalid_del ) ) );
   }

   return true;
}

void FunctionDef::parseFuncParams( ParamList& params, Tokenizer& tok )
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


String FunctionDef::toString() const
{
   if ( m_return == 0 )
   {
      return m_name;
   }

   String ret = m_return->toString();
   ret += "(" + m_params.toString() + ")";
   return ret;
}


void FunctionDef::gcMark( uint32 )
{
   // nothing to mark
}

FalconData *FunctionDef::clone() const
{
   return new FunctionDef( *this );
}

//===================================================
// ParamValues
//

ParamValue::ParamValue( Parameter* type ):
   m_param( type ),
   m_size(0),
   m_cstring(0),
   m_wstring(0),
   m_next(0),
   m_itemByRef(0)
{
}


ParamValue::~ParamValue()
{
   delete m_cstring;
   delete m_wstring;
}


bool ParamValue::transform( const Item& item )
{
   // if we have a base parameter, it regulates the item
   // transformation.
   if( m_param != 0 )
   {
      return transformWithParam( item );
   }
   else
   {
      return transformFree( item );
   }
}


bool ParamValue::transformWithParam( const Item& item )
{
   switch( m_param->m_type )
   {
   case Parameter::e_void:
      if( m_param->indirections() > 0 )
      {
         switch( item.type() )
         {
         case FLC_ITEM_NIL: transform( (void*) 0 ); return true;
         case FLC_ITEM_INT: transform( (void*) item.asInteger() ); return true;
         case FLC_ITEM_STRING: makeCString(*item.asString() ); return true;
         case FLC_ITEM_MEMBUF: transform( (void*) item.asMemBuf()->data() ); return true;
         }
      }
      return false;

   case Parameter::e_char:
      if( m_param->indirections() > 0 )
      {
         // char*?
         if( item.isString() )
         {
            makeCString( *item.asString() );
            return true;
         }
         else if ( item.isMemBuf() )
         {
            transform( (void*) item.asMemBuf()->data() );
            return true;
         }
      }
      else
      {
         switch( item.type() )
         {
         case FLC_ITEM_INT: transform( (char) item.asInteger() ); return true;
         case FLC_ITEM_NUM: transform( (char) item.asNumeric() ); return true;
         case FLC_ITEM_STRING:
            {
               String* s = item.asString();
               if( s->length() >= 1 )
                  transform( (char) s->getCharAt(0) );
               else
                  transform( (char) 0 );

            }
            return true;
         }
      }
      return false;


   case Parameter::e_wchar_t:
      if( m_param->indirections() > 0 )
      {
         // char*?
         if( item.isString() )
         {
            makeWString( *item.asString() );
            return true;
         }
         else if ( item.isMemBuf() )
         {
            transform( (void*) item.asMemBuf()->data() );
            return true;
         }
      }
      else
      {
         switch( item.type() )
         {
         case FLC_ITEM_INT: transform( (int) item.asInteger() ); return true;
         case FLC_ITEM_NUM: transform( (int) item.asNumeric() ); return true;
         case FLC_ITEM_STRING:
            {
               String* s = item.asString();
               if( s->length() >= 1 )
                  transform( (int) s->getCharAt(0) );
               else
                  transform( (int) 0 );

            }
            return true;
         }
      }
      return false;


   case Parameter::e_unsigned_char:
      if( m_param->indirections() > 0 )
      {
         // unsigned char*?
         if( item.isString() )
         {
            transform( (void*) item.asString()->getRawStorage() );
            return true;
         }
         else if ( item.isMemBuf() )
         {
            transform( (void*) item.asMemBuf()->data() );
            return true;
         }
      }
      else
      {
         switch( item.type() )
         {
         case FLC_ITEM_INT: transform( (unsigned char) item.asInteger() ); return true;
         case FLC_ITEM_NUM: transform( (unsigned char) item.asNumeric() ); return true;
         }
      }
      return false;

   case Parameter::e_struct:
   case Parameter::e_union:
      if( item.isOfClass( "DynOpaque" ) )
      {
         DynOpaque* obj = static_cast<DynOpaque*>( item.asObject() );
         transform((void *) obj->data() );
         return true;
      }
      return false;

   // Number-like parameters; we treat indirections at the end.

   case Parameter::e_signed_char:
      switch( item.type() )
      {
      case FLC_ITEM_INT: transform( (char) item.asInteger() ); break;
      case FLC_ITEM_NUM: transform( (char) item.asNumeric() ); break;
      default: return false;
      }
      break;

   case Parameter::e_short:
   case Parameter::e_int:
   case Parameter::e_enum:
      switch( item.type() )
      {
      case FLC_ITEM_BOOL: transform( item.isTrue() ? 1 : 0 ); break;
      case FLC_ITEM_INT: transform( (int) item.asInteger() ); break;
      case FLC_ITEM_NUM: transform( (int) item.asNumeric() ); break;
      default: return false;
      }
      break;


   case Parameter::e_unsigned_short:
   case Parameter::e_unsigned_int:
      switch( item.type() )
      {
      case FLC_ITEM_BOOL: transform( item.isTrue() ? 1 : 0 ); break;
      case FLC_ITEM_INT: transform( (unsigned int) item.asInteger() ); break;
      case FLC_ITEM_NUM: transform( (unsigned int) item.asNumeric() ); break;
      default: return false;
      }
      break;

   case Parameter::e_long:
      switch( item.type() )
      {
      case FLC_ITEM_INT: transform( (long) item.asInteger() ); break;
      case FLC_ITEM_NUM: transform( (long) item.asNumeric() ); break;
      default: return false;
      }
      break;

   case Parameter::e_unsigned_long:
      switch( item.type() )
      {
      case FLC_ITEM_INT: transform( (unsigned long) item.asInteger() ); break;
      case FLC_ITEM_NUM: transform( (unsigned long) item.asNumeric() ); break;
      default: return false;
      }
      break;

   case Parameter::e_long_long:
      switch( item.type() )
      {
      case FLC_ITEM_INT: transform( (int64) item.asInteger() ); break;
      case FLC_ITEM_NUM: transform( (int64) item.asNumeric() ); break;
      default: return false;
      }
      break;

   case Parameter::e_unsigned_long_long:
      switch( item.type() )
      {
      case FLC_ITEM_INT: transform( (int64) item.asInteger() ); break;
      case FLC_ITEM_NUM: transform( (int64) item.asNumeric() ); break;
      default: return false;
      }
      break;

   case Parameter::e_float:
      switch( item.type() )
      {
      case FLC_ITEM_INT: transform( (float) item.asInteger() ); break;
      case FLC_ITEM_NUM: transform( (float) item.asNumeric() ); break;
      default: return false;
      }
      break;

   case Parameter::e_double:
      switch( item.type() )
      {
      case FLC_ITEM_INT: transform( (double) item.asInteger() ); break;
      case FLC_ITEM_NUM: transform( (double) item.asNumeric() ); break;
      default: return false;
      }
      break;

   case Parameter::e_long_double:
      switch( item.type() )
      {
      case FLC_ITEM_INT: transform( (long double) item.asInteger() ); break;
      case FLC_ITEM_NUM: transform( (long double) item.asNumeric() ); break;
      default: return false;
      }
      break;


   default:
      return false;

   }

   // ok, treat indirections
   if( m_param->indirections() )
   {
      // the transformed value goes in the buffer, and we must point to it.
      toReference();
   }
   return true;
}


bool ParamValue::transformFree( const Item& item )
{
   switch( item.type() )
   {
   case FLC_ITEM_NIL: transform( (void*) 0 ); return true;
   case FLC_ITEM_BOOL: transform( item.asBoolean() ? 1 : 0 ); return true;
   case FLC_ITEM_INT: transform( (int) item.asInteger() ); return true;
   case FLC_ITEM_NUM: transform( (double) item.asNumeric() ); return true;
   case FLC_ITEM_STRING: makeCString( *item.asString() ); return true;
   case FLC_ITEM_MEMBUF: transform( (void*) item.asMemBuf()->data() ); return true;
   }

   return false;
}

void ParamValue::makeCString( const String& value )
{
   delete m_cstring;
   m_cstring = new AutoCString( value );
   m_buffer.vptr = const_cast<char*>(m_cstring->c_str());
   m_size = sizeof( char* );
}

void ParamValue::makeWString( const String& value )
{
   delete m_wstring;
   m_wstring = new AutoWString( value );
   m_buffer.vptr = const_cast<wchar_t*>(m_wstring->w_str());
   m_size = sizeof( wchar_t* );
}


bool ParamValue::prepareReturn()
{
   fassert( m_param != 0);

   if( m_param->indirections() > 0 )
   {
      m_buffer.vint = sizeof( void* );
      return true;
   }
   else switch( m_param->m_type )
   {
   case Parameter::e_void: m_buffer.vint = 0; break;

   case Parameter::e_char:
   case Parameter::e_wchar_t:
   case Parameter::e_unsigned_char:
   case Parameter::e_signed_char:
   case Parameter::e_short:
   case Parameter::e_unsigned_short:
   case Parameter::e_int:
   case Parameter::e_unsigned_int:
      m_buffer.vint = sizeof(int);
      break;

   case Parameter::e_long:
   case Parameter::e_unsigned_long:
      m_buffer.vint = sizeof(long);
      break;

   case Parameter::e_long_long:
   case Parameter::e_unsigned_long_long:
      m_buffer.vint = sizeof(int64);
      break;

   case Parameter::e_float:
      m_buffer.vint = 0x80|sizeof(float);
      break;

   case Parameter::e_double:
      m_buffer.vint = 0x80|sizeof(double);
      break;

   case Parameter::e_long_double:
      m_buffer.vint = 0x80|sizeof(long double);
      break;

   default:
      m_buffer.vint = 0;
      return false;

   }

   return true;
}

void ParamValue::toReference( Item* item )
{
   m_itemByRef = item;
   memcpy( m_target, m_buffer.vbuffer, sizeof( m_target ) );
   m_buffer.vptr = m_target;
   m_size = sizeof( void* );
}

void ParamValue::toReference()
{
   memcpy( m_target, m_buffer.vbuffer, sizeof( m_target ) );
   m_buffer.vptr = m_target;
   m_size = sizeof( void* );
}


void ParamValue::fromReference()
{
   memcpy( m_buffer.vbuffer, m_target, sizeof( m_target ) );
}

void ParamValue::itemRef( Item* item )
{
   m_itemByRef = item;
}

void ParamValue::derefItem()
{
   fromReference();
   if( m_itemByRef != 0 )
      toItem( *m_itemByRef );
}

bool ParamValue::toItem( Item& target, void* deletorPtr )
{
   fassert( m_param != 0);

   if( m_param->indirections() > 0 )
   {

      switch( m_param->m_type )
      {
      case Parameter::e_char:
         {
            CoreString* str = new CoreString;
            str->fromUTF8( (const char*)(m_buffer.vptr) );
            target = str;
         }
         return true;

      case Parameter::e_wchar_t:
         {
            CoreString* str = new CoreString( (const wchar_t*)(m_buffer.vptr) );
            str->bufferize();
            target = str;
         }
         return true;

      case Parameter::e_struct:
      case Parameter::e_union:
         {
            Item* i_dynop = VMachine::getCurrent()->findWKI( "DynOpaque" );
            DynOpaque* obj = new DynOpaque( i_dynop->asClass(), m_param->m_tag, m_buffer.vptr, deletorPtr );
            target = obj;
         }
         return true;

      case Parameter::e_signed_char:
      case Parameter::e_short:
      case Parameter::e_int:
         target.setInteger( (int64) m_buffer.vint );
         break;

      case Parameter::e_unsigned_char:
      case Parameter::e_unsigned_short:
      case Parameter::e_unsigned_int:
        target.setInteger( (int64) ((unsigned) m_buffer.vint) );
        break;

      case Parameter::e_long:
        target.setInteger( (int64) m_buffer.vlong );
        break;

      case Parameter::e_unsigned_long:
        target.setInteger( (int64) ((unsigned) m_buffer.vlong) );
        break;

      case Parameter::e_long_long:
      case Parameter::e_unsigned_long_long:
         target.setInteger( (int64) m_buffer.vint64 );
         break;

      case Parameter::e_float:
         target.setNumeric( m_buffer.vfloat );
         break;

      case Parameter::e_double:
         target.setNumeric( m_buffer.vdouble );
         break;

      case Parameter::e_long_double:
          target.setNumeric( (numeric) m_buffer.vld );
          break;

      default:
         // return the data as a pointer
         target.setInteger( (int64) m_buffer.vptr );
         return true;
      }

   }
   else switch( m_param->m_type )
   {
   case Parameter::e_char:
   case Parameter::e_wchar_t:
      {
         CoreString* str = new CoreString;
         str->append( m_buffer.vint );
         target = str;
      }
      break;

   case Parameter::e_unsigned_char:
   case Parameter::e_unsigned_short:
   case Parameter::e_unsigned_int:
      {
         unsigned int uint = (unsigned int) m_buffer.vint;
         target = (int64) uint;
      }
      break;

   case Parameter::e_signed_char:
   case Parameter::e_short:
   case Parameter::e_int:
   case Parameter::e_enum:
      target = (int64) m_buffer.vint;
      break;

   case Parameter::e_long:
      target = (int64) m_buffer.vlong;
      break;

   case Parameter::e_unsigned_long:
      {
         unsigned long ul = (unsigned long) m_buffer.vlong;
         target = (int64) ul;
      }
      break;

   case Parameter::e_long_long:
   case Parameter::e_unsigned_long_long:
      target = (int64) m_buffer.vint64;
      break;

   case Parameter::e_float:
      target = (numeric) m_buffer.vfloat;
      break;

   case Parameter::e_double:
      target = (numeric) m_buffer.vdouble;
      break;

   case Parameter::e_long_double:
      target = (numeric) m_buffer.vdouble;
      break;

   default:
      return false;

   }

   return true;
}

//===================================================
// ParamValueList
//

ParamValueList::ParamValueList():
   m_head(0),
   m_tail(0),
   m_size(0),
   m_compiledParams(0),
   m_compiledSizes(0)
{}

ParamValueList::~ParamValueList()
{
   if( m_compiledParams )
   {
      memFree( m_compiledParams );
      memFree( m_compiledSizes );
   }

   ParamValue* p = m_head;
   while( p != 0 )
   {
      ParamValue* old = p;
      p = p->m_next;
      delete old;
   }
}

void ParamValueList::add( ParamValue* v )
{
   if( m_head == 0 )
   {
      m_tail = m_head = v;
   }
   else
   {
      v->m_next = m_head;
      m_head = v;

      /*m_tail->m_next = v;
      m_tail = v;*/

   }

   ++m_size;
   //v->m_next = 0;
}


void ParamValueList::compile()
{
   if( m_compiledParams )
   {
      memFree( m_compiledParams );
      memFree( m_compiledSizes );
   }

   m_compiledSizes = (int*) memAlloc( sizeof(int) * (m_size+1) );

   if ( m_head != 0 )
   {
      m_compiledParams = (void**) memAlloc( sizeof(void*) * m_size );

      int count = 0;
      ParamValue* p = m_head;
      while( p != 0 )
      {
         m_compiledParams[count] = const_cast<byte*>(p->buffer());
         m_compiledSizes[count] = p->size();

         p = p->m_next;
         ++count;
      }
      // add an end marker
      m_compiledSizes[m_size] = 0;
   }
   else
   {
      m_compiledParams = 0;
      m_compiledSizes[0] = 0;
   }
}

//==========================================================
// DynOpaque
//

DynOpaque::DynOpaque( const CoreClass* cls, const String &tagName, void* data, void* deletor ):
   CoreObject( cls ),
   m_tagName( tagName ),
   m_opaqueData( data ),
   m_deletor( deletor )
{
}

DynOpaque::DynOpaque( const DynOpaque &other ):
   CoreObject( other ),
   m_tagName( other.m_tagName ),
   m_opaqueData( other.m_opaqueData ),
   m_deletor( other.m_deletor )
{
}

DynOpaque::~DynOpaque()
{
   if( m_deletor != 0 )
   {
      void(*func)(void*) = (void(*)(void*)) m_deletor;
      func( m_opaqueData );
   }
}

CoreObject *DynOpaque::clone() const
{
   return new DynOpaque( *this );
}

bool DynOpaque::setProperty( const String &prop, const Item &value )
{
   uint32 pos;
   if( generator()->properties().findKey( prop, pos ) )
      readOnlyError( prop );
   return false;
}

bool DynOpaque::getProperty( const String &prop, Item &value ) const
{
   if( prop == "type" )
   {
      value = new CoreString( m_tagName );
   }
   else if ( prop == "ptr" )
   {
      value = (int64) m_opaqueData;
   }
   else
      return defaultProperty( prop, value );

   return true;
}


}

/* end of dynlib_mod.cpp */
