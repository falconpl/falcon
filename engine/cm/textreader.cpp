/*
   FALCON - The Falcon Programming Language.
   FILE: textreader.cpp

   Falcon core module -- Wrapping for text reader
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 11 Dec 2011 17:32:58 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/textreader.cpp"

#include <falcon/stderrors.h>
#include <falcon/cm/textreader.h>
#include <falcon/transcoder.h>
#include <falcon/vmcontext.h>
#include <falcon/stdhandlers.h>
#include <falcon/function.h>

#include <falcon/cm/textstream.h>

#include "falcon/itemarray.h"

namespace Falcon {
namespace Ext {

//=================================================================
// Properties
//

static void set_encoding( const Class*, const String&, void* instance, const Item& value )
{
   static Engine* eng = Engine::instance();
   
   TextReader* sc = static_cast<TextReader*>(instance);
   if( value.isNil() )
   {
      sc->setEncoding( eng->getTranscoder("C") );
   }
   else if( value.isString() )
   {
      Transcoder* tc = eng->getTranscoder(*value.asString());
      if( tc == 0 )
      {      
         // unknown encoding
         throw new ParamError( ErrorParam( e_param_range, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime )
            .extra("Unknown encoding " + *value.asString()));
      }  
      sc->setEncoding(tc);
   }
   else
   {
      throw new ParamError( ErrorParam(e_inv_prop_value, __LINE__, SRC ).extra("S|Nil"));
   }   
}


static void get_encoding( const Class*, const String&, void* instance, Item& value )
{
   TextReader* sc = static_cast<TextReader*>(instance);
   value = sc->transcoder()->name();
}

static void get_underlying( const Class*, const String&, void* instance, Item& value )
{
   TextReader* sc = static_cast<TextReader*>(instance);
   value = Item( sc->underlying()->handler(), sc->underlying());
}

static void get_eof( const Class*, const String&, void* instance, Item& value )
{
   TextReader* sc = static_cast<TextReader*>(instance);
   value.setBoolean(sc->eof());
}

static void get_status( const Class*, const String&, void* instance, Item& value )
{
   TextReader* sc = static_cast<TextReader*>(instance);
   value.setInteger((int64)sc->underlying()->status());
}



//=================================================================
// Methods
//

namespace CTextReader {

FALCON_DECLARE_FUNCTION( read, "text:S, count:N" );
FALCON_DECLARE_FUNCTION( grab, "count:N" );
FALCON_DECLARE_FUNCTION( readLine, "text:S, count:[N]" );
FALCON_DECLARE_FUNCTION( grabLine, "count:[N]" );
FALCON_DECLARE_FUNCTION( readEof, "text:S" );
FALCON_DECLARE_FUNCTION( grabEof, "" );
FALCON_DECLARE_FUNCTION( readRecord, "text:S, marker:S, count:[N]" );
FALCON_DECLARE_FUNCTION( grabRecord, "marker:S, count:[N]" );
FALCON_DECLARE_FUNCTION( readToken, "text:S, tokens:A, count:[N]" );
FALCON_DECLARE_FUNCTION( grabToken, "tokens:A, count:[N]" );

FALCON_DECLARE_FUNCTION( readChar, "text:S, append:[B]" );
FALCON_DECLARE_FUNCTION( getChar, "" );
FALCON_DECLARE_FUNCTION( ungetChar, "char:S|N" );   

FALCON_DECLARE_FUNCTION( getStream, "" );      
FALCON_DECLARE_FUNCTION( sync, "" );
FALCON_DECLARE_FUNCTION( close, "" );

void Function_read::invoke(VMContext* ctx, int32 )
{
   Item* i_data = ctx->param(0);
   Item* i_count = ctx->param(1);
   
   if( (i_data == 0 || ! i_data->isString())
      || (i_count == 0 || !i_count->isOrdinal()) )
   {
      ctx->raiseError(paramError());
      return;
   }
           
   TextReader* sc = static_cast<TextReader*>(ctx->self().asInst());
   String* data = i_data->asString();
   if( data->isImmutable() )
   {
      throw new ParamError( ErrorParam(e_param_type, __LINE__, SRC).extra("Immutable string") );
   }

   bool value = sc->read( *data, (length_t) i_count->forceInteger() );
   ctx->returnFrame( value );   
}


void Function_grab::invoke(VMContext* ctx, int32 )
{
   static Class* clsString = Engine::handlers()->stringClass();
   Item* i_count = ctx->param(0);
   if( i_count == 0 || !(i_count->isOrdinal()) )
   {
      ctx->raiseError(paramError());
      return;
   }

   String* str = new String;   
   
   int64 icount = i_count->forceInteger();
   // shall we read?
   if( icount > 0 )
   {      
      TextReader* sc = static_cast<TextReader*>(ctx->self().asInst());
      try {
         sc->read( *str, (length_t) icount );
      }
      catch( ... )
      {
         delete str;
         throw;
      }
   }
   
   // Return the string.
   // force to garbage the string NOW!
   ctx->returnFrame( FALCON_GC_STORE( clsString, str ) );
}


void Function_readLine::invoke(VMContext* ctx, int32 )
{
   Item* i_data = ctx->param(0);
   Item* i_count = ctx->param(1);
   
   if( (i_data == 0 || ! i_data->isString())
      || (i_count != 0 && !i_count->isOrdinal()) )
   {
      ctx->raiseError(paramError());
      return;
   }
   
   length_t count =  i_count == 0 ? 4096 : 
         static_cast<length_t>(i_count->forceInteger());
   
   if( count > 0 )
   {
      String* data = i_data->asString();
      if( data->isImmutable() )
      {
         throw new ParamError( ErrorParam(e_param_type, __LINE__, SRC).extra("Immutable string") );
      }
      TextReader* sc = static_cast<TextReader*>(ctx->self().asInst());
      bool value = sc->readLine( *data, count );
      ctx->returnFrame( value );   
   }
   else {
      ctx->returnFrame(false);
   }
}


void Function_grabLine::invoke(VMContext* ctx, int32 )
{
   static Class* clsString = Engine::handlers()->stringClass();
   Item* i_count = ctx->param(0);
   if( i_count != 0 && !(i_count->isOrdinal()) )
   {
      ctx->raiseError(paramError());
      return;
   }

   String* str = new String;   
   
   length_t count =  i_count == 0 ? 4096 : 
         static_cast<length_t>(i_count->forceInteger());

   // shall we read?
   if( count > 0 )
   {      
      TextReader* sc = static_cast<TextReader*>(ctx->self().asInst());
      try {
         sc->readLine( *str, count );
      }
      catch( ... )
      {
         delete str;
         throw;
      }
   }
   
   // Return the string.
   ctx->returnFrame( FALCON_GC_STORE( clsString, str ) );
}


void Function_readEof::invoke(VMContext* ctx, int32 )
{
   Item* i_data = ctx->param(0);   
   
   if( (i_data == 0 || ! i_data->isString()))
   {
      ctx->raiseError(paramError());
      return;
   }      
   
   String* data = i_data->asString();
   if( data->isImmutable() )
   {
      throw new ParamError( ErrorParam(e_param_type, __LINE__, SRC).extra("Immutable string") );
   }

   TextReader* sc = static_cast<TextReader*>(ctx->self().asInst());
   bool value = sc->readEof( *data );
   ctx->returnFrame( value );   
}


void Function_grabEof::invoke(VMContext* ctx, int32 )
{  
   static Class* clsString = Engine::handlers()->stringClass();
   String* str = new String;   
   TextReader* sc = static_cast<TextReader*>(ctx->self().asInst());
   try {
      sc->readEof( *str );
   }
   catch( ... )
   {
      delete str;
      throw;
   }
   // Return the string.
   ctx->returnFrame( FALCON_GC_STORE( clsString, str ) );
}


void Function_readRecord::invoke(VMContext* ctx, int32 )
{
   Item* i_data = ctx->param(0);
   Item* i_sep = ctx->param(1);   
   Item* i_count = ctx->param(2);
   
   if( (i_data == 0 || ! i_data->isString())
      || (i_sep == 0 || ! i_sep->isString())
      || (i_count != 0 && !i_count->isOrdinal()) )
   {
      ctx->raiseError(paramError());
      return;
   }
   
   length_t count =  i_count == 0 ? 4096 : 
         static_cast<length_t>(i_count->forceInteger());
   
   String* data = i_data->asString();
   if( data->isImmutable() )
   {
      throw new ParamError( ErrorParam(e_param_type, __LINE__, SRC).extra("Immutable string") );
   }

   if( count > 0 )
   {
      TextReader* sc = static_cast<TextReader*>(ctx->self().asInst());
      bool value = sc->readRecord( *data, *i_sep->asString(), count );
      ctx->returnFrame( value );
   }
   else {
      ctx->returnFrame(false);
   }
}


void Function_grabRecord::invoke(VMContext* ctx, int32 )
{
   static Class* clsString = Engine::handlers()->stringClass();

   Item* i_sep = ctx->param(0);   
   Item* i_count = ctx->param(1);
   
   if((i_sep == 0 || ! i_sep->isString())
      || (i_count != 0 && !i_count->isOrdinal()) )
   {
      ctx->raiseError(paramError());
      return;
   }
   
   String* str = new String;   
   
   length_t count =  i_count == 0 ? 4096 : 
         static_cast<length_t>(i_count->forceInteger());
   
   if( count > 0 )
   {
      TextReader* sc = static_cast<TextReader*>(ctx->self().asInst());
      try {
         sc->readRecord( *str, *i_sep->asString(), count );
      }
      catch( ... )
      {
         delete str;
         throw;
      }
   }
   
   ctx->returnFrame( FALCON_GC_STORE( clsString, str ) );
}


static bool internal_readToken( VMContext* ctx, String& target, ItemArray* iarr, Item* i_count )
{
   length_t count =  i_count == 0 ? 4096 : 
         static_cast<length_t>(i_count->forceInteger());
   
   if( target.isImmutable() )
   {
      throw new ParamError( ErrorParam(e_param_type, __LINE__, SRC).extra("Immutable string") );
   }

   if( count > 0 )
   {
      String tokens[32];

      uint32 stri;
      for( stri = 0; stri < iarr->length() && stri < 32; ++stri )
      {
         Item& sep = iarr->at(stri);
         if( ! sep.isString() )
         {
            // unknown encoding
            ctx->raiseError( new ParamError( ErrorParam( e_param_type, __LINE__, SRC )
               .origin( ErrorParam::e_orig_runtime )
               .extra("Elements in tokens must be Strings")));
            return false;
         }

         tokens[stri] = *sep.asString();
      }
  
      TextReader* sc = static_cast<TextReader*>(ctx->self().asInst());
      int value = sc->readToken( target, tokens, stri, count );
      return value > 0;
   }
   else {
      return false;
   }
}

void Function_readToken::invoke(VMContext* ctx, int32 )
{
   Item* i_data = ctx->param(0);
   Item* i_seps = ctx->param(1);   
   Item* i_count = ctx->param(2);
   
   if( (i_data == 0 || ! i_data->isString())
      || (i_seps == 0 || ! i_seps->isArray())
      || (i_count != 0 && !i_count->isOrdinal()) )
   {
      ctx->raiseError(paramError());
      return;
   }
   
   bool retval = internal_readToken( 
                  ctx, *i_data->asString(), i_seps->asArray(), i_count );
   ctx->returnFrame( retval );
}


void Function_grabToken::invoke(VMContext* ctx, int32 )
{
   static Class* clsString = Engine::handlers()->stringClass();

   Item* i_seps = ctx->param(0);   
   Item* i_count = ctx->param(1);
   
   if((i_seps == 0 || ! i_seps->isArray())
      || (i_count != 0 && !i_count->isOrdinal()) )
   {
      ctx->raiseError(paramError());
      return;
   }
   
   String* str = new String;   
   try
   {
      internal_readToken( 
                  ctx, *str, i_seps->asArray(), i_count );
   }
   catch( ... )
   {
      delete str;
      throw;
   }

   ctx->returnFrame( FALCON_GC_STORE( clsString, str ) );
}


void Function_readChar::invoke(VMContext* ctx, int32 )
{
   Item* i_data = ctx->param(0);
   Item* i_append = ctx->param(1);
   
   if( (i_data == 0 || ! i_data->isString()) )
   {
      ctx->raiseError(paramError());
      return;
   }
   
   TextReader* sc = static_cast<TextReader*>(ctx->self().asInst());
   char_t chr = sc->getChar();
   if( chr == (char_t)-1 )
   {
      ctx->returnFrame(false);
   }
   else
   {
      String& str = *i_data->asString();
      if (i_append == 0 && ! i_append->isTrue())
      {
         str.reserve(1);
         str.size(0);         
      }
      str.append( chr );
   }
   ctx->returnFrame(true); 
}


void Function_getChar::invoke(VMContext* ctx, int32 )
{   
   TextReader* sc = static_cast<TextReader*>(ctx->self().asInst());
   char_t chr = sc->getChar();
   if( chr == (char_t)-1 )
   {
      ctx->returnFrame((int64)-1);
   }
   else
   {
      ctx->returnFrame((int64)chr);
   }
}



void Function_ungetChar::invoke(VMContext* ctx, int32 )
{
   Item* i_data = ctx->param(0);

   if( i_data == 0 || ! (i_data->isString() || ! i_data->isOrdinal() ) )
   {
      ctx->raiseError(this->paramError());
      return;
   }
   
   TextReader* sc = static_cast<TextReader*>(ctx->self().asInst());
   char_t chr;
   if( i_data->isString() )
   {
      String& str = *i_data->asString();
      if( str.length() == 0 )
      {
         ctx->raiseError( new ParamError( ErrorParam(e_param_range, __LINE__, SRC)
         .origin(ErrorParam::e_orig_runtime)
         .extra( "String parameter empty") )         
         );
         return;
      }
      
      chr = str.getCharAt(0);
   }
   else
   {
      chr = (char_t) i_data->forceInteger();
   }
   
   sc->ungetChar( chr );
   ctx->returnFrame();
}


void Function_sync::invoke(VMContext* ctx, int32 )
{   
   TextReader* sc = static_cast<TextReader*>(ctx->self().asInst());
   sc->changeStream( sc->underlying(), true );
   ctx->returnFrame();
}


void Function_close::invoke(VMContext* ctx, int32 )
{
   TextReader* sc = static_cast<TextReader*>(ctx->self().asInst());
   sc->underlying()->close();
   ctx->returnFrame();
}
}

//================================================================
//

ClassTextReader::ClassTextReader( ClassStream* clsStream ):
   Class( "TextReader" ),
   m_clsStream( clsStream )
{
   addProperty( "encoding", &get_encoding, &set_encoding );
   addProperty( "underlying", &get_underlying );
   addProperty( "eof", &get_eof );
   addProperty( "status", &get_status );
   
   addMethod( new CTextReader::Function_read );
   addMethod( new CTextReader::Function_grab );
   addMethod( new CTextReader::Function_readLine );
   addMethod( new CTextReader::Function_grabLine );
   addMethod( new CTextReader::Function_readEof );
   addMethod( new CTextReader::Function_grabEof );
   addMethod( new CTextReader::Function_readRecord );
   addMethod( new CTextReader::Function_grabRecord );
   addMethod( new CTextReader::Function_readToken );
   addMethod( new CTextReader::Function_grabToken );
   
   addMethod( new CTextReader::Function_readChar );
   addMethod( new CTextReader::Function_getChar );
   addMethod( new CTextReader::Function_ungetChar );   
      
   addMethod( new CTextReader::Function_sync );
   addMethod( new CTextReader::Function_close );
}

ClassTextReader::~ClassTextReader()
{
}


void* ClassTextReader::createInstance() const
{
   return FALCON_CLASS_CREATE_AT_INIT;
}

void ClassTextReader::dispose( void* instance ) const
{
   TextReader* tr = static_cast<TextReader*>(instance);
   tr->decref();
}

void* ClassTextReader::clone( void* ) const
{
   return 0;
}

void ClassTextReader::gcMarkInstance( void* instance, uint32 mark ) const
{
   TextReader* tr = static_cast<TextReader*>(instance);
   tr->gcMark( mark );
}

bool ClassTextReader::gcCheckInstance( void* instance, uint32 mark ) const
{
   TextReader* tr = static_cast<TextReader*>(instance);
   return tr->currentMark() > mark;
}

bool ClassTextReader::op_init( VMContext* ctx, void* , int pcount ) const
{
   static Engine* eng = Engine::instance();
   
   // if we have 2 parameters, the second one is the encoding.
   String* encoding = 0;
   Stream* stc = 0;
   Transcoder* tc = 0;
   
   if( pcount > 0 )
   {
      Class* cls;
      void* data;
      Item* params = ctx->opcodeParams(pcount);
      
      if( params[0].asClassInst(cls, data) && cls->isDerivedFrom(m_clsStream) )
      {
         stc = static_cast<Stream*>(data);
         if( pcount > 1 )
         {
            if ( params[1].isString() )
            {
               encoding = params[1].asString();
            }
            else
            {
               // force a param error.
               stc = 0;
            }
         }
      }
   }
   
   // if we're here with stc == 0, we have a param error.
   if( stc == 0 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
         .origin(ErrorParam::e_orig_runtime)
         .extra( "Stream,[S]" ) );      
   }
   
   // is the encoding correct?
   if( encoding != 0 ) 
   {
      tc = eng->getTranscoder( *encoding );
      if( tc == 0 )
      {
         throw new ParamError( ErrorParam( e_param_range, __LINE__, SRC )
            .origin(ErrorParam::e_orig_runtime)
            .extra( "Unknown encoding " + *encoding ) );      
      }
   }
   
   TextReader* twc = new TextReader(stc);
   if( tc != 0 )
   {
      twc->setEncoding(tc);
   }
   ctx->opcodeParam(pcount) = FALCON_GC_STORE(this, twc);
   
   return false;
}

}
}

/* end of textreader.cpp */
