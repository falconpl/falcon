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

#include <falcon/errors/paramerror.h>
#include <falcon/cm/textreader.h>
#include <falcon/transcoder.h>
#include <falcon/vmcontext.h>

#include <falcon/cm/textstream.h>

#include "falcon/itemarray.h"

namespace Falcon {
namespace Ext {


TextReaderCarrier::TextReaderCarrier( StreamCarrier* stc ):
   UserCarrierT<StreamCarrier>(stc),
   m_reader( new TextReader(stc->m_underlying ) )
{
   stc->incref();
}

TextReaderCarrier::~TextReaderCarrier()
{
   delete m_reader;
   carried()->decref();
}

StreamCarrier* TextReaderCarrier::cloneData() const
{
   carried()->incref();
   return carried();
}
   
void TextReaderCarrier::gcMark( uint32 mark )
{
   if( mark != m_gcMark )
   {
      m_gcMark = mark;
      carried()->m_stream->gcMark( mark );
   }
}


//================================================================
//

ClassTextReader::ClassTextReader( ClassStream* clsStream ):
   ClassUser( "TextReader" ),
   m_clsStream( clsStream ),
   
   FALCON_INIT_PROPERTY( encoding ),
   
   FALCON_INIT_METHOD( read ),
   FALCON_INIT_METHOD( grab ),
   FALCON_INIT_METHOD( readLine ),
   FALCON_INIT_METHOD( grabLine ),
   FALCON_INIT_METHOD( readEof ),
   FALCON_INIT_METHOD( grabEof ),
   FALCON_INIT_METHOD( readRecord ),
   FALCON_INIT_METHOD( grabRecord ),
   FALCON_INIT_METHOD( readToken ),
   FALCON_INIT_METHOD( grabToken ),
   
   FALCON_INIT_METHOD( readChar ),
   FALCON_INIT_METHOD( getChar ),
   FALCON_INIT_METHOD( ungetChar ),   
   
   FALCON_INIT_METHOD( getStream ),      
   FALCON_INIT_METHOD( sync ),
   FALCON_INIT_METHOD( close )
{
   
}

ClassTextReader::~ClassTextReader()
{
}


void* ClassTextReader::createInstance() const
{
   return FALCON_CLASS_CREATE_AT_INIT;
}

bool ClassTextReader::op_init( VMContext* ctx, void* , int pcount ) const
{
   static Engine* eng = Engine::instance();
   
   // if we have 2 parameters, the second one is the encoding.
   String* encoding = 0;
   StreamCarrier* stc = 0;
   Transcoder* tc = 0;
   
   if( pcount > 0 )
   {
      Class* cls;
      void* data;
      Item* params = ctx->opcodeParams(pcount);
      
      if( params[0].asClassInst(cls, data) && cls->isDerivedFrom(m_clsStream) )
      {
         stc = static_cast<StreamCarrier*>(data);
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
   
   TextReaderCarrier* twc = new TextReaderCarrier(stc);
   if( tc != 0 )
   {
      twc->m_reader->setEncoding(tc);
   }
   ctx->opcodeParam(pcount) = FALCON_GC_STORE(this, twc);
   
   return false;
}

//=================================================================
// Properties
//

FALCON_DEFINE_PROPERTY_SET_P( ClassTextReader, encoding )
{
   static Engine* eng = Engine::instance();
   
   TextReaderCarrier* sc = static_cast<TextReaderCarrier*>(instance);
   if( value.isNil() )
   {
      sc->m_reader->setEncoding( eng->getTranscoder("C") );
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
      sc->m_reader->setEncoding(tc);
   }
   else
   {
      throw new ParamError( ErrorParam(e_inv_prop_value, __LINE__, SRC ).extra("S|Nil"));
   }   
}


FALCON_DEFINE_PROPERTY_GET_P( ClassTextReader, encoding )
{
   TextReaderCarrier* sc = static_cast<TextReaderCarrier*>(instance);     
   value = sc->m_reader->transcoder()->name();
}

//=================================================================
// Methods
//

FALCON_DEFINE_METHOD_P1( ClassTextReader, read )
{
   Item* i_data = ctx->param(0);
   Item* i_count = ctx->param(1);
   
   if( (i_data == 0 || ! i_data->isString())
      || (i_count == 0 || !i_count->isOrdinal()) )
   {
      ctx->raiseError(paramError());
      return;
   }
           
   TextReaderCarrier* sc = static_cast<TextReaderCarrier*>(ctx->self().asInst());
   bool value = sc->m_reader->read( *i_data->asString(), i_count->forceInteger() );
   ctx->returnFrame( value );   
}


FALCON_DEFINE_METHOD_P1( ClassTextReader, grab )
{
   static Class* clsString = Engine::instance()->stringClass();
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
      TextReaderCarrier* sc = static_cast<TextReaderCarrier*>(ctx->self().asInst());
      try {
         sc->m_reader->read( *str, icount );
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


FALCON_DEFINE_METHOD_P1( ClassTextReader, readLine )
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
      TextReaderCarrier* sc = static_cast<TextReaderCarrier*>(ctx->self().asInst());
      bool value = sc->m_reader->readLine( *i_data->asString(), count );
      ctx->returnFrame( value );   
   }
   else {
      ctx->returnFrame(false);
   }
}


FALCON_DEFINE_METHOD_P1( ClassTextReader, grabLine )
{
   static Class* clsString = Engine::instance()->stringClass();
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
      TextReaderCarrier* sc = static_cast<TextReaderCarrier*>(ctx->self().asInst());
      try {
         sc->m_reader->readLine( *str, count );
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


FALCON_DEFINE_METHOD_P1( ClassTextReader, readEof )
{
   Item* i_data = ctx->param(0);   
   
   if( (i_data == 0 || ! i_data->isString()))
   {
      ctx->raiseError(paramError());
      return;
   }      
   
   TextReaderCarrier* sc = static_cast<TextReaderCarrier*>(ctx->self().asInst());
   bool value = sc->m_reader->readEof( *i_data->asString() );
   ctx->returnFrame( value );   
}


FALCON_DEFINE_METHOD_P1( ClassTextReader, grabEof )
{  
   static Class* clsString = Engine::instance()->stringClass();
   String* str = new String;   
   TextReaderCarrier* sc = static_cast<TextReaderCarrier*>(ctx->self().asInst());
   try {
      sc->m_reader->readEof( *str );
   }
   catch( ... )
   {
      delete str;
      throw;
   }
   // Return the string.
   ctx->returnFrame( FALCON_GC_STORE( clsString, str ) );
}


FALCON_DEFINE_METHOD_P1( ClassTextReader, readRecord )
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
   
   if( count > 0 )
   {
      TextReaderCarrier* sc = static_cast<TextReaderCarrier*>(ctx->self().asInst());
      bool value = sc->m_reader->readRecord( *i_data->asString(), *i_sep->asString(), count );
      ctx->returnFrame( value );
   }
   else {
      ctx->returnFrame(false);
   }
}


FALCON_DEFINE_METHOD_P1( ClassTextReader, grabRecord )
{
   static Class* clsString = Engine::instance()->stringClass();

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
      TextReaderCarrier* sc = static_cast<TextReaderCarrier*>(ctx->self().asInst());
      try {
         sc->m_reader->readRecord( *str, *i_sep->asString(), count );
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
  
      TextReaderCarrier* sc = static_cast<TextReaderCarrier*>(ctx->self().asInst());
      int value = sc->m_reader->readToken( target, tokens, stri, count );
      return value > 0;
   }
   else {
      return false;
   }
}

FALCON_DEFINE_METHOD_P1( ClassTextReader, readToken )
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


FALCON_DEFINE_METHOD_P1( ClassTextReader, grabToken )
{
   static Class* clsString = Engine::instance()->stringClass();

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


FALCON_DEFINE_METHOD_P1( ClassTextReader, readChar )
{
   Item* i_data = ctx->param(0);
   Item* i_append = ctx->param(1);
   
   if( (i_data == 0 || ! i_data->isString()) )
   {
      ctx->raiseError(paramError());
      return;
   }
   
   
   TextReaderCarrier* sc = static_cast<TextReaderCarrier*>(ctx->self().asInst());
   char_t chr = sc->m_reader->getChar();
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

FALCON_DEFINE_METHOD_P1( ClassTextReader, getChar )
{   
   TextReaderCarrier* sc = static_cast<TextReaderCarrier*>(ctx->self().asInst());
   char_t chr = sc->m_reader->getChar();
   if( chr == (char_t)-1 )
   {
      ctx->returnFrame((int64)-1);
   }
   else
   {
      ctx->returnFrame((int64)chr);
   }
}



FALCON_DEFINE_METHOD_P1( ClassTextReader, ungetChar )
{
   Item* i_data = ctx->param(0);

   if( i_data == 0 || ! (i_data->isString() || ! i_data->isOrdinal() ) )
   {
      ctx->raiseError(this->paramError());
      return;
   }
   
   TextReaderCarrier* sc = static_cast<TextReaderCarrier*>(ctx->self().asInst());
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
   
   sc->m_reader->ungetChar( chr );
   ctx->returnFrame();
}


FALCON_DEFINE_METHOD_P1( ClassTextReader, sync )
{   
   TextReaderCarrier* sc = static_cast<TextReaderCarrier*>(ctx->self().asInst());
   sc->m_reader->changeStream( sc->carried()->m_underlying, true );
   ctx->returnFrame();
}

FALCON_DEFINE_METHOD_P1( ClassTextReader, close )
{
   TextReaderCarrier* sc = static_cast<TextReaderCarrier*>(ctx->self().asInst());
   sc->carried()->m_stream->close();
   ctx->returnFrame();
}

FALCON_DEFINE_METHOD_P1( ClassTextReader, getStream )
{     
   TextReaderCarrier* sc = static_cast<TextReaderCarrier*>(ctx->self().asInst());
   // the token is held already in the collector.
   ClassTextReader* ctw = static_cast<ClassTextReader*>(m_methodOf);
   
   ctx->returnFrame( Item( ctw->m_clsStream, sc->carried() ) );
}


}
}

/* end of textreader.cpp */
