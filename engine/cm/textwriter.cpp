/*
   FALCON - The Falcon Programming Language.
   FILE: textwriter.cpp

   Falcon core module -- Wrapping for text writer
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 11 Dec 2011 17:32:58 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/errors/paramerror.h>
#include <falcon/cm/textwriter.h>
#include <falcon/transcoder.h>
#include <falcon/vmcontext.h>

#include <falcon/cm/textstream.h>

namespace Falcon {
namespace Ext {


TextWriterCarrier::TextWriterCarrier( StreamCarrier* stc ):
   UserCarrierT<StreamCarrier>(stc),
   m_writer( stc->m_underlying )
{
}

TextWriterCarrier::~TextWriterCarrier()
{
}
   
void TextWriterCarrier::gcMark( uint32 mark )
{
   if( mark != m_gcMark )
   {
      m_gcMark = mark;
      carried()->m_gcMark = mark;
   }
}


//================================================================
//

ClassTextWriter::ClassTextWriter( ClassStream* clsStream ):
   ClassUser( "TextWriter" ),
   m_clsStream( clsStream ),
   
   FALCON_INIT_PROPERTY( encoding ),
   FALCON_INIT_PROPERTY( crlf ),
   FALCON_INIT_PROPERTY( lineflush ),
   FALCON_INIT_PROPERTY( buffer ),
   
   FALCON_INIT_METHOD( write ),
   FALCON_INIT_METHOD( writeLine ),   
   FALCON_INIT_METHOD( putChar ),   
   FALCON_INIT_METHOD( getStream ),      
   FALCON_INIT_METHOD( flush )
{
   
}

ClassTextWriter::~ClassTextWriter()
{
}


void* ClassTextWriter::createInstance( Item* params, int pcount ) const
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
   
   TextWriterCarrier* twc = new TextWriterCarrier( stc );
   if( tc != 0 )
   {
      twc->m_writer.setEncoding(tc);
   }
   
   return twc;
}

//=================================================================
// Properties
//

FALCON_DEFINE_PROPERTY_SET_P( ClassTextWriter, encoding )
{
   static Engine* eng = Engine::instance();
   
   TextWriterCarrier* sc = static_cast<TextWriterCarrier*>(instance);
   if( value.isNil() )
   {
      sc->m_writer.setEncoding( eng->getTranscoder("C") );
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
      sc->m_writer.setEncoding(tc);
   }
   else
   {
      throw new ParamError( ErrorParam(e_inv_prop_value, __LINE__, SRC ).extra("S|Nil"));
   }   
}


FALCON_DEFINE_PROPERTY_GET_P( ClassTextWriter, encoding )
{
   TextWriterCarrier* sc = static_cast<TextWriterCarrier*>(instance);     
   value = sc->m_writer.transcoder()->name();
}


FALCON_DEFINE_PROPERTY_SET_P( ClassTextWriter, crlf )
{   
   TextWriterCarrier* sc = static_cast<TextWriterCarrier*>(instance);
   sc->m_writer.setCRLF(value.isTrue());
}


FALCON_DEFINE_PROPERTY_GET_P( ClassTextWriter, crlf )
{
   TextWriterCarrier* sc = static_cast<TextWriterCarrier*>(instance);     
   value.setBoolean(sc->m_writer.isCRLF());
}

FALCON_DEFINE_PROPERTY_SET_P( ClassTextWriter, lineflush )
{   
   TextWriterCarrier* sc = static_cast<TextWriterCarrier*>(instance);
   sc->m_writer.lineFlush(value.isTrue());
}


FALCON_DEFINE_PROPERTY_GET_P( ClassTextWriter, lineflush )
{
   TextWriterCarrier* sc = static_cast<TextWriterCarrier*>(instance);     
   value.setBoolean(sc->m_writer.lineFlush());
}


FALCON_DEFINE_PROPERTY_SET_P( ClassTextWriter, buffer )
{   
   TextWriterCarrier* sc = static_cast<TextWriterCarrier*>(instance);
   
   if( !value.isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime )
            .extra("N"));
   }
   sc->m_writer.setBufferSize( static_cast<length_t>(value.forceInteger()));
}


FALCON_DEFINE_PROPERTY_GET_P( ClassTextWriter, buffer )
{
   TextWriterCarrier* sc = static_cast<TextWriterCarrier*>(instance);     
   value.setBoolean(sc->m_writer.bufferSize());
}

//=================================================================
// Methods
//


static void internal_write( Method* called, VMContext* ctx, bool asLine )
{   
   Item* i_data = ctx->param(0);
   if( i_data == 0 || !i_data->isString() )
   {
      ctx->raiseError(called->paramError());
      return;
   }
   
   Item* i_count = ctx->param(1);
   Item* i_start = ctx->param(2);
   
   if ( (i_count != 0 && !(i_count->isOrdinal() || i_count->isNil())) ||
        (i_start != 0 && !(i_start->isOrdinal() || i_start->isNil()))
      )
   {
      ctx->raiseError(called->paramError());
      return;
   }
   
   length_t count = i_count != 0 && i_count->isOrdinal() ? 
      static_cast<length_t>(i_count->forceInteger()) : String::npos;
   
   length_t start = i_start != 0 && i_start->isOrdinal() ? 
      static_cast<length_t>(i_start->forceInteger()) : 0;
   
   TextWriterCarrier* sc = static_cast<TextWriterCarrier*>(ctx->self().asInst());
   
   bool value;
   if( asLine )
   {
      value = sc->m_writer.writeLine( *i_data->asString(), start, count );   
   }
   else
   {
      value = sc->m_writer.write( *i_data->asString(), start, count );   
   }
   
   ctx->returnFrame( value );   
}

FALCON_DEFINE_METHOD_P1( ClassTextWriter, write )
{
   internal_write(this, ctx, false );
}

FALCON_DEFINE_METHOD_P1( ClassTextWriter, writeLine )
{   
   internal_write(this, ctx, true );
}


FALCON_DEFINE_METHOD_P1( ClassTextWriter, putChar )
{
   Item* i_data = ctx->param(0);

   if( i_data == 0 || ! (i_data->isString() || ! i_data->isOrdinal() ) )
   {
      ctx->raiseError(this->paramError());
      return;
   }
   
   TextWriterCarrier* sc = static_cast<TextWriterCarrier*>(ctx->self().asInst());
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
   
   sc->m_writer.putChar( chr );
   ctx->returnFrame();
}


FALCON_DEFINE_METHOD_P1( ClassTextWriter, flush )
{   
   TextWriterCarrier* sc = static_cast<TextWriterCarrier*>(ctx->self().asInst());
   sc->m_writer.flush();
}

FALCON_DEFINE_METHOD_P1( ClassTextWriter, getStream )
{     
   TextWriterCarrier* sc = static_cast<TextWriterCarrier*>(ctx->self().asInst());
   // the token is held already in the collector.
   ClassTextWriter* ctw = static_cast<ClassTextWriter*>(m_methodOf);
   
   ctx->returnFrame( Item( ctw->m_clsStream, sc->carried() ) );
}

}
}

/* end of textwriter.cpp */
