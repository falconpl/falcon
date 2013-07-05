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

#define SRC "engine/textwriter.cpp"

#include <falcon/stderrors.h>
#include <falcon/cm/textwriter.h>
#include <falcon/transcoder.h>
#include <falcon/function.h>
#include <falcon/vmcontext.h>

#include <falcon/cm/textstream.h>

namespace Falcon {
namespace Ext {

//=================================================================
// Properties
//

static void set_encoding( const Class*, const String&, void* instance, const Item& value )
{
   static Engine* eng = Engine::instance();
   
   TextWriter* sc = static_cast<TextWriter*>(instance);
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
   TextWriter* sc = static_cast<TextWriter*>(instance);
   value = sc->transcoder()->name();
}


static void set_crlf( const Class*, const String&, void* instance, const Item& value )
{   
   TextWriter* sc = static_cast<TextWriter*>(instance);
   sc->setCRLF(value.isTrue());
}


static void get_crlf( const Class*, const String&, void* instance, Item& value )
{
   TextWriter* sc = static_cast<TextWriter*>(instance);
   value.setBoolean(sc->isCRLF());
}


static void set_lineflush( const Class*, const String&, void* instance, const Item& value )
{   
   TextWriter* sc = static_cast<TextWriter*>(instance);
   sc->lineFlush(value.isTrue());
}


static void get_lineflush( const Class*, const String&, void* instance, Item& value )
{
   TextWriter* sc = static_cast<TextWriter*>(instance);
   value.setBoolean(sc->lineFlush());
}


static void set_buffer( const Class*, const String&, void* instance, const Item& value )
{   
   TextWriter* sc = static_cast<TextWriter*>(instance);
   
   if( !value.isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime )
            .extra("N"));
   }
   sc->setBufferSize( static_cast<length_t>(value.forceInteger()));
}


static void get_buffer( const Class*, const String&, void* instance, Item& value )
{
   TextWriter* sc = static_cast<TextWriter*>(instance);
   value.setInteger(sc->bufferSize());
}

static void get_underlying( const Class*, const String&, void* instance, Item& value )
{
   TextWriter* sc = static_cast<TextWriter*>(instance);
   value.setUser(sc->underlying()->handler(), sc->underlying());
}

//=================================================================
// Methods
//

namespace CTextWriter {

FALCON_DECLARE_FUNCTION( write, "text:S, count:[N], start:[N]" );
FALCON_DECLARE_FUNCTION( writeLine, "text:S, count:[N], start:[N]" );   
FALCON_DECLARE_FUNCTION( putChar, "char:S|N" );   
FALCON_DECLARE_FUNCTION( getStream, "" );      
FALCON_DECLARE_FUNCTION( flush, "" );
FALCON_DECLARE_FUNCTION( close, "" );

static void internal_write( Function* called, VMContext* ctx, bool asLine )
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
   
   TextWriter* sc = static_cast<TextWriter*>(ctx->self().asInst());
   
   bool value;
   if( asLine )
   {
      value = sc->writeLine( *i_data->asString(), start, count );
   }
   else
   {
      value = sc->write( *i_data->asString(), start, count );
   }
   
   ctx->returnFrame( value );   
}


void Function_write::invoke(VMContext* ctx, int32 )
{
   internal_write(this, ctx, false );
}


void Function_writeLine::invoke(VMContext* ctx, int32 )
{   
   internal_write(this, ctx, true );
}


void Function_putChar::invoke(VMContext* ctx, int32 )
{
   Item* i_data = ctx->param(0);

   if( i_data == 0 || ! (i_data->isString() || ! i_data->isOrdinal() ) )
   {
      ctx->raiseError(this->paramError());
      return;
   }
   
   TextWriter* sc = static_cast<TextWriter*>(ctx->self().asInst());
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
   
   sc->putChar( chr );
   ctx->returnFrame();
}


void Function_flush::invoke(VMContext* ctx, int32 )
{   
   TextWriter* sc = static_cast<TextWriter*>(ctx->self().asInst());
   sc->flush();
   ctx->returnFrame();
}


void Function_close::invoke(VMContext* ctx, int32 )
{
   TextWriter* sc = static_cast<TextWriter*>(ctx->self().asInst());
   sc->flush();
   sc->underlying()->close();
   ctx->returnFrame();
}
}

//================================================================
//

ClassTextWriter::ClassTextWriter( ClassStream* clsStream ):
   Class( "TextWriter" ),
   m_clsStream( clsStream )
{
   addProperty( "encoding", &get_encoding, &set_encoding );
   addProperty( "crlf", &get_crlf, &set_crlf );
   addProperty( "lineflush", &get_lineflush, &set_lineflush );
   addProperty( "buffer", &get_buffer, &set_buffer );
   addProperty( "underlying", &get_underlying );
   
   addMethod( new CTextWriter::Function_write );
   addMethod( new CTextWriter::Function_writeLine );   
   addMethod( new CTextWriter::Function_putChar );    
   addMethod( new CTextWriter::Function_flush );
   addMethod( new CTextWriter::Function_close );  
}

ClassTextWriter::~ClassTextWriter()
{
}

void* ClassTextWriter::createInstance() const
{
   return FALCON_CLASS_CREATE_AT_INIT;
}


void ClassTextWriter::dispose( void* instance ) const
{
   TextWriter* tw = static_cast<TextWriter*>(instance);
   delete tw;
}

void* ClassTextWriter::clone( void* ) const
{
   return 0;
}

void ClassTextWriter::gcMarkInstance( void* instance, uint32 mark ) const
{
   TextWriter* tw = static_cast<TextWriter*>(instance);
   tw->gcMark( mark );
}

bool ClassTextWriter::gcCheckInstance( void* instance, uint32 mark ) const
{
   TextWriter* tw = static_cast<TextWriter*>(instance);
   return tw->currentMark() >= mark;
}

bool ClassTextWriter::op_init( VMContext* ctx, void* , int pcount ) const
{
   static Engine* eng = Engine::instance();
   
   // if we have 2 parameters, the second one is the encoding.
   String* encoding = 0;
   Stream* stc = 0;
   Transcoder* tc = 0;
   
   if( pcount > 0 )
   {
      Item* params = ctx->opcodeParams(pcount);
      
      Class* cls;
      void* data;
      if( params[0].asClassInst(cls, data) && cls->isDerivedFrom(m_clsStream) )
      {
         stc = static_cast<Stream*>(cls->getParentData( m_clsStream, data) );
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
   
   TextWriter* twc = new TextWriter(stc);
   if( tc != 0 )
   {
      twc->setEncoding(tc);
   }
   ctx->opcodeParam(pcount) = FALCON_GC_STORE(this, twc);
   
   return false;
}

}
}

/* end of textwriter.cpp */
