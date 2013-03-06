/*
   FALCON - The Falcon Programming Language.
   FILE: textstream.cpp

   Falcon core module -- Text oriented stream abstraction.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 19 Jul 2011 21:49:29 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/cm/textstream.cpp"

#include <falcon/stream.h>
#include <falcon/vmcontext.h>
#include <falcon/errors/paramerror.h>
#include <falcon/stdhandlers.h>
#include <falcon/function.h>

#include <falcon/cm/textstream.h>
#include <math.h>


namespace Falcon {
namespace Ext {

TextStreamCarrier::TextStreamCarrier( Stream* stream ):
   m_reader( new TextReader(stream ) ),
   m_writer( new TextWriter(stream ) )
{
}

TextStreamCarrier::~TextStreamCarrier()
{
   m_reader->decref();
   m_writer->decref();
}
   
void TextStreamCarrier::onFlushingOperation()
{
   m_writer->flush();
   m_reader->changeStream( m_reader->underlying(), true );
}


bool TextStreamCarrier::setEncoding( const String& encName )
{
   static Engine* eng = Engine::instance();
   
   Transcoder* tc = eng->getTranscoder(encName);
   if( tc == 0 )
   {
      return false;
   }
   
   m_reader->setEncoding(tc);
   m_writer->setEncoding(tc);
   return true;
}




//==========================================================
// Properties
//

static void set_encoding( const Class*, const String&, void* instance, const Item& value )
{
   TextStreamCarrier* sc = static_cast<TextStreamCarrier*>(instance);
   if( value.isNil() )
   {
      sc->setEncoding( "C" );
   }
   else if( value.isString() )
   {
      if( ! sc->setEncoding( *value.asString() ) )
      {      
         // unknown encoding
         throw new ParamError( ErrorParam( e_param_range, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime )
            .extra("Unknown encoding " + *value.asString()));
      }      
   }
   else
   {
      throw new ParamError( ErrorParam(e_inv_prop_value, __LINE__, SRC ).extra("S|Nil"));
   }   
}


static void get_encoding( const Class*, const String&, void* instance, Item& value )
{
   TextStreamCarrier* sc = static_cast<TextStreamCarrier*>(instance);     
   value = sc->m_encoding;
}

//========================================================
// Methods
//
namespace CTextStream {

FALCON_DECLARE_FUNCTION( write, "text:S, count:[N], start:[N]" );
FALCON_DECLARE_FUNCTION( read, "text:S, count:[N]" );   
FALCON_DECLARE_FUNCTION( grab, "count:N" );
FALCON_DECLARE_FUNCTION( readLine, "text:S, maxCount:[N]" );   
FALCON_DECLARE_FUNCTION( grabLine, "maxCount:[N]" );   
FALCON_DECLARE_FUNCTION( readChar, "text:S, append:[B]" );
FALCON_DECLARE_FUNCTION( getChar, "" );
FALCON_DECLARE_FUNCTION( ungetChar, "char:S|N" );
FALCON_DECLARE_FUNCTION( putChar, "char:S|N" );
FALCON_DECLARE_FUNCTION( sync, "" );
FALCON_DECLARE_FUNCTION( flush, "" );
FALCON_DECLARE_FUNCTION( close, "" );

void Function_write::invoke(VMContext* ctx, int32 )
{   
   Item* i_data = ctx->param(0);
   if( i_data == 0 || !i_data->isString() )
   {
      ctx->raiseError(paramError());
      return;
   }
   
   Item* i_count = ctx->param(1);
   Item* i_start = ctx->param(2);
   
   if ( (i_count != 0 && !(i_count->isOrdinal() || i_count->isNil())) ||
        (i_start != 0 && !(i_start->isOrdinal() || i_start->isNil()))
      )
   {
      ctx->raiseError(paramError());
      return;
   }
   
   length_t count = i_count != 0 && i_count->isOrdinal() ? 
      static_cast<length_t>(i_count->forceInteger()) : String::npos;
   
   length_t start = i_start != 0 && i_start->isOrdinal() ? 
      static_cast<length_t>(i_start->forceInteger()) : 0;
   
   TextStreamCarrier* sc = static_cast<TextStreamCarrier*>(ctx->self().asInst());
   bool value = sc->m_writer->write( *i_data->asString(), start, count );
   ctx->returnFrame( value );   
}


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
           
   TextStreamCarrier* sc = static_cast<TextStreamCarrier*>(ctx->self().asInst());
   bool value = sc->m_reader->read( *i_data->asString(), (length_t) i_count->forceInteger() );
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
      TextStreamCarrier* sc = static_cast<TextStreamCarrier*>(ctx->self().asInst());
      try {
         sc->m_reader->read( *str, (length_t) icount );
      }
      catch( ... ) {
         delete str;
         throw;
      }
   }
   
   // Return the string.
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
      TextStreamCarrier* sc = static_cast<TextStreamCarrier*>(ctx->self().asInst());
      bool value = sc->m_reader->readLine( *i_data->asString(), count );
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
      TextStreamCarrier* sc = static_cast<TextStreamCarrier*>(ctx->self().asInst());
      try {
         sc->m_reader->readLine( *str, count );
      }
      catch(...)
      {
         delete str;
         throw;
      }
   }
   
   // Return the string.
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
   
   
   TextStreamCarrier* sc = static_cast<TextStreamCarrier*>(ctx->self().asInst());
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


void Function_getChar::invoke(VMContext* ctx, int32 )
{   
   TextStreamCarrier* sc = static_cast<TextStreamCarrier*>(ctx->self().asInst());
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


static void inner_putchar( Function* called, VMContext* ctx, bool isUnget )
{
   Item* i_data = ctx->param(0);

   if( i_data == 0 || ! (i_data->isString() || ! i_data->isOrdinal() ) )
   {
      ctx->raiseError(called->paramError());
      return;
   }
   
   TextStreamCarrier* sc = static_cast<TextStreamCarrier*>(ctx->self().asInst());
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
   
   if( isUnget )
   {
      sc->m_reader->ungetChar( chr );
   }
   else
   {
      sc->m_writer->putChar( chr );
   }
   ctx->returnFrame();
}


void Function_ungetChar::invoke(VMContext* ctx, int32 )
{
   inner_putchar( this, ctx, true );
}


void Function_putChar::invoke(VMContext* ctx, int32 )
{
   inner_putchar( this, ctx, false );
}


void Function_close::invoke(VMContext* ctx, int32 )
{
   TextStreamCarrier* sc = static_cast<TextStreamCarrier*>(ctx->self().asInst());
   sc->m_writer->flush();
   sc->m_reader->underlying()->close();
   ctx->returnFrame();
}


void Function_flush::invoke(VMContext* ctx, int32 )
{
   TextStreamCarrier* sc = static_cast<TextStreamCarrier*>(ctx->self().asInst());
   sc->m_writer->flush();
   ctx->returnFrame();
}


void Function_sync::invoke(VMContext* ctx, int32 )
{
   TextStreamCarrier* sc = static_cast<TextStreamCarrier*>(ctx->self().asInst());
   sc->m_reader->changeStream( sc->m_reader->underlying(), true );
   ctx->returnFrame();
}
}

//===============================================================
// Main class
//

ClassTextStream::ClassTextStream( ClassStream* parent ):
   ClassStream("TextStream"),
   m_stream( parent )
{
   setParent( parent );
   m_stream = parent;
      
   addProperty( "encoding", &get_encoding, &set_encoding );
   
   addMethod( new CTextStream::Function_write );
   addMethod( new CTextStream::Function_read );  
   addMethod( new CTextStream::Function_grab );
   addMethod( new CTextStream::Function_readLine );
   addMethod( new CTextStream::Function_grabLine );
   addMethod( new CTextStream::Function_readChar );
   addMethod( new CTextStream::Function_getChar );
   addMethod( new CTextStream::Function_ungetChar );
   addMethod( new CTextStream::Function_putChar );
   addMethod( new CTextStream::Function_sync );
   addMethod( new CTextStream::Function_flush );
   addMethod( new CTextStream::Function_close );
}

   
ClassTextStream::~ClassTextStream()
{}

void* ClassTextStream::createInstance() const
{
   return FALCON_CLASS_CREATE_AT_INIT;
}

bool ClassTextStream::op_init( VMContext* ctx, void* , int32 pcount ) const
{
   static Engine* eng = Engine::instance();
   
   Stream* scarrier = 0;
   String* sEncoding = 0;
   
   if ( pcount > 0 )
   {
      Item& streamInst = ctx->opcodeParam(pcount-1);
      Class* cls; 
      void* data;
      if( streamInst.asClassInst( cls, data ) && cls->isDerivedFrom( m_stream ) )
      {
         scarrier = static_cast<Stream*>(data);
      }
      
      if ( pcount > 1 )
      {
         Item& i_enc = ctx->opcodeParam(pcount-2);
         if( ! i_enc.isString() )
         {
            // force to generate the error
            scarrier = 0;
         }
         else
         {
            sEncoding = i_enc.asString();
         }
      }
   }      
         
   if( scarrier == 0 )
   {
      throw new ParamError( ErrorParam(e_inv_params, __LINE__,SRC)
            .origin(ErrorParam::e_orig_runtime)
            .extra("Stream,[S]") );
   }
   
   
   TextStreamCarrier* tsc = new TextStreamCarrier( scarrier );
   if( sEncoding != 0 )
   {
      Transcoder* tcode = eng->getTranscoder(*sEncoding);
      if( tcode == 0 )
      {
         throw new ParamError( ErrorParam(e_param_range, __LINE__,SRC)
               .origin(ErrorParam::e_orig_runtime)
               .extra("Unknown encoding " + *sEncoding) );
      }
      
      tsc->m_reader->setEncoding( tcode );
      tsc->m_writer->setEncoding( tcode );
      tsc->m_encoding = *sEncoding;
   }   
   
   return false;
}

}
}
