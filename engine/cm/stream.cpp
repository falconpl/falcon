/*
   FALCON - The Falcon Programming Language.
   FILE: stream.cpp

   Falcon core module -- Interface to Stream class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 19 Jul 2011 21:49:29 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/cm/stream.cpp"


#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/errors/paramerror.h>
#include <falcon/errors/codeerror.h>

#include <falcon/cm/uri.h>

#include <falcon/cm/stream.h>
#include <falcon/errors/unsupportederror.h>
#include <falcon/errors/ioerror.h>
#include <falcon/stream.h>

#include <falcon/streambuffer.h>
#include <falcon/transcoder.h>
#include <falcon/memory.h>
#include <falcon/errors/encodingerror.h>

#include <string.h>

namespace Falcon {
namespace Ext {
 
StreamCarrier::StreamCarrier( Stream* stream ):
   m_gcMark(0),
   m_stream(stream),
   m_sbuf(0),
   m_underlying(stream)
{}


StreamCarrier::~StreamCarrier()
{   
   delete m_sbuf;
   delete m_underlying;
}


void StreamCarrier::setBuffering( uint32 size )
{
   // shall we initialize a buffer?
   if( m_sbuf == 0 && size != 0 )
   {
      m_sbuf = new StreamBuffer( m_underlying, false, size );
      m_stream = m_sbuf;
   }
   else {
      // with size 0 remove buffering.
      if( size == 0 )
      {
         if( m_sbuf != 0 )
         {
            m_stream = m_underlying;
            delete m_sbuf;
            m_sbuf = 0;
         }
         // otherwise, we have nothing to do, size = 0 but it's already so.
      }
      else {
         m_sbuf->resizeBuffer( size );
      }
   }
}

void StreamCarrier::onFlushingOperation()
{
   // normally, do nothing.
}

//==================================================
//

ClassStream::ClassStream():
   ClassUser("Stream"),
   
   FALCON_INIT_PROPERTY( error ),
   FALCON_INIT_PROPERTY( moved ),
   FALCON_INIT_PROPERTY( position ),
   FALCON_INIT_PROPERTY( status ),
   FALCON_INIT_PROPERTY( eof ),
   FALCON_INIT_PROPERTY( interrupted ),
   FALCON_INIT_PROPERTY( bad ),
   FALCON_INIT_PROPERTY( good ),
   FALCON_INIT_PROPERTY( isopen ),
   FALCON_INIT_PROPERTY( buffer ),

   FALCON_INIT_METHOD( write ),
   FALCON_INIT_METHOD( read ),
   FALCON_INIT_METHOD( grab ),
   FALCON_INIT_METHOD( close ),
   FALCON_INIT_METHOD( seekBeg ),
   FALCON_INIT_METHOD( seekCur ),
   FALCON_INIT_METHOD( seekEnd ),
   FALCON_INIT_METHOD( seek ),
   FALCON_INIT_METHOD( tell ),
   FALCON_INIT_METHOD( flush ),   
   FALCON_INIT_METHOD( trunc ),
   FALCON_INIT_METHOD( ravail ),
   FALCON_INIT_METHOD( wavail )
{
}

ClassStream::~ClassStream()
{}


void ClassStream::dispose( void* instance ) const
{
   delete static_cast<StreamCarrier*>(instance);
}

void* ClassStream::clone( void* insatnce ) const
{
   // TODO: Clone the underlying streams to have new file pointers
   return insatnce;
}


void ClassStream::gcMark( void* instance, uint32 mark ) const
{
   StreamCarrier* carrier = static_cast<StreamCarrier*>(instance);
   carrier->m_gcMark = mark;
}

bool ClassStream::gcCheck( void* instance, uint32 mark ) const
{
   StreamCarrier* carrier = static_cast<StreamCarrier*>(instance);
   return carrier->m_gcMark < mark;
}
   
void ClassStream::op_create( VMContext*, int32 ) const
{
   throw new UnsupportedError( ErrorParam(e_abstract_init, __LINE__, __FILE__ )
      .extra("Stream") 
      );
}

void* ClassStream::createInstance( Item*, int ) const
{
   // never really called
   return 0;
}
   
//====================================================
// Properties.
//

FALCON_DEFINE_PROPERTY_SET( ClassStream, error )( void*, const Item& )
{
   throw new ParamError( ErrorParam(e_prop_ro, __LINE__, SRC ).extra(name()));
}

FALCON_DEFINE_PROPERTY_GET_P( ClassStream, error )
{
   value.setBoolean(static_cast<StreamCarrier*>(instance)->m_stream->error());
}


FALCON_DEFINE_PROPERTY_SET( ClassStream, moved )( void*, const Item& )
{
   throw new ParamError( ErrorParam(e_prop_ro, __LINE__, SRC ).extra(name()));
}

FALCON_DEFINE_PROPERTY_GET( ClassStream, moved )( void*, Item& value)
{
   value = (int64) 0; // not implemented
}


FALCON_DEFINE_PROPERTY_SET_P( ClassStream, position )
{
   checkType( value.isOrdinal(), "N" );
   int64 pos = static_cast<StreamCarrier*>(instance)
               ->m_stream->seekBegin( value.forceInteger() );
   
   if( ! pos < 0 )
   {
      throw new IOError( ErrorParam( e_io_seek, __LINE__, SRC ) );
   }
}


FALCON_DEFINE_PROPERTY_GET_P( ClassStream, position )
{
   value = static_cast<StreamCarrier*>(instance)->m_stream->tell();
}
  
  
FALCON_DEFINE_PROPERTY_SET( ClassStream, status )( void*, const Item& )
{
   throw new ParamError( ErrorParam(e_prop_ro, __LINE__, SRC ).extra(name()));
}

FALCON_DEFINE_PROPERTY_GET_P( ClassStream, status )
{
   value = (int64) static_cast<StreamCarrier*>(instance)->m_stream->status();
}
   

FALCON_DEFINE_PROPERTY_SET( ClassStream, eof )( void*, const Item& )
{
   throw new ParamError( ErrorParam(e_prop_ro, __LINE__, SRC ).extra(name()));
}

FALCON_DEFINE_PROPERTY_GET_P( ClassStream, eof )
{
   value.setBoolean(static_cast<StreamCarrier*>(instance)->m_stream->eof());
}

FALCON_DEFINE_PROPERTY_SET( ClassStream, interrupted )( void*, const Item& )
{
   throw new ParamError( ErrorParam(e_prop_ro, __LINE__, SRC ).extra(name()));
}

FALCON_DEFINE_PROPERTY_GET_P( ClassStream, interrupted )
{
   value.setBoolean(static_cast<StreamCarrier*>(instance)->m_stream->interrupted());
}

FALCON_DEFINE_PROPERTY_SET( ClassStream, bad )( void*, const Item& )
{
   throw new ParamError( ErrorParam(e_prop_ro, __LINE__, SRC ).extra(name()));
}

FALCON_DEFINE_PROPERTY_GET_P( ClassStream, bad )
{
   value.setBoolean(static_cast<StreamCarrier*>(instance)->m_stream->bad());
}

FALCON_DEFINE_PROPERTY_SET( ClassStream, good )( void*, const Item& )
{
   throw new ParamError( ErrorParam(e_prop_ro, __LINE__, SRC ).extra(name()));
}

FALCON_DEFINE_PROPERTY_GET_P( ClassStream, good )
{
   value.setBoolean(static_cast<StreamCarrier*>(instance)->m_stream->good());
}


FALCON_DEFINE_PROPERTY_SET( ClassStream, isopen )( void*, const Item& )
{
   throw new ParamError( ErrorParam(e_prop_ro, __LINE__, SRC ).extra(name()));
}

FALCON_DEFINE_PROPERTY_GET_P( ClassStream, isopen )
{
   value.setBoolean(static_cast<StreamCarrier*>(instance)->m_stream->open());
}


FALCON_DEFINE_PROPERTY_SET_P( ClassStream, buffer )
{
   if ( ! value.isInteger() )
   {
      throw new ParamError( ErrorParam(e_inv_prop_value, __LINE__, SRC ).extra("N"));
   }
   
   StreamCarrier* sc = static_cast<StreamCarrier*>(instance);
   uint32 v = value.asInteger() < 0 ? 0 : value.asInteger();
   sc->setBuffering( v );
}

FALCON_DEFINE_PROPERTY_GET_P( ClassStream, buffer )
{
   StreamCarrier* sc = static_cast<StreamCarrier*>(instance);
   uint32 bufSize = sc->m_sbuf == 0 ? 0 : sc->m_sbuf->bufferSize();   
   value = (int64) bufSize;
}

//======================================================
// Methods
//

FALCON_DEFINE_METHOD_P1( ClassStream, write )
{
   StreamCarrier* sc = static_cast<StreamCarrier*>(ctx->self().asInst());
   
   Item* i_data = ctx->param(0);
   if( i_data == 0 || !(i_data->isString()||i_data->isMemBuf()) )
   {
      throw paramError();
   }
   
   Item* i_count = ctx->param(1);
   Item* i_start = ctx->param(2);
   
   if ( (i_count != 0 && !(i_count->isOrdinal() || i_count->isNil())) ||
        (i_start != 0 && !(i_start->isOrdinal() || i_start->isNil()))
      )
   {
      throw paramError();
   }
   
   byte* dataSource; 
   uint32 dataSize;
   
   uint32 count = String::npos;
   uint32 start = 0;
   
   if( i_start != 0 && ! i_start->isNil() )
   {
      int64 istart = i_start->forceInteger();
      if ( istart > 0 )
      {
         start = (uint32) start;
      }
   }
   
   
   if( i_count != 0 && ! i_count->isNil() )
   {
      int64 icount = i_start->forceInteger();
      if ( icount > 0 )
      {
         count = (uint32) icount;
      }
   }
      
   if( i_data->isString() )
   {
      dataSource = i_data->asString()->getRawStorage();
      dataSize = i_data->asString()->size();
   }
   else
   {
      /* TODO: Membuf
      dataSource = i_data->isMem;
      dataSize = i_data->asString()->size();
       */
   }
   
   if( count == String::npos ) 
   {
      count = dataSize;
   }
   
   if( start + count > dataSize )
   {
      count = dataSize - start;
   }
   
   if( start >= dataSize || count <= 0 )
   {
      // nothing to do.
      ctx->returnFrame( (int64) 0 );
      return;
   }
   
   int64 retval = (int64) sc->m_stream->write(dataSource+start, count);   
   ctx->returnFrame( retval );
}


FALCON_DEFINE_METHOD_P1( ClassStream, read )
{
   Item* i_data = ctx->param(0);
   if( i_data == 0 || !(i_data->isString()||i_data->isMemBuf()) )
   {
      throw paramError();
   }
   
   Item* i_count = ctx->param(1);
   Item* i_start = ctx->param(2);
   
   if ( (i_count != 0 && !(i_count->isOrdinal() || i_count->isNil())) ||
        (i_start != 0 && !(i_start->isOrdinal() || i_start->isNil()))
      )
   {
      throw paramError();
   }
   
   uint32 dataSize = i_data->asString()->size();
   uint32 count = dataSize;
   uint32 start = 0;
   
   if( i_start != 0 && ! i_start->isNil() )
   {
      int64 istart = i_start->forceInteger();
      if ( istart > 0 )
      {
         start = (uint32) start;
      }
   }
   
   
   if( i_count != 0 && ! i_count->isNil() )
   {
      int64 icount = i_start->forceInteger();
      if ( icount > 0 )
      {
         count = (uint32) icount;
      }
   }
   
   if( count == 0 )
   {
      // nothing to do
      ctx->returnFrame( (int64) 0 );
      return;
   }
   
   if( start + count > dataSize )
   {
      i_data->asString()->reserve( start+count );
   }
   
   byte* dataSource = i_data->asString()->getRawStorage();
   StreamCarrier* sc = static_cast<StreamCarrier*>(ctx->self().asInst());
   int64 retval = (int64) sc->m_stream->read(dataSource+start, count);   
   ctx->returnFrame( retval );   
}


FALCON_DEFINE_METHOD_P1( ClassStream, grab )
{
   Item* i_count = ctx->param(0);
   if( i_count == 0 || !(i_count->isOrdinal()) )
   {
      throw paramError();
   }

   String* str = new String;
   Item rv(str, true); // force to garbage the string NOW!
   
   int64 icount = i_count->forceInteger();
   // shall we read?
   if( icount > 0 )
   {
      str->reserve( icount );
      StreamCarrier* sc = static_cast<StreamCarrier*>(ctx->self().asInst());
      int64 retval = (int64) sc->m_stream->read( str->getRawStorage(), icount );
      str->size( retval );
   }
   
   // Return the string.
   ctx->returnFrame( rv );
}

FALCON_DEFINE_METHOD_P1( ClassStream, close )
{
   StreamCarrier* sc = static_cast<StreamCarrier*>(ctx->self().asInst());
   sc->m_stream->close();
   ctx->returnFrame();
}


FALCON_DEFINE_METHOD_P1( ClassStream, seekBeg )
{
   Item* i_loc = ctx->param(0);
   if( i_loc == 0 || !(i_loc->isOrdinal()) )
   {
      throw paramError();
   }
   StreamCarrier* sc = static_cast<StreamCarrier*>(ctx->self().asInst());

   sc->onFlushingOperation();
   ctx->returnFrame( sc->m_stream->seekBegin( i_loc->forceInteger() ) );
}


FALCON_DEFINE_METHOD_P1( ClassStream, seekCur )
{   
   Item* i_loc = ctx->param(0);
   if( i_loc == 0 || !(i_loc->isOrdinal()) )
   {
      throw paramError();
   }
   StreamCarrier* sc = static_cast<StreamCarrier*>(ctx->self().asInst());
   
   sc->onFlushingOperation();
   ctx->returnFrame( sc->m_stream->seekCurrent( i_loc->forceInteger() ) );
}


FALCON_DEFINE_METHOD_P1( ClassStream, seekEnd )
{   
   Item* i_loc = ctx->param(0);
   if( i_loc == 0 || !(i_loc->isOrdinal()) )
   {
      throw paramError();
   }
   StreamCarrier* sc = static_cast<StreamCarrier*>(ctx->self().asInst());
   
   sc->onFlushingOperation();
   ctx->returnFrame( sc->m_stream->seekEnd( i_loc->forceInteger() ) );
}

FALCON_DEFINE_METHOD_P1( ClassStream, seek )
{   
   Item* i_loc = ctx->param(0);
   Item* i_whence = ctx->param(1);
   if( i_loc == 0 || !(i_loc->isOrdinal()) || 
       ( i_whence != 0 && ! i_whence->isOrdinal() )
      )
   {
      throw paramError();
   }
   
   StreamCarrier* sc = static_cast<StreamCarrier*>(ctx->self().asInst());
   
   sc->onFlushingOperation();
   ctx->returnFrame( sc->m_stream->seek( i_loc->forceInteger(), 
         (Stream::e_whence) i_whence->forceInteger() ) );
}


FALCON_DEFINE_METHOD_P1( ClassStream, tell )
{   
   StreamCarrier* sc = static_cast<StreamCarrier*>(ctx->self().asInst());   
   ctx->returnFrame( sc->m_stream->tell() );
}


FALCON_DEFINE_METHOD_P1( ClassStream, flush )
{   
   StreamCarrier* sc = static_cast<StreamCarrier*>(ctx->self().asInst());   
   sc->onFlushingOperation();
   ctx->returnFrame( sc->m_stream->flush() );
}

FALCON_DEFINE_METHOD_P1( ClassStream, trunc )
{   
   Item* i_loc = ctx->param(0);
   if( i_loc != 0 && !(i_loc->isOrdinal()) )
   {
      throw paramError();
   }
   StreamCarrier* sc = static_cast<StreamCarrier*>(ctx->self().asInst());
   sc->onFlushingOperation();
   int64 loc = i_loc != 0 ? i_loc->forceInteger() : -1 ;
   ctx->returnFrame( sc->m_stream->truncate( loc ) );
}


FALCON_DEFINE_METHOD_P1( ClassStream, ravail )
{   
   Item* i_loc = ctx->param(0);
   if( i_loc != 0 && !(i_loc->isOrdinal()) )
   {
      throw paramError();
   }
   StreamCarrier* sc = static_cast<StreamCarrier*>(ctx->self().asInst());
   int64 loc = i_loc != 0 ? i_loc->forceInteger() : -1 ;
   ctx->returnFrame( (int64) sc->m_stream->readAvailable( loc ) );
}


FALCON_DEFINE_METHOD_P1( ClassStream, wavail )
{   
   Item* i_loc = ctx->param(0);
   if( i_loc != 0 && !(i_loc->isOrdinal()) )
   {
      throw paramError();
   }
   StreamCarrier* sc = static_cast<StreamCarrier*>(ctx->self().asInst());
   int64 loc = i_loc != 0 ? i_loc->forceInteger() : -1 ;
   ctx->returnFrame( (int64) sc->m_stream->writeAvailable( loc ) );
}

}
}

/* end of stream.cpp */
