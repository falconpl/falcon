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
#define SRC "falcon/classes/classstream.cpp"


#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/stderrors.h>

#include <falcon/classes/classstream.h>
#include <falcon/stream.h>

#include <falcon/streambuffer.h>
#include <falcon/transcoder.h>

#include <falcon/selectable.h>
#include <falcon/fstream.h>

#include <string.h>

namespace Falcon {

      
//====================================================
// Properties.
//

static void get_error( const Class*, const String&, void *instance, Item& value )
{
   value.setBoolean(static_cast<Stream*>(instance)->error());
}


static void get_moved( const Class*, const String&, void *, Item& value )
{
   value = (int64) 0; // not implemented
}


static void set_position( const Class*, const String&, void *instance, const Item& value )
{
   if( ! value.isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_prop_value, __LINE__, SRC)
         .extra("N"));
   }

   int64 pos = static_cast<Stream*>(instance)
               ->seekBegin( value.forceInteger() );
   
   if( pos < 0 )
   {
      throw new IOError( ErrorParam( e_io_seek, __LINE__, SRC ) );
   }
}


static void get_position( const Class*, const String&, void *instance, Item& value )
{
   value = static_cast<Stream*>(instance)->tell();
}
  

static void get_status( const Class*, const String&, void *instance, Item& value )
{
   value = (int64) static_cast<Stream*>(instance)->status();
}


static void get_eof( const Class*, const String&, void *instance, Item& value )
{
   value.setBoolean(static_cast<Stream*>(instance)->eof());
}


static void get_bad( const Class*, const String&, void *instance, Item& value )
{
   value.setBoolean(static_cast<Stream*>(instance)->bad());
}


static void get_good( const Class*, const String&, void *instance, Item& value )
{
   value.setBoolean(static_cast<Stream*>(instance)->good());
}


static void get_isopen( const Class*, const String&, void *instance, Item& value )
{
   value.setBoolean(static_cast<Stream*>(instance)->open());
}


static void set_buffer( const Class*, const String&, void *instance, const Item& value )
{
   if ( ! value.isInteger() )
   {
      throw new ParamError( ErrorParam(e_inv_prop_value, __LINE__, SRC ).extra("N"));
   }
   
   Stream* sc = static_cast<Stream*>(instance);
   uint32 v = (uint32) (value.asInteger() < 0 ? 0 : value.asInteger());
   if( sc->underlying() == 0 )
   {
      sc = new StreamBuffer(sc, v);
   }
   else {
      static_cast<StreamBuffer*>(sc)->resizeBuffer(v);
   }
}

static void get_buffer( const Class*, const String&, void *instance, Item& value )
{
   Stream* sc = static_cast<Stream*>(instance);
   uint32 bufSize = sc->underlying() == 0 ? 0 : static_cast<StreamBuffer*>(sc)->bufferSize();
   value = (int64) bufSize;
}


static void set_userItem( const Class*, const String&, void *instance, const Item& value )
{
   Stream* sc = static_cast<Stream*>(instance);
   sc->userItem() = value;
}


static void get_userItem( const Class*, const String&, void *instance, Item& value )
{
   Stream* sc = static_cast<Stream*>(instance);
   value = sc->userItem();
}


//======================================================
// Methods
//
FALCON_DECLARE_FUNCTION( write, "data:S, count:[N], start:[N]" );
void Function_write::invoke( ::Falcon::VMContext* ctx, ::Falcon::int32 )
{
   Stream* sc = static_cast<Stream*>(ctx->self().asInst());
   
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
   
   byte* dataSource=0; 
   uint32 dataSize=0;
   
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
   
   int64 retval = (int64) sc->write(dataSource+start, count);
   ctx->returnFrame( retval );
}

FALCON_DECLARE_FUNCTION( read, "data:S, count:[N], start:[N]" );
void Function_read::invoke( ::Falcon::VMContext* ctx, ::Falcon::int32 )
{
   Item* i_data = ctx->param(0);
   if( i_data == 0 || ! i_data->isString() )
   {
      throw paramError();
   }
   String* data = i_data->asString();
   if( data->isImmutable() )
   {
      throw new ParamError( ErrorParam(e_param_type, __LINE__, SRC).extra("Immutable string") );
   }
   
   Item* i_count = ctx->param(1);
   Item* i_start = ctx->param(2);
   
   if ( (i_count != 0 && !(i_count->isOrdinal() || i_count->isNil())) ||
        (i_start != 0 && !(i_start->isOrdinal() || i_start->isNil()))
      )
   {
      throw paramError();
   }
   
   uint32 dataSize = i_data->asString()->allocated();
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
      int64 icount = i_count->forceInteger();
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
   Stream* sc = static_cast<Stream*>(ctx->self().asInst());
   int64 retval = (int64) sc->read(dataSource+start, count);
   if( retval > 0 )
   {
      i_data->asString()->size( (length_t)(start + retval) );
   }
   ctx->returnFrame( retval );   
}


FALCON_DECLARE_FUNCTION( grab, "count:N" );
void Function_grab::invoke( ::Falcon::VMContext* ctx, ::Falcon::int32 )
{
   Item* i_count = ctx->param(0);
   if( i_count == 0 || !(i_count->isOrdinal()) )
   {
      throw paramError();
   }

   String* str = new String;
   Item rv(FALCON_GC_HANDLE(str)); // force to garbage the string NOW!
   
   int64 icount = i_count->forceInteger();
   // shall we read?
   if( icount > 0 )
   {
      str->reserve( (uint32) icount );
      Stream* sc = static_cast<Stream*>(ctx->self().asInst());
      int64 retval = (int64) sc->read( str->getRawStorage(), (size_t) icount );
      str->size( (length_t) retval );
   }
   
   // Return the string.
   ctx->returnFrame( rv );
}


FALCON_DECLARE_FUNCTION( close, "" );
void Function_close::invoke( ::Falcon::VMContext* ctx, ::Falcon::int32 )
{
   Stream* sc = static_cast<Stream*>(ctx->self().asInst());
   sc->close();
   ctx->returnFrame();
}


FALCON_DECLARE_FUNCTION( seekBeg, "position:N" );
void Function_seekBeg::invoke( ::Falcon::VMContext* ctx, ::Falcon::int32 )
{
   Item* i_loc = ctx->param(0);
   if( i_loc == 0 || !(i_loc->isOrdinal()) )
   {
      throw paramError();
   }
   Stream* sc = static_cast<Stream*>(ctx->self().asInst());
   ctx->returnFrame( sc->seekBegin( i_loc->forceInteger() ) );
}


FALCON_DECLARE_FUNCTION( seekCur, "position:N" );
void Function_seekCur::invoke( ::Falcon::VMContext* ctx, ::Falcon::int32 )
{   
   Item* i_loc = ctx->param(0);
   if( i_loc == 0 || !(i_loc->isOrdinal()) )
   {
      throw paramError();
   }
   Stream* sc = static_cast<Stream*>(ctx->self().asInst());
   ctx->returnFrame( sc->seekCurrent( i_loc->forceInteger() ) );
}


FALCON_DECLARE_FUNCTION( seekEnd, "position:N" );
void Function_seekEnd::invoke( ::Falcon::VMContext* ctx, ::Falcon::int32 )
{   
   Item* i_loc = ctx->param(0);
   if( i_loc == 0 || !(i_loc->isOrdinal()) )
   {
      throw paramError();
   }
   Stream* sc = static_cast<Stream*>(ctx->self().asInst());
   ctx->returnFrame( sc->seekEnd( i_loc->forceInteger() ) );
}

FALCON_DECLARE_FUNCTION( seek, "position:N,whence:N" );
void Function_seek::invoke( ::Falcon::VMContext* ctx, ::Falcon::int32 )
{   
   Item* i_loc = ctx->param(0);
   Item* i_whence = ctx->param(1);
   if( i_loc == 0 || !(i_loc->isOrdinal()) || 
       ( i_whence != 0 && ! i_whence->isOrdinal() )
      )
   {
      throw paramError();
   }
   
   Stream* sc = static_cast<Stream*>(ctx->self().asInst());
   Stream::e_whence wh = i_whence == 0 ? Stream::ew_begin : (Stream::e_whence) i_whence->forceInteger();
   
   int64 loc = i_loc->forceInteger();
   int64 pos = sc->seek( loc, wh);
   ctx->returnFrame( Item().setInteger(pos) );
}


FALCON_DECLARE_FUNCTION( tell, "" );
void Function_tell::invoke( ::Falcon::VMContext* ctx, ::Falcon::int32 )
{   
   Stream* sc = static_cast<Stream*>(ctx->self().asInst());
   ctx->returnFrame( sc->tell() );
}


FALCON_DECLARE_FUNCTION( flush, "" );
void Function_flush::invoke( ::Falcon::VMContext* ctx, ::Falcon::int32 )
{   
   Stream* sc = static_cast<Stream*>(ctx->self().asInst());
   ctx->returnFrame( sc->flush() );
}


FALCON_DECLARE_FUNCTION( trunc, "position:[N]" );
void Function_trunc::invoke( ::Falcon::VMContext* ctx, ::Falcon::int32 )
{   
   Item* i_loc = ctx->param(0);
   if( i_loc != 0 && !(i_loc->isOrdinal()) )
   {
      throw paramError();
   }
   Stream* sc = static_cast<Stream*>(ctx->self().asInst());
   int64 loc = i_loc != 0 ? i_loc->forceInteger() : -1 ;
   ctx->returnFrame( sc->truncate( loc ) );
}


static void register_on(ClassStream* cls )
{
   cls->addProperty( "error", &get_error );
   cls->addProperty( "moved", &get_moved );
   // property is hidden, as many streams don't provide this
   cls->addProperty( "position", &get_position, &set_position, false, true );
   cls->addProperty( "status", &get_status );
   cls->addProperty( "eof", &get_eof );
   cls->addProperty( "bad", &get_bad );
   cls->addProperty( "good", &get_good );
   cls->addProperty( "isopen", &get_isopen );
   cls->addProperty( "buffer", &get_buffer, &set_buffer );
   // Hidden, as not normally used
   cls->addProperty( "userItem", &get_userItem, &set_userItem, false, true );
   
   cls->addMethod( new Function_read );
   cls->addMethod( new Function_write );
   cls->addMethod( new Function_grab );
   cls->addMethod( new Function_close );
   cls->addMethod( new Function_seekBeg );
   cls->addMethod( new Function_seekCur );
   cls->addMethod( new Function_seekEnd );
   cls->addMethod( new Function_seek );
   cls->addMethod( new Function_tell );
   cls->addMethod( new Function_flush );
   cls->addMethod( new Function_trunc ); 
}

ClassStream::ClassStream():
   Class("Stream")
{
   register_on(this);
}

ClassStream::ClassStream( const String& subclassName ):
   Class(subclassName)
{
   register_on(this);
}

ClassStream::~ClassStream()
{}


void ClassStream::dispose( void* instance ) const
{
   static_cast<Stream*>(instance)->decref();
}

void* ClassStream::clone( void* insatnce ) const
{
   // TODO: Clone the underlying streams to have new file pointers
   Stream* sc = static_cast<Stream*>(insatnce);
   sc->incref();
   return sc;
}


void ClassStream::gcMarkInstance( void* instance, uint32 mark ) const
{
   Stream* carrier = static_cast<Stream*>(instance);
   carrier->gcMark(mark);
}

bool ClassStream::gcCheckInstance( void* instance, uint32 mark ) const
{
   Stream* carrier = static_cast<Stream*>(instance);
   return carrier->gcMark() >= mark;
}
   
void* ClassStream::createInstance() const
{
   // never really called
   return 0;
}


namespace {
   class StreamSelectable: public FDSelectable
   {
   public:
      StreamSelectable( const Class* cls, Stream* inst ):
         FDSelectable( cls, inst )
      {
         inst->incref();
      }

      virtual ~StreamSelectable() {
         static_cast<Stream*>(instance())->decref();
      }

      const Multiplex::Factory* factory() const {
         return static_cast<Stream*>(instance())->multiplexFactory();
      }

      // we're using FD files on POXSIX only. On windows, we're using handles,
      // and the windows file multiplexer uses the base Selectable interface.
      // Also, getFD is called only by those multiplex that know they're handling
      // fstreams or sublcasses (it's the stream that creates the multiplex!)
      virtual int getFd() const
      {
         #ifdef FALCON_SYSTEM_WIN
         return 0;
         #else
         // warning: not all streams are fstreams!. Let the multiplex only
         // call this function!
         return static_cast<FStream*>(instance())->fileData()->fdFile;
         #endif
      }
   };
}


Selectable* ClassStream::getSelectableInterface( void* instance ) const
{
   Stream* stream = static_cast<Stream*>(instance);
   return new StreamSelectable( this, stream );
}

}

/* end of classstream.cpp */
