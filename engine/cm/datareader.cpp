/*
   FALCON - The Falcon Programming Language.
   FILE: datareader.cpp

   Falcon core module -- Wrapping for DataReader
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 11 Dec 2011 17:32:58 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/datareader.cpp"

#include <falcon/cm/datareader.h>

#include <falcon/datareader.h>
#include <falcon/stdsteps.h>
#include <falcon/module.h>
#include <falcon/classes/classstream.h>

#include <falcon/stderrors.h>
#include <falcon/vmcontext.h>
#include <falcon/stdhandlers.h>

namespace Falcon {
namespace Ext {

   
FALCON_DECLARE_FUNCTION( read, "data:S|M, count:[N]" );
FALCON_DECLARE_FUNCTION( readBool, "" );
FALCON_DECLARE_FUNCTION( readChar, "" );
FALCON_DECLARE_FUNCTION( readByte, "" );
FALCON_DECLARE_FUNCTION( readI16, "" );
FALCON_DECLARE_FUNCTION( readU16, "" );
FALCON_DECLARE_FUNCTION( readI32, "" );
FALCON_DECLARE_FUNCTION( readU32, "" );
FALCON_DECLARE_FUNCTION( readI64, "" );
FALCON_DECLARE_FUNCTION( readU64, "" );
FALCON_DECLARE_FUNCTION( readF32, "" );
FALCON_DECLARE_FUNCTION( readF64, "" );
FALCON_DECLARE_FUNCTION( readString, "" );
FALCON_DECLARE_FUNCTION( readItem, "model:Class" );

FALCON_DECLARE_FUNCTION( sync, "" );
FALCON_DECLARE_FUNCTION( eof, "" );

//=================================================================
// Properties
//

static void set_endianity( const Class*, const String&, void* instance, const Item& value )
{
   DataReader* sc = static_cast<DataReader*>(instance);
     
   if( !value.isOrdinal() )
   {
      // unknown encoding
      throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
         .origin( ErrorParam::e_orig_runtime )
         .extra( "N" ));
   }
   
   int64 edty = (int64) value.forceInteger();
   if( edty == DataReader::e_BE || edty == DataReader::e_LE
       || edty == DataReader::e_reverseEndian || edty == DataReader::e_sameEndian )
   {
      sc->setEndianity( (DataReader::t_endianity) edty );
   }
   else
   {
      throw new ParamError( ErrorParam( e_param_range, __LINE__, SRC )
         .origin( ErrorParam::e_orig_runtime )
         .extra( "Not a valid endianity setting" ));
   }
}


static void get_endianity( const Class*, const String&, void* instance, Item& value )
{
   DataReader* sc = static_cast<DataReader*>(instance);
   value = (int64)sc->endianity();
}


static void get_sysEndianity( const Class*, const String&, void* instance, Item& value )
{
   DataReader* sc = static_cast<DataReader*>(instance);
   DataReader::t_endianity edity = sc->endianity();
   if( sc->isSameEndianity() )
   {
      value = (int64)edity; 
   }
   else {
      value = (int64) (edity == DataReader::e_LE ? DataReader::e_BE : DataReader::e_LE );
   }
}

//=================================================================
// Methods
//

void Function_read::invoke(VMContext* ctx, int32 )
{
   Item* i_data = ctx->param(0);
   if( i_data == 0 || !(i_data->isString()||i_data->isMemBuf()) )
   {
      throw paramError();
   }
   
   Item* i_count = ctx->param(1);
   
   if ( (i_count != 0 && !(i_count->isOrdinal() || i_count->isNil())) 
      )
   {
      throw paramError();
   }
   
   uint32 dataSize = i_data->asString()->size();
   uint32 count = dataSize;  
      
   if( i_count != 0 && ! i_count->isNil() )
   {      
      count = (uint32) i_count->forceInteger();
   }
   
   if( count == 0 )
   {
      // nothing to do
      ctx->returnFrame( (int64) 0 );
      return;
   }
   
   if( count > dataSize )
   {
      i_data->asString()->reserve( count );
   }
   
   byte* dataSource = i_data->asString()->getRawStorage();
   DataReader* sc = static_cast<DataReader*>(ctx->self().asInst());
   int64 retval = (int64) sc->read(dataSource, count);   
   ctx->returnFrame( retval );   
}


void Function_readBool::invoke(VMContext* ctx, int32 )
{   
   bool bValue;
   DataReader* dw = static_cast<DataReader*>(ctx->self().asInst());
   dw->read(bValue);
   Item retVal;
   retVal.setBoolean(bValue);
   ctx->returnFrame( retVal );
}


void Function_readChar::invoke(VMContext* ctx, int32 )
{
   char chr;   
   DataReader* dw = static_cast<DataReader*>(ctx->self().asInst());
   dw->read(chr);
   ctx->returnFrame((int64) chr);
}


void Function_readByte::invoke(VMContext* ctx, int32 )
{
   byte value;   
   DataReader* dw = static_cast<DataReader*>(ctx->self().asInst());
   dw->read(value);
   ctx->returnFrame((int64) value);
}

void Function_readI16::invoke(VMContext* ctx, int32 )
{
   int16 value;   
   DataReader* dw = static_cast<DataReader*>(ctx->self().asInst());
   dw->read(value);
   ctx->returnFrame((int64) value);
}


void Function_readU16::invoke(VMContext* ctx, int32 )
{
   uint16 value;   
   DataReader* dw = static_cast<DataReader*>(ctx->self().asInst());
   dw->read(value);
   ctx->returnFrame((int64) value);
}


void Function_readI32::invoke(VMContext* ctx, int32 )
{
   int32 value;   
   DataReader* dw = static_cast<DataReader*>(ctx->self().asInst());
   dw->read(value);
   ctx->returnFrame((int64) value);
}


void Function_readU32::invoke(VMContext* ctx, int32 )
{
   uint16 value;   
   DataReader* dw = static_cast<DataReader*>(ctx->self().asInst());
   dw->read(value);
   ctx->returnFrame((int64) value);
}


void Function_readI64::invoke(VMContext* ctx, int32 )
{
   int64 value;   
   DataReader* dw = static_cast<DataReader*>(ctx->self().asInst());
   dw->read(value);
   ctx->returnFrame((int64) value);
}


void Function_readU64::invoke(VMContext* ctx, int32 )
{
   uint64 value;   
   DataReader* dw = static_cast<DataReader*>(ctx->self().asInst());
   dw->read(value);
   ctx->returnFrame((int64) value);
}


void Function_readF32::invoke(VMContext* ctx, int32 )
{
   float value;   
   DataReader* dw = static_cast<DataReader*>(ctx->self().asInst());
   dw->read(value);
   ctx->returnFrame((numeric) value);
}


void Function_readF64::invoke(VMContext* ctx, int32 )
{
   double value;   
   DataReader* dw = static_cast<DataReader*>(ctx->self().asInst());
   dw->read(value);
   ctx->returnFrame((numeric) value);
}


void Function_readString::invoke(VMContext* ctx, int32 )
{
   static Class* stringClass = Engine::handlers()->stringClass();
   String* str = new String;
   DataReader* dw = static_cast<DataReader*>(ctx->self().asInst());
   try
   {
      dw->read(*str);
   }
   catch( ... )
   {
      delete str;
      throw;
   }
   
   ctx->returnFrame( FALCON_GC_STORE( stringClass, str ) );
}


void Function_readItem::invoke(VMContext* ctx, int32 )
{
   static StdSteps* steps = Engine::instance()->stdSteps();

   Item* i_data = ctx->param(0);
   if( i_data == 0 || ! i_data->isClass() )
   {
      ctx->raiseError( paramError() );
      return;
   }
   
   DataReader* dw = static_cast<DataReader*>(ctx->self().asInst());
   Class* cls = static_cast<Class*>(i_data->asInst());
   
   // store may go deep, so we use a conditional return construct
   ctx->pushCode( &static_cast<ClassDataReader*>(m_methodOf)->m_readItemNext );
   
   long depth = ctx->codeDepth();
   
   // this will punch in later on...
   ctx->pushCode( &steps->m_returnFrameWithTop );
   cls->restore( ctx, dw );
   
   // ... if the restore process isn't done.
   if( depth == ctx->codeDepth() )
   {
      ctx->returnFrame( ctx->topData() );
   }
}


void Function_sync::invoke(VMContext* ctx, int32 )
{
  DataReader* dw = static_cast<DataReader*>(ctx->self().asInst());
  dw->sync();  
  ctx->returnFrame();
}


void Function_eof::invoke(VMContext* ctx, int32 )
{
  DataReader* dw = static_cast<DataReader*>(ctx->self().asInst());
  Item ret;
  ret.setBoolean(dw->eof());  
  ctx->returnFrame( ret );
}

//==================================================================



ClassDataReader::ClassDataReader( Class* clsStream ):
   Class("DataReader"),
   m_clsStream( clsStream ),
   m_readItemNext( this )  
{
   addProperty( "endianity", &get_endianity, &set_endianity );
   addProperty( "sysEndianity", &get_sysEndianity);
   
   addMethod( new Function_read );
   addMethod( new Function_readBool );
   addMethod( new Function_readChar );
   addMethod( new Function_readByte );
   addMethod( new Function_readI16 );
   addMethod( new Function_readU16 );
   addMethod( new Function_readI32 );
   addMethod( new Function_readU32 );
   addMethod( new Function_readI64 );
   addMethod( new Function_readU64 );
   addMethod( new Function_readF32 );
   addMethod( new Function_readF64 );
   addMethod( new Function_readString );
   addMethod( new Function_readItem );
   
   addMethod( new Function_sync );
   addMethod( new Function_eof );
}

ClassDataReader::~ClassDataReader()
{
}

   
void ClassDataReader::dispose( void* instance ) const
{
   DataReader* wr = static_cast<DataReader*>(instance);
   wr->decref();
}


void* ClassDataReader::clone( void* instance ) const
{
   // nothing to clone, it can be shared.
   DataReader* wr = static_cast<DataReader*>(instance);
   return new DataReader(*wr);
}


void ClassDataReader::gcMarkInstance( void* instance, uint32 mark ) const
{
   DataReader* wr = static_cast<DataReader*>(instance);
   wr->gcMark( mark );
}


bool ClassDataReader::gcCheckInstance( void* instance, uint32 mark ) const
{
   DataReader* wr = static_cast<DataReader*>(instance);
   return wr->gcMark() >= mark;
}


void* ClassDataReader::createInstance() const
{
   return new DataReader;
}


bool ClassDataReader::op_init( VMContext* ctx, void* instance, int pcount ) const
{
   if( pcount >= 1 )
   {
      Class* cls=0;
      void* data=0;
      
      Item* params = ctx->opcodeParams(pcount);
      params[0].asClassInst( cls, data );
      if( cls->isDerivedFrom(m_clsStream) )
      {
         DataReader* wr = static_cast<DataReader*>( instance );
         wr->changeStream(
               static_cast<Stream*>(cls->getParentData(m_clsStream, data)), false );
         // this data is going to be added to gc very soon.
         return false;
      }      
   }
   
   throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
         .origin(ErrorParam::e_orig_runtime)
         .extra( "Stream" ) );      
}


void ClassDataReader::ReadItemNext::apply_( const PStep*, VMContext* ctx )
{
   // we just got to return our good item
   //ClassDataReader::ReadItemNext* self = static_cast<ClassDataReader::ReadItemNext*>(ps);   
   ctx->returnFrame( *ctx->local(0) );
}

}
}

/* end of datareader.cpp */
