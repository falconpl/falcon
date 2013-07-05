/*
   FALCON - The Falcon Programming Language.
   FILE: datawriter.cpp

   Falcon core module -- Wrapping for DataWriter
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 11 Dec 2011 17:32:58 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/datawriter.cpp"

#include <falcon/cm/datawriter.h>

#include <falcon/datawriter.h>
#include <falcon/module.h>
#include <falcon/classes/classstream.h>

#include <falcon/stderrors.h>
#include <falcon/vmcontext.h>
#include <falcon/function.h>

namespace Falcon {
namespace Ext {

FALCON_DECLARE_FUNCTION( write, "data:S|M, count:[N], start:[N]" );
FALCON_DECLARE_FUNCTION( writeBool, "data:B" );
FALCON_DECLARE_FUNCTION( writeChar, "data:S" );
FALCON_DECLARE_FUNCTION( writeByte, "data:N" );
FALCON_DECLARE_FUNCTION( writeI16, "data:N" );
FALCON_DECLARE_FUNCTION( writeU16, "data:N" );
FALCON_DECLARE_FUNCTION( writeI32, "data:N" );
FALCON_DECLARE_FUNCTION( writeU32, "data:N" );
FALCON_DECLARE_FUNCTION( writeI64, "data:N" );
FALCON_DECLARE_FUNCTION( writeU64, "data:N" );
FALCON_DECLARE_FUNCTION( writeF32, "data:N" );
FALCON_DECLARE_FUNCTION( writeF64, "data:N" );
FALCON_DECLARE_FUNCTION( writeString, "data:S" );
FALCON_DECLARE_FUNCTION( writeItem, "data:X" );

FALCON_DECLARE_FUNCTION( flush, "" );

//=================================================================
// Properties
//

static void set_endianity( const Class*, const String&, void* instance, const Item& value )
{
   DataWriter* sc = static_cast<DataWriter*>(instance);
     
   if( !value.isOrdinal() )
   {
      // unknown encoding
      throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
         .origin( ErrorParam::e_orig_runtime )
         .extra( "N" ));
   }
   
   int64 edty = (int64) value.forceInteger();
   if( edty == DataWriter::e_BE || edty == DataWriter::e_LE
       || edty == DataWriter::e_reverseEndian || edty == DataWriter::e_sameEndian )
   {
      sc->setEndianity( (DataWriter::t_endianity) edty );
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
   DataWriter* sc = static_cast<DataWriter*>(instance);
   value = (int64)sc->endianity();
}


static void get_sysEndianity( const Class*, const String&, void* instance, Item& value )
{
   DataWriter* sc = static_cast<DataWriter*>(instance);
   DataWriter::t_endianity edity = sc->endianity();
   if( sc->isSameEndianity() )
   {
      value = (int64)edity; 
   }
   else {
      value = (int64) (edity == DataWriter::e_LE ? DataWriter::e_BE : DataWriter::e_LE );
   }
}

//=================================================================
// Methods
//

void Function_write::invoke(VMContext* ctx, int32 )
{
   DataWriter* dw = static_cast<DataWriter*>(ctx->self().asInst());
   
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

   int64 retval = (int64) dw->writeRaw(dataSource+start, count);   
   ctx->returnFrame( retval );
   ctx->returnFrame();
}


void Function_writeBool::invoke(VMContext* ctx, int32 )
{
   Item* i_data = ctx->param(0);
   if( i_data == 0 )
   {
      ctx->raiseError( paramError() );
      return;
   }
   
   bool bValue = i_data->isTrue();
   DataWriter* dw = static_cast<DataWriter*>(ctx->self().asInst());
   dw->write(bValue);
   ctx->returnFrame();
}


void Function_writeChar::invoke(VMContext* ctx, int32 )
{
   Item* i_data = ctx->param(0);
   if( i_data == 0 )
   {
      ctx->raiseError( paramError() );
      return;
   }
   
   char value = 0;
   if( i_data->isOrdinal() )
   {
      value = (char) i_data->forceInteger();
   }
   else if( i_data->isString() && i_data->asString()->length() > 0 )
   {
      value = (char) i_data->asString()->getCharAt(0);
   }
   else
   {
      ctx->raiseError( paramError() );
      return;
   }
   
   DataWriter* dw = static_cast<DataWriter*>(ctx->self().asInst());
   dw->write(value);
   ctx->returnFrame();
}

void Function_writeByte::invoke(VMContext* ctx, int32 )
{
   Item* i_data = ctx->param(0);
   if( i_data == 0 || ! i_data->isOrdinal() )
   {
      ctx->raiseError( paramError() );
      return;
   }
   
   byte value = (byte) i_data->forceInteger();   
   DataWriter* dw = static_cast<DataWriter*>(ctx->self().asInst());
   dw->write(value);
   ctx->returnFrame();
}

void Function_writeI16::invoke(VMContext* ctx, int32 )
{
   Item* i_data = ctx->param(0);
   if( i_data == 0 || ! i_data->isOrdinal() )
   {
      ctx->raiseError( paramError() );
      return;
   }
   
   int16 value = (int16) i_data->forceInteger();   
   DataWriter* dw = static_cast<DataWriter*>(ctx->self().asInst());
   dw->write(value);
   ctx->returnFrame();
}



void Function_writeU16::invoke(VMContext* ctx, int32 )
{
   Item* i_data = ctx->param(0);
   if( i_data == 0 || ! i_data->isOrdinal() )
   {
      ctx->raiseError( paramError() );
      return;
   }
   
   uint16 value = (uint16) i_data->forceInteger();   
   DataWriter* dw = static_cast<DataWriter*>(ctx->self().asInst());
   dw->write(value);
   ctx->returnFrame();
}


void Function_writeI32::invoke(VMContext* ctx, int32 )
{
   Item* i_data = ctx->param(0);
   if( i_data == 0 || ! i_data->isOrdinal() )
   {
      ctx->raiseError( paramError() );
      return;
   }
   
   int32 value = (int32) i_data->forceInteger();   
   DataWriter* dw = static_cast<DataWriter*>(ctx->self().asInst());
   dw->write(value);
   ctx->returnFrame();
}


void Function_writeU32::invoke(VMContext* ctx, int32 )
{
   Item* i_data = ctx->param(0);
   if( i_data == 0 || ! i_data->isOrdinal() )
   {
      ctx->raiseError( paramError() );
      return;
   }
   
   uint32 value = (uint32) i_data->forceInteger();   
   DataWriter* dw = static_cast<DataWriter*>(ctx->self().asInst());
   dw->write(value);
   ctx->returnFrame();
}


void Function_writeI64::invoke(VMContext* ctx, int32 )
{
   Item* i_data = ctx->param(0);
   if( i_data == 0 || ! i_data->isOrdinal() )
   {
      ctx->raiseError( paramError() );
      return;
   }
   
   int64 value = (int64) i_data->forceInteger();   
   DataWriter* dw = static_cast<DataWriter*>(ctx->self().asInst());
   dw->write(value);
   ctx->returnFrame();
}


void Function_writeU64::invoke(VMContext* ctx, int32 )
{
   Item* i_data = ctx->param(0);
   if( i_data == 0 || ! i_data->isOrdinal() )
   {
      ctx->raiseError( paramError() );
      return;
   }
   
   uint64 value = (uint64) i_data->forceInteger();   
   DataWriter* dw = static_cast<DataWriter*>(ctx->self().asInst());
   dw->write(value);
   ctx->returnFrame();
}


void Function_writeF32::invoke(VMContext* ctx, int32 )
{
   Item* i_data = ctx->param(0);
   if( i_data == 0 || ! i_data->isOrdinal() )
   {
      ctx->raiseError( paramError() );
      return;
   }
   
   float value = (float) i_data->forceInteger();   
   DataWriter* dw = static_cast<DataWriter*>(ctx->self().asInst());
   dw->write(value);
   ctx->returnFrame();
}


void Function_writeF64::invoke(VMContext* ctx, int32 )
{
   Item* i_data = ctx->param(0);
   if( i_data == 0 || ! i_data->isOrdinal() )
   {
      ctx->raiseError( paramError() );
      return;
   }
   
   double value = (double) i_data->forceInteger();   
   DataWriter* dw = static_cast<DataWriter*>(ctx->self().asInst());
   dw->write(value);
   ctx->returnFrame();
}


void Function_writeString::invoke(VMContext* ctx, int32 )
{
   Item* i_data = ctx->param(0);
   if( i_data == 0 || ! i_data->isString() )
   {
      ctx->raiseError( paramError() );
      return;
   }
   
   DataWriter* dw = static_cast<DataWriter*>(ctx->self().asInst());
   dw->write(*i_data->asString());
   ctx->returnFrame();
}


void Function_writeItem::invoke(VMContext* ctx, int32 )
{
   Item* i_data = ctx->param(0);
   if( i_data == 0 )
   {
      ctx->raiseError( paramError() );
      return;
   }
   
   DataWriter* dw = static_cast<DataWriter*>(ctx->self().asInst());
   Class* cls;
   void* data;
   i_data->forceClassInst( cls, data );
   
   // store may go deep, so we use a conditional return construct
   ctx->pushReturn();
   long depth = ctx->codeDepth();
   cls->store( ctx, dw, data );
   if( depth == ctx->codeDepth() )
   {
      ctx->returnFrame();
   }
}


void Function_flush::invoke(VMContext* ctx, int32 )
{
  DataWriter* dw = static_cast<DataWriter*>(ctx->self().asInst());
  dw->flush();
  ctx->returnFrame();
}


//=======================================================================


ClassDataWriter::ClassDataWriter( Class* clsStream ):
   Class("DataWriter"),
   m_clsStream( clsStream )
{
   addProperty( "endianity", &get_endianity, &set_endianity );
   addProperty( "sysEndianity", &get_sysEndianity);
   
   addMethod( new Function_write );
   addMethod( new Function_writeBool );
   addMethod( new Function_writeChar );
   addMethod( new Function_writeByte );
   addMethod( new Function_writeI16 );
   addMethod( new Function_writeU16 );
   addMethod( new Function_writeI32 );
   addMethod( new Function_writeU32 );
   addMethod( new Function_writeI64 );
   addMethod( new Function_writeU64 );
   addMethod( new Function_writeF32 );
   addMethod( new Function_writeF64 );
   addMethod( new Function_writeString );
   addMethod( new Function_writeItem );
   
   addMethod( new Function_flush );
}


ClassDataWriter::~ClassDataWriter()
{
}

   
void ClassDataWriter::dispose( void* instance ) const
{
   DataWriter* wr = static_cast<DataWriter*>(instance);
   wr->decref();
}


void* ClassDataWriter::clone( void* instance ) const
{
   // nothing to clone, it can be shared.
   DataWriter* wr = static_cast<DataWriter*>(instance);
   return new DataWriter(*wr);
}


void ClassDataWriter::gcMarkInstance( void* instance, uint32 mark ) const
{
   DataWriter* wr = static_cast<DataWriter*>(instance);
   wr->gcMark( mark );
}


bool ClassDataWriter::gcCheckInstance( void* instance, uint32 mark ) const
{
   DataWriter* wr = static_cast<DataWriter*>(instance);
   return wr->gcMark() >= mark;
}


void* ClassDataWriter::createInstance() const
{
   return new DataWriter;
}

bool ClassDataWriter::op_init( VMContext* ctx, void* instance, int pcount ) const
{   
   if( pcount >= 1 )
   {
      Class* cls=0;
      void* data=0;
      
      Item* params = ctx->opcodeParams(pcount);
      params[0].asClassInst( cls, data );
      if( cls->isDerivedFrom(m_clsStream) )
      {
         DataWriter* wr = static_cast<DataWriter*>(instance);            
         wr->changeStream( 
                  static_cast<Stream*>(cls->getParentData(m_clsStream,data)),
                  false );
         return false;
      }      
   }
   
   throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
         .origin(ErrorParam::e_orig_runtime)
         .extra( "Stream" ) );      
}


}
}

/* end of datawriter.cpp */
