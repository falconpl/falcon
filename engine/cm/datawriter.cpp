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

#include <falcon/cm/datawriter.h>

#include <falcon/datawriter.h>
#include <falcon/module.h>
#include <falcon/cm/stream.h>

#include <falcon/errors/accesserror.h>
#include <falcon/errors/paramerror.h>
#include <falcon/vmcontext.h>

namespace Falcon {
namespace Ext {

ClassDataWriter::ClassDataWriter():
   ClassUser("DataWriter"),
   FALCON_INIT_PROPERTY( endianity ),
   FALCON_INIT_PROPERTY( sysEndianity ),
   
   FALCON_INIT_METHOD( write ),
   FALCON_INIT_METHOD( writeBool ),
   FALCON_INIT_METHOD( writeChar ),
   FALCON_INIT_METHOD( writeByte ),
   FALCON_INIT_METHOD( writeI16 ),
   FALCON_INIT_METHOD( writeU16 ),
   FALCON_INIT_METHOD( writeI32 ),
   FALCON_INIT_METHOD( writeU32 ),
   FALCON_INIT_METHOD( writeI64 ),
   FALCON_INIT_METHOD( writeU64 ),
   FALCON_INIT_METHOD( writeF32 ),
   FALCON_INIT_METHOD( writeF64 ),
   FALCON_INIT_METHOD( writeString ),
   FALCON_INIT_METHOD( writeItem ),
   
   FALCON_INIT_METHOD( flush )
{
}

ClassDataWriter::~ClassDataWriter()
{
}

   
void ClassDataWriter::dispose( void* instance ) const
{
   DataWriter* wr = static_cast<DataWriter*>(instance);
   delete wr;
}


void* ClassDataWriter::clone( void* instance ) const
{
   // nothing to clone, it can be shared.
   DataWriter* wr = static_cast<DataWriter*>(instance);
   return new DataWriter(*wr);
}


void ClassDataWriter::gcMark( void* instance, uint32 mark ) const
{
   DataWriter* wr = static_cast<DataWriter*>(instance);
   wr->gcMark( mark );
}


bool ClassDataWriter::gcCheck( void* instance, uint32 mark ) const
{
   DataWriter* wr = static_cast<DataWriter*>(instance);
   return wr->gcMark() >= mark;
}


void* ClassDataWriter::createInstance( Item* params, int pcount ) const
{
   static Class* streamCls = m_module->getClass("Stream");
   
   if( pcount >= 1 )
   {
      Class* cls;
      void* data;
      params[0].asClassInst( cls, data );
      if( cls->isDerivedFrom(streamCls) )
      {
         DataWriter* wr = new DataWriter( static_cast<StreamCarrier*>(data)->m_underlying );
         // this data is going to be added to gc very soon.
         wr->gcMark(1);
         return wr;
      }      
   }
   
   throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
         .origin(ErrorParam::e_orig_runtime)
         .extra( "Stream" ) );      
}


//=================================================================
// Properties
//

FALCON_DEFINE_PROPERTY_SET_P( ClassDataWriter, endianity )
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


FALCON_DEFINE_PROPERTY_GET_P( ClassDataWriter, endianity )
{
   DataWriter* sc = static_cast<DataWriter*>(instance);
   value = (int64)sc->endianity();
}


FALCON_DEFINE_PROPERTY_SET( ClassDataWriter, sysEndianity )( void*, const Item& )
{
   throw new AccessError( ErrorParam( e_prop_ro, __LINE__, SRC )
      .origin( ErrorParam::e_orig_runtime )
      .extra( "sysEndianity" ));
}


FALCON_DEFINE_PROPERTY_GET_P( ClassDataWriter, sysEndianity )
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

FALCON_DEFINE_METHOD_P1( ClassDataWriter, write )
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

   int64 retval = (int64) dw->writeRaw(dataSource+start, count);   
   ctx->returnFrame( retval );
   ctx->returnFrame();
}


FALCON_DEFINE_METHOD_P1( ClassDataWriter, writeBool )
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


FALCON_DEFINE_METHOD_P1( ClassDataWriter, writeChar )
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

FALCON_DEFINE_METHOD_P1( ClassDataWriter, writeByte )
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

FALCON_DEFINE_METHOD_P1( ClassDataWriter, writeI16 )
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


FALCON_DEFINE_METHOD_P1( ClassDataWriter, writeU16 )
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


FALCON_DEFINE_METHOD_P1( ClassDataWriter, writeI32 )
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


FALCON_DEFINE_METHOD_P1( ClassDataWriter, writeU32 )
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


FALCON_DEFINE_METHOD_P1( ClassDataWriter, writeI64 )
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


FALCON_DEFINE_METHOD_P1( ClassDataWriter, writeU64 )
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


FALCON_DEFINE_METHOD_P1( ClassDataWriter, writeF32 )
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


FALCON_DEFINE_METHOD_P1( ClassDataWriter, writeF64 )
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


FALCON_DEFINE_METHOD_P1( ClassDataWriter, writeString )
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


FALCON_DEFINE_METHOD_P1( ClassDataWriter, writeItem )
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


FALCON_DEFINE_METHOD_P1( ClassDataWriter, flush )
{
  DataWriter* dw = static_cast<DataWriter*>(ctx->self().asInst());
  dw->flush();
  ctx->returnFrame();
}

}
}

/* end of datawriter.cpp */
