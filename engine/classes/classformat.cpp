/*
   FALCON - The Falcon Programming Language.
   FILE: classformat.cpp

   Format type handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 01 Feb 2013 23:31:18 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/classes/classformat.cpp"

#include <falcon/classes/classformat.h>
#include <falcon/format.h>
#include <falcon/vmcontext.h>
#include <falcon/optoken.h>
#include <falcon/errors/accesserror.h>
#include <falcon/errors/paramerror.h>
#include <falcon/stdsteps.h>
#include <falcon/engine.h>
#include <falcon/processor.h>
#include <falcon/function.h>

#include <falcon/datareader.h>
#include <falcon/datawriter.h>


namespace Falcon {

//====================================================
// Properties.
//
static void get_original( const Class*, const String&, void* instance, Item& value )
{
   Format* self = static_cast<Format*>( instance );
   value.setUser( FALCON_GC_HANDLE(new String( self->originalFormat() ) ));
}

//=====================================================================
// Methods
//
FALCON_DECLARE_FUNCTION( parse, "S" );
void Function_parse::invoke(VMContext* ctx, int32 pCount )
{
   Format* self = static_cast<Format*>(ctx->self().asInst());

   Item* pFmt;
   if( pCount < 1 || ! (pFmt = ctx->param(0))->isString() )
   {
      throw paramError();
   }

   String* str = pFmt->asString();
   self->parse(*str);
   ctx->returnFrame();
}

FALCON_DECLARE_FUNCTION( format, "X" );
void Function_format::invoke(VMContext* ctx, int32 pCount )
{
   static StdSteps* steps = Engine::instance()->stdSteps();

   Format* self = static_cast<Format*>(ctx->self().asInst());

   if( pCount < 1 )
   {
      throw paramError();
   }

   Item* pFmt = ctx->param(0);

   ctx->pushCode(&steps->m_returnFrameWithTop);
   self->opFormat( ctx, *pFmt );
}

//
// Class properties used for enumeration
//

ClassFormat::ClassFormat():
   Class( "Format" )
{
   addProperty( "original", &get_original );
   addMethod( new Function_format );
   addMethod( new Function_parse );
}


ClassFormat::~ClassFormat()
{
}

int64 ClassFormat::occupiedMemory( void* instance ) const
{
   Format* format = static_cast<Format*>( instance );
   const String& s = format->originalFormat();
   return sizeof(Format) + s.allocated() + 16 + (s.allocated()?16:0);
}


void ClassFormat::dispose( void* self ) const
{
   delete static_cast<Format*>( self );
}


void* ClassFormat::clone( void* source ) const
{
   return new Format( *( static_cast<Format*>( source ) ) );
}

void* ClassFormat::createInstance() const
{
   return new Format;
}

void ClassFormat::store( VMContext*, DataWriter* dw, void* data ) const
{
   Format* value = static_cast<Format*>( data );
   TRACE2( "ClassFormat::store -- \"%s\"", value->originalFormat().c_ize() );
   dw->write( value->originalFormat() );
}


void ClassFormat::restore( VMContext* ctx, DataReader* dr ) const
{
   String fmt;

   dr->read( fmt );
   TRACE2( "ClassFormat::restore -- \"%s\"", fmt.c_ize() );

   Format* ffmt = new Format(fmt);
   ctx->pushData( Item( this, ffmt ) );
}


void ClassFormat::describe( void* instance, String& target, int, int ) const
{
   Format* self = static_cast<Format*>( instance );

   target.size( 0 );
   target += "Format(\"" + self->originalFormat() + "\")";
}


void ClassFormat::gcMarkInstance( void* instance, uint32 mark ) const
{
   static_cast<Format*>( instance )->gcMark( mark );
}


bool ClassFormat::gcCheckInstance( void* instance, uint32 mark ) const
{
   return static_cast<Format*>( instance )->currentMark() >= mark;
}


//=======================================================================
// Operands
//

bool ClassFormat::op_init( VMContext* ctx, void* instance, int pcount ) const
{
   Format* self = static_cast<Format*>(instance);
   
   // no param?
   if ( pcount > 0 )
   {
      // the parameter is a string?
      Item* itm = ctx->opcodeParams( pcount );

      if ( itm->isString() )
      {
         // copy it.
         self->parse( *itm->asString() );
      }
      else {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
                  .origin(ErrorParam::e_orig_runtime)
                  .extra( "S" )
            );
      }
   }
   
   return false;
}


//=======================================================================
// Comparation
//

void ClassFormat::op_compare( VMContext* ctx, void* self ) const
{
   Item* op1, *op2;

   OpToken token( ctx, op1, op2 );

   Format* fmt = static_cast<Format*>( self );

   Class* otherClass;
   void* otherData;

   if ( op2->asClassInst( otherClass, otherData ) )
   {
      if ( otherClass->isDerivedFrom( this ) )
      {
         Format* fmt2 = static_cast<Format*>( otherClass->getParentData(const_cast<ClassFormat*>(this), otherData) );
         token.exit( fmt->originalFormat().compare( fmt2->originalFormat() ) );
      }
      else
      {
         token.exit( (int64)(static_cast<const char*>(self) - static_cast<const char*>(otherData)) );
      }
   }
   else
   {
      token.exit( typeID() - op2->type() );
   }
}

}

/* end of classformat.cpp */
