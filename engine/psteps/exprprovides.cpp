/*
   FALCON - The Falcon Programming Language.
   FILE: exprnamed.h

   Syntactic tree item definitions -- Named expression
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 04 Feb 2013 19:39:39 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/exprnamed.cpp"

#include <falcon/psteps/exprnamed.h>
#include <falcon/trace.h>
#include <falcon/vmcontext.h>
#include <falcon/pstep.h>
#include <falcon/stderrors.h>
#include <falcon/synclasses.h>

#include <falcon/datareader.h>
#include <falcon/datawriter.h>
#include <falcon/textwriter.h>

namespace Falcon {


ExprNamed::ExprNamed( const String& name, Expression* op1, int line, int chr ):
         UnaryExpression( op1, line, chr ),
         m_name(name)
{
   FALCON_DECLARE_SYN_CLASS( expr_named );
   m_trait = e_trait_named;
   apply = apply_;
}

ExprNamed::ExprNamed(int line, int chr):
            UnaryExpression(line,chr)
{
   FALCON_DECLARE_SYN_CLASS( expr_named );
   m_trait = e_trait_named;
   apply = apply_;
}

ExprNamed::ExprNamed( const ExprNamed& other ):
            UnaryExpression( other ),
            m_name(other.m_name)
{
   FALCON_DECLARE_SYN_CLASS( expr_named );
   m_trait = e_trait_named;
   apply = apply_;
}

ExprNamed::~ExprNamed()
{}


bool ExprNamed::simplify( Item& ) const
{
   return false;
}

void ExprNamed::render( TextWriter* tw, int32 depth ) const
{
   tw->write( renderPrefix(depth) );
   if( first() == 0 )
   {
     tw->write( "/* Blank named expression */" );
   }
   else {
      tw->write("(");
      tw->write( m_name );
      tw->write(" | " );
      first()->render(tw, relativeDepth(depth));
      tw->write( ")" );
   }

   if( depth >= 0 )
   {
      tw->write( "\n" );
   }
}


const String& ExprNamed::exprName() const
{
   static String name("|");
   return name;
}


void ExprNamed::apply_( const PStep*ps , VMContext* ctx )
{
   const ExprNamed* self = static_cast<const ExprNamed*>(ps);
   #ifndef NDEBUG
   CodeFrame& cf = ctx->currentCode();
   TRACE2( "ExprNamed::apply_ %d/1 \"%s\"", cf.m_seqId, self->describe().c_ize() );
   #endif
   ctx->resetCode( self->first() );
}

//======================================================================
// We do all ClassProvides here.
//

void* SynClasses::ClassNamed::createInstance() const
{
   return new ExprNamed;
}

bool SynClasses::ClassNamed::hasProperty( void* instance, const String& prop ) const
{
   if( prop == "name" )
   {
      return true;
   }
   return Class::hasProperty(instance, prop);
}

void SynClasses::ClassNamed::store( VMContext* ctx, DataWriter* dw, void* instance ) const
{
   ExprNamed* prov = static_cast<ExprNamed *>(instance);
   dw->write( prov->name() );
   m_parent->store(ctx, dw, instance);
}

void SynClasses::ClassNamed::restore( VMContext* ctx, DataReader* dw ) const
{
   String name;
   dw->read( name );
   ExprNamed* prov = new ExprNamed;
   prov->name(name);
   ctx->pushData( Item(this, prov) );
   m_parent->restore(ctx, dw);
}


bool SynClasses::ClassNamed::op_init(VMContext* ctx, void* instance, int pcount) const
{
   Item& iname = ctx->opcodeParam(1);
   Item& ioperand = ctx->opcodeParam(0);

   if( pcount != 2 || ! iname.isString()  )
   {
      throw new ParamError( ErrorParam(e_inv_params, ctx).extra("S,X") );
   }

   bool make = true;
   Expression* first = TreeStep::checkExpr(ioperand, make);
   if( first == 0 )
   {
      throw new ParamError( ErrorParam(e_inv_params, ctx).extra("Incompatible type of expression at 0") );
   }

   ExprNamed* named = static_cast<ExprNamed*>( instance );
   named->setInGC();
   String* prop = iname.asString();

   named->first( first );
   named->name( *prop );

   return false;
}

void SynClasses::ClassNamed::op_getProperty( VMContext* ctx, void* instance, const String& prop) const
{
   ExprNamed* prov = static_cast<ExprNamed *>(instance);
   if( prop == "name" )
   {
      ctx->topData() = FALCON_GC_HANDLE(new String(prov->name()));
      return;
   }

   Class::op_getProperty(ctx, instance, prop);
}

void SynClasses::ClassNamed::op_setProperty( VMContext* ctx, void* instance, const String& prop ) const
{
   ExprNamed* named = static_cast<ExprNamed *>(instance);
   if( prop == "name" )
   {
      Item& value = ctx->opcodeParam(1);
      if( ! value.isString() )
      {
         throw new AccessTypeError( ErrorParam(e_inv_prop_value, __LINE__, SRC )
                  .extra("S"));
      }

      named->name(*value.asString());
      ctx->popData();
      return;
   }

   Class::op_setProperty(ctx, instance, prop);
}

}

/* end of exprnamed.cpp */
