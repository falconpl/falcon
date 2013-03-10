/*
   FALCON - The Falcon Programming Language.
   FILE: exprprovides.h

   Syntactic tree item definitions -- Operator "provides"
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 04 Feb 2013 19:39:39 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/exprprovides.cpp"

#include <falcon/psteps/exprprovides.h>
#include <falcon/trace.h>
#include <falcon/vmcontext.h>
#include <falcon/pstep.h>
#include <falcon/errors/operanderror.h>
#include <falcon/errors/paramerror.h>
#include <falcon/synclasses.h>

#include <falcon/datareader.h>
#include <falcon/datawriter.h>

namespace Falcon {


ExprProvides::ExprProvides( Expression* op1, const String& property, int line, int chr ):
         UnaryExpression( op1, line, chr ),
         m_property(property)
{
   FALCON_DECLARE_SYN_CLASS( expr_provides );
   apply = apply_;
}

ExprProvides::ExprProvides(int line, int chr):
            UnaryExpression(line,chr)
{
   FALCON_DECLARE_SYN_CLASS( expr_provides );
   apply = apply_;
}

ExprProvides::ExprProvides( const ExprProvides& other ):
            UnaryExpression( other ),
            m_property(other.m_property)
{
   FALCON_DECLARE_SYN_CLASS( expr_provides );
   apply = apply_;
}

ExprProvides::~ExprProvides()
{}


bool ExprProvides::simplify( Item& ) const
{
   return false;
}

void ExprProvides::describeTo( String& target, int depth ) const
{
   if( first() == 0 )
   {
      target = "<Blank provides>";
      return;
   }

   target = "(" +first()->describe(depth+1) + " provides " + m_property + ")";
}


void ExprProvides::apply_( const PStep*ps , VMContext* ctx )
{  
   const ExprProvides* self = static_cast<const ExprProvides*>(ps);
   CodeFrame& cf = ctx->currentCode();
   TRACE2( "ExprProvides::apply_ %d/1 \"%s\"", cf.m_seqId, self->describe().c_ize() );
   fassert( self->first() != 0 );
   
   switch( cf.m_seqId )
   {
   case 0:
      cf.m_seqId = 1;
      if( ctx->stepInYield(self->first(), cf) ) {
         return;
      }
      break;
   }
   ctx->popCode();

   Class* cls;
   void* inst;
   ctx->topData().forceClassInst(cls, inst);

   cls->op_provides(ctx, inst, self->property() );
}

//======================================================================
// We do all ClassProvides here.
//

void* SynClasses::ClassProvides::createInstance() const
{
   return new ExprProvides;
}

bool SynClasses::ClassProvides::hasProperty( void* instance, const String& prop ) const
{
   if( prop == "property" )
   {
      return true;
   }
   return Class::hasProperty(instance, prop);
}

void SynClasses::ClassProvides::store( VMContext* ctx, DataWriter* dw, void* instance ) const
{
   ExprProvides* prov = static_cast<ExprProvides *>(instance);
   dw->write( prov->property() );
   m_parent->store(ctx, dw, instance);
}

void SynClasses::ClassProvides::restore( VMContext* ctx, DataReader* dw ) const
{
   String property;
   dw->read( property );
   ExprProvides* prov = new ExprProvides;
   prov->property(property);
   ctx->pushData( Item(this, prov) );
   m_parent->restore(ctx, dw);
}


bool SynClasses::ClassProvides::op_init(VMContext* ctx, void* instance, int pcount) const
{
   Item& ioperand = ctx->opcodeParam(1);
   Item& iprop = ctx->opcodeParam(0);

   if( pcount != 2 || ! iprop.isString()  )
   {
      throw new ParamError( ErrorParam(e_inv_params, ctx).extra("X,S") );
   }

   bool make = true;
   Expression* first = TreeStep::checkExpr(ioperand, make);
   if( first == 0 )
   {
      throw new ParamError( ErrorParam(e_inv_params, ctx).extra("Incompatible type of expression at 0") );
   }

   ExprProvides* prov = static_cast<ExprProvides*>( instance );
   prov->setInGC();
   String* prop = iprop.asString();

   prov->first( first );
   prov->property( *prop );

   return false;
}

void SynClasses::ClassProvides::op_getProperty( VMContext* ctx, void* instance, const String& prop) const
{
   ExprProvides* prov = static_cast<ExprProvides *>(instance);
   if( prop == "property" )
   {
      ctx->topData() = FALCON_GC_HANDLE(new String(prov->property()));
      return;
   }

   Class::op_getProperty(ctx, instance, prop);
}

void SynClasses::ClassProvides::op_setProperty( VMContext* ctx, void* instance, const String& prop ) const
{
   ExprProvides* prov = static_cast<ExprProvides *>(instance);
   if( prop == "property" )
   {
      Item& value = ctx->opcodeParam(1);
      if( ! value.isString() )
      {
         throw new OperandError( ErrorParam(e_invalid_op, __LINE__, SRC )
                  .extra("S"));
      }

      prov->property(*value.asString());
      ctx->popData();
      return;
   }

   Class::op_setProperty(ctx, instance, prop);
}

}

/* end of exprprovides.cpp */
