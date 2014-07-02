/*
   FALCON - The Falcon Programming Language.
   FILE: classtreestep.cpp

   Class handling basic TreeStep common behavior.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 28 Dec 2011 09:40:45 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/classes/classtreestep.cpp"

#include <falcon/trace.h>
#include <falcon/classes/classtreestep.h>
#include <falcon/treestep.h>
#include <falcon/modspace.h>
#include <falcon/stringstream.h>
#include <falcon/textwriter.h>
#include <falcon/module.h>

#include <falcon/statement.h>
#include <falcon/expression.h>
#include <falcon/psteps/exprvalue.h>

#include <falcon/vmcontext.h>
#include <falcon/itemarray.h>
#include <falcon/datawriter.h>
#include <falcon/datareader.h>
#include <falcon/stdhandlers.h>
#include <falcon/stderrors.h>

#ifdef NDEBUG
#include <falcon/stdhandlers.h>
#endif

namespace Falcon {

ClassTreeStep::ClassTreeStep():
   Class("TreeStep - abstract", FLC_CLASS_ID_TREESTEP)
{
   m_insertMethod.methodOf(this);
   m_removeMethod.methodOf(this);
   m_appendMethod.methodOf(this);
}

ClassTreeStep::ClassTreeStep( const String& name ):
   Class(name, FLC_CLASS_ID_TREESTEP)
{
   m_insertMethod.methodOf(this);
   m_removeMethod.methodOf(this);
   m_appendMethod.methodOf(this);
}


ClassTreeStep::~ClassTreeStep()
{}


void ClassTreeStep::describe( void* instance, String& target, int, int ) const
{
   TreeStep* ts = static_cast<TreeStep*>(instance);
   ts->describeTo( target );
}


void ClassTreeStep::dispose( void* instance ) const
{
   TRACE( "ClassTreeStep::dispose %p ", instance );
   TreeStep* ts = static_cast<TreeStep*>(instance);
   // we can kill the step only if it has no parent.
   if( ts->parent() == 0 )
   {
      TRACE1( "ClassTreeStep::dispose performing delete at %p ", instance );
      delete ts;
   }
}

void* ClassTreeStep::clone( void* instance ) const
{
   TRACE( "ClassTreeStep::clone %p ", instance );
   TreeStep* ts = static_cast<TreeStep*>(instance);
   ts->setInGC();
   return ts->clone();
}

void* ClassTreeStep::createInstance() const
{
   // we're abstract.
   return 0;
}

void ClassTreeStep::gcMarkInstance( void* instance, uint32 mark ) const
{
   TRACE( "ClassTreeStep::gcMark %p, %d ", instance, mark );
   // the mark is performed on the topmost unparented parent.
   TreeStep* ts = static_cast<TreeStep*>(instance);
   while( ts->parent() != 0 )
   {
      ts = ts->parent();
   }
   TRACE1( "ClassTreeStep::gcMark parent %p, %d ", ts, mark );
   ts->gcMark( mark );
}

bool ClassTreeStep::gcCheckInstance( void* instance, uint32 mark ) const
{
   TRACE( "ClassTreeStep::gcCheck %p, %d ", instance, mark );

   // the check is performed on the topmost parent.
   TreeStep* ts = static_cast<TreeStep*>(instance);
   if( ts->parent() != 0 )
   {
      TRACE1( "ClassTreeStep::gcCheck %p has a parent, ignored", ts );
      return true;
   }
   TRACE1( "ClassTreeStep::gcCheck %p, %d -- %s", instance, mark, ts->gcMark() >= mark ? "PASS" : "FAIL" );
   return ts->gcMark() >= mark;
}




void ClassTreeStep::enumerateProperties( void*, Class::PropertyEnumerator& cb ) const
{
   cb("arity");
   cb("insert");
   cb("parent" );
   cb("remove");
   cb("append");
}


void ClassTreeStep::enumeratePV( void* instance, Class::PVEnumerator& cb ) const
{
   Statement* stmt = static_cast<Statement*>(instance);
   Item temp = (int64) stmt->arity();

   cb("arity", temp );
   TreeStep* expr = stmt->selector();
   if( expr != 0 )
   {
      temp.setUser( expr->handler(), expr );
   }
   else
   {
      temp.setNil();
   }
   cb("selector", temp);
}


bool ClassTreeStep::hasProperty( void*, const String& prop ) const
{
   return
         prop == "arity"
      || prop == "len"
      || prop == "parent"
      || prop == "selector"
      || prop == "insert"
      || prop == "remove"
      || prop == "append";
}


void ClassTreeStep::op_call( VMContext* ctx, int32 paramCount, void* instance ) const
{
   TreeStep* ts = static_cast<TreeStep*>(instance);
   ctx->pushCode(ts);
   ctx->popData(paramCount+1);
}


void ClassTreeStep::op_getProperty( VMContext* ctx, void* instance, const String& prop) const
{
   TreeStep* stmt = static_cast<TreeStep*>(instance);
   if( prop == "insert" )
   {
      ctx->topData().methodize( &m_insertMethod );
   }
   else if( prop == "remove")
   {
      ctx->topData().methodize( &m_removeMethod );
   }
   else if( prop == "append")
   {
      ctx->topData().methodize( &m_appendMethod );
   }
   else if( prop == "selector")
   {
      TreeStep* expr = stmt->selector();
      if( expr != 0 )
      {
         ctx->topData().setUser( expr->handler(), expr );
      }
      else {
         ctx->topData().setNil();
      }
   }
   else if( prop == "len" || prop == "arity" )
   {
      ctx->topData().setInteger( (int64) stmt->arity() );
   }
   else if( prop == "parent" )
   {
      if( stmt->parent() == 0 )
         ctx->topData().setNil();
      else
         ctx->topData().setUser( stmt->parent()->handler(), stmt->parent() );
   }
   else
   {
      Class::op_getProperty( ctx, instance, prop );
   }
}


void ClassTreeStep::op_setProperty( VMContext* ctx, void* instance, const String& prop ) const
{
   TreeStep* stmt = static_cast<TreeStep*>(instance);

   // the value to be stored.
   Item& value = ctx->opcodeParam(1);

   if( prop == "selector" )
   {
      bool bCreate = true;
      if( value.isNil() )
      {
         // set it and ingore the result, we don't care.
         stmt->selector(0);
      }
      else {
         Expression* expr = TreeStep::checkExpr( value, bCreate );
         if( expr == 0 )
         {
            throw new AccessError( ErrorParam( e_inv_params, __LINE__, SRC )
               .origin( ErrorParam::e_orig_vm )
               .extra( "Expression|Nil" ) );
         }

         if( expr->parent() != 0 )
         {
            throw new CodeError( ErrorParam(e_invalid_op, __LINE__, SRC)
               .origin( ErrorParam::e_orig_vm)
               .extra( "Parented entity cannot be inserted" ) );
         }

         if( ! stmt->selector( expr ) )
         {
            throw new CodeError( ErrorParam(e_invalid_op, __LINE__, SRC)
               .origin( ErrorParam::e_orig_vm)
               .extra( "Entity is not accepting that kind of element" ) );
         }
      }

      ctx->popData();
   }
   else if( hasProperty( instance, prop) )
   {
      FALCON_RESIGN_ROPROP_ERROR( prop, ctx );
   }
   else
   {
      Class::op_setProperty( ctx, instance, prop );
   }
}


void ClassTreeStep::store( VMContext*, DataWriter* dw, void* instance ) const
{
   TreeStep* ts = static_cast<TreeStep*>( instance );
   TRACE2("ClassTreeStep::store %s", ts->describe().c_ize());

   // save the position
   dw->write( ts->line() );
   int32 chr = ts->chr();
   if( ts->isTracedCatch() )
   {
      chr = - chr;
      if( chr == 0 ) {
         chr = -1;
      }
   }
   dw->write( chr );
}

void ClassTreeStep::restore( VMContext* ctx, DataReader*dr ) const
{
   fassert( ctx->topData().asClass()->isDerivedFrom( this ) ) ;
   TreeStep* ts = static_cast<TreeStep*>( ctx->topData().asInst() );
   TRACE2("ClassTreeStep::restore instance of %s", ts->handler()->name().c_ize() );

   int32 line, chr;
   dr->read( line );
   dr->read( chr );
   if( chr < 0 )
   {
      chr = -chr;
      ts->setTracedCatch();
   }
   ts->decl( line, chr );
}

void ClassTreeStep::flatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   TreeStep* ts = static_cast<TreeStep*>( instance );
   TRACE2("ClassTreeStep::flatten %s", ts->describe().c_ize());

   int arity = ts->arity();
   subItems.resize( arity + 1 );
   if( ts->selector() != 0 )
   {
      subItems.at(0).setUser( ts->selector()->handler(), ts->selector() );
   }
   // else, let it be nil.

   if( arity > 0 )
   {
      for( int i = 0; i < arity; ++i )
      {
         TreeStep* element = ts->nth(i);
         if( element != 0 )
         {
            subItems[i+1].setUser( element->handler(), element );
         }
         // else, let it be nil.
      }
   }

}

void ClassTreeStep::unflatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   TreeStep* ts = static_cast<TreeStep*>( instance );
   TRACE2("ClassTreeStep::unflatten %s", ts->describe().c_ize());

   if( subItems[0].isUser() )
   {
      // we have a selector.
#ifndef NDEBUG
      static Class* clsTreeStep = Engine::instance()->stdHandlers()->treeStepClass();
      fassert( subItems[0].asClass()->isDerivedFrom(clsTreeStep) );
#endif
      ts->selector( static_cast<TreeStep*>(subItems[0].asInst()) );
   }

   for( int i = 1; i < (int) subItems.length(); ++i )
   {
      Class* cls = 0;
      void* data = 0;
      if( subItems[i].asClassInst(cls, data) )
      {
         ts->setNth(i-1, static_cast<TreeStep*>( data ) );
      }
      // else, it was nil and unused.
   }
}

bool ClassTreeStep::gcCheckMyself( uint32 )
{
   // if we went in the GC for error, remove us.
   return false;
}

//=============================================================================
// The block handler
//


void ClassTreeStep::op_getIndex(VMContext* ctx, void* instance ) const
{
   TreeStep* stmt = static_cast<TreeStep*>(instance);
   Item& index = ctx->opcodeParam(0);
   if( index.isOrdinal() )
   {
      int64 num = ctx->opcodeParam(0).forceInteger();
      if( num < 0 ) num = stmt->arity() + num;
      if( num < 0 || num >= stmt->arity() )
      {
         throw new AccessError( ErrorParam( e_arracc, __LINE__, SRC )
            .origin(ErrorParam::e_orig_vm));
      }

      TreeStep* st = stmt->nth( (int32) num );
      if( st == 0 )
      {
         // nil might be valid for optional blocks
         ctx->stackResult( 2, Item() );
      }
      else {
         ctx->stackResult( 2, Item( st->handler(), st ) );
      }
   }
   else {
      throw new AccessError( ErrorParam( e_inv_params, __LINE__, SRC )
         .origin( ErrorParam::e_orig_vm )
         .extra( "N" ) );
   }
}


void ClassTreeStep::op_setIndex(VMContext* ctx, void* instance ) const
{
   static Class* treeStepClass = Engine::instance()->stdHandlers()->treeStepClass();

   TreeStep* self = static_cast<TreeStep*>(instance);
   Item& index = ctx->opcodeParam(0);
   if( index.isOrdinal() )
   {
      int64 num = ctx->opcodeParam(0).forceInteger();
      if( num < 0 ) num = self->arity() + num;
      if( num < 0 || num >= self->arity() )
      {
         throw new AccessError( ErrorParam( e_arracc, __LINE__, SRC )
            .origin(ErrorParam::e_orig_vm));
      }

      Item& i_tree = ctx->opcodeParam(2);
      if( i_tree.isNil() )
      {
         if( ! self->setNth( (int32) num, 0 ) )
         {
            throw new CodeError( ErrorParam(e_invalid_op, __LINE__, SRC)
               .origin( ErrorParam::e_orig_vm)
               .extra( "Statement is not allowing an optional block" ) );
         }
      }
      else {
         Class* cls; void* inst;
         ExprValue* ev = 0;
         bool isInst = i_tree.asClassInst(cls, inst);
         if( ! isInst || ! cls->isDerivedFrom( treeStepClass ) )
         {
            ev = new ExprValue( i_tree );
            cls = ev->handler();
            inst = ev;
         }

         TreeStep* ts = static_cast<TreeStep*>(inst);
         // check TreeStep category.
         if( ts->parent() != 0 )
         {
            // cannot be our ev...
            throw new CodeError( ErrorParam(e_invalid_op, __LINE__, SRC)
               .origin( ErrorParam::e_orig_vm)
               .extra( "Parented entity cannot be inserted" ) );
         }

         if( ! self->setNth( (int32) num, ts ) )
         {
            delete ev; // this can be our ev
            throw new CodeError( ErrorParam(e_invalid_op, __LINE__, SRC)
               .origin( ErrorParam::e_orig_vm)
               .extra( "Entity is not accepting that kind of element" ) );
         }
      }

      ctx->popData(2); // keep the 3d value, that we just assigned.
   }
   else {
      throw new AccessError( ErrorParam( e_inv_params, __LINE__, SRC )
         .origin( ErrorParam::e_orig_vm )
         .extra( "N" ) );
   }
}


void ClassTreeStep::op_iter( VMContext* ctx, void* ) const
{
   // seq
   ctx->pushData((int64)0);
}


void ClassTreeStep::op_next( VMContext* ctx, void* instance ) const
{
   Statement* stmt = static_cast<Statement*>(instance);

   // (seq)(iter)
   Item& iter = ctx->opcodeParam(0);
   fassert( iter.isInteger() );
   int32 pos = (int32) iter.asInteger();
   if( pos >= stmt->arity() )
   {
      ctx->addDataSlot().setBreak();
   }
   else
   {

      TreeStep* st = stmt->nth(pos++);
      iter.setInteger( pos );  // here the stack is still valid

      if( st == 0 )
      {
         ctx->pushData(Item()); // a nil
      }
      else {
         Item value(st->handler(), st);
         ctx->pushData( value );
         if( pos < stmt->arity() )
         {
            ctx->topData().setDoubt();
         }
      }
   }
}



//===============================================================
// Insert method
//
ClassTreeStep::InsertMethod::InsertMethod():
   Function("insert")
{
   signature("N,SynTree");
   addParam("pos");
   addParam("syntree");
}

ClassTreeStep::InsertMethod::~InsertMethod()
{}


void ClassTreeStep::InsertMethod::invoke( VMContext* ctx, int32 pcount )
{
   static Class* treeStepClass = Engine::instance()->stdHandlers()->treeStepClass();

   Item& self = ctx->self();
   fassert( self.isUser() );

   if( pcount < 2 )
   {
      ctx->raiseError(paramError(__LINE__, SRC ));
      return;
   }

   Item* i_pos = ctx->param(0);
   Item* i_treestep = ctx->param(1);
   Class* cls;
   void* inst;

   if( (! i_pos->isOrdinal()) ||
        !( i_treestep->asClassInst(cls, inst) && cls->isDerivedFrom(treeStepClass) )
      )
   {
      ctx->raiseError(paramError(__LINE__, SRC ));
      return;
   }

   TreeStep* ts = static_cast<TreeStep*>( inst );
   if( ts->parent() !=  0 )
   {
      ctx->raiseError( new CodeError( ErrorParam(e_invalid_op, __LINE__, SRC)
      .origin( ErrorParam::e_orig_runtime)
      .extra( "Parented syntree cannot be inserted" ) ) );
      return;
   }

   // check TreeStep category.
   TreeStep* self_step = static_cast<TreeStep*>(self.asInst());
   if( ! self_step->insert( (int32) i_pos->forceInteger(), ts) ) {
      ctx->raiseError( new CodeError( ErrorParam(e_invalid_op, __LINE__, SRC)
      .origin( ErrorParam::e_orig_runtime)
      .extra( "This Statement is not accepting insert" ) ) );

      return;
   }

   ctx->returnFrame();
}

//===============================================================
// Remove method
//
ClassTreeStep::RemoveMethod::RemoveMethod():
   Function("remove")
{
   signature("N");
   addParam("pos");
}

ClassTreeStep::RemoveMethod::~RemoveMethod()
{}


void ClassTreeStep::RemoveMethod::invoke( VMContext* ctx, int32 )
{
   Item& self = ctx->self();
   fassert( self.isUser() );

   Item* i_pos = ctx->param(0);

   if( i_pos == 0 || ! i_pos->isOrdinal() )
   {
      ctx->raiseError(paramError(__LINE__, SRC ));
      return;
   }

   TreeStep* self_step = static_cast<TreeStep*>(self.asInst());
   if( ! self_step->remove( (int32) i_pos->forceInteger() ) ) {
      ctx->raiseError( new CodeError( ErrorParam(e_invalid_op, __LINE__, SRC)
      .origin( ErrorParam::e_orig_runtime)
      .extra( "This Statement is not accepting remove" ) ) );

      return;
   }

   ctx->returnFrame();
}

//===============================================================
// AppendMethod method
//
ClassTreeStep::AppendMethod::AppendMethod():
   Function("append")
{
   signature("SynTree");
   addParam("syntree");
}

ClassTreeStep::AppendMethod::~AppendMethod()
{}


void ClassTreeStep::AppendMethod::invoke( VMContext* ctx, int32 )
{
   Item& self = ctx->self();
   fassert( self.isUser() );

   Item* i_treestep = ctx->param(0);
   if( i_treestep == 0 || i_treestep->type() != FLC_CLASS_ID_TREESTEP )
   {
      ctx->raiseError(paramError(__LINE__, SRC ));
      return;
   }

   TreeStep* ts = static_cast<TreeStep*>( i_treestep->asInst() );
   if( ts->parent() !=  0 )
   {
     ctx->raiseError( new CodeError( ErrorParam(e_invalid_op, __LINE__, SRC)
        .origin( ErrorParam::e_orig_runtime)
        .extra( "Parented syntree cannot be inserted" ) ) );
     return;
   }

   TreeStep* self_step = static_cast<TreeStep*>(self.asInst());
   if( ! self_step->append(ts) )
   {
      ctx->raiseError( new CodeError( ErrorParam(e_invalid_op, __LINE__, SRC)
           .origin( ErrorParam::e_orig_runtime)
           .extra( "Element cannot be appended" ) ) );
           return;
   }

   ctx->returnFrame();
}


}
/* end of classtreestep.cpp */
