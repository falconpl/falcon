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

#include <falcon/trace.h>
#include <falcon/classes/classtreestep.h>
#include <falcon/treestep.h>

#include <falcon/statement.h>
#include <falcon/expression.h>

#include <falcon/vmcontext.h>
#include <falcon/itemarray.h>
#include <falcon/datawriter.h>
#include <falcon/datareader.h>

#include <falcon/errors/accesserror.h>

namespace Falcon {

ClassTreeStep::ClassTreeStep():
   Class("TreeStep - abstract")
{
   m_lenMethod.methodOf(this);   
   m_insertMethod.methodOf(this);
   m_removeMethod.methodOf(this);
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
   return ts->clone();
}

void ClassTreeStep::gcMark( void* instance, uint32 mark ) const
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

bool ClassTreeStep::gcCheck( void* instance, uint32 mark ) const
{
   TRACE( "ClassTreeStep::gcCheck %p, %d ", instance, mark );
   
   // the check is performed on the topmost parent.
   TreeStep* ts = static_cast<TreeStep*>(instance);
   while( ts->parent() != 0 )
   {
      ts = ts->parent();
   }
   TRACE1( "ClassTreeStep::gcCheck %p, %d -- %s", instance, mark, ts->gcMark() >= mark ? "PASS" : "FAIL" );
   return ts->gcMark() >= mark;
}
 


         
void ClassTreeStep::enumerateProperties( void*, Class::PropertyEnumerator& cb ) const
{  
   cb("arity", false);
   cb("insert", false);
   cb("parent", false );
   cb("remove", true);
   
   //cb("append", true);
}


void ClassTreeStep::enumeratePV( void* instance, Class::PVEnumerator& cb ) const
{     
   Statement* stmt = static_cast<Statement*>(instance);
   Item temp = (int64) stmt->arity();
   
   cb("arity", temp );
   Expression* expr = stmt->selector();
   if( expr != 0 )
   {
      temp.setUser( expr->cls(), expr );
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
      || prop == "len_"
      || prop == "parent"
      || prop == "selector"
      || prop == "insert" 
      || prop == "remove";
}
   

void ClassTreeStep::op_getProperty( VMContext* ctx, void* instance, const String& prop) const
{
   TreeStep* stmt = static_cast<TreeStep*>(instance);
   if( prop == "len" )
   {
      ctx->topData().methodize( &m_lenMethod );
   }
   else if( prop == "insert" )
   {
      ctx->topData().methodize( &m_insertMethod );
   }
   else if( prop == "remove")
   {
      ctx->topData().methodize( &m_removeMethod );
   }
   else if( prop == "selector")
   {
      Expression* expr = stmt->selector();
      if( expr != 0 )
      {
         ctx->topData().setUser( expr->cls(), expr );
      }
      else {
         ctx->topData().setNil();
      }
   }
   else if( prop == "len_" || prop == "arity" )
   {
      ctx->topData().setInteger( (int64) stmt->arity() );
   }
   else if( prop == "parent" )
   {
      if( stmt->parent() == 0 )
         ctx->topData().setNil();
      else
         ctx->topData().setUser( stmt->parent()->cls(), stmt->parent() );
   }
   else
   {
      Class::op_getProperty( ctx, instance, prop );
   }
}


void ClassTreeStep::op_setProperty( VMContext* ctx, void* instance, const String& prop ) const
{
   TreeStep* stmt = static_cast<TreeStep*>(instance);
   
   if( prop == "selector" )
   {
      bool bCreate = true;
      if( ctx->opcodeParam(2).isNil() )
      {
         // set it and ingore the result, we don't care.
         stmt->selector(0);
      }
      else {
         Expression* expr = TreeStep::checkExpr( ctx->topData(), bCreate );      
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
      
      ctx->stackResult(3, ctx->topData());
   }
   else if( hasProperty( instance, prop) )
   {
      throw ropError( prop, __LINE__, SRC );
   }
   else
   {
      Class::op_setProperty( ctx, instance, prop );
   }
}


void ClassTreeStep::store( VMContext*, DataWriter* dw, void* instance ) const
{
   TreeStep* ts = static_cast<TreeStep*>( instance );
   // save the position
   dw->write( ts->line() );
   dw->write( ts->chr() );   
}

void ClassTreeStep::restore( VMContext*, DataReader*dr, void*& empty ) const
{
   TreeStep* ts = createInstance();
   int32 line, chr;
   dr->read( line );
   dr->read( chr );
   ts->decl( line, chr );
   empty = ts;
}

void ClassTreeStep::flatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   TreeStep* ts = static_cast<TreeStep*>( instance );
   subItems.resize( ts->arity() + 1 );
   if( ts->selector() != 0 )
   {
      subItems.at(0).setUser( ts->selector()->cls(), ts->selector() );
   }
   // else, let it be nil.
   
   if( ts->arity() > 0 )
   {
      for( int i = 0; i < ts->arity(); ++i )
      {
         TreeStep* element = ts->nth(i);
         if( element != 0 )
         {
            subItems[i+1].setUser( element->cls(), element );
         }
         // else, let it be nil.
      }
   }
   
}

void ClassTreeStep::unflatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   TreeStep* ts = static_cast<TreeStep*>( instance );
   for( int i = 0; i < (int) subItems.length(); ++i )
   {
      if( subItems[i].isUser() )
      {
         ts->nth(i, static_cast<TreeStep*>( subItems[i].asInst() ) );         
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
   Item& index = ctx->opcodeParam(1);
   if( index.isOrdinal() )
   {
      int64 num = ctx->opcodeParam(1).forceInteger();
      if( num < 0 ) num = stmt->arity() + num;
      if( num < 0 || num >= stmt->arity() )
      {
         throw new AccessError( ErrorParam( e_arracc, __LINE__, SRC )
            .origin(ErrorParam::e_orig_vm));
      }

      TreeStep* st = stmt->nth( num );
      if( st == 0 )
      {
         // nil might be valid for optional blocks
         ctx->stackResult( 2, Item() );
      }
      else {
         ctx->stackResult( 2, Item( st->cls(), st ) );
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
   TreeStep* self = static_cast<TreeStep*>(instance);
   Item& index = ctx->opcodeParam(1);
   if( index.isOrdinal() )
   {
      int64 num = ctx->opcodeParam(1).forceInteger();
      if( num < 0 ) num = self->arity() + num;
      if( num < 0 || num >= self->arity() )
      {
         throw new AccessError( ErrorParam( e_arracc, __LINE__, SRC )
            .origin(ErrorParam::e_orig_vm));
      }
      
      Item& i_tree = ctx->opcodeParam(2);
      if( i_tree.isNil() )
      {
         if( ! self->nth( num, 0 ) )
         {
            throw new CodeError( ErrorParam(e_invalid_op, __LINE__, SRC)
               .origin( ErrorParam::e_orig_vm)
               .extra( "Statement is not allowing an optional block" ) );
         }
      }
      else {
         Class* cls; void* inst;
         if( ! i_tree.asClassInst(cls, inst) 
               || ! cls->isDerivedFrom( this ) )
         {
            throw new AccessError( ErrorParam( e_inv_params, __LINE__, SRC )
               .origin(ErrorParam::e_orig_vm)
               .extra( "SynTree" ) );
         }


         TreeStep* ts = static_cast<TreeStep*>(inst);         
         // check TreeStep category.
         if( ! self->canHost( ts ) )
         {
            throw new CodeError( ErrorParam(e_invalid_op, __LINE__, SRC)
               .origin( ErrorParam::e_orig_vm)
               .extra( "Incompatible type of Step to be inserted here" ) );
         }
         
         if( ts->parent() != 0 )
         {
            throw new CodeError( ErrorParam(e_invalid_op, __LINE__, SRC)
               .origin( ErrorParam::e_orig_vm)
               .extra( "Parented entity cannot be inserted" ) );
         }

         if( ! self->nth( num, ts ) )
         {
            throw new CodeError( ErrorParam(e_invalid_op, __LINE__, SRC)
               .origin( ErrorParam::e_orig_vm)
               .extra( "Entity is not accepting that kind of element" ) );
         }
      }
      
      ctx->stackResult( 3, ctx->topData() );
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
         Item value(st->cls(), st);
         ctx->pushData( value );
         if( pos >= stmt->arity() )
         {
            ctx->topData().setLast();
         }
      }
   }
}


//===============================================================
// Len method
//
ClassTreeStep::LenMethod::LenMethod():
   Function("len")
{ 
}

ClassTreeStep::LenMethod::~LenMethod()
{}


void ClassTreeStep::LenMethod::invoke( VMContext* ctx, int32 )
{
   Item& self = ctx->self();
   fassert( self.isUser() );
      
   Statement* stmt = static_cast<Statement*>(self.asInst());
   ctx->returnFrame( (int64) stmt->arity() );
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
   
   ClassTreeStep* owner = static_cast<ClassTreeStep*>(methodOf());
   if( (! i_pos->isOrdinal()) ||
        !( i_treestep->asClassInst(cls, inst) && cls->isDerivedFrom(owner) ) 
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
   if( ! self_step->canHost( ts ) )
   {
      throw new CodeError( ErrorParam(e_invalid_op, __LINE__, SRC)
         .origin( ErrorParam::e_orig_vm)
         .extra( "Incompatible type of Step to be inserted here" ) );
   }

   if( ! self_step->insert( i_pos->forceInteger(), ts) ) {
      ctx->raiseError( new CodeError( ErrorParam(e_invalid_op, __LINE__, SRC)
      .origin( ErrorParam::e_orig_runtime)
      .extra( "This Statement is not accepting insert" ) ) );
      
      return;
   }
   
   ctx->returnFrame( pcount );
}

//===============================================================
// Insert method
//
ClassTreeStep::RemoveMethod::RemoveMethod():
   Function("remove")
{ 
   signature("N");
   addParam("pos");
   addParam("syntree");
}

ClassTreeStep::RemoveMethod::~RemoveMethod()
{}


void ClassTreeStep::RemoveMethod::invoke( VMContext* ctx, int32 pcount )
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
   if( ! self_step->remove( i_pos->forceInteger() ) ) {
      ctx->raiseError( new CodeError( ErrorParam(e_invalid_op, __LINE__, SRC)
      .origin( ErrorParam::e_orig_runtime)
      .extra( "This Statement is not accepting remove" ) ) );
      
      return;
   }
   
   ctx->returnFrame( pcount );
}

}
/* end of classtreestep.cpp */
