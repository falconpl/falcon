/*
   FALCON - The Falcon Programming Language.
   FILE: generator.cpp

   Falcon core module -- Generator class to help iterating
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 30 Mar 2013 17:59:00 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/cm/generator.h>
#include <falcon/datareader.h>
#include <falcon/datawriter.h>
#include <falcon/vmcontext.h>
#include <falcon/itemarray.h>

#include <falcon/stderrors.h>

namespace Falcon {
namespace Ext {


class Generator {
public:
   Item m_function;
   Item m_data;
   Item m_iterator;
   Item m_next;

   bool m_bHasCache;
   bool m_bComplete;
   bool m_bMore;

   uint32 m_mark;

   Generator():
      m_bHasCache(false),
      m_bComplete(false),
      m_bMore(false),
      m_mark(0)
   {}

   Generator(const Generator& o)
   {
      m_function = o.m_function;
      m_data = o.m_data;
      m_iterator = o.m_iterator;
      m_next = o.m_next;

      m_bHasCache = o.m_bHasCache;
      m_bComplete = o.m_bComplete;
      m_bMore = o.m_bMore;
      m_mark = 0;
   }

};

ClassGenerator::ClassGenerator():
         Class("Generator")
{
}

ClassGenerator::~ClassGenerator()
{
}


void* ClassGenerator::createInstance() const
{
   return new Generator;
}


void ClassGenerator::dispose( void* instance ) const
{
   Generator* gen = static_cast<Generator*>(instance);
   delete gen;
}


void* ClassGenerator::clone( void* instance ) const
{
   Generator* gen = static_cast<Generator*>(instance);
   return new Generator(*gen);
}


void ClassGenerator::enumerateProperties( void*, PropertyEnumerator& cb ) const
{
   //Generator* gen = static_cast<Generator*>(instance);
   cb("hasNext");
   cb("next");
   cb("iterator");
   cb("func");
   cb("data");
}


bool ClassGenerator::hasProperty( void*, const String& prop ) const
{
   return
            prop == "hasNext"
         || prop == "next"
         || prop == "func"
         || prop == "data"
         || prop == "iterator";
}


void ClassGenerator::gcMarkInstance( void* instance, uint32 mark ) const
{
   Generator* gen = static_cast<Generator*>(instance);
   if( mark != gen->m_mark )
   {
      gen->m_mark = mark;
      gen->m_function.gcMark(mark);
      gen->m_data.gcMark(mark);
      gen->m_iterator.gcMark(mark);
      gen->m_next.gcMark(mark);
   }
}


bool ClassGenerator::gcCheckInstance( void* instance, uint32 mark ) const
{
   Generator* gen = static_cast<Generator*>(instance);
   return gen->m_mark >= mark;

}


void ClassGenerator::store( VMContext*, DataWriter* dw, void* instance ) const
{
   Generator* gen = static_cast<Generator*>(instance);
   dw->write( gen->m_bComplete );
   dw->write( gen->m_bHasCache );
   dw->write( gen->m_bMore );
}

void ClassGenerator::restore( VMContext* ctx, DataReader* dr) const
{
   bool bHasFirst = false;
   bool bHasNext = false;
   bool bMore = false;
   dr->read( bHasFirst );
   dr->read( bHasNext );
   dr->read( bMore );

   Generator* gen = new Generator;
   gen->m_bComplete = bHasFirst;
   gen->m_bHasCache = bHasNext;
   gen->m_bMore = bMore;
   ctx->pushData( Item(this, gen) );
}


void ClassGenerator::flatten( VMContext*, ItemArray& arr, void* instance ) const
{
   Generator* gen = static_cast<Generator*>(instance);
   arr.resize(4);
   arr[0] = gen->m_function;
   arr[1] = gen->m_data;
   arr[2] = gen->m_iterator;
   arr[3] = gen->m_next;

}


void ClassGenerator::unflatten( VMContext*, ItemArray& arr, void* instance ) const
{
   Generator* gen = static_cast<Generator*>(instance);
   fassert( arr.length() == 4 );

   gen->m_function = arr[0];
   gen->m_data = arr[1];
   gen->m_iterator = arr[2];
   gen->m_next = arr[3];
}



bool ClassGenerator::op_init( VMContext* ctx, void* instance, int pcount ) const
{
   Generator* gen = static_cast<Generator*>(instance);

   if ( pcount < 2 )
   {
      throw FALCON_SIGN_XERROR( ParamError, e_inv_params, .extra("C,X,[X]") );
   }

   Item* params = ctx->opcodeParams(pcount);
   if( pcount >= 3 )
   {
      gen->m_iterator = params[2];
   }

   gen->m_function = params[0];
   gen->m_data = params[1];

   return false;
}


void ClassGenerator::op_getProperty( VMContext* ctx, void* instance, const String& prop) const
{
   Generator* gen = static_cast<Generator*>(instance);

   if( prop == "hasNext" )
   {
      if( ! gen->m_bHasCache && ! gen->m_bComplete )
      {
         ctx->pushData( Item(this, gen) );
         ctx->pushCode(&m_stepAfterHasNext);
         Item params[] = {gen->m_data, Item(this, gen) };
         ctx->callItem(gen->m_function, 2, params );
      }
      else {
         ctx->topData().setBoolean(! gen->m_bComplete);
      }
   }
   else if( prop == "next" )
   {
      if( gen->m_bComplete )
      {
         throw FALCON_SIGN_XERROR( AccessError, e_acc_forbidden, .extra("No more items"));
      }
      else if( ! gen->m_bMore )
      {
         gen->m_bComplete = true;
         ctx->topData() =  gen->m_next;
      }
      else {
         ctx->pushData( Item(this, gen) );
         ctx->pushCode(&m_stepAfterNext);
         Item params[] = {gen->m_data, Item(this, gen) };
         ctx->callItem(gen->m_function, 2, params );
      }
   }
   else if( prop == "func" )
   {
      ctx->topData() = gen->m_function;
   }
   else if( prop == "data" )
   {
      ctx->topData() = gen->m_data;
   }
   else if( prop == "iterator" )
   {
      ctx->topData() = gen->m_iterator;
   }
   else
   {
      Class::op_getProperty(ctx, instance, prop);
   }

}

void ClassGenerator::op_setProperty( VMContext* ctx, void* instance, const String& prop ) const
{
   Generator* gen = static_cast<Generator*>(instance);
   if( prop == "iterator" )
   {
      ctx->popData();
      gen->m_iterator = ctx->topData();
   }
   else
   {
      Class::op_setProperty(ctx, instance, prop);
   }
}


void ClassGenerator::op_iter( VMContext* ctx, void* instance ) const
{
   ctx->pushData( Item(this, instance) );
}

void ClassGenerator::op_next( VMContext* ctx, void* instance ) const
{
   Generator* gen = static_cast<Generator*>(instance);
   Item params[] = {gen->m_data, Item(this, gen) };
   ctx->callerLine(__LINE__+1);
   ctx->callItem(gen->m_function, 2, params );
}


void ClassGenerator::PStepAfterHasNext::apply_( const PStep*, VMContext* ctx )
{
   ctx->popCode();

   // we have called the function, do we have a next?
   Item next = ctx->topData();
   ctx->popData();
   Generator* gen = static_cast<Generator*>(ctx->topData().asInst());
   gen->m_next = next;
   gen->m_bHasCache = ! next.isBreak();
   gen->m_bComplete = next.isBreak();
   gen->m_bMore = next.isDoubt();

   ctx->popData();
   ctx->topData().setBoolean(!gen->m_bComplete);
}


void ClassGenerator::PStepAfterNext::apply_( const PStep*, VMContext* ctx )
{
   ctx->popCode();

   // we have called the function, do we have a next?
   Item next = ctx->topData();
   ctx->popData();
   Generator* gen = static_cast<Generator*>(ctx->topData().asInst());
   ctx->popData();

   if( gen->m_bHasCache )
   {
      ctx->topData() = gen->m_next;
      if( next.isBreak() )
      {
         gen->m_bMore = false;
         gen->m_bHasCache = false;
      }
      else if( ! next.isDoubt() ){

         gen->m_bMore = false;
         gen->m_bHasCache = true;
      }
      else {
         gen->m_bMore = true;
         gen->m_bHasCache = true;
      }

      gen->m_next = next;
   }
   else {
      if( next.isBreak() )
      {
         throw FALCON_SIGN_XERROR( AccessError, e_acc_forbidden, .extra("No more items"));
      }
      else {
         ctx->topData() = next;
         gen->m_bMore = ! next.isDoubt();
      }
   }
}

}
}
/* end of generator.cpp */
