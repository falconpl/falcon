/*
   FALCON - The Falcon Programming Language.
   FILE: classcomposition.cpp

   A Functional composition. 
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 01 Apr 2012 16:36:55 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/classes/classcomposition.cpp"

#include <falcon/fassert.h>
#include <falcon/item.h>
#include <falcon/vmcontext.h>
#include <falcon/datawriter.h>
#include <falcon/datareader.h>
#include <falcon/itemarray.h>
#include <falcon/errors/paramerror.h>
#include <falcon/classes/classcomposition.h>

namespace Falcon {

class CompositionInstance
{
public:
   CompositionInstance():
      m_times( -1 ),
      m_mark(0)
   {}
      
   CompositionInstance( Item first, Item second ):
      m_first( first ),
      m_second( second ),
      m_times( -1 ),
      m_mark(0)
   {
   }
      
   CompositionInstance( Item first, int64 times ):
      m_first( first ),
      m_times( times ),
      m_mark(0)
   {
   }
   
   CompositionInstance( const CompositionInstance& other ):
      m_first( other.m_first ),
      m_second( other.m_second ),
      m_times( other.m_times ),
      m_mark(0)
   {
   }
      
   ~CompositionInstance() {}
   
   int64 times() const { return m_times; }
   Item& first() { return m_first; }
   Item& second() { return m_second; }
      
   void times( int64 t ) { m_times = t; }
   void first( const Item& item ) { m_first = item; }
   void second( const Item& item ) { m_second = item; }
   
   void gcMark( uint32 mark )
   {
      if( mark != m_mark )
      {
         m_mark = mark;
         first().gcMark( mark );
         second().gcMark( mark );
      }
   }
   
   uint32 gcMark() const { return m_mark; }
   
private:
   Item m_first;
   Item m_second;
   int64 m_times;
   
   uint32 m_mark;
};


ClassComposition::ClassComposition():
   Class("Composition")
{}

void* ClassComposition::createInstance() const
{ 
   return new CompositionInstance;
}

void ClassComposition::dispose( void* instance ) const
{
   CompositionInstance* ci = static_cast<CompositionInstance*>(instance);
   delete ci;
}

void* ClassComposition::clone( void* instance ) const
{
   CompositionInstance* ci = static_cast<CompositionInstance*>(instance);
   return new CompositionInstance(*ci);
}


void ClassComposition::store( VMContext*, DataWriter* stream, void* instance ) const
{
   CompositionInstance* ci = static_cast<CompositionInstance*>(instance);
   stream->write( ci->times() );
}

void ClassComposition::restore( VMContext* ctx, DataReader* stream ) const
{
   int64 times;
   stream->read( times );
   CompositionInstance* ci = new CompositionInstance;
   ci->times( times );
   ctx->pushData( Item( this, ci) );
}

void ClassComposition::flatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   CompositionInstance* ci = static_cast<CompositionInstance*>(instance);
   subItems.resize(2);
   subItems[0] = ci->first();
   subItems[1] = ci->second();
}

void ClassComposition::unflatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   fassert( subItems.length() == 2 );
   CompositionInstance* ci = static_cast<CompositionInstance*>(instance);
   ci->first( subItems[0] );
   ci->second( subItems[1] );   
}


void ClassComposition::gcMarkInstance( void* instance, uint32 mark ) const
{
   CompositionInstance* ci = static_cast<CompositionInstance*>(instance);
   ci->gcMark( mark );
}

bool ClassComposition::gcCheckInstance( void* instance, uint32 mark ) const
{
   CompositionInstance* ci = static_cast<CompositionInstance*>(instance);
   return ci->gcMark() >= mark;
}

void ClassComposition::enumerateProperties( void* instance, PropertyEnumerator& cb ) const
{
   CompositionInstance* ci = static_cast<CompositionInstance*>(instance);
   cb( "first" );
   if( ci->times() > -1 )
   {
      cb("times");
   }
   else {
      cb("second");
   }
}

void ClassComposition::enumeratePV( void* instance, PVEnumerator& cb ) const
{
   CompositionInstance* ci = static_cast<CompositionInstance*>(instance);
   cb( "first", ci->first() );
   if( ci->times() > -1 )
   {
      Item value = ci->times();
      cb("times", value );
      ci->times( value.forceInteger() );
   }
   else {
      cb("second", ci->second() );
   }
}

bool ClassComposition::hasProperty( void* instance, const String& prop ) const
{
   CompositionInstance* ci = static_cast<CompositionInstance*>(instance);
   if( prop == "first" )
   {
      return true;
   }
   
   if ( ci->times() < 0 && prop == "second" )
   {
      return true;
   }
   
   if( ci->times() >= 0 && prop == "times" )
   {
      return true;
   }
   
   return false;
}

void ClassComposition::describe( void* instance, String& target, int depth, int maxlen ) const
{
   CompositionInstance* ci = static_cast<CompositionInstance*>(instance);
   
   target = "Composition{ first=";
   String temp;
   Class* cls = 0;
   void* inst = 0;
   ci->first().forceClassInst( cls, inst );
   cls->describe( inst, temp, depth-1, maxlen );
   target += temp;
   
   if( ci->times() >= 0 )
   {
      target += ", times=";
      target.N(ci->times());
   }
   else
   {
      target += ", second=";
      ci->second().forceClassInst( cls, inst );
      cls->describe( inst, temp, depth-1, maxlen );
      target += temp;
   }
   
   target += " }";
}

bool ClassComposition::op_init( VMContext* ctx, void* instance, int32 pcount ) const
{
   if( pcount < 2 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__, SRC )
         .extra("C,C|N"));
   }
   
   CompositionInstance* ci = static_cast<CompositionInstance*>(instance);
   Item* params = ctx->opcodeParams(pcount);
   Item& first = params[0];
   Item& second = params[1];
  
   if( second.isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params,  __LINE__, SRC )
         .extra("C,C|N"));
      
   }
   
   ci->first( first );
   if( second.isOrdinal() )
   {
      ci->times( second.asInteger() );
   }
   else {
      ci->second( second );
   }
   
   ctx->stackResult( pcount + 1, Item( this, ci ) );
   return false;
}


void* ClassComposition::createComposition( const Item& first, const Item& second ) const
{   
   CompositionInstance* ci = new CompositionInstance;
   ci->first( first );
   if( second.isOrdinal() )
   {
      ci->times( second.asInteger() );
   }
   else {
      ci->second( second );
   }
   
   return ci;
}

void ClassComposition::op_getProperty( VMContext* ctx, void* instance, const String& prop) const
{
   CompositionInstance* ci = static_cast<CompositionInstance*>(instance);
   if( prop == "first" )
   {
      ctx->topData() = ci->first();
   }
   else if( prop == "second" && ci->times() < 0 ) 
   {
      ctx->topData() = ci->second();
   }
   else if( prop == "times" && ci->times() >= 0 )
   {
      ctx->topData() = ci->times();
   }
   else {
      Class::op_getProperty( ctx, instance, prop );
   }
}

void ClassComposition::op_setProperty( VMContext* ctx, void* instance, const String& prop ) const
{
   CompositionInstance* ci = static_cast<CompositionInstance*>(instance);
   ctx->popData();
   if( prop == "first" )
   {
      ci->first().assignFromLocal(ctx->topData());
   }
   else if( prop == "second" && ci->times() < 0 ) 
   {
      ci->second().assignFromLocal(ctx->topData());
   }
   else if( prop == "times" && ci->times() >= 0 )
   {
      if( ! ctx->topData().isOrdinal() )
      {
         throw new ParamError( ErrorParam( e_inv_prop_value, __LINE__, SRC )
            .extra("N"));
      }
      ci->times(ctx->topData().forceInteger());
   }
   else {
      Class::op_setProperty( ctx, instance, prop );
   }
}


void ClassComposition::op_call( VMContext* ctx, int32 paramCount, void* instance ) const
{
   CompositionInstance* ci = static_cast<CompositionInstance*>(instance);
   
   if( ci->times() == 0 )
   {
      Item topCopy = *ctx->opcodeParams(paramCount);
      ctx->stackResult( paramCount + 1, topCopy );
      return;
   }
   else if( ci->times() < 0 )
   {
      *ctx->opcodeParams(paramCount+1) = ci->second();
      Class* cls;
      void* data;
      ci->second().forceClassInst( cls, data );

      ctx->insertData(paramCount+1, &ci->first(), 1, 0 );

      ctx->pushCode( &m_applyFirst );
      ctx->currentCode().m_seqId = 1;

      cls->op_call( ctx, paramCount, data );
   }
   else
   {
      Class* cls;
      void* data;
      *ctx->opcodeParams(paramCount+1) = ci->first();
      ci->first().forceClassInst( cls, data );

      if ( ci->times() > 1 )
      {
         ctx->insertData(paramCount+1, &ci->first(), 1, 0 );
         ctx->pushCode( &m_applyFirst );
         ctx->currentCode().m_seqId = (int) ci->times()-1;
      }
      
      cls->op_call( ctx, paramCount, data );
   }
}


ClassComposition::ApplyFirst::ApplyFirst()
{
   apply = apply_;
}

void ClassComposition::ApplyFirst::apply_( const PStep*, VMContext* ctx )
{
   CodeFrame& cf = ctx->currentCode();
   
   if( cf.m_seqId == 0 )
   {
      ctx->popCode();
      ctx->opcodeParam(1) = ctx->topData();
      ctx->popData();
      return;
   }
   
   cf.m_seqId--;
   Item& callee = ctx->opcodeParam(1);
   Class* cls;
   void* udata;
   callee.forceClassInst( cls, udata );
   Item topData = ctx->topData();
   ctx->topData() = callee;
   ctx->pushData( topData );
   
   cls->op_call( ctx, 1, udata );
}


   
}

/* end of classcomposition.cpp */
